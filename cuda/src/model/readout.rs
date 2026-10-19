use super::*;
use blckbaux_semantics::hnet::{CausalPlan, LossEstimate, ObjectiveStats, PairShape, WaveShape};

pub(super) struct HnetState {
    fine: DeviceBuffer<u32>,
    expected: DeviceBuffer<u32>,
    seeds: DeviceBuffer<u64>,
    centroids: DeviceBuffer<u16>,
    output: DeviceBuffer<u32>,
    losses: DeviceBuffer<f32>,
    patch_nlls: DeviceBuffer<f32>,
    stats: DeviceBuffer<ObjectiveStats>,
    baseline: DeviceBuffer<f32>,
    patch_gains: DeviceBuffer<f32>,
    paired: bool,
    pub(super) index_incumbent: DeviceBuffer<u32>,
    pub(super) index_candidate: DeviceBuffer<u32>,
    pub(super) index_enabled: bool,
    index_ready: bool,
    pub(super) index_shadow: bool,
    index_force_full: bool,
    blocks: usize,
    tokens: usize,
    proposal_start: usize,
    score: bool,
    generation_graph: Option<(u32, crate::graph::CudaGraph)>,
}

pub(super) struct HnetEvaluation {
    pub stats: Option<ScoreStats>,
    /// One mean target NLL per causal H-Net patch. Paired differences provide
    /// acceptance and reliability evidence; they do not train a proposal model.
    pub patch_nlls: Option<Vec<f32>>,
    pub estimate: Option<LossEstimate>,
    pub patch_gains: Option<Vec<f32>>,
    pub device_ms: f32,
    pub core_ms: f32,
    pub generation_ms: f32,
    pub objective_ms: f32,
    pub wall_ms: f32,
    pub visits: usize,
    pub core_profile: Option<WaveProfile>,
    pub generation_profile: Option<HnetGenerationProfile>,
}

pub(super) struct HnetGenerationProfile {
    pub init_ms: f32,
    pub gemm_ms: f32,
    pub project_ms: f32,
    pub reduce_ms: f32,
    pub total_ms: f32,
}

#[derive(Clone, Copy)]
enum HnetGenerationStage {
    Init,
    Gemm,
    Project,
    Reduce,
}

impl FbtModel {
    pub(super) fn initialized_trace(
        &mut self,
    ) -> CudaResult<blckbaux_artifact::forward::ForwardTrace> {
        use blckbaux_artifact::forward::ForwardTrace;
        use blckbaux_semantics::hnet::probe::initialized as case;
        let weights = self.weights.logical_hashes(&self.stream)?;
        let sequence = case::sequence();
        let batch = blckbaux_artifact::pretrain::CausalBatch {
            tokens: sequence[..case::TOKENS].to_vec(),
            labels: sequence[1..].to_vec(),
            loss_mask: vec![1; case::TOKENS],
            offset: 0,
        };
        self.configure_causal(&batch)?;
        self.evaluate_hnet(2, 0.0, blckbaux_semantics::random::splitmix64(0), false)?;
        let state = self
            .hnet
            .as_ref()
            .ok_or("CUDA initialized trace lost its state")?;
        let trace = ForwardTrace {
            format: ForwardTrace::FORMAT.into(),
            seed: case::SEED,
            input: batch.tokens,
            labels: batch.labels,
            rows: case::ROWS,
            width: 512,
            weights,
            encoded: read_prefix(&self.scratch.embedded, &self.stream, case::WORDS)?,
            hidden: read_prefix(&self.scratch.normalized, &self.stream, case::WORDS)?,
            predictions: read_prefix(&state.output, &self.stream, case::TARGETS)?,
            loss_bits: read_prefix(&state.losses, &self.stream, case::TARGETS)?
                .iter()
                .map(|v| v.to_bits())
                .collect(),
        };
        trace.validate()?;
        Ok(trace)
    }

    /// Finite lifecycle fixture through the production encoder, core, readout,
    /// loss capture and graph replay. Uniform fixture weights have uniform
    /// target likelihood; this is not an optimizer or learning experiment.
    pub(super) fn check_causal(&mut self) -> CudaResult<()> {
        self.weights.hnet = Some(super::weights::HnetWeights {
            mask: DeviceBuffer::zeroed(&self.stream, 512).map_err(cuda_error)?,
            positions: DeviceBuffer::zeroed(&self.stream, 64 * 512).map_err(cuda_error)?,
        });
        for (index, length) in [128, 128, 64].into_iter().enumerate() {
            let mut batch = blckbaux_artifact::pretrain::CausalBatch {
                tokens: vec![index as u32; length],
                labels: vec![u32::from(index != 0); length],
                loss_mask: vec![1; length],
                offset: index * length,
            };
            self.configure_causal(&batch)?;
            if index == 1
                && self
                    .hnet
                    .as_ref()
                    .is_none_or(|s| s.generation_graph.is_none())
            {
                return Err("same-shape corpus switch discarded its readout graph".into());
            }
            let first = self.evaluate_hnet(2, 0.0, 17, false)?;
            let estimate = first.estimate.ok_or("causal fixture omitted estimate")?;
            let stats = first.stats.ok_or("causal fixture omitted counts")?;
            let targets = length - 63;
            if estimate.improvement.is_some()
                || first.patch_gains.is_some()
                || stats.tokens as usize != targets
                || stats.mismatches as usize != if index == 0 { 0 } else { targets }
                || (estimate.mean - (VOCAB as f32).ln()).abs() > 0.001
            {
                return Err(format!(
                    "causal fixture differs at condition {index}: mean={} tokens={} mismatches={}",
                    estimate.mean, stats.tokens, stats.mismatches
                ));
            }
            self.capture_hnet_pair()?;
            let repeat = self.evaluate_hnet(2, 0.0, 17, false)?;
            let paired = repeat.estimate.ok_or("causal repeat omitted estimate")?;
            if paired.mean.to_bits() != estimate.mean.to_bits()
                || paired.improvement != Some(0.0)
                || paired.paired_variance != Some(0.0)
                || repeat
                    .patch_gains
                    .as_ref()
                    .is_none_or(|v| v.len() != targets.div_ceil(64) || v.iter().any(|&v| v != 0.0))
            {
                return Err(
                    "causal repeated condition changed or lost token-aligned capture".into(),
                );
            }
            self.clear_hnet_pair()?;
            batch.loss_mask[0] = 0;
            if self.configure_causal(&batch).is_ok() {
                return Err("causal fixture accepted an unsupported loss mask".into());
            }
        }
        println!(
            "HNET_CAUSAL_PARITY ok=true targets=65,65,1 graph_reused=true labels_reloaded=true capture_reset=true identical_pair=exact"
        );
        Ok(())
    }

    pub(super) fn configure_hnet(
        &mut self,
        prompt: &[u32],
        generated: usize,
        targets: Option<&[u32]>,
    ) -> CudaResult<()> {
        if self.weights.hnet.is_none()
            || prompt.is_empty()
            || prompt.len() % 64 != 0
            || generated == 0
            || generated % 64 != 0
            || prompt
                .len()
                .checked_add(generated)
                .is_none_or(|tokens| tokens > self.rows * 64)
            || prompt.iter().any(|&token| token >= VOCAB as u32)
            || targets.is_some_and(|values| {
                values.len() != generated || values.iter().any(|&token| token >= VOCAB as u32)
            })
        {
            return Err("CUDA H-Net configuration requires matching learned H-Net weights and aligned vocabulary-valid 64-token patches".into());
        }
        let shape =
            blckbaux_semantics::hnet::shape(prompt.len(), generated, 0.0, targets.is_some())?;
        self.configure_window(prompt, targets, shape)
    }

    /// Score reference corpus tokens without feeding sampled tokens back into
    /// the scored state. The preceding complete patch determines label offset.
    pub(super) fn configure_causal(
        &mut self,
        batch: &blckbaux_artifact::pretrain::CausalBatch,
    ) -> CudaResult<()> {
        let plan = CausalPlan::new(
            batch.tokens.len(),
            batch.labels.len(),
            &batch.loss_mask,
            self.rows,
        )?;
        self.configure_window(
            &batch.tokens,
            Some(&batch.labels[plan.start..]),
            plan.wave()?,
        )
    }

    fn configure_window(
        &mut self,
        input: &[u32],
        targets: Option<&[u32]>,
        shape: WaveShape,
    ) -> CudaResult<()> {
        let generated = shape.tokens as usize;
        if self.weights.hnet.is_none()
            || input.len() > self.rows * 64
            || input.iter().any(|&t| t >= VOCAB as u32)
            || targets.is_some_and(|values| {
                values.len() != generated || values.iter().any(|&t| t >= VOCAB as u32)
            })
        {
            return Err("CUDA H-Net window has invalid capacity, weights or token IDs".into());
        }
        let blocks = shape.blocks as usize;
        let proposal_start = shape.proposal_start as usize;
        let mut fine = vec![u32::MAX; self.rows * 64];
        fine[..input.len()].copy_from_slice(input);
        let expected = targets.map_or_else(|| vec![0; generated], <[u32]>::to_vec);
        if let Some(state) = self.hnet.as_mut().filter(|state| {
            state.tokens == generated
                && state.proposal_start == proposal_start
                && state.score == targets.is_some()
                && state.fine.len() == fine.len()
        }) {
            copy_prefix(&state.fine, &fine, &self.stream)?;
            copy_prefix(&state.expected, &expected, &self.stream)?;
            state.paired = false;
            state.index_ready = false;
            state.index_shadow = false;
            state.index_force_full = false;
            // Kernel addresses and shapes are unchanged. Captured readout
            // graphs read the newly uploaded data and the current weights.
            return Ok(());
        }
        self.hnet = Some(HnetState {
            fine: DeviceBuffer::from_host(&self.stream, &fine).map_err(cuda_error)?,
            expected: DeviceBuffer::from_host(&self.stream, &expected).map_err(cuda_error)?,
            seeds: DeviceBuffer::zeroed(&self.stream, 1).map_err(cuda_error)?,
            centroids: DeviceBuffer::zeroed(&self.stream, 37_448 * 64).map_err(cuda_error)?,
            output: DeviceBuffer::zeroed(&self.stream, generated).map_err(cuda_error)?,
            losses: DeviceBuffer::zeroed(&self.stream, generated).map_err(cuda_error)?,
            patch_nlls: DeviceBuffer::zeroed(&self.stream, blocks).map_err(cuda_error)?,
            stats: DeviceBuffer::zeroed(&self.stream, 1).map_err(cuda_error)?,
            baseline: DeviceBuffer::zeroed(&self.stream, generated).map_err(cuda_error)?,
            patch_gains: DeviceBuffer::zeroed(&self.stream, blocks).map_err(cuda_error)?,
            paired: false,
            index_incumbent: DeviceBuffer::zeroed(&self.stream, self.rows * 8 * 11)
                .map_err(cuda_error)?,
            index_candidate: DeviceBuffer::zeroed(&self.stream, self.rows * 8 * 11)
                .map_err(cuda_error)?,
            index_enabled: std::env::var_os("BLCKBAUX_CUDA_PISA_SHADOW").is_some(),
            index_ready: false,
            index_shadow: false,
            index_force_full: false,
            blocks,
            tokens: generated,
            proposal_start,
            score: targets.is_some(),
            generation_graph: None,
        });
        Ok(())
    }

    pub(super) fn clear_hnet_pair(&mut self) -> CudaResult<()> {
        self.hnet
            .as_mut()
            .ok_or("CUDA H-Net state is not configured")?
            .paired = false;
        Ok(())
    }

    /// Retain aligned incumbent token losses on the same ordered stream.
    pub(super) fn capture_hnet_pair(&mut self) -> CudaResult<()> {
        let state = self
            .hnet
            .as_mut()
            .ok_or("CUDA H-Net state is not configured")?;
        if !state.score {
            return Err("CUDA paired capture requires reference targets".into());
        }
        unsafe {
            cuda_core::simt::memory::memcpy_dtod_async(
                state.baseline.cu_deviceptr(),
                state.losses.cu_deviceptr(),
                state.tokens * size_of::<f32>(),
                self.stream.cu_stream(),
            )
            .map_err(cuda_error)?;
        }
        state.paired = true;
        Ok(())
    }

    pub(super) fn evaluate_hnet(
        &mut self,
        visits: usize,
        temperature: f32,
        seed: u64,
        profile: bool,
    ) -> CudaResult<HnetEvaluation> {
        if !temperature.is_finite() || temperature < 0.0 {
            return Err("CUDA H-Net temperature must be finite and nonnegative".into());
        }
        let (blocks, tokens, proposal_start, score, paired) = self
            .hnet
            .as_ref()
            .map(|state| {
                (
                    state.blocks,
                    state.tokens,
                    state.proposal_start,
                    state.score,
                    state.paired,
                )
            })
            .ok_or("CUDA H-Net state is not configured")?;
        let blocks_u32 = u32::try_from(blocks).map_err(cuda_error)?;
        let tokens_u32 = u32::try_from(tokens).map_err(cuda_error)?;
        let plan = recurrence::RecurrentCore::selective_fbt().layer_visits(5, visits)?;
        if plan.len() > 11 {
            return Err("CUDA H-Net PISA state does not cover the recurrent plan".into());
        }
        let index_shadow = {
            let state = self.hnet.as_mut().ok_or("CUDA H-Net state disappeared")?;
            state.index_shadow =
                state.index_enabled && state.index_ready && !state.index_force_full;
            state.index_shadow
        };
        self.prepare_hnet_cache(plan.len())?;
        let encode_shape = HnetEncodeShape {
            rows: self.rows as u32,
            width: 512,
            vocabulary: VOCAB as u32,
            patch: 64,
        };
        let encode_launch = self
            .module
            .prepare_hnet_encode(LaunchConfig1D::new(
                (encode_shape.rows * 512).div_ceil(256),
                256,
                0,
            ))
            .map_err(cuda_error)?;
        let replicate_launch = self
            .module
            .prepare_mhc_replicate(LaunchConfig1D::new(
                (encode_shape.rows * 512).div_ceil(256),
                256,
                0,
            ))
            .map_err(cuda_error)?;
        let hierarchy_launch = self
            .module
            .prepare_hnet_hierarchy_centroids(LaunchConfig1D::new(
                (37_448_u32 * 64).div_ceil(256),
                256,
                0,
            ))
            .map_err(cuda_error)?;
        let expand_launch = self
            .module
            .prepare_hnet_hierarchical_expand(LaunchConfig1D::new(blocks_u32, 64, 0))
            .map_err(cuda_error)?;
        let objective_launch = score
            .then(|| {
                self.module
                    .prepare_hnet_objective(LaunchConfig1D::new(1, 256, 0))
                    .map_err(cuda_error)
            })
            .transpose()?;
        let patch_launch = score
            .then(|| {
                self.module
                    .prepare_patch_mean_nll(LaunchConfig1D::new(blocks_u32, 64, 0))
                    .map_err(cuda_error)
            })
            .transpose()?;
        copy_prefix(
            &self
                .hnet
                .as_ref()
                .ok_or("CUDA H-Net state disappeared")?
                .seeds,
            &[seed],
            &self.stream,
        )?;
        let wall = Instant::now();
        let start = timing_event(&self.stream)?;
        {
            let state = self.hnet.as_ref().ok_or("CUDA H-Net state disappeared")?;
            let weights = self
                .weights
                .hnet
                .as_ref()
                .ok_or("checkpoint missing H-Net mask/index tensors")?;
            self.module
                .hnet_encode(
                    &self.stream,
                    &encode_launch,
                    &self.weights.readout,
                    &state.fine,
                    &weights.positions,
                    &weights.mask,
                    &mut self.scratch.embedded,
                    encode_shape,
                )
                .map_err(cuda_error)?;
        }
        self.module
            .mhc_replicate(
                &self.stream,
                &replicate_launch,
                &self.scratch.embedded,
                &mut self.scratch.streams[0],
                self.rows as u32,
            )
            .map_err(cuda_error)?;
        let mut core_events = Vec::new();
        let visits = if profile {
            let mut events = Some(&mut core_events);
            mark(&mut events, &self.stream, Stage::Embed)?;
            let visits = self.enqueue_core(&plan, temperature, seed, events, None, false)?;
            let mut events = Some(&mut core_events);
            mark(&mut events, &self.stream, Stage::Update)?;
            visits
        } else {
            self.enqueue_core(&plan, temperature, seed, None, None, false)?
        };
        let core_end = timing_event(&self.stream)?;
        if let Some(state) = self.hnet.as_mut() {
            if state.index_enabled && !index_shadow {
                state.index_ready = true;
            }
        }
        let mut generation_events = Vec::new();
        {
            let state = self.hnet.as_mut().ok_or("CUDA H-Net state disappeared")?;
            let weights = self
                .weights
                .hnet
                .as_ref()
                .ok_or("checkpoint missing H-Net mask/index tensors")?;
            let prompt_tokens = proposal_start
                .checked_add(1)
                .and_then(|patches| patches.checked_mul(blckbaux_semantics::hnet::PATCH))
                .ok_or("H-Net prompt size overflow")?;
            let shape = blckbaux_semantics::hnet::shape(
                prompt_tokens,
                tokens_u32 as usize,
                temperature,
                score,
            )?;
            let graph_key = temperature.to_bits();
            let cached = (!profile)
                .then(|| state.generation_graph.take())
                .flatten()
                .filter(|(key, _)| *key == graph_key);
            if let Some(graph) = cached {
                graph.1.launch()?;
                state.generation_graph = Some(graph);
            } else {
                let mut record = || -> CudaResult<()> {
                    self.module
                        .hnet_hierarchy_centroids(
                            &self.stream,
                            &hierarchy_launch,
                            &self.weights.readout,
                            &mut state.centroids,
                            shape,
                        )
                        .map_err(cuda_error)?;
                    if profile {
                        generation_events
                            .push((HnetGenerationStage::Gemm, timing_event(&self.stream)?));
                    }
                    self.module
                        .hnet_hierarchical_expand(
                            &self.stream,
                            &expand_launch,
                            &self.scratch.normalized,
                            &self.weights.layers[0].qkv,
                            &weights.positions,
                            &self.weights.readout,
                            &state.centroids,
                            &state.seeds,
                            &state.expected,
                            &mut state.output,
                            &mut state.losses,
                            shape,
                        )
                        .map_err(cuda_error)?;
                    if profile {
                        generation_events
                            .push((HnetGenerationStage::Reduce, timing_event(&self.stream)?));
                    }
                    Ok(())
                };
                if profile {
                    record()?;
                } else {
                    let graph = crate::graph::CudaGraph::capture(
                        self.context.clone(),
                        self.stream.clone(),
                        record,
                    )?;
                    graph.launch()?;
                    state.generation_graph = Some((graph_key, graph));
                }
            }
        }
        let generation_end = timing_event(&self.stream)?;
        if let Some(objective_launch) = objective_launch.as_ref() {
            let state = self.hnet.as_mut().ok_or("CUDA H-Net state disappeared")?;
            self.module
                .hnet_objective(
                    &self.stream,
                    objective_launch,
                    &state.losses,
                    &state.output,
                    &state.expected,
                    &mut state.stats,
                    &state.baseline,
                    PairShape {
                        tokens: tokens_u32,
                        paired: u32::from(paired),
                        batch_tokens: 64,
                    },
                    &mut state.patch_gains,
                )
                .map_err(cuda_error)?;
            self.module
                .patch_mean_nll(
                    &self.stream,
                    patch_launch
                        .as_ref()
                        .ok_or("CUDA H-Net patch reduction launch is missing")?,
                    &state.losses,
                    &mut state.patch_nlls,
                    tokens_u32,
                )
                .map_err(cuda_error)?;
        }
        let end = timing_event(&self.stream)?;
        let raw = if score {
            Some(
                read_prefix(
                    &self
                        .hnet
                        .as_ref()
                        .ok_or("CUDA H-Net state disappeared")?
                        .stats,
                    &self.stream,
                    1,
                )?[0],
            )
        } else {
            None
        };
        let estimate = raw
            .as_ref()
            .map(|raw| {
                raw.estimate(PairShape {
                    tokens: tokens_u32,
                    paired: u32::from(paired),
                    batch_tokens: 64,
                })
            })
            .transpose()
            .map_err(str::to_string)?;
        let stats = raw
            .as_ref()
            .zip(estimate.as_ref())
            .map(|(raw, estimate)| ScoreStats {
                reward: -estimate.mean,
                mean_nll: estimate.mean,
                maximum_nll: raw.maximum,
                positional_accuracy: raw.matches as f32 / tokens as f32,
                tokens: tokens_u32,
                mismatches: raw.mismatches,
                patch_nll_octiles: raw.patch_nll_octiles,
            });
        let patch_nlls = if score {
            Some(read_prefix(
                &self
                    .hnet
                    .as_ref()
                    .ok_or("CUDA H-Net state disappeared")?
                    .patch_nlls,
                &self.stream,
                blocks,
            )?)
        } else {
            None
        };
        let patch_gains = if paired {
            Some(read_prefix(
                &self
                    .hnet
                    .as_ref()
                    .ok_or("CUDA H-Net state disappeared")?
                    .patch_gains,
                &self.stream,
                blocks,
            )?)
        } else {
            None
        };
        self.context.check_err().map_err(cuda_error)?;
        let core_profile = profile
            .then(|| WaveProfile::from_events(&start, &core_events))
            .transpose()?;
        let generation_profile = profile
            .then(|| HnetGenerationProfile::from_events(&core_end, &generation_events))
            .transpose()?;
        Ok(HnetEvaluation {
            stats,
            patch_nlls,
            estimate,
            patch_gains,
            device_ms: start.elapsed_ms(&end).map_err(cuda_error)?,
            core_ms: start.elapsed_ms(&core_end).map_err(cuda_error)?,
            generation_ms: core_end.elapsed_ms(&generation_end).map_err(cuda_error)?,
            objective_ms: generation_end.elapsed_ms(&end).map_err(cuda_error)?,
            wall_ms: wall.elapsed().as_secs_f32() * 1000.0,
            visits,
            core_profile,
            generation_profile,
        })
    }

    pub(super) fn commit_hnet_index(&mut self, accepted: bool) {
        if let Some(state) = self.hnet.as_mut() {
            if state.index_enabled && state.index_shadow && accepted {
                std::mem::swap(&mut state.index_incumbent, &mut state.index_candidate);
            }
            state.index_shadow = false;
        }
    }

    pub(super) fn force_full_hnet_index(&mut self) -> CudaResult<()> {
        let state = self
            .hnet
            .as_mut()
            .ok_or("CUDA H-Net state is not configured")?;
        if !state.index_enabled || !state.index_ready {
            return Err("CUDA H-Net index audit requires initialized shadow state".into());
        }
        state.index_force_full = true;
        Ok(())
    }

    pub(super) fn hnet_index_overlap(&self, visits: usize) -> CudaResult<(f32, u64, u64)> {
        let state = self
            .hnet
            .as_ref()
            .ok_or("CUDA H-Net state is not configured")?;
        let values = visits
            .checked_mul(self.rows)
            .and_then(|rows| rows.checked_mul(8))
            .ok_or("CUDA H-Net index audit size overflow")?;
        let shadow = read_prefix(&state.index_candidate, &self.stream, values)?;
        let exact = read_prefix(&state.index_incumbent, &self.stream, values)?;
        let mut matches = 0_u64;
        let mut exact_rows = 0_u64;
        for (left, right) in shadow.chunks_exact(8).zip(exact.chunks_exact(8)) {
            let overlap = left.iter().filter(|leaf| right.contains(leaf)).count() as u64;
            matches += overlap;
            exact_rows += u64::from(overlap == 8);
        }
        let rows = (visits * self.rows) as u64;
        Ok((matches as f32 / (rows * 8) as f32, exact_rows, rows))
    }

    pub(super) fn hnet_tokens(&self) -> CudaResult<Vec<u32>> {
        let state = self
            .hnet
            .as_ref()
            .ok_or("CUDA H-Net state is not configured")?;
        read_prefix(&state.output, &self.stream, state.output.len())
    }

    pub(super) fn score_targets(&mut self, targets: &[u32]) -> CudaResult<()> {
        if targets.is_empty() || targets.iter().any(|&token| token >= VOCAB as u32) {
            return Err("CUDA H-Net scoring requires vocabulary-valid target macros".into());
        }
        self.scoring = Some(Scoring {
            targets: DeviceBuffer::from_host(&self.stream, targets).map_err(cuda_error)?,
            losses: DeviceBuffer::zeroed(&self.stream, targets.len()).map_err(cuda_error)?,
            count: targets.len(),
        });
        Ok(())
    }

    pub(super) fn expand_patches(&self, macros: &[u32]) -> CudaResult<(Vec<u32>, f32)> {
        let (tokens, elapsed) = self.expand_patches_device(macros)?;
        Ok((read_prefix(&tokens, &self.stream, tokens.len())?, elapsed))
    }

    pub(super) fn expand_patches_device(
        &self,
        macros: &[u32],
    ) -> CudaResult<(DeviceBuffer<u32>, f32)> {
        if macros.is_empty() {
            return Err("H-Net expansion requires macro tokens".into());
        }
        let count = u32::try_from(macros.len()).map_err(|_| "too many H-Net patches")?;
        let macros = DeviceBuffer::from_host(&self.stream, macros).map_err(cuda_error)?;
        self.expand(&macros, 0, count)
    }

    pub(super) fn expand_generated(
        &self,
        first: usize,
        count: usize,
    ) -> CudaResult<(DeviceBuffer<u32>, f32)> {
        let end = first
            .checked_add(count)
            .ok_or("H-Net patch range overflow")?;
        if count == 0 || end > self.scratch.sampled.len() {
            return Err("H-Net generated patch range is invalid".into());
        }
        self.expand(
            &self.scratch.sampled,
            u32::try_from(first).map_err(|_| "H-Net patch offset is too large")?,
            u32::try_from(count).map_err(|_| "too many H-Net patches")?,
        )
    }

    fn expand(
        &self,
        macros: &DeviceBuffer<u32>,
        first: u32,
        count: u32,
    ) -> CudaResult<(DeviceBuffer<u32>, f32)> {
        let mut tokens =
            DeviceBuffer::zeroed(&self.stream, count as usize * 64).map_err(cuda_error)?;
        let launch = self
            .module
            .prepare_expand_patches(LaunchConfig1D::new(count, 256, 0))
            .map_err(cuda_error)?;
        let start = timing_event(&self.stream)?;
        self.module
            .expand_patches(
                &self.stream,
                &launch,
                &macros,
                &self.weights.readout,
                &mut tokens,
                count,
                first,
            )
            .map_err(cuda_error)?;
        let end = timing_event(&self.stream)?;
        Ok((tokens, start.elapsed_ms(&end).map_err(cuda_error)?))
    }

    pub(super) fn objective(
        &self,
        generated: &DeviceBuffer<u32>,
        targets: &DeviceBuffer<u32>,
    ) -> CudaResult<(ScoreStats, f32)> {
        let scoring = self
            .scoring
            .as_ref()
            .ok_or("CUDA H-Net target scoring was not configured")?;
        if generated.len() == 0 || generated.len() != targets.len() {
            return Err("CUDA H-Net objective requires aligned generated and target tokens".into());
        }
        let losses = u32::try_from(scoring.count).map_err(|_| "too many scored positions")?;
        let tokens = u32::try_from(generated.len()).map_err(|_| "too many generated tokens")?;
        let mut output = DeviceBuffer::zeroed(&self.stream, 1).map_err(cuda_error)?;
        let launch = self
            .module
            .prepare_objective(LaunchConfig1D::new(1, 256, 0))
            .map_err(cuda_error)?;
        let start = timing_event(&self.stream)?;
        self.module
            .objective(
                &self.stream,
                &launch,
                &scoring.losses,
                generated,
                targets,
                &mut output,
                ObjectiveShape {
                    losses,
                    tokens,
                    patch: 1,
                },
            )
            .map_err(cuda_error)?;
        let end = timing_event(&self.stream)?;
        let stats = read_prefix(&output, &self.stream, 1)?[0];
        Ok((stats, start.elapsed_ms(&end).map_err(cuda_error)?))
    }

    /// Reuse bounded vocabulary scratch. Sampling is keyed by absolute row,
    /// so changing chunk boundaries preserves the generated token sequence.
    pub(super) fn readout(
        &mut self,
        temperature: f32,
        seed: u64,
        mut score_events: Option<&mut Vec<(CudaEvent, CudaEvent)>>,
    ) -> CudaResult<()> {
        let capacity = self.scratch.logits.len() / VOCAB;
        // Prefix logits before the final prompt position cannot contribute to
        // generation. Keep absolute row seeds for the positions we do sample.
        let start = self.cache.as_ref().map_or(self.readout_from, |cache| {
            cache.sample_from.saturating_sub(self.first).min(self.rows)
        });
        for first in (start..self.rows).step_by(capacity) {
            let count = capacity.min(self.rows - first);
            matmul_at(
                self.synth.as_ref(),
                &self.module,
                &self.stream,
                &self.scratch.normalized,
                &self.weights.readout,
                &mut self.scratch.logits,
                count,
                VOCAB,
                WIDTH,
                first,
            )?;
            self.sample(count, temperature, seed, first)?;
            let score_start = score_events
                .as_ref()
                .map(|_| timing_event(&self.stream))
                .transpose()?;
            self.score(count, first.saturating_sub(start))?;
            if let (Some(score_start), Some(events)) = (score_start, score_events.as_deref_mut()) {
                events.push((score_start, timing_event(&self.stream)?));
            }
        }
        Ok(())
    }

    fn sample(
        &mut self,
        count: usize,
        temperature: f32,
        seed: u64,
        first: usize,
    ) -> CudaResult<()> {
        let position = (self.first + first) as u32;
        if let Some(state) = &mut self.diffusion {
            let launch = state
                .module
                .prepare_diffusion_sample(LaunchConfig1D::new(count as u32, 256, 0))
                .map_err(cuda_error)?;
            state
                .module
                .diffusion_sample(
                    &self.stream,
                    &launch,
                    &self.scratch.logits,
                    &mut self.scratch.sampled,
                    &mut state.confidence,
                    count as u32,
                    temperature,
                    seed,
                    position,
                )
                .map_err(cuda_error)
        } else {
            let launch = self
                .module
                .prepare_sample(LaunchConfig1D::new(count as u32, 256, 0))
                .map_err(cuda_error)?;
            self.module
                .sample(
                    &self.stream,
                    &launch,
                    &self.scratch.logits,
                    &mut self.scratch.sampled,
                    count as u32,
                    temperature,
                    seed,
                    position,
                )
                .map_err(cuda_error)
        }
    }

    fn score(&mut self, count: usize, offset: usize) -> CudaResult<()> {
        let Some(scoring) = &mut self.scoring else {
            return Ok(());
        };
        let count = count.min(scoring.count.saturating_sub(offset));
        if count == 0 {
            return Ok(());
        }
        let launch = self
            .module
            .prepare_score(LaunchConfig1D::new(count as u32, 256, 0))
            .map_err(cuda_error)?;
        self.module
            .score(
                &self.stream,
                &launch,
                &self.scratch.logits,
                &scoring.targets,
                &mut scoring.losses,
                count as u32,
                offset as u32,
                offset as u32,
            )
            .map_err(cuda_error)
    }
}

impl HnetGenerationProfile {
    fn from_events(
        start: &CudaEvent,
        events: &[(HnetGenerationStage, CudaEvent)],
    ) -> CudaResult<Self> {
        let mut init_ms = 0.0;
        let mut gemm_ms = 0.0;
        let mut project_ms = 0.0;
        let mut reduce_ms = 0.0;
        let mut previous = start;
        for (stage, event) in events {
            let elapsed = previous.elapsed_ms(event).map_err(cuda_error)?;
            match stage {
                HnetGenerationStage::Init => init_ms += elapsed,
                HnetGenerationStage::Gemm => gemm_ms += elapsed,
                HnetGenerationStage::Project => project_ms += elapsed,
                HnetGenerationStage::Reduce => reduce_ms += elapsed,
            }
            previous = event;
        }
        let end = &events.last().ok_or("H-Net generation profile is empty")?.1;
        Ok(Self {
            init_ms,
            gemm_ms,
            project_ms,
            reduce_ms,
            total_ms: start.elapsed_ms(end).map_err(cuda_error)?,
        })
    }
}
