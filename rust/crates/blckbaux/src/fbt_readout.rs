use super::scorer::Scorer;
use super::*;
use crate::forward_program::hnet::{self, WaveShape};

#[cfg(test)]
#[test]
fn wave_compile() {
    let runtime = Runtime::shared().expect("Metal runtime required");
    HnetWaves::new(&runtime, 2).expect("production H-Net kernels must compile with generated ABI");
}

#[cfg(test)]
#[test]
fn objective_parity() {
    metal::objc::rc::autoreleasepool(|| {
        let runtime = Runtime::shared().unwrap();
        let source = hnet::metal_source(include_str!("fbt_draft.metal"));
        let pipeline = runtime
            .precise(&source, "H-Net objective parity", "fbt_hnet_objective")
            .unwrap();
        for count in [1, 65, 131, 1025] {
            for paired in [false, true] {
                let fixture = hnet::probe::objective::Fixture::new(count, paired);
                let losses = runtime.buffer_with(&fixture.losses);
                let baseline = runtime.buffer_with(&fixture.baseline);
                let generated = runtime.buffer_with(&fixture.generated);
                let targets = runtime.buffer_with(&fixture.targets);
                let stats = runtime.buffer::<hnet::ObjectiveStats>(1);
                let patches = runtime.buffer::<f32>(count.div_ceil(64));
                let command = runtime.queue.new_command_buffer();
                let encoder = command.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&pipeline);
                for (index, buffer) in [&losses, &generated, &targets, &stats, &baseline]
                    .iter()
                    .enumerate()
                {
                    encoder.set_buffer(index as u64, Some(buffer), 0);
                }
                encoder.set_bytes(
                    5,
                    size_of_val(&fixture.shape) as u64,
                    std::ptr::from_ref(&fixture.shape).cast(),
                );
                encoder.set_buffer(6, Some(&patches), 0);
                encoder.dispatch_thread_groups(thread_group(1), thread_group(256));
                encoder.end_encoding();
                complete(command).unwrap();
                // SAFETY: completed GPU writes and exact fixture allocations.
                unsafe {
                    let raw = stats.contents().cast::<hnet::ObjectiveStats>().read();
                    let patches = if paired {
                        std::slice::from_raw_parts(
                            patches.contents().cast::<f32>(),
                            count.div_ceil(64),
                        )
                    } else {
                        &[]
                    };
                    fixture.verify(&raw, patches).unwrap();
                }
            }
        }
    });
}

#[cfg(test)]
#[test]
fn encode_parity() {
    metal::objc::rc::autoreleasepool(|| {
        use hnet::probe::encode::{self, Fixture};
        let runtime = Runtime::shared().unwrap();
        let fixture = Fixture::new();
        let readout = runtime.buffer_with(&fixture.readout);
        let tokens = runtime.buffer_with(&fixture.tokens);
        let positions = runtime.buffer_with(&fixture.positions);
        let mask = runtime.buffer_with(&fixture.mask);
        let output = runtime.buffer::<u16>(fixture.expected.len());
        let pipeline = runtime
            .precise(
                &blckbaux_semantics::hnet::metal_source(include_str!("fbt_draft.metal")),
                "H-Net encoding parity",
                "fbt_hnet_encode",
            )
            .unwrap();
        let shape = hnet::EncodeShape {
            rows: encode::ROWS as u32,
            width: encode::WIDTH as u32,
            vocabulary: encode::VOCABULARY as u32,
            patch: encode::PATCH as u32,
        };
        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        for (index, buffer) in [&readout, &tokens, &positions, &mask, &output]
            .iter()
            .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.set_bytes(
            5,
            size_of_val(&shape) as u64,
            std::ptr::from_ref(&shape).cast(),
        );
        encoder.dispatch_threads(
            thread_group(fixture.expected.len() as u64),
            thread_group(64),
        );
        encoder.end_encoding();
        complete(command).unwrap();
        // SAFETY: the completed encoder wrote exactly this many FP16 values.
        let actual = unsafe {
            std::slice::from_raw_parts(output.contents().cast::<u16>(), fixture.expected.len())
        };
        fixture.verify(actual).unwrap();
    });
}

#[cfg(test)]
#[test]
fn wave_parity() {
    metal::objc::rc::autoreleasepool(|| {
        let runtime = Runtime::shared().unwrap();
        let fixture = hnet::probe::Fixture::new();
        let waves = HnetWaves::new(&runtime, 2).unwrap();
        let hidden = runtime.buffer_with(&fixture.hidden);
        let qkv = runtime.buffer_with(&fixture.qkv);
        let positions = runtime.buffer_with(&fixture.positions);
        let readout = runtime.buffer_with(&fixture.readout);
        let expected = runtime.buffer_with(&fixture.targets);
        let seeds = runtime.buffer_with(&[fixture.seed]);
        let output = runtime.buffer::<u32>(fixture.targets.len());
        let losses = runtime.buffer::<f32>(fixture.targets.len());
        let quality = runtime.buffer::<HnetObjective>(1);
        let paired = runtime.buffer::<f32>(2);
        for temperature in [0.0, 0.8] {
            let plan = ExpandPlan {
                prompt: hnet::PATCH,
                tokens: fixture.targets.len(),
                patch: hnet::PATCH,
                objective: Some(Objective {
                    losses: &losses,
                    baseline_losses: None,
                    paired_patches: &paired,
                    expected: &expected,
                    output: &quality,
                }),
            };
            let command = runtime.queue.new_command_buffer();
            waves
                .encode(
                    command,
                    &hidden,
                    &output,
                    (&qkv, 0),
                    (&positions, 0),
                    (&readout, 0),
                    &seeds,
                    &plan,
                    temperature,
                )
                .unwrap();
            complete(command).unwrap();
            // SAFETY: completed command; buffers have the checked fixture lengths.
            unsafe {
                fixture
                    .verify(
                        temperature,
                        std::slice::from_raw_parts(
                            output.contents().cast::<u32>(),
                            fixture.targets.len(),
                        ),
                        std::slice::from_raw_parts(
                            losses.contents().cast::<f32>(),
                            fixture.targets.len(),
                        ),
                        std::slice::from_raw_parts(
                            waves.centroids.contents().cast::<u16>(),
                            fixture.centroids.len(),
                        ),
                    )
                    .unwrap();
            }
        }
    });
}

pub(super) struct HnetWaves {
    centroids_pipeline: ComputePipelineState,
    expand_pipeline: ComputePipelineState,
    grad_pipeline: ComputePipelineState,
    centroid_grad_pipeline: ComputePipelineState,
    centroids: Buffer,
    grad_status: Buffer,
}

impl HnetWaves {
    pub(super) fn new(runtime: &Runtime, _blocks: usize) -> Result<Self, String> {
        let source = hnet::metal_source(include_str!("fbt_hnet_wave.metal"));
        let pipeline =
            |name| runtime.precise_metal4(&source, "H-Net batched causal waves", name, &[]);
        let grad_source = include_str!("fbt_hnet_grad.metal");
        Ok(Self {
            centroids_pipeline: pipeline("hnet_hierarchy_centroids")?,
            expand_pipeline: pipeline("hnet_hierarchical_expand")?,
            grad_pipeline: runtime.precise(grad_source, "H-Net loss gradient", "hnet_loss_grad")?,
            centroid_grad_pipeline: runtime.precise(
                grad_source,
                "H-Net centroid gradient",
                "hnet_centroid_grad",
            )?,
            centroids: runtime.buffer::<u16>(hnet::HIERARCHY_NODES * hnet::QUERY),
            grad_status: runtime.buffer::<u32>(8),
        })
    }

    pub(super) fn centroids(&self) -> &BufferRef {
        &self.centroids
    }

    #[allow(clippy::too_many_arguments)]
    fn encode(
        &self,
        command: &CommandBufferRef,
        coarse: &BufferRef,
        output: &BufferRef,
        qkv: (&BufferRef, u64),
        positions: (&BufferRef, u64),
        readout: (&BufferRef, u64),
        seeds: &BufferRef,
        plan: &ExpandPlan<'_>,
        temperature: f32,
    ) -> Result<(), String> {
        let shape = hnet::shape(
            plan.prompt,
            plan.tokens,
            temperature,
            plan.objective.is_some(),
        )?;
        let encoder =
            command.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Serial);
        encoder.set_compute_pipeline_state(&self.centroids_pipeline);
        encoder.set_buffer(0, Some(readout.0), readout.1);
        encoder.set_buffer(1, Some(&self.centroids), 0);
        encoder.set_bytes(
            2,
            size_of::<WaveShape>() as u64,
            std::ptr::from_ref(&shape).cast(),
        );
        let elements = u64::try_from(hnet::HIERARCHY_NODES * hnet::QUERY)
            .map_err(|error| error.to_string())?;
        encoder.dispatch_threads(thread_group(elements), thread_group(128));
        let expected = plan
            .objective
            .as_ref()
            .map_or(output, |objective| objective.expected);
        let losses = plan
            .objective
            .as_ref()
            .map_or(output, |objective| objective.losses);
        encoder.set_compute_pipeline_state(&self.expand_pipeline);
        for (index, (buffer, offset)) in [
            (coarse, 0),
            qkv,
            positions,
            readout,
            (&*self.centroids, 0),
            (seeds, 0),
            (expected, 0),
            (output, 0),
            (losses, 0),
        ]
        .into_iter()
        .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), offset);
        }
        encoder.set_bytes(
            9,
            size_of::<WaveShape>() as u64,
            std::ptr::from_ref(&shape).cast(),
        );
        encoder.dispatch_thread_groups(thread_group(u64::from(shape.blocks)), thread_group(64));
        encoder.end_encoding();
        Ok(())
    }
}

pub(super) use hnet::ObjectiveStats as HnetObjective;

pub(super) struct Objective<'a> {
    pub losses: &'a BufferRef,
    pub baseline_losses: Option<&'a BufferRef>,
    pub paired_patches: &'a BufferRef,
    pub expected: &'a BufferRef,
    pub output: &'a BufferRef,
}

pub(super) struct ExpandPlan<'a> {
    pub prompt: usize,
    pub tokens: usize,
    pub patch: usize,
    pub objective: Option<Objective<'a>>,
}

pub(super) struct Expansion {
    pub seconds: f64,
    pub quality: Option<decode::TargetQuality>,
    pub paired_patch_improvements: Vec<f32>,
}

#[repr(C)]
struct HnetShape {
    tokens: u32,
    proposal_start: u32,
    patch: u32,
    qkv_width: u32,
    index_width: u32,
    vocabulary: u32,
    temperature: f32,
    score: u32,
}

#[allow(clippy::too_many_arguments)]
fn dispatch_head(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    coarse: &BufferRef,
    output: &BufferRef,
    qkv: (&BufferRef, u64),
    positions: (&BufferRef, u64),
    readout: (&BufferRef, u64),
    seeds: &BufferRef,
    plan: &ExpandPlan<'_>,
    shape: &HnetShape,
) {
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipelines.hnet_expand);
    encoder.set_buffer(0, Some(coarse), 0);
    encoder.set_buffer(1, Some(qkv.0), qkv.1);
    encoder.set_buffer(2, Some(positions.0), positions.1);
    encoder.set_buffer(3, Some(readout.0), readout.1);
    encoder.set_buffer(4, Some(seeds), 0);
    let expected = plan
        .objective
        .as_ref()
        .map_or(output, |value| value.expected);
    let losses = plan.objective.as_ref().map_or(output, |value| value.losses);
    encoder.set_buffer(5, Some(expected), 0);
    encoder.set_buffer(6, Some(output), 0);
    encoder.set_buffer(7, Some(losses), 0);
    encoder.set_bytes(
        8,
        std::mem::size_of_val(shape) as u64,
        std::ptr::from_ref(shape).cast(),
    );
    encoder.dispatch_thread_groups(
        thread_group(u64::from(shape.tokens.div_ceil(shape.patch))),
        thread_group(256),
    );
    encoder.end_encoding();
}

fn expansion_shape(
    coarse: &BufferRef,
    output: &BufferRef,
    plan: &ExpandPlan<'_>,
) -> Result<(usize, usize), String> {
    if plan.patch == 0 || plan.tokens == 0 {
        return Err("H-Net expansion requires positive token and patch counts".into());
    }
    let patches = plan.tokens.div_ceil(plan.patch);
    let proposal_start = plan
        .prompt
        .checked_div(plan.patch)
        .and_then(|value| value.checked_sub(1))
        .ok_or("H-Net expansion requires at least one complete prompt patch")?;
    let proposal_end = proposal_start
        .checked_add(patches)
        .ok_or("H-Net proposal range overflow")?;
    if coarse.length() < (proposal_end * WIDTH as usize * size_of::<u16>()) as u64 {
        return Err("H-Net coarse-state buffer is smaller than the expansion range".into());
    }
    if output.length() < (plan.tokens * size_of::<u32>()) as u64 {
        return Err("H-Net output buffer is smaller than the raw token trajectory".into());
    }
    if let Some(objective) = &plan.objective {
        if objective.losses.length() < (plan.tokens * size_of::<f32>()) as u64
            || objective.expected.length() < (plan.tokens * size_of::<u32>()) as u64
            || objective.output.length() < size_of::<HnetObjective>() as u64
            || objective.paired_patches.length() < (patches * size_of::<f32>()) as u64
        {
            return Err("H-Net objective buffers do not cover the requested trajectory".into());
        }
    }
    Ok((patches, proposal_start))
}

fn qualify_scalar(
    runtime: &Runtime,
    pipelines: &Pipelines,
    coarse: &BufferRef,
    output: &BufferRef,
    qkv: (&BufferRef, u64),
    positions: (&BufferRef, u64),
    readout: (&BufferRef, u64),
    seeds: &BufferRef,
    plan: &ExpandPlan<'_>,
    shape: &HnetShape,
) -> Result<(), String> {
    let reference = runtime.buffer::<u32>(plan.tokens);
    let reference_losses = runtime.buffer::<f32>(plan.tokens);
    let reference_quality = runtime.buffer::<HnetObjective>(1);
    let reference_plan = ExpandPlan {
        prompt: plan.prompt,
        tokens: plan.tokens,
        patch: plan.patch,
        objective: plan.objective.as_ref().map(|objective| Objective {
            losses: &reference_losses,
            baseline_losses: objective.baseline_losses,
            paired_patches: objective.paired_patches,
            expected: objective.expected,
            output: &reference_quality,
        }),
    };
    let reference_command = runtime.queue.new_command_buffer();
    dispatch_head(
        reference_command,
        pipelines,
        coarse,
        &reference,
        qkv,
        positions,
        readout,
        seeds,
        &reference_plan,
        shape,
    );
    complete(reference_command)?;
    let generated =
        unsafe { std::slice::from_raw_parts(output.contents().cast::<u32>(), plan.tokens) };
    let scalar =
        unsafe { std::slice::from_raw_parts(reference.contents().cast::<u32>(), plan.tokens) };
    let mismatches = generated.iter().zip(scalar).filter(|(a, b)| a != b).count();
    let mut maximum_error = 0.0f32;
    if let Some(objective) = &plan.objective {
        let losses = unsafe {
            std::slice::from_raw_parts(objective.losses.contents().cast::<f32>(), plan.tokens)
        };
        let scalar_losses = unsafe {
            std::slice::from_raw_parts(reference_losses.contents().cast::<f32>(), plan.tokens)
        };
        for (&a, &b) in losses.iter().zip(scalar_losses) {
            if !a.is_finite() || !b.is_finite() {
                return Err("H-Net qualification produced nonfinite NLL".into());
            }
            maximum_error = maximum_error.max((a - b).abs());
        }
    }
    eprintln!(
        "hnet.qualification tokens={} mismatches={} maximum_nll_error={} tolerance=0.00002",
        plan.tokens, mismatches, maximum_error
    );
    if mismatches != 0 || maximum_error > 0.00002 {
        return Err("H-Net batched waves failed scalar trajectory qualification".into());
    }
    Ok(())
}

fn patch_responses(plan: &ExpandPlan<'_>) -> Result<Vec<f32>, String> {
    let Some(objective) = plan
        .objective
        .as_ref()
        .filter(|objective| objective.baseline_losses.is_some())
    else {
        return Ok(Vec::new());
    };
    let patches = plan.tokens.div_ceil(plan.patch);
    let values = unsafe {
        std::slice::from_raw_parts(objective.paired_patches.contents().cast::<f32>(), patches)
    };
    if values.iter().any(|value| !value.is_finite()) {
        return Err("H-Net paired patch response contains a nonfinite value".into());
    }
    Ok(values.to_vec())
}

pub(super) fn expand_hnet(
    runtime: &Runtime,
    pipelines: &Pipelines,
    waves: &HnetWaves,
    coarse: &BufferRef,
    output: &BufferRef,
    qkv: (&BufferRef, u64),
    positions: (&BufferRef, u64),
    readout: (&BufferRef, u64),
    seeds: &BufferRef,
    temperature: f32,
    plan: ExpandPlan<'_>,
) -> Result<Expansion, String> {
    let (_, proposal_start) = expansion_shape(coarse, output, &plan)?;
    let shape = HnetShape {
        tokens: u32::try_from(plan.tokens).map_err(|error| error.to_string())?,
        proposal_start: u32::try_from(proposal_start).map_err(|error| error.to_string())?,
        patch: u32::try_from(plan.patch).map_err(|error| error.to_string())?,
        qkv_width: QKV_WIDTH,
        index_width: WIDTH,
        vocabulary: VOCAB,
        temperature,
        score: u32::from(plan.objective.is_some()),
    };
    let command = runtime.queue.new_command_buffer();
    if std::env::var_os("BLCKBAUX_HNET_SCALAR").is_some() {
        dispatch_head(
            command, pipelines, coarse, output, qkv, positions, readout, seeds, &plan, &shape,
        );
    } else {
        waves.encode(
            command,
            coarse,
            output,
            qkv,
            positions,
            readout,
            seeds,
            &plan,
            temperature,
        )?;
    }
    if let Some(objective) = &plan.objective {
        let objective_shape = hnet::PairShape {
            tokens: shape.tokens,
            paired: u32::from(objective.baseline_losses.is_some()),
            batch_tokens: u32::try_from(plan.patch).map_err(|error| error.to_string())?,
        };
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipelines.hnet_objective);
        encoder.set_buffer(0, Some(objective.losses), 0);
        encoder.set_buffer(1, Some(output), 0);
        encoder.set_buffer(2, Some(objective.expected), 0);
        encoder.set_buffer(3, Some(objective.output), 0);
        encoder.set_buffer(
            4,
            Some(objective.baseline_losses.unwrap_or(objective.losses)),
            0,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of_val(&objective_shape) as u64,
            std::ptr::from_ref(&objective_shape).cast(),
        );
        encoder.set_buffer(6, Some(objective.paired_patches), 0);
        encoder.dispatch_thread_groups(thread_group(1), thread_group(256));
        encoder.end_encoding();
    }
    let seconds = complete(command)?;
    if std::env::var_os("BLCKBAUX_HNET_QUALIFY").is_some() {
        qualify_scalar(
            runtime, pipelines, coarse, output, qkv, positions, readout, seeds, &plan, &shape,
        )?;
    }
    let paired_patch_improvements = patch_responses(&plan)?;
    let quality = plan
        .objective
        .map(|objective| -> Result<_, String> {
            let value = unsafe { objective.output.contents().cast::<HnetObjective>().read() };
            let tokens = plan.tokens as f32;
            let estimate = value.estimate(hnet::PairShape {
                tokens: shape.tokens,
                paired: u32::from(objective.baseline_losses.is_some()),
                batch_tokens: u32::try_from(plan.patch).map_err(|error| error.to_string())?,
            })?;
            let codelength_bits = estimate.bits;
            Ok(decode::TargetQuality {
                scored_tokens: plan.tokens,
                codelength_bits,
                bits_per_token: (codelength_bits / plan.tokens as f64) as f32,
                mean_nll: estimate.mean,
                mean_nll_variance: estimate.variance,
                maximum_nll: value.maximum,
                worst_window_nll: value.sum / tokens,
                window_tokens: plan.tokens,
                positional_accuracy: value.matches as f32 / tokens,
                positional_mismatches: value.mismatches as usize,
                first_positional_mismatch: (value.first_mismatch != u32::MAX)
                    .then_some(value.first_mismatch as usize),
                mismatch_target_nll: (value.mismatches > 0)
                    .then_some(value.mismatch_sum / value.mismatches as f32),
                paired_improvement: estimate.improvement,
                paired_improvement_variance: estimate.paired_variance,
                patch_nll_octiles: Some(value.patch_nll_octiles),
            })
        })
        .transpose()?;
    Ok(Expansion {
        seconds,
        quality,
        paired_patch_improvements,
    })
}

impl Scorer<'_> {
    pub(super) fn readout(&self) -> Result<(), String> {
        let encoder = self.active_encoder();
        let pipelines = self.pipelines;
        let buffers = self.buffers;
        let loss_partials = buffers.loss_partials()?;
        let shape = [self.rows, VOCAB];
        encoder.set_compute_pipeline_state(&self.tensorops.readout_loss_tiles);
        encoder.set_buffer(0, Some(&buffers.normalized), 0);
        encoder.set_buffer(1, Some(self.weights.buffer), self.weights.readout);
        encoder.set_buffer(2, Some(&buffers.labels), 0);
        encoder.set_buffer(3, Some(loss_partials), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of_val(&shape) as u64,
            shape.as_ptr().cast(),
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: u64::from(VOCAB / 64),
                height: u64::from(self.rows / 128),
                depth: 1,
            },
            thread_group(128),
        );
        encoder.memory_barrier_with_resources(&[loss_partials]);
        encoder.set_compute_pipeline_state(&pipelines.readout_loss_reduce);
        encoder.set_buffer(0, Some(loss_partials), 0);
        encoder.set_buffer(1, Some(&buffers.losses), 0);
        let tiles = VOCAB / 64;
        encoder.set_bytes(
            2,
            std::mem::size_of_val(&tiles) as u64,
            (&tiles as *const u32).cast(),
        );
        encoder.dispatch_thread_groups(thread_group(u64::from(self.rows)), thread_group(128));
        encoder.memory_barrier_with_resources(&[&buffers.losses]);
        encoder.set_compute_pipeline_state(&pipelines.sequence_loss);
        encoder.set_buffer(0, Some(&buffers.losses), 0);
        encoder.set_buffer(1, Some(&buffers.score_mask), 0);
        encoder.set_buffer(2, Some(&buffers.sequence_scores), 0);
        encoder.dispatch_thread_groups(
            thread_group(u64::from(self.rows / CONTEXT)),
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    pub(super) fn readout_proposals(
        &self,
        output: &BufferRef,
        seeds: &BufferRef,
        temperature: f32,
        score_targets: bool,
    ) -> Result<(), String> {
        if !temperature.is_finite() || temperature < 0.0 {
            return Err("proposal readout temperature must be finite and nonnegative".into());
        }
        let loss_partials = self.buffers.loss_partials()?;
        let proposal_partials = self.buffers.proposal_partials()?;
        let encoder = self.active_encoder();
        let shape = [self.rows, VOCAB, self.row_start, self.pisa1.context()];
        let activation_offset = self.activation_offset();
        encoder.set_compute_pipeline_state(if self.diffusion.is_some() {
            &self.tensorops.denoise_tiles
        } else {
            &self.tensorops.readout_proposal_tiles
        });
        encoder.set_buffer(0, Some(&self.buffers.normalized), activation_offset);
        encoder.set_buffer(1, Some(self.weights.buffer), self.weights.readout);
        encoder.set_buffer(2, Some(loss_partials), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of_val(&shape) as u64,
            shape.as_ptr().cast(),
        );
        encoder.set_buffer(4, Some(seeds), 0);
        encoder.set_bytes(
            5,
            std::mem::size_of_val(&temperature) as u64,
            std::ptr::from_ref(&temperature).cast(),
        );
        encoder.set_buffer(6, Some(&self.buffers.labels), 0);
        encoder.set_buffer(7, Some(proposal_partials), 0);
        encoder.set_bytes(
            8,
            std::mem::size_of_val(&score_targets) as u64,
            std::ptr::from_ref(&score_targets).cast(),
        );
        if let Some(input) = self.diffusion {
            encoder.set_buffer(9, Some(input.moments), 0);
        }
        encoder.dispatch_thread_groups(
            MTLSize {
                width: u64::from(VOCAB / 64),
                height: u64::from(self.rows / 128),
                depth: 1,
            },
            thread_group(128),
        );
        encoder.memory_barrier_with_resources(&[loss_partials, proposal_partials]);
        if let Some(input) = self.diffusion {
            encoder.memory_barrier_with_resources(&[input.moments]);
        }
        let tiles = VOCAB / 64;
        self.start_stage("proposal_reduce")?;
        let encoder = self.active_encoder();
        self.proposal_reduce(&encoder, loss_partials, output, temperature, tiles);
        if score_targets {
            self.start_stage("target_losses")?;
            self.target_losses(self.active_encoder(), proposal_partials, tiles);
        }
        Ok(())
    }

    fn target_losses(
        &self,
        encoder: &ComputeCommandEncoderRef,
        proposal_partials: &BufferRef,
        tiles: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.readout_loss_reduce);
        encoder.set_buffer(0, Some(proposal_partials), 0);
        encoder.set_buffer(
            1,
            Some(&self.buffers.losses),
            u64::from(self.row_start) * std::mem::size_of::<f32>() as u64,
        );
        encoder.set_bytes(
            2,
            std::mem::size_of_val(&tiles) as u64,
            std::ptr::from_ref(&tiles).cast(),
        );
        encoder.dispatch_thread_groups(thread_group(u64::from(self.rows)), thread_group(128));
        encoder.memory_barrier_with_resources(&[&self.buffers.losses]);
    }
    fn proposal_reduce(
        &self,
        encoder: &ComputeCommandEncoderRef,
        loss_partials: &BufferRef,
        output: &BufferRef,
        temperature: f32,
        tiles: u32,
    ) {
        encoder.set_compute_pipeline_state(if self.diffusion.is_some() {
            &self.pipelines.denoise_reduce
        } else {
            &self.pipelines.readout_proposal_reduce
        });
        encoder.set_buffer(0, Some(loss_partials), 0);
        encoder.set_buffer(
            1,
            Some(output),
            u64::from(self.row_start) * std::mem::size_of::<u32>() as u64,
        );
        if let Some(input) = self.diffusion {
            encoder.set_buffer(2, Some(input.confidence), u64::from(self.row_start) * 4);
            encoder.set_buffer(3, Some(input.moments), 0);
            encoder.set_bytes(
                4,
                std::mem::size_of_val(&temperature) as u64,
                std::ptr::from_ref(&temperature).cast(),
            );
        } else {
            encoder.set_bytes(
                2,
                std::mem::size_of_val(&tiles) as u64,
                std::ptr::from_ref(&tiles).cast(),
            );
        }
        encoder.dispatch_thread_groups(thread_group(u64::from(self.rows)), thread_group(128));
        encoder.memory_barrier_with_resources(&[output]);
    }
}

#[repr(C)]
struct GradShape {
    tokens: u32,
    proposal_start: u32,
    vocabulary: u32,
    loss_scale: f32,
}

pub(crate) struct HnetGrad<'a> {
    pub hidden: &'a BufferRef,
    pub qkv: (&'a BufferRef, u64),
    pub positions: (&'a BufferRef, u64),
    pub readout: (&'a BufferRef, u64),
    pub expected: &'a BufferRef,
    pub score_mask: &'a BufferRef,
    pub hidden_grad: (&'a BufferRef, u64),
    pub qkv_grad: (&'a BufferRef, u64),
    pub position_grad: (&'a BufferRef, u64),
    pub readout_grad: (&'a BufferRef, u64),
    pub centroid_grad: &'a BufferRef,
    pub tokens: u32,
    pub prompt: u32,
    pub loss_scale: f32,
}

impl HnetWaves {
    pub(crate) fn backward(
        &self,
        command: &CommandBufferRef,
        grad: HnetGrad<'_>,
    ) -> Result<(), String> {
        let proposal_start = grad
            .prompt
            .checked_div(64)
            .and_then(|value| value.checked_sub(1))
            .ok_or("H-Net gradient requires one complete prompt patch")?;
        if grad.tokens == 0 || !grad.loss_scale.is_finite() || grad.loss_scale <= 0.0 {
            return Err("invalid H-Net gradient extent or scale".into());
        }
        let shape = GradShape {
            tokens: grad.tokens,
            proposal_start,
            vocabulary: VOCAB,
            loss_scale: grad.loss_scale,
        };
        let clear = command.new_blit_command_encoder();
        clear.fill_buffer(
            &self.grad_status,
            metal::NSRange::new(0, self.grad_status.length()),
            0,
        );
        clear.end_encoding();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.grad_pipeline);
        for (slot, (buffer, offset)) in [
            (grad.hidden, 0),
            grad.qkv,
            grad.positions,
            grad.readout,
            (&self.centroids, 0),
            (grad.expected, 0),
            (grad.score_mask, 0),
            grad.hidden_grad,
            grad.qkv_grad,
            grad.position_grad,
            grad.readout_grad,
            (grad.centroid_grad, 0),
        ]
        .into_iter()
        .enumerate()
        {
            encoder.set_buffer(slot as u64, Some(buffer), offset);
        }
        encoder.set_bytes(
            12,
            size_of::<GradShape>() as u64,
            std::ptr::from_ref(&shape).cast(),
        );
        encoder.set_buffer(13, Some(&self.grad_status), 0);
        encoder.dispatch_thread_groups(
            thread_group(u64::from(grad.tokens.div_ceil(64))),
            thread_group(64),
        );
        encoder.memory_barrier_with_resources(&[grad.centroid_grad, grad.readout_grad.0]);
        encoder.set_compute_pipeline_state(&self.centroid_grad_pipeline);
        encoder.set_buffer(0, Some(grad.centroid_grad), 0);
        encoder.set_buffer(1, Some(grad.readout_grad.0), grad.readout_grad.1);
        encoder.set_bytes(
            2,
            size_of::<GradShape>() as u64,
            std::ptr::from_ref(&shape).cast(),
        );
        encoder.dispatch_threads(
            thread_group(u64::from(VOCAB) * 64),
            thread_group(
                self.centroid_grad_pipeline
                    .max_total_threads_per_threadgroup()
                    .min(256),
            ),
        );
        encoder.end_encoding();
        Ok(())
    }

    pub(super) fn check_grad(&self) -> Result<(), String> {
        let counts =
            unsafe { std::slice::from_raw_parts(self.grad_status.contents().cast::<u32>(), 8) };
        if counts.iter().all(|count| *count == 0) {
            return Ok(());
        }
        Err(format!(
            "H-Net adjoint invariant failed: query={} logits={} softmax={} query-grad={} prefix-grad={} transition-grad={} base-grad={} projection-grad={}",
            counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], counts[6], counts[7]
        ))
    }
}
