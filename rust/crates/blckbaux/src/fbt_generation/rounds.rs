use super::generation::*;
use crate::traits::Oracle;
pub(super) struct LoopState {
    pub(super) search: SearchState,
    pub(super) updates: UpdateLog,
    pub(super) journal: journal::Journal,
    pub(super) records: BufWriter<File>,
    pub(super) walls: Vec<f64>,
    pub(super) minimum_changed_weights: u64,
    pub(super) minimum_generated_tokens: usize,
    pub(super) minimum_context_tokens: usize,
    pub(super) accepted: u32,
    pub(super) incumbent_version: u64,
    pub(super) incumbent_rollouts: Vec<decode::Rollout>,
    pub(super) regions: RegionState,
    pub(super) temperature: f32,
    pub(super) loop_seconds: f64,
    pub(super) rank_audit: Option<RankAudit>,
    pub(super) pool_rank_audited: u32,
    pub(super) pool_rank_report: Option<blckbaux_artifact::json::Value>,
    pub(super) causal_value: Option<f32>,
    pub(super) condition_baselines: Vec<Option<ConditionBaseline>>,
}
pub(super) struct RegionState {
    pub(super) versions: Vec<u64>,
    pub(super) rollouts: Vec<Vec<decode::Rollout>>,
    pub(super) temperatures: Vec<f32>,
}
pub(super) struct RoundCondition {
    pub(super) config: GenerationConfig,
    pub(super) condition_seed: u64,
    pub(super) evaluation_seed: u64,
    pub(super) corpus_token_offset: usize,
}
#[derive(Clone)]
pub(super) struct ConditionBaseline {
    pub(super) condition_seed: u64,
    pub(super) reward: f32,
    pub(super) variance: f32,
    pub(super) rollouts: Vec<decode::Rollout>,
}
impl ConditionBaseline {
    fn measured(condition_seed: u64, evaluation: &Evaluation) -> Self {
        Self {
            condition_seed,
            reward: evaluation.mean,
            variance: evaluation.variance,
            rollouts: evaluation.rollouts.clone(),
        }
    }
    fn accepted(condition_seed: u64, evaluation: &Evaluation) -> Result<Self, String> {
        let paired = evaluation
            .paired
            .ok_or("accepted paired evaluation lost its measured candidate response")?;
        Ok(Self {
            condition_seed,
            reward: paired.candidate_reward,
            variance: paired.candidate_variance,
            rollouts: evaluation.rollouts.clone(),
        })
    }
    fn gpu_seconds(&self) -> f64 {
        self.rollouts
            .iter()
            .map(|rollout| rollout.gpu_seconds)
            .sum()
    }
}
impl RegionState {
    pub(super) fn initial(
        search: &SearchState,
        model_seed: u64,
        rollouts: &[decode::Rollout],
        temperature: f32,
    ) -> Self {
        let count = search.morbo_regions().max(1);
        let version = crate::hash::splitmix64(model_seed ^ 0x656e_6e78_2d62_6173);
        Self {
            versions: vec![version; count],
            rollouts: vec![rollouts.to_vec(); count],
            temperatures: vec![temperature; count],
        }
    }
}
impl LoopState {
    fn commit(
        &mut self,
        evaluation: &Evaluation,
        accepted: bool,
        region: usize,
        version: u64,
        temperature: f32,
    ) {
        let generated = evaluation
            .rollouts
            .iter()
            .map(|rollout| rollout.tokens.len())
            .min()
            .unwrap_or(0);
        self.minimum_generated_tokens = self.minimum_generated_tokens.min(generated);
        self.minimum_context_tokens = self
            .minimum_context_tokens
            .min(evaluation.prompt_tokens + generated);
        if accepted {
            self.accept_region(region, version, &evaluation.rollouts, temperature);
            if let Some(causal) = evaluation.causal {
                self.causal_value = Some(causal.optimizer_value);
            }
        }
        self.temperature = self.regions.temperatures[region];
        self.accepted += u32::from(accepted);
    }
    fn accept_region(
        &mut self,
        region: usize,
        version: u64,
        rollouts: &[decode::Rollout],
        temperature: f32,
    ) {
        self.incumbent_version = version;
        self.incumbent_rollouts = rollouts.to_vec();
        self.regions.versions[region] = version;
        self.regions.rollouts[region] = rollouts.to_vec();
        self.regions.temperatures[region] = temperature;
    }
    pub(super) fn finalize_state(
        &mut self,
        record: bool,
        output: &Path,
        weights: &CandidateWeights,
    ) -> Result<metal::Buffer, String> {
        if let Some(region) = self.search.finalize_morbo()? {
            self.incumbent_version = self.regions.versions[region];
            self.incumbent_rollouts = self.regions.rollouts[region].clone();
            self.temperature = self.regions.temperatures[region];
        }
        let final_weights = self.search.base_buffer();
        if record {
            self.updates.write(&output.join("tensor-updates.jsonl"))?;
            weights.write_stats(&final_weights, &output.join("tensor-state.jsonl"))?;
        }
        self.records.flush().map_err(|error| error.to_string())?;
        self.journal.finish()?;
        Ok(final_weights)
    }

    fn record_rank(
        &mut self,
        evaluation: &Evaluation,
        record: &mut blckbaux_artifact::json::Value,
    ) {
        if let (Some(audit), Some(nll)) = (
            self.rank_audit.as_mut(),
            evaluation
                .rollouts
                .first()
                .and_then(|rollout| rollout.conditional_target_nll),
        ) {
            record["model_selection_audit"] = audit.observe(-nll, evaluation.mean);
        }
    }

    fn record_updates(
        &mut self,
        enabled: bool,
        step: u32,
        proposal: &crate::bf16_metal::Proposals,
        accepted: bool,
        changes: Vec<(u64, f64)>,
    ) -> Result<(), String> {
        if enabled {
            self.updates.push_scaled(
                step,
                proposal.seed,
                proposal.length,
                accepted,
                changes,
                proposal.block_scales.clone(),
            )?;
        }
        Ok(())
    }

    fn write_record(
        &mut self,
        record: &blckbaux_artifact::json::Value,
        flush: bool,
    ) -> Result<(), String> {
        blckbaux_artifact::json::write_line(&mut self.records, record)
            .map_err(|error| error.to_string())?;
        if flush {
            self.records.flush().map_err(|error| error.to_string())?;
        }
        Ok(())
    }
}

fn token_sum(
    rollouts: &[decode::Rollout],
    field: fn(&decode::Rollout) -> usize,
) -> Result<u64, String> {
    let total = rollouts
        .iter()
        .try_fold(0usize, |total, rollout| total.checked_add(field(rollout)))
        .ok_or("Generation token count overflow")?;
    u64::try_from(total).map_err(|_| "Generation token count exceeds u64".to_string())
}

fn token_accounting(
    context: &LoopContext<'_>,
    evaluation: &Evaluation,
) -> Result<crate::response_observation::TokenAccounting, String> {
    let prompt =
        u64::try_from(evaluation.prompt_tokens).map_err(|_| "Prompt token count exceeds u64")?;
    let drafted = evaluation
        .rollouts
        .iter()
        .filter_map(|rollout| rollout.draft.as_ref())
        .try_fold(0usize, |total, draft| {
            total.checked_add(draft.generated_tokens)
        })
        .ok_or("Drafted token count overflow")?;
    let drafted = u64::try_from(drafted).map_err(|_| "Drafted token count exceeds u64")?;
    let generated = token_sum(&evaluation.rollouts, |rollout| rollout.tokens.len())?;
    let committed = token_sum(&evaluation.rollouts, |rollout| rollout.committed_tokens)?;
    let evaluated = if context.weights.architecture == ResidualArchitecture::HnetMhc4 {
        token_sum(&evaluation.rollouts, |rollout| rollout.token_count)?
    } else {
        token_sum(&evaluation.rollouts, |rollout| rollout.evaluated_positions)?
    };
    let repaired = token_sum(&evaluation.rollouts, |rollout| rollout.repaired_tokens)?;
    let output = evaluation
        .rollouts
        .iter()
        .map(|rollout| rollout.token_count)
        .max()
        .ok_or("Generation response has no rollout")?;
    let output = u64::try_from(output).map_err(|_| "Context token count exceeds u64")?;
    let context_tokens = prompt
        .checked_add(output)
        .ok_or("Context token count overflow")?;
    crate::response_observation::TokenAccounting {
        samples: evaluation.rollouts.len() as u64,
        context: context_tokens,
        prompt,
        drafted,
        evaluated,
        repaired,
        generated,
        committed,
    }
    .validate()
}

fn response_observation(
    context: &LoopContext<'_>,
    proposal: &crate::bf16_metal::Proposals,
    evaluation: &Evaluation,
    decision: crate::bf16_metal::NoisyDecision,
    initializing: bool,
) -> Result<crate::response_observation::ResponseObservation, String> {
    use crate::objective_observation::ObjectiveEstimate;
    use crate::response_observation::{ResponseObservation, ResponseReference};

    let reference = if evaluation.paired.is_some() {
        ResponseReference::PairedIncumbent
    } else if initializing {
        ResponseReference::Initial
    } else if context.run.objective_acquisition.is_some() {
        ResponseReference::VectorPolicy
    } else if evaluation.causal.is_some() {
        ResponseReference::PairedIncumbent
    } else {
        ResponseReference::SurrogateIncumbent
    };
    let improvement_variance = evaluation.paired.map_or_else(
        || {
            evaluation.causal.map_or_else(
                || evaluation.variance + decision.incumbent_variance,
                |causal| causal.variance,
            )
        },
        |paired| paired.improvement_variance,
    );
    let coordinates = u64::try_from(context.weights.architecture.parameter_count())
        .map_err(|_| "Model parameter count exceeds u64")?;
    ResponseObservation {
        proposal: proposal.measure(coordinates)?,
        reference,
        candidate: ObjectiveEstimate {
            mean: evaluation.mean,
            variance: evaluation.variance,
        },
        baseline: ObjectiveEstimate {
            mean: decision.incumbent_value,
            variance: decision.incumbent_variance,
        },
        improvement: ObjectiveEstimate {
            mean: decision.improvement as f32,
            variance: improvement_variance,
        },
        accepted: decision.accepted,
        tokens: Some(token_accounting(context, evaluation)?),
    }
    .validate()
}

fn response_json(
    context: &LoopContext<'_>,
    response: crate::response_observation::ResponseObservation,
) -> Result<blckbaux_artifact::json::Value, String> {
    use crate::response_observation::{RESPONSE_SCHEMA, ResponseReference};

    let proposal = response.proposal;
    let tokens = response
        .tokens
        .ok_or("Generation response lost token accounting")?;
    let reference =
        blckbaux_artifact::json::to_value(response.reference).map_err(|error| error.to_string())?;
    let distribution = match context.run.proposal_method {
        Some(crate::procedural_pool::ProposalMethod::QuadraticFlow) => "lie_algebra_flow_v1",
        Some(crate::procedural_pool::ProposalMethod::PairMix) => "pair_rotation_mix_v1",
        _ => context.run.proposal_kernel().name(),
    };
    Ok(blckbaux_artifact::json::json!({
        "schema": RESPONSE_SCHEMA,
        "reference": reference,
        "measured_pair": matches!(response.reference, ResponseReference::PairedIncumbent),
        "proposal": {
            "distribution": distribution,
            "seed": proposal.seed,
            "candidate": proposal.candidate,
            "coordinates": proposal.coordinates,
            "changed_coordinates": proposal.changed_coordinates,
            "quantized_survival": proposal.quantized_survival(),
            "squared_displacement": proposal.squared_displacement,
            "metric_radius": proposal.metric_radius,
            "trust_radius": proposal.trust_radius,
        },
        "candidate": {
            "mean": response.candidate.mean,
            "variance": response.candidate.variance,
        },
        "baseline": {
            "mean": response.baseline.mean,
            "variance": response.baseline.variance,
        },
        "improvement": {
            "mean": response.improvement.mean,
            "variance": response.improvement.variance,
        },
        "accepted": response.accepted,
        "tokens": {
            "samples": tokens.samples,
            "context": tokens.context,
            "prompt": tokens.prompt,
            "drafted": tokens.drafted,
            "evaluated": tokens.evaluated,
            "repaired": tokens.repaired,
            "generated": tokens.generated,
            "committed": tokens.committed,
        },
    }))
}

fn response_record(
    context: &LoopContext<'_>,
    proposal: &crate::bf16_metal::Proposals,
    evaluation: &Evaluation,
    decision: crate::bf16_metal::NoisyDecision,
    initializing: bool,
) -> Result<blckbaux_artifact::json::Value, String> {
    let mut record = response_json(
        context,
        response_observation(context, proposal, evaluation, decision, initializing)?,
    )?;
    record["posterior_response"] = blckbaux_artifact::json::json!({
        "components": evaluation
            .paired
            .map_or(0, |paired| paired.response_components),
    });
    Ok(record)
}

pub(super) struct LoopContext<'a> {
    pub(super) run: &'a RunConfig,
    pub(super) config: &'a GenerationConfig,
    pub(super) output: &'a Path,
    pub(super) runtime: &'a Runtime,
    pub(super) decoder: &'a decode::Decoder,
    pub(super) verifier: &'a Option<block_decode::BlockDecoder>,
    pub(super) weights: &'a CandidateWeights,
    pub(super) sampling_seed: u64,
    pub(super) tokenizer: Option<&'a ByteDecoder>,
    pub(super) evaluator: &'a mut Option<Evaluator>,
    pub(super) validation_tasks: &'a [GenerationTask],
    pub(super) dataset: Option<&'a crate::pretrain_data::PretrainDataset>,
    pub(super) corpus_prompt_cap: usize,
    pub(super) corpus_output_cap: usize,
    pub(super) policy: crate::config::ResidentEnnConfig,
    pub(super) proposal_seed: u64,
    pub(super) acquisition_seed: u64,
    pub(super) causal: &'a mut Option<CausalScorer>,
    pub(super) causal_log: &'a mut Option<CausalLog>,
}

pub(super) fn attach_pair(
    candidate: &mut Evaluation,
    incumbent: &ConditionBaseline,
    baseline: f32,
    baseline_variance: f32,
    condition_seed: u64,
    sampling_seed: u64,
    corpus_token_offset: usize,
    incumbent_gpu_seconds: f64,
    incumbent_reused: bool,
) -> Result<(), String> {
    if candidate.observation.is_vector() {
        return Err(
            "paired corpus evaluation currently requires one scalar control objective".into(),
        );
    }
    let candidate_reward = candidate.mean;
    let candidate_variance = candidate.variance;
    let incumbent_reward = incumbent.reward;
    let incumbent_variance = incumbent.variance;
    let paired = candidate
        .rollouts
        .iter()
        .map(|rollout| {
            rollout.target_quality.and_then(|quality| {
                Some((
                    quality.paired_improvement?,
                    quality.paired_improvement_variance?,
                ))
            })
        })
        .collect::<Option<Vec<_>>>();
    let response_components = candidate
        .rollouts
        .iter()
        .map(|rollout| rollout.paired_patch_improvements.len())
        .sum();
    let (improvement, improvement_variance, difference_statistic) =
        if let Some(paired) = paired.filter(|values| !values.is_empty()) {
            let count = paired.len() as f32;
            (
                paired.iter().map(|value| value.0).sum::<f32>() / count,
                paired.iter().map(|value| value.1).sum::<f32>() / (count * count),
                if response_components > 0 {
                    "causal-patch-vector"
                } else {
                    "causal-block-means"
                },
            )
        } else {
            (
                candidate_reward - incumbent_reward,
                candidate_variance + incumbent_variance,
                "independent-marginals",
            )
        };
    let optimizer_value = baseline + improvement;
    let optimizer_variance = baseline_variance + improvement_variance;
    if !improvement.is_finite()
        || !improvement_variance.is_finite()
        || improvement_variance < 0.0
        || !optimizer_value.is_finite()
        || !optimizer_variance.is_finite()
        || optimizer_variance < 0.0
    {
        return Err("paired generation produced a non-finite response".into());
    }
    candidate.mean = optimizer_value;
    candidate.variance = optimizer_variance;
    candidate.observation = crate::objective_observation::ObjectiveObservation::scalar(
        optimizer_value,
        optimizer_variance,
    )?;
    candidate.paired = Some(PairedGeneration {
        condition_seed,
        sampling_seed,
        corpus_token_offset,
        candidate_reward,
        candidate_variance,
        incumbent_reward,
        incumbent_variance,
        improvement,
        improvement_variance,
        response_components,
        difference_statistic,
        optimizer_value,
        optimizer_variance,
        incumbent_generated_tokens: incumbent
            .rollouts
            .iter()
            .map(|rollout| rollout.tokens.len())
            .sum(),
        incumbent_evaluated_positions: incumbent
            .rollouts
            .iter()
            .map(|rollout| rollout.evaluated_positions)
            .sum(),
        incumbent_committed_tokens: incumbent
            .rollouts
            .iter()
            .map(|rollout| rollout.committed_tokens)
            .sum(),
        incumbent_gpu_seconds,
        incumbent_reused,
        incumbent_expert_mask: decode::route_union(&incumbent.rollouts),
        candidate_expert_mask: decode::route_union(&candidate.rollouts),
    });
    Ok(())
}

impl LoopContext<'_> {
    fn alignment(
        &mut self,
        probe: &mut gen_alignment::Probe,
        step: u32,
        temperature: f32,
        candidate: &BufferRef,
        candidate_nll: f32,
        path: &Path,
    ) -> Result<gen_alignment::Alignment, String> {
        let condition = self.round_condition(step, temperature)?;
        let verifier = self
            .verifier
            .as_ref()
            .ok_or("optimizer alignment requires accepted-prefix verification")?;
        let incumbent = score(
            self.runtime,
            self.decoder,
            Some((verifier, None)),
            self.weights.row(probe.incumbent())?,
            &condition.config,
            condition.evaluation_seed,
            &path.join("alignment-incumbent"),
            self.evaluator.as_mut(),
            self.tokenizer,
            false,
        )?;
        let incumbent_nll = conditional_nll(&incumbent)?;
        let mut alignment = probe.update(
            self.runtime,
            verifier,
            &condition.config.tasks[0],
            condition.config.max_tokens,
            candidate,
            incumbent_nll,
            candidate_nll,
        )?;
        let muon = score(
            self.runtime,
            self.decoder,
            Some((verifier, None)),
            self.weights.row(probe.muon())?,
            &condition.config,
            condition.evaluation_seed,
            &path.join("alignment-muon"),
            self.evaluator.as_mut(),
            self.tokenizer,
            false,
        )?;
        let muon_nll = conditional_nll(&muon)?;
        alignment.muon_nll = Some(muon_nll);
        alignment.muon_nll_change = Some(muon_nll - incumbent_nll);
        Ok(alignment)
    }

    fn round_condition(&self, step: u32, temperature: f32) -> Result<RoundCondition, String> {
        let mut config = self.config.clone();
        config.temperature = temperature;
        // Acquisition ranks the complete candidate pool, then evaluates only
        // its selected member. Each completed evaluation therefore advances
        // the corpus condition exactly once.
        let condition = step;
        let condition_seed = crate::hash::splitmix64(
            self.sampling_seed ^ u64::from(condition + 1) ^ 0x636f_7270_7573_2d78,
        );
        let sampling_seed = crate::hash::splitmix64(condition_seed ^ 0x7361_6d70_6c65_2d78);
        let corpus_token_offset = if let Some(dataset) = self.dataset {
            let (task, offset) = setup::corpus_window(
                dataset,
                &config.tasks[0],
                self.corpus_prompt_cap,
                self.corpus_output_cap,
                self.sampling_seed,
                condition,
            )?;
            config.max_tokens = u32::try_from(task.expected.len())
                .map_err(|_| "sampled natural target exceeds u32 token capacity")?;
            config.tasks[0] = task;
            offset
        } else {
            0
        };
        Ok(RoundCondition {
            config,
            condition_seed,
            evaluation_seed: if self.dataset.is_some() {
                sampling_seed
            } else {
                self.sampling_seed
            },
            corpus_token_offset,
        })
    }

    fn preview(
        &self,
        retain: bool,
        evaluation: &Evaluation,
        path: &Path,
    ) -> Result<Option<String>, String> {
        if !retain {
            return Ok(None);
        }
        decoded_completion(self.tokenizer, &evaluation.rollouts[0], path, false)
    }

    fn retain_round(
        &self,
        step: u32,
        path: &Path,
        evaluation: &mut Evaluation,
    ) -> Result<Option<String>, String> {
        if !self.checkpoint_step(step) {
            return Ok(None);
        }
        if transient(&self.config.reward) {
            retain(&path.join("candidate"), evaluation)?;
        }
        self.preview(true, evaluation, path)
    }

    pub(super) fn checkpoint_step(&self, step: u32) -> bool {
        step == 0
            || step + 1 == self.run.rounds()
            || learning_checkpoint(step + 1)
            || self
                .run
                .validation_interval
                .is_some_and(|interval| (step + 1) % interval == 0)
    }
    fn annotate_evaluation(
        evaluation: &Evaluation,
        record: &mut blckbaux_artifact::json::Value,
    ) -> Result<(), String> {
        if let Some(causal) = evaluation.causal {
            record["causal_pretraining"] =
                blckbaux_artifact::json::to_value(causal).map_err(|error| error.to_string())?;
        }
        if let Some(paired) = evaluation.paired {
            record["paired_generation"] =
                blckbaux_artifact::json::to_value(paired).map_err(|error| error.to_string())?;
        }
        evaluation.annotate(record)?;
        Ok(())
    }

    fn validate_causal(&mut self, state: &LoopState, round: u32) -> Result<(), String> {
        if let (Some(scorer), Some(log)) = (self.causal.as_ref(), self.causal_log.as_mut()) {
            let base = state.search.base_buffer();
            log.observe(scorer, self.runtime, self.weights.row(&base)?, round)?;
        }
        Ok(())
    }

    fn observe_incumbent(
        &mut self,
        state: &LoopState,
        config: &GenerationConfig,
        buffer: &metal::Buffer,
        path: &Path,
        seed: u64,
        step: u32,
    ) -> Result<Option<Evaluation>, String> {
        if self.causal.is_some() || matches!(config.reward, GenerationReward::CausalObjectives) {
            return Ok(None);
        }
        let region = state.search.morbo_region().unwrap_or(0);
        let mut config = config.clone();
        config.temperature = state.regions.temperatures[region];
        let persist = !transient(&config.reward) || self.checkpoint_step(step);
        Ok(Some(score(
            self.runtime,
            self.decoder,
            self.verifier.as_ref().map(|verifier| (verifier, None)),
            self.weights.row(buffer)?,
            &config,
            seed,
            &path.join("incumbent"),
            self.evaluator.as_mut(),
            self.tokenizer,
            persist,
        )?))
    }

    fn condition_baseline(
        &mut self,
        state: &mut LoopState,
        region: usize,
        condition: &RoundCondition,
        incumbent: &metal::Buffer,
        path: &Path,
        step: u32,
    ) -> Result<Option<(ConditionBaseline, f64, bool)>, String> {
        if self.causal.is_some()
            || matches!(condition.config.reward, GenerationReward::CausalObjectives)
        {
            return Ok(None);
        }
        if let Some(baseline) = state.condition_baselines[region]
            .as_ref()
            .filter(|baseline| baseline.condition_seed == condition.condition_seed)
        {
            return Ok(Some((baseline.clone(), 0.0, true)));
        }
        if let Some(verifier) = self.verifier {
            verifier.clear_pair();
        }
        let measured = self
            .observe_incumbent(
                state,
                &condition.config,
                incumbent,
                path,
                condition.evaluation_seed,
                step,
            )?
            .ok_or("pretraining did not produce an incumbent observation")?;
        if let Some(verifier) = self.verifier {
            verifier.capture_pair(self.runtime, condition.config.max_tokens as usize)?;
        }
        let baseline = ConditionBaseline::measured(condition.condition_seed, &measured);
        let gpu_seconds = baseline.gpu_seconds();
        state.condition_baselines[region] = Some(baseline.clone());
        Ok(Some((baseline, gpu_seconds, false)))
    }

    fn evaluate(
        &mut self,
        state: &mut LoopState,
        row: &metal::Buffer,
        proposal: &crate::bf16_metal::Proposals,
        path: &Path,
        initial: bool,
        step: u32,
    ) -> Result<(Evaluation, f32, crate::bf16_metal::NoisyDecision, Duration), String> {
        let region = state.search.morbo_region().unwrap_or(0);
        let temperature = proposal
            .axis_value()
            .unwrap_or(state.regions.temperatures[region]);
        let condition = self.round_condition(step, temperature)?;
        let incumbent_buffer = state.search.base_buffer();
        let incumbent =
            self.condition_baseline(state, region, &condition, &incumbent_buffer, path, step)?;
        let candidate_path = path.join("candidate");
        let oracle_path = if incumbent.is_some() {
            candidate_path.as_path()
        } else {
            path
        };
        let mut evaluation = if self
            .run
            .pool_rank_audit
            .is_some_and(|points| state.pool_rank_audited < points)
            && !initial
        {
            self.audit_pool(
                state,
                proposal,
                &condition,
                incumbent.as_ref(),
                region,
                path,
                step,
            )?
        } else {
            self.score_candidate(
                state,
                row,
                &condition,
                incumbent.as_ref(),
                region,
                oracle_path,
                step,
            )?
        };
        let observation = evaluation.observation;
        let tell_start = Instant::now();
        let decision = if observation.is_vector() {
            state.search.modeled_objectives(proposal, observation)?
        } else if evaluation.paired.is_some() {
            super::gen_pool::tell_paired(&mut state.search, proposal, &evaluation)?
        } else if initial {
            state.search.initial_objectives(proposal, observation)?
        } else if let Some(causal) = evaluation.causal {
            state.search.paired_modeled(
                proposal,
                causal.optimizer_value,
                causal.variance,
                state.search.best()?,
                state.search.best_variance()?,
                causal.improvement,
                causal.variance,
            )?
        } else {
            state.search.modeled_objectives(proposal, observation)?
        };
        if state.search.sync()? != vec![decision.accepted] {
            return Err("resident search acceptance mismatch".into());
        }
        if decision.accepted && evaluation.paired.is_some() {
            if let Some(verifier) = self.verifier {
                verifier.capture_pair(self.runtime, condition.config.max_tokens as usize)?;
            }
            state.condition_baselines[region] = Some(ConditionBaseline::accepted(
                condition.condition_seed,
                &evaluation,
            )?);
        }
        let tell_time = tell_start.elapsed();
        Ok((evaluation, temperature, decision, tell_time))
    }

    fn propose_step(
        &self,
        state: &mut LoopState,
        step: u32,
    ) -> Result<(metal::Buffer, crate::bf16_metal::Proposals, bool), String> {
        let mut ask = self.policy.ask;
        ask.acquisition.seed = self.acquisition_seed + u64::from(step);
        let pool_count = state.search.pool_layout().count() as usize;
        let forced = (self.run.selection == Some(crate::config::PretrainSelection::Random))
            .then(|| (crate::hash::splitmix64(ask.acquisition.seed) as usize) % pool_count);
        state.search.propose(
            self.proposal_seed + u64::from(step),
            ask,
            step as usize % pool_count,
            forced,
        )
    }

    fn validate_checkpoint(&mut self, state: &mut LoopState, round: u32) -> Result<(), String> {
        let scheduled = learning_checkpoint(round)
            || self
                .run
                .validation_interval
                .is_some_and(|interval| round % interval == 0);
        if !scheduled || round == self.run.rounds() {
            return Ok(());
        }
        let started = Instant::now();
        let mut config = self.config.clone();
        config.temperature = state.temperature;
        let rewards = validate_generation(
            self.runtime,
            self.decoder,
            self.verifier.as_ref(),
            self.weights,
            &state.search.base_buffer(),
            &config,
            self.sampling_seed,
            &self.output.join(format!("validation-{round:06}")),
            self.evaluator.as_mut(),
            self.tokenizer,
            self.validation_tasks,
        )?;
        state
            .journal
            .validation(round, &rewards, started.elapsed().as_secs_f64())?;
        self.validate_causal(state, round)
    }

    pub(super) fn run(&mut self, state: &mut LoopState) -> Result<(), String> {
        let mut alignment = self
            .config
            .record_tensor_updates
            .then(|| {
                let rows = self
                    .verifier
                    .as_ref()
                    .ok_or("optimizer alignment requires accepted-prefix verification")?
                    .context;
                gen_alignment::Probe::new(self.runtime, self.weights, rows)
            })
            .transpose()?;
        let rounds = if std::env::var_os("BLCKBAUX_UPDATE_ALIGNMENT_SINGLE").is_some() {
            1
        } else {
            self.run.rounds()
        };
        for step in 0..rounds {
            let start = Instant::now();
            let (row, proposal, initializing) = self.propose_step(state, step)?;
            let region = state.search.morbo_region().unwrap_or(0);
            let tensor_version = state
                .search
                .tensor_version(state.regions.versions[region], &proposal)?;
            let proposal_seconds = start.elapsed().as_secs_f64();
            let changes = state.search.describe(&proposal)?.remove(0).3;
            let round_path = self.output.join(format!("round-{:04}", step + 1));
            let diagnose = self.checkpoint_step(step);
            if diagnose && let Some(probe) = alignment.as_ref() {
                probe.preserve(self.runtime, &state.search.base_buffer())?;
            }
            let evaluation_start = Instant::now();
            let (mut evaluation, candidate_temperature, decision, tell_time) =
                self.evaluate(state, &row, &proposal, &round_path, initializing, step)?;
            let evaluation_seconds =
                evaluation_start.elapsed().as_secs_f64() - tell_time.as_secs_f64();
            let diagnostic_start = Instant::now();
            let mut alignment = if diagnose {
                alignment
                    .as_mut()
                    .map(|probe| {
                        self.alignment(
                            probe,
                            step,
                            candidate_temperature,
                            &row,
                            conditional_nll(&evaluation)?,
                            &round_path,
                        )
                    })
                    .transpose()?
            } else {
                None
            };
            let diagnostic_seconds = alignment
                .as_mut()
                .map(|alignment| {
                    let seconds = diagnostic_start.elapsed().as_secs_f64();
                    alignment.diagnostic_ms = seconds * 1000.0;
                    seconds
                })
                .unwrap_or(0.0);
            let tell_seconds = tell_time.as_secs_f64();
            let changed_weights = changes.iter().map(|change| change.0).sum::<u64>();
            state.minimum_changed_weights = state.minimum_changed_weights.min(changed_weights);
            let changed_fraction =
                changed_weights as f64 / self.weights.architecture.parameter_count() as f64;
            let code = codelength(&evaluation.rollouts);
            state.commit(
                &evaluation,
                decision.accepted,
                region,
                tensor_version.id,
                candidate_temperature,
            );
            state.record_updates(
                self.config.record_tensor_updates,
                step + 1,
                &proposal,
                decision.accepted,
                changes,
            )?;
            let text = self.retain_round(step, &round_path, &mut evaluation)?;
            let controller = state.search.controller_info()?;
            let mut record = optimization_record(
                step + 1,
                initializing,
                &proposal,
                decision,
                controller,
                state.search.reliability_info()?,
                start.elapsed().as_secs_f64() - diagnostic_seconds,
                proposal_seconds + tell_time.as_secs_f64(),
            );
            annotate_proposal(&state.search, &proposal, &mut record);
            if let Some(alignment) = alignment {
                record["muon_alignment"] = blckbaux_artifact::json::to_value(alignment)
                    .map_err(|error| error.to_string())?;
            }
            record["response"] =
                response_record(self, &proposal, &evaluation, decision, initializing)?;
            record["generation"] = blckbaux_artifact::json::json!({"reward":evaluation.reported_reward(),"optimizer_value":evaluation.mean,"variance":evaluation.variance,
            "diagnostics":evaluation.diagnostics,"selection":self.run.selection.unwrap_or_default(),
            "temperature":candidate_temperature,"incumbent_temperature":state.temperature,
                "proposal_seconds":proposal_seconds,"reward_seconds":evaluation.reward_seconds,
            "evaluation_seconds":evaluation_seconds,"tell_seconds":tell_seconds,
            "draft":evaluation.rollouts.iter().map(|r| &r.draft).collect::<Vec<_>>(),
            "target_evaluated_positions":evaluation.rollouts.iter().map(|r| r.evaluated_positions - r.draft.as_ref().map_or(0, |d| d.evaluated_positions)).sum::<usize>(),
            "target_gpu_seconds":evaluation.rollouts.iter().map(|r| r.gpu_seconds - r.draft.as_ref().map_or(0.0, |d| d.generation_gpu_seconds + d.cache_gpu_seconds)).sum::<f64>(),
            "position_amplification":evaluation.rollouts.iter().map(|r| r.evaluated_positions).sum::<usize>() as f64 / evaluation.rollouts.iter().map(|r| r.tokens.len()).sum::<usize>().max(1) as f64,
            "changed_weights":changed_weights,"changed_fraction":changed_fraction,
            "generated_tokens":evaluation.rollouts.iter().map(|r| r.tokens.len()).sum::<usize>(),
            "prompt_tokens":evaluation.prompt_tokens,
            "context_tokens":evaluation.prompt_tokens + evaluation.rollouts.iter().map(|r| r.token_count).max().unwrap_or(0),
            "generation_samples":evaluation.rollouts.len(),
            "drafted_tokens":evaluation.rollouts.iter().filter_map(|r| r.draft.as_ref()).map(|draft| draft.generated_tokens).sum::<usize>(),
            "scored_tokens":if code.tokens == 0 && matches!(self.config.reward, GenerationReward::SequenceKernel) { evaluation.rollouts.iter().map(|r| r.token_count).sum::<usize>() } else { code.tokens },
            "codelength_bits":if code.tokens == 0 { None } else { Some(code.bits) },
            "bits_per_token":if code.tokens == 0 { None } else { Some(code.bits / code.tokens as f64) },
            "codelength_scope":"complete-reference-continuation-under-true-causal-prefix",
            "evaluated_core_positions":evaluation.rollouts.iter().map(|r| r.evaluated_positions).sum::<usize>(),
            "evaluated_tokens":evaluation.rollouts.iter().map(|r| if self.weights.architecture == ResidualArchitecture::HnetMhc4 { r.token_count } else { r.evaluated_positions }).sum::<usize>(),
            "evaluated_positions":evaluation.rollouts.iter().map(|r| r.evaluated_positions).sum::<usize>(),
            "committed_tokens":evaluation.rollouts.iter().map(|r| r.committed_tokens).sum::<usize>(),
            "rollout_gpu_seconds":evaluation.rollouts.iter().map(|r| r.gpu_seconds).sum::<f64>(),
            "broad_passes":evaluation.rollouts.iter().map(|r| r.broad_passes).sum::<usize>(),
            "correction_waves":evaluation.rollouts.iter().map(|r| r.correction_waves).sum::<usize>(),
            "repair_batches":evaluation.rollouts.iter().map(|r| r.repair_batches).sum::<usize>(),
            "repaired_tokens":evaluation.rollouts.iter().map(|r| r.repaired_tokens).sum::<usize>(),
            "accepted_tokens":evaluation.rollouts.iter().map(|r| r.accepted_tokens).sum::<usize>(),
            "first_mismatch":evaluation.rollouts.iter().map(|r| r.first_mismatch).collect::<Vec<_>>(),
            "evaluated_lengths":evaluation.rollouts.iter().map(|r| &r.evaluated_lengths).collect::<Vec<_>>(),
            "accepted_lengths":evaluation.rollouts.iter().map(|r| &r.accepted_lengths).collect::<Vec<_>>(),
            "committed_lengths":evaluation.rollouts.iter().map(|r| &r.committed_lengths).collect::<Vec<_>>(),
            "route_samples":evaluation.rollouts.iter().map(|r| &r.route_samples).collect::<Vec<_>>()});
            Self::annotate_evaluation(&evaluation, &mut record["generation"])?;
            if state.pool_rank_report.is_some() {
                record["pool_rank_audit"] = state.pool_rank_report.take().unwrap();
            }
            state.record_rank(&evaluation, &mut record);
            if let Some(report) = state.search.objective_report()? {
                let width = report.width as usize;
                record["acquisition"]["objectives"] = blckbaux_artifact::json::json!({
                    "method": "vector",
                    "values": report.values.map(|values| values[..width].to_vec()),
                    "weights": report.weights[..width].to_vec(),
                    "nondominated_mask": report.nondominated_mask,
                    "selected": report.selected,
                    "region": region,
                    "regions": state.search.morbo_regions(),
                });
            }
            record["tensor_version"] = blckbaux_artifact::json::json!({
                "id": tensor_version.id,
                "parent": tensor_version.parent,
                "seed": tensor_version.seed,
                "radius": tensor_version.radius,
                "distribution": tensor_version.distribution,
                "blocks": tensor_version.blocks.len(),
                "basis_seed": tensor_version.basis_seed,
                "threshold_words": tensor_version.threshold_words,
                "accepted_parent": state.incumbent_version,
            });
            if let Some(bounds) = proposal.bound_report() {
                record["history_bounds"] = blckbaux_artifact::json::json!({
                    "history": bounds.history,
                    "neighbors": bounds.neighbors,
                    "raw_survivors": bounds.raw_survivors,
                    "final_survivors": bounds.final_survivors,
                    "bound_violations": bounds.bound_violations,
                    "rank_violations": bounds.rank_violations,
                    "pruning_enabled": false,
                });
            }
            state.write_record(&record, self.checkpoint_step(step))?;
            let wall = start.elapsed().as_secs_f64() - diagnostic_seconds;
            state.loop_seconds += wall;
            state.walls.push(wall);
            state.journal.training(
                step + 1,
                wall,
                &evaluation,
                decision.accepted,
                if initializing {
                    "initialization"
                } else {
                    "optimization"
                },
            )?;
            report_round(
                step + 1,
                self.run.rounds(),
                RoundTiming {
                    wall,
                    proposal: proposal_seconds,
                    evaluation: evaluation_seconds,
                    tell: tell_seconds,
                },
                RoundTrust {
                    proposed: proposal.length,
                    next: controller.length,
                    outcome: decision.trust_outcome,
                    successes: controller.success_counter,
                    failures: controller.failure_counter,
                },
                candidate_temperature,
                state.temperature,
                changed_weights,
                changed_fraction,
                &evaluation,
                decision,
                text.as_deref(),
            );
            state
                .journal
                .gate(step + 1, self.config.signal_gate.as_ref())?;
            self.validate_checkpoint(state, step + 1)?;
        }
        Ok(())
    }
}

fn conditional_nll(evaluation: &Evaluation) -> Result<f32, String> {
    evaluation
        .rollouts
        .first()
        .and_then(|rollout| rollout.conditional_target_nll)
        .ok_or_else(|| "optimizer comparison requires conditional target NLL".into())
}

pub(super) fn transient(reward: &GenerationReward) -> bool {
    matches!(
        reward,
        GenerationReward::ConditionalCrossEntropy
            | GenerationReward::CausalObjectives
            | GenerationReward::BoundedCrossEntropy
            | GenerationReward::SequenceKernel
    )
}
