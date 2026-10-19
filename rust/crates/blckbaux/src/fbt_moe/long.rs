//! Fixed-corpus long-context pretraining for the H-Net forward program.
//!
//! The objective is corpus next-token cross-entropy, not sampled-text quality.
//! Each round proposes one full parameter row, selects a corpus window, and
//! scores incumbent and candidate on that same window. Their measured loss
//! difference supplies acceptance evidence and the relative reward accumulated
//! in ENN history. Warm-up and subsequent paired-model decisions have distinct
//! thresholds; ask-time predictions supply model-agreement evidence, not the
//! measured loss difference itself.
//!
//! Validation scores the current incumbent on fixed per-source windows without
//! contributing observations or acceptance decisions. Training window scores,
//! modeled incumbent values, and held-out mixture means are distinct quantities.
//! This module does not establish calibration, regret, or a latency guarantee.
//!
//! Per-round wall time excludes validation and initial setup. `loop_seconds`
//! includes interleaved validation; GPU round time covers paired scoring,
//! not proposal kernels. Artifact serialization is performed by the worker.

use super::*;

/// One evaluated corpus window: negative NLL in nats/token, its estimated mean
/// variance, total codelength in bits, actual scored tokens, and scorer GPU time
/// in seconds. The variance is telemetry from the scorer, not a certified bound.
struct BatchScore {
    reward: f32,
    variance: f32,
    bits: f64,
    tokens: usize,
    gpu_seconds: f64,
    paired_gain: Option<f32>,
    paired_var: Option<f32>,
    patch_gains: Vec<f32>,
}

/// Run the H-Net causal scorer with supplied weights and corpus tokens.
/// Converts mean NLL to higher-is-better reward without changing parameter
/// storage or search state. Mutates scorer/decoder scratch; the caller clears
/// stale paired-loss state when no paired measurement is intended.
fn score_batch(
    scorer: &block_decode::BlockDecoder,
    runtime: &Runtime,
    decoder: &decode::Decoder,
    weights: CandidateRow<'_>,
    batch: &crate::pretrain_data::CausalBatch,
) -> Result<BatchScore, String> {
    let result = scorer.score_batch(runtime, decoder, weights, batch)?;
    Ok(BatchScore {
        reward: -result.quality.mean_nll,
        variance: result.quality.mean_nll_variance,
        bits: result.quality.codelength_bits,
        tokens: result.quality.scored_tokens,
        gpu_seconds: result.gpu_seconds,
        paired_gain: result.quality.paired_improvement,
        paired_var: result.quality.paired_improvement_variance,
        patch_gains: result.patch_gains,
    })
}

/// Score a fixed window per validation source and weight source means.
/// The seed is reused independently of `round`, which is only a record label.
/// Requires an evaluation `CorpusSet`; it does not send observations to ENN.
/// `scored_tokens` in the record is the requested window size, whereas
/// `evaluated_tokens` sums actual scored positions over sources. These counts
/// can differ because of H-Net boundary context and multiple source evaluations.
fn validate(
    scorer: &block_decode::BlockDecoder,
    runtime: &Runtime,
    decoder: &decode::Decoder,
    weights: &CandidateWeights,
    row: &BufferRef,
    dataset: &crate::pretrain_data::CorpusSet,
    window_tokens: usize,
    seed: u64,
    round: u32,
) -> Result<blckbaux_artifact::json::Value, String> {
    scorer.clear_pair();
    let mut nll = 0.0f64;
    let mut bits = 0.0f64;
    let mut gpu_seconds = 0.0f64;
    let mut evaluated_tokens = 0usize;
    let mut sources = Vec::new();
    for batch in dataset.causal_batches(window_tokens, seed)? {
        let score = score_batch(scorer, runtime, decoder, weights.row(row)?, &batch.batch)?;
        let weight = f64::from(batch.weight);
        nll += f64::from(-score.reward) * weight;
        bits += score.bits / score.tokens as f64 * weight;
        gpu_seconds += score.gpu_seconds;
        evaluated_tokens += score.tokens;
        sources.push(blckbaux_artifact::json::json!({
            "source":batch.source,
            "weight":batch.weight,
            "nll":-score.reward,
            "bits_per_token":score.bits / score.tokens as f64,
            "corpus_offset":batch.batch.offset,
            "scored_tokens":score.tokens,
            "gpu_seconds":score.gpu_seconds,
        }));
    }
    let total = dataset.total_weight() as f64;
    Ok(blckbaux_artifact::json::json!({
        "round":round,
        "nll":nll / total,
        "bits_per_token":bits / total,
        "scored_tokens":window_tokens,
        "evaluated_tokens":evaluated_tokens,
        "gpu_seconds":gpu_seconds,
        "used_for_acceptance":false,
        "scope":"mixture-weighted-causal-likelihood",
        "sources":sources,
    }))
}

/// Print a completed validation record without measuring or updating a model.
/// Missing reporting fields fall back to zero/NaN; this display adapter is not
/// schema validation and never makes an acceptance decision.
fn emit_validation(record: &blckbaux_artifact::json::Value) {
    let sources = record["sources"]
        .as_seq()
        .map(|sources| {
            sources
                .iter()
                .map(|source| {
                    format!(
                        "{}:{:.9}",
                        source["source"].as_str().unwrap_or("unknown"),
                        source["nll"].as_f64().unwrap_or(f64::NAN),
                    )
                })
                .collect::<Vec<_>>()
                .join(",")
        })
        .unwrap_or_default();
    eprintln!(
        "BLCKBAUX_PRETRAIN_VALIDATION round={} nll={:.9} bits_per_token={:.9} scored_tokens={} evaluated_tokens={} sources={sources} used_for_acceptance=false",
        record["round"].as_u64().unwrap_or(0),
        record["nll"].as_f64().unwrap_or(f64::NAN),
        record["bits_per_token"].as_f64().unwrap_or(f64::NAN),
        record["scored_tokens"].as_u64().unwrap_or(0),
        record["evaluated_tokens"].as_u64().unwrap_or(0),
    );
}

/// Execute the fixed-corpus Metal/H-Net experiment and return measurements.
/// Requires a validated nonzero-round configuration, window size, supported
/// H-Net architecture, and loadable corpora. Initializes weights/search once;
/// each round uses `propose`, same-window incumbent/candidate scores,
/// initialization or paired modeled tell, and `sync`. Rejected trials still
/// provide ENN observations.
/// Optional validation is held out of those decisions. Returned timing samples
/// use the scopes in the module contract; no learned checkpoint is saved here.
/// Errors propagate rather than starting a replacement experiment or rolling
/// back completed rounds. Successful execution does not imply successful learning.
pub(super) fn run(
    run: &crate::config::RunConfig,
    dataset_path: &std::path::Path,
) -> Result<ActualBoResult, String> {
    let window_tokens =
        run.window_tokens
            .ok_or("long pretraining requires objective.window-tokens")? as usize;
    let architecture =
        ResidualArchitecture::from_model(run.model.ok_or("long pretraining requires a model")?);
    if architecture != ResidualArchitecture::HnetMhc4 {
        return Err("long pretraining requires fbt-pisa1-hnet-mhc4-v1".into());
    }
    let dataset = crate::pretrain_data::CorpusSet::load(dataset_path)?;
    let validation = run
        .validation_dataset
        .as_deref()
        .map(crate::pretrain_data::CorpusSet::load_eval)
        .transpose()?;
    let runtime = Runtime::shared()?;
    let core_rows = window_tokens
        .div_ceil(architecture.patch_size())
        .next_power_of_two()
        .max(CONTEXT as usize);
    let context = u32::try_from(core_rows).map_err(|error| error.to_string())?;
    let decoder = decode::Decoder::with_visits(
        &runtime,
        context,
        context > ROWS,
        architecture.layer_steps().len(),
    )?;
    let mut scorer =
        block_decode::BlockDecoder::with_architecture(&runtime, context, architecture)?;
    scorer.profile(&runtime, run.scorer_stage_samples)?;
    let weights = CandidateWeights::initialized(
        &runtime,
        run.model_seed(),
        crate::config::ModelInitialization::FbtPisa1HnetMhc4,
        architecture,
    )?;
    let shape = run
        .trust_region_shape
        .unwrap_or(crate::config::TrustRegionShape::TensorFamilyStatic);
    let (mut search, mut updates) =
        weights.search_shaped(run.proposal_kernel(), run.length(), shape)?;
    let policy = run.resident_enn(run.acquisition_seed())?;
    search.configure_enn(policy)?;
    if let Some(controller) = run.reliability_controller() {
        search.configure_controller(controller)?;
    }
    search.set_profiling(run.trace());

    let initial_batch = dataset.causal_batch(window_tokens, run.reference_seed(), 0)?;
    scorer.clear_pair();
    let initial = score_batch(
        &scorer,
        &runtime,
        &decoder,
        weights.row(&search.base_buffer())?,
        &initial_batch.batch,
    )?;
    search.observe_initial(0.0, 0.0)?;
    eprintln!(
        "BLCKBAUX_PRETRAIN_INITIAL source={} window_tokens={window_tokens} scored_tokens={} corpus_offset={} nll={:.9} bits_per_token={:.9} gpu_ms={:.3}",
        initial_batch.source,
        initial.tokens,
        initial_batch.batch.offset,
        -initial.reward,
        initial.bits / initial.tokens as f64,
        initial.gpu_seconds * 1000.0,
    );

    let mut validation_records = Vec::new();
    if let Some(validation) = validation.as_ref() {
        let record = validate(
            &scorer,
            &runtime,
            &decoder,
            &weights,
            &search.base_buffer(),
            validation,
            window_tokens,
            run.reference_seed(),
            0,
        )?;
        emit_validation(&record);
        validation_records.push(record);
    }

    let started = Instant::now();
    let mut wall = Vec::with_capacity(run.rounds() as usize);
    let mut gpu = Vec::with_capacity(run.rounds() as usize);
    let mut optimizer_seconds = Vec::with_capacity(run.rounds() as usize);
    let mut optimization_records = Vec::with_capacity(run.rounds() as usize);
    let mut accepted = 0u32;
    let mut cumulative_tokens = 0usize;
    let mut cumulative_wall_seconds = 0.0f64;
    let mut cumulative_gpu_seconds = 0.0f64;
    let mut incumbent_nll = -initial.reward;
    let mut reuse = blckbaux_semantics::optimizer::SeedReuse::default();
    for step in 0..run.rounds() {
        let round_start = Instant::now();
        let mut ask = policy.ask;
        ask.acquisition.seed = run
            .acquisition_seed()
            .checked_add(u64::from(step))
            .ok_or("acquisition seed overflow")?;
        let pool_count = search.pool_layout().count() as usize;
        let forced = (run.selection == Some(crate::config::PretrainSelection::Random))
            .then(|| crate::hash::splitmix64(ask.acquisition.seed) as usize % pool_count);
        let (prop_seed, transported) = reuse.select(run.proposal_seed(), u64::from(step))?;
        if let Some(target) = shape.active_group(step, prop_seed) {
            search.set_active(Some(target))?;
            eprintln!(
                "TURBO_ENN_ACTIVE round={} family={} name={}",
                step + 1,
                target,
                match target {
                    0 => "experts",
                    1 => "projections",
                    2 => "router_feedback",
                    3 => "norms_control",
                    _ => "unknown",
                }
            );
        }
        let ask_start = Instant::now();
        let (row, proposal, initializing) =
            search.propose(prop_seed, ask, step as usize % pool_count, forced)?;
        let ask_seconds = ask_start.elapsed().as_secs_f64();
        let batch = dataset.causal_batch(window_tokens, run.reference_seed(), step)?;

        scorer.clear_pair();
        let incumbent = score_batch(
            &scorer,
            &runtime,
            &decoder,
            weights.row(&search.base_buffer())?,
            &batch.batch,
        )?;
        scorer.capture_pair(&runtime, incumbent.tokens)?;
        let candidate = score_batch(
            &scorer,
            &runtime,
            &decoder,
            weights.row(&row)?,
            &batch.batch,
        )?;
        scorer.clear_pair();

        let improvement = candidate
            .paired_gain
            .ok_or("paired score did not return improvement")?;
        let gain_variance = candidate
            .paired_var
            .ok_or("paired score did not return variance")?;
        if candidate.patch_gains.is_empty() {
            return Err("paired long-context score returned no patch evidence".into());
        }
        let incumbent_value = search.best()?;
        let incumbent_variance = search.best_variance()?;
        let value = incumbent_value + improvement;
        let variance = incumbent_variance + gain_variance;
        let changes = search.describe(&proposal)?.remove(0).3;

        let tell_start = Instant::now();
        let decision = if initializing {
            search.tell_initial(&proposal, value, variance)?
        } else {
            search.paired_blocks(
                &proposal,
                value,
                variance,
                incumbent_value,
                incumbent_variance,
                improvement,
                gain_variance,
                &candidate.patch_gains,
            )?
        };
        if search.sync()? != vec![decision.accepted] {
            return Err("long pretraining acceptance synchronization failed".into());
        }
        let tell_seconds = tell_start.elapsed().as_secs_f64();
        accepted += u32::from(decision.accepted);
        let changed_coords: u64 = changes.iter().map(|(count, _)| *count).sum();
        updates.push_scaled(
            step + 1,
            proposal.seed,
            proposal.length,
            decision.accepted,
            changes,
            proposal.block_scales.clone(),
        )?;
        let gpu_seconds = incumbent.gpu_seconds + candidate.gpu_seconds;
        let wall_seconds = round_start.elapsed().as_secs_f64();
        wall.push(wall_seconds);
        gpu.push(gpu_seconds);
        cumulative_tokens += candidate.tokens;
        cumulative_wall_seconds += wall_seconds;
        cumulative_gpu_seconds += gpu_seconds;
        if decision.accepted {
            incumbent_nll = -candidate.reward;
        }
        reuse.observe(prop_seed, decision.accepted, initializing);
        let paired_gain = candidate.paired_gain;
        let paired_se = candidate.paired_var.map(|v| v.max(0.0).sqrt());
        let next_radius = search.length()?;
        optimizer_seconds.push(ask_seconds + tell_seconds);
        optimization_records.push(blckbaux_artifact::json::json!({
            "round": step + 1,
            "initializing": initializing,
            "transported": transported,
            "accepted": decision.accepted,
            "candidate_nll": -candidate.reward,
            "incumbent_nll": incumbent_nll,
            "paired_gain": paired_gain,
            "paired_se": paired_se,
            "optimizer_value": value,
            "optimizer_variance": variance,
            "scored_tokens": candidate.tokens,
            "cumulative_tokens": cumulative_tokens,
            "codelength_bits": candidate.bits,
            "bits_per_token": candidate.bits / candidate.tokens as f64,
            "source": &batch.source,
            "corpus_offset": batch.batch.offset,
            "window_tokens": window_tokens,
            "objective": "fixed-corpus-causal-likelihood",
            "candidate_gpu_ms": candidate.gpu_seconds * 1000.0,
            "base_gpu_ms": incumbent.gpu_seconds * 1000.0,
            "round_gpu_ms": gpu_seconds * 1000.0,
            "round_wall_ms": wall_seconds * 1000.0,
            "cumulative_wall_seconds": cumulative_wall_seconds,
            "cumulative_gpu_seconds": cumulative_gpu_seconds,
            "ask_ms": ask_seconds * 1000.0,
            "tell_ms": tell_seconds * 1000.0,
            "proposal_radius": proposal.length,
            "next_radius": next_radius,
            "changed_coords": changed_coords,
            "active_family": proposal.active_family,
        }));
        eprintln!(
            "BLCKBAUX_PRETRAIN_ROUND round={} total={} source={} offset={} scored_tokens={} nll={:.9} accepted={} transported={} paired_gain={:.9} paired_se={:.9} proposal_radius={:.6} next_radius={:.6} gpu_ms={:.3} wall_ms={:.3}",
            step + 1,
            run.rounds(),
            batch.source,
            batch.batch.offset,
            candidate.tokens,
            -candidate.reward,
            decision.accepted,
            transported,
            paired_gain.unwrap_or(0.0),
            paired_se.unwrap_or(0.0),
            proposal.length,
            next_radius,
            gpu_seconds * 1000.0,
            wall_seconds * 1000.0,
        );
        let round = step + 1;
        if let Some(validation) = validation.as_ref()
            && (round % run.validation_interval.unwrap_or(run.rounds()) == 0
                || round == run.rounds())
        {
            let record = validate(
                &scorer,
                &runtime,
                &decoder,
                &weights,
                &search.base_buffer(),
                validation,
                window_tokens,
                run.reference_seed(),
                round,
            )?;
            emit_validation(&record);
            validation_records.push(record);
        }
    }
    let loop_seconds = started.elapsed().as_secs_f64();
    wall.sort_by(f64::total_cmp);
    gpu.sort_by(f64::total_cmp);
    Ok(ActualBoResult {
        parameters: architecture.parameter_count(),
        loop_seconds,
        median_wall_seconds: wall[wall.len() / 2],
        min_wall_seconds: wall[0],
        max_wall_seconds: wall[wall.len() - 1],
        median_gpu_seconds: gpu[gpu.len() / 2],
        accepted,
        learning_seconds: loop_seconds,
        validation: validation_records,
        optimizer_seconds,
        optimization_records,
        updates,
    })
}
