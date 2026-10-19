use blckbaux::experimental::{
    Backend, BackendSchedule, HNET_COEFFS, HNET_PARAMS, HNET_PATCH, ModelProgram, PISA_BYTES,
    ProgramKind, perf_digest, perf_sources,
};
use blckbaux_artifact::json::{Value, json, pretty_string};
use blckbaux_tooling::radar::{PerformanceAction, PerformanceBackend};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

const SCHEMA: &str = "blckbaux.performance-certificate.v1";
const REGRESSION_SCHEMA: &str = "blckbaux.experiment-regression.v1";
const REGRESSION_BASELINE: &str = "performance/hnet-1m-round-parity.json";
const CONTEXT_GOAL: u64 = 1_048_576;
const NOISE_MS: f64 = 2.0;

pub(super) fn run(root: &Path, action: PerformanceAction) -> Result<(), String> {
    match action {
        PerformanceAction::Show {
            config,
            backend,
            run,
            json,
        } => {
            let certificate = certificate(root, &config, backend, run.as_deref())?;
            if json {
                print(&certificate)
            } else {
                summary(&certificate)
            }
        }
        PerformanceAction::Record {
            config,
            backend,
            run,
            output,
        } => record(root, &config, backend, &run, &output),
        PerformanceAction::Gate {
            config,
            backend,
            run,
            baseline,
            tolerance_percent,
        } => gate(
            root,
            &config,
            backend,
            run.as_deref(),
            &baseline,
            tolerance_percent,
        ),
    }
}

fn certificate(
    root: &Path,
    config_path: &Path,
    backend: PerformanceBackend,
    run: Option<&Path>,
) -> Result<Value, String> {
    let config_path = root.join(config_path);
    let (config, _) = blckbaux::config::load_tune(&config_path)?;
    let model = config
        .model
        .ok_or("performance certificate requires a model")?;
    if model.id() != "fbt-pisa1-hnet-mhc4-v1" {
        return Err("performance certificate currently covers fbt-pisa1-hnet-mhc4-v1".into());
    }
    let (mode, prompt, generated, retained, input_positions) =
        if let Some(generation) = config.generation.as_ref() {
            let prompt = prompt_tokens(generation)?;
            let generated = u64::from(generation.max_tokens);
            let retained = prompt
                .checked_add(generated)
                .ok_or("retained context length overflow")?
                .min(CONTEXT_GOAL);
            (
                "generation",
                Some(prompt),
                Some(generated),
                retained,
                retained - u64::from(generation.draft.is_none()),
            )
        } else {
            let tokens = config
                .window_tokens
                .map(u64::from)
                .ok_or("performance certificate requires objective.window-tokens")?;
            ("causal-likelihood", None, None, tokens, tokens)
        };
    let core_positions = input_positions.div_ceil(HNET_PATCH);
    let broad_positions = core_positions.div_ceil(128) * 128;
    let cache_capacity = core_positions
        .checked_next_power_of_two()
        .ok_or("cache capacity overflow")?
        .max(4096);

    let program = ModelProgram::hnet_program();
    let schedule = schedule(backend);
    let missing = schedule.coverage(&program, ProgramKind::ExperimentRound)?;
    let recurrent = program
        .recurrent_core
        .ok_or("model program has no recurrent core")?;
    let executions = program.layer_visits(recurrent.training_visits)?.len() as u64;
    let contraction_flops = HNET_COEFFS
        .checked_mul(2)
        .and_then(|flops| flops.checked_mul(broad_positions))
        .ok_or("contraction work overflow")?;
    let cache_bytes = cache_capacity
        .checked_mul(PISA_BYTES)
        .and_then(|bytes| bytes.checked_mul(executions))
        .ok_or("cache byte count overflow")?;
    let digest = perf_digest(root, backend_kind(backend))?;
    let sources = perf_sources(root, backend_kind(backend))?
        .into_iter()
        .map(|path| {
            path.strip_prefix(root)
                .unwrap_or(&path)
                .display()
                .to_string()
        })
        .collect::<Vec<_>>();
    let active_kernels = schedule.kernels_for(&program, ProgramKind::ExperimentRound)?;
    let kernels = kernel_rows(&active_kernels);
    let static_contract = json!({
        "model": model.id(),
        "program-version": program.version,
        "backend": backend.name(),
        "compiler": compiler_name(schedule.compiler),
        "source-digest": digest,
        "source-files": sources,
        "workload": {
            "mode": mode,
            "prompt-token-cap": prompt,
            "generated-token-cap": generated,
            "window-tokens": (mode == "causal-likelihood").then_some(retained),
            "retained-context-tokens": retained,
            "target-input-positions": input_positions,
            "hnet-patch-tokens": HNET_PATCH,
            "core-positions": core_positions,
            "broad-pass-positions": broad_positions,
            "rounds": config.rounds(),
            "target-round-ms": config.target_ms(),
            "target-tokens-per-second": retained as f64 * 1000.0 / f64::from(config.target_ms()),
        },
        "architecture": {
            "parameters": HNET_PARAMS,
            "physical-layers": program.physical_layers,
            "executed-layer-visits": executions,
            "contraction-coefficients-per-position": HNET_COEFFS,
            "contraction-flops": contraction_flops,
            "pisa-state-bytes": cache_bytes,
            "cache-capacity": cache_capacity,
        },
        "lowering": {
            "kernel-count": active_kernels.len(),
            "kernels": kernels,
            "missing-count": missing.len(),
            "missing": missing.iter().map(|op| format!("{op:?}")).collect::<Vec<_>>(),
        },
        "goal": {
            "maximum-context-tokens": CONTEXT_GOAL,
            "context-met": retained == CONTEXT_GOAL,
            "lowering-complete": missing.is_empty(),
            "static-goal-met": retained == CONTEXT_GOAL && missing.is_empty(),
        },
        "counting-boundary": "HNet core dense contractions and resident PISA state; excludes routing logistics, normalization, sampling, BO proposal, reward, and tell",
    });
    let mut measurement = run
        .map(|path| {
            measured(
                root,
                path,
                static_contract["source-digest"].as_str().unwrap(),
                f64::from(config.target_ms()),
            )
        })
        .transpose()?
        .unwrap_or(Value::null());
    if !measurement.is_null() {
        measurement["matches-workload"] = json!(truth_matches(&measurement, retained));
    }
    Ok(json!({
        "schema": SCHEMA,
        "config": config_path.strip_prefix(root).unwrap_or(&config_path),
        "static-contract": static_contract,
        "measurement": measurement,
    }))
}

fn prompt_tokens(generation: &blckbaux::config::GenerationConfig) -> Result<u64, String> {
    generation
        .corpus_prompt_tokens
        .map(u64::from)
        .or_else(|| {
            (!generation.corpus_prompt.is_empty()).then_some(generation.corpus_prompt.len() as u64)
        })
        .or_else(|| {
            generation
                .tasks
                .iter()
                .map(|task| task.prompt.len() as u64)
                .max()
        })
        .ok_or_else(|| "generation workload has no prompt length".into())
}

fn truth_matches(measurement: &Value, context: u64) -> bool {
    let truth = &measurement["truth"];
    if truth["transaction-kind"] == "causal-hnet-bo" {
        return truth["window-tokens"]
            .as_u64()
            .or_else(|| truth["sequence-tokens"].as_u64())
            == Some(context)
            && truth["evaluations-per-round"].as_u64() == Some(1)
            && truth["complete-candidate-transaction"].as_bool() == Some(true);
    }
    let prompt = truth["prompt-tokens"].as_u64();
    let generated = truth["generated-tokens"].as_u64();
    let geometry_matches = truth["context-tokens"].as_u64() == Some(context)
        && prompt
            .zip(generated)
            .is_some_and(|(prompt, generated)| prompt.checked_add(generated) == Some(context));
    if truth["transaction-kind"] == "paired-hnet-bo" {
        return geometry_matches
            && truth["evaluations-per-round"].as_u64() == Some(2)
            && truth["complete-candidate-transaction"].as_bool() == Some(true)
            && truth["same-condition-pair"].as_bool() == Some(true);
    }
    geometry_matches
        && truth["committed-tokens"].as_u64() == generated
        && truth["exact-per-token-generation"].as_bool() == Some(true)
}

fn kernel_rows(kernels: &[&blckbaux::experimental::ScheduledKernel]) -> Vec<Value> {
    kernels
        .iter()
        .map(|kernel| {
            json!({
                "name": kernel.kernel,
                "nodes": kernel.nodes,
                "tile": kernel.tile,
                "threads": kernel.threads,
            })
        })
        .collect()
}

fn measured(
    root: &Path,
    run: &Path,
    current_digest: &str,
    target_ms: f64,
) -> Result<Value, String> {
    let run = root.join(run);
    let source = std::fs::read_to_string(run.join("performance-source.sha256"))
        .ok()
        .map(|text| text.trim().to_owned());
    if run.join("optimization.jsonl").is_file() {
        let first = BufReader::new(
            File::open(run.join("optimization.jsonl")).map_err(|error| error.to_string())?,
        )
        .lines()
        .next()
        .transpose()
        .map_err(|error| error.to_string())?
        .ok_or("optimization trace is empty")?;
        let first: Value =
            blckbaux_artifact::json::from_str(&first).map_err(|error| error.to_string())?;
        if first["objective"] == "fixed-corpus-causal-likelihood" {
            return causal_measure(root, &run, source, current_digest, target_ms);
        }
    }
    if run.join("run.log").is_file() && !run.join("result.json").is_file() {
        return causal_log(root, &run, source, current_digest, target_ms);
    }
    let result: Value = blckbaux_artifact::json::from_reader(
        File::open(run.join("result.json")).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    if run.join("optimization.jsonl").is_file() {
        return metal_measure(root, &run, result, source, current_digest, target_ms);
    }
    cuda_measure(root, &run, result, source, current_digest, target_ms)
}

fn causal_log(
    root: &Path,
    run: &Path,
    source: Option<String>,
    current_digest: &str,
    target_ms: f64,
) -> Result<Value, String> {
    let file = File::open(run.join("run.log")).map_err(|error| error.to_string())?;
    let mut wall = Vec::new();
    let mut gpu = Vec::new();
    let mut window_tokens = None;
    let mut scored_tokens = None;
    let mut expected_rounds = None;
    for line in BufReader::new(file).lines() {
        let line = line.map_err(|error| error.to_string())?;
        if let Some((_, fields)) = line.split_once("BLCKBAUX_PRETRAIN_INITIAL ") {
            window_tokens = log_field(fields, "window_tokens")
                .or_else(|| log_field(fields, "sequence_tokens"))
                .map(str::parse::<u64>)
                .transpose()
                .map_err(|error| error.to_string())?;
            continue;
        }
        let Some((_, fields)) = line.split_once("BLCKBAUX_PRETRAIN_ROUND ") else {
            continue;
        };
        wall.push(
            log_field(fields, "wall_ms")
                .ok_or("causal round log lacks wall_ms")?
                .parse::<f64>()
                .map_err(|error| error.to_string())?,
        );
        gpu.push(
            log_field(fields, "gpu_ms")
                .ok_or("causal round log lacks gpu_ms")?
                .parse::<f64>()
                .map_err(|error| error.to_string())?,
        );
        let scored = log_field(fields, "scored_tokens")
            .ok_or("causal round log lacks scored_tokens")?
            .parse::<u64>()
            .map_err(|error| error.to_string())?;
        let rounds = log_field(fields, "total")
            .ok_or("causal round log lacks total")?
            .parse::<u64>()
            .map_err(|error| error.to_string())?;
        if scored_tokens
            .replace(scored)
            .is_some_and(|old| old != scored)
            || expected_rounds
                .replace(rounds)
                .is_some_and(|old| old != rounds)
        {
            return Err("causal round log changes geometry between rounds".into());
        }
    }
    if wall.is_empty() {
        return Err("causal round log has no completed rounds".into());
    }
    let window_tokens = window_tokens.ok_or("causal log lacks window_tokens")?;
    let scored_tokens = scored_tokens.unwrap();
    let samples = wall.len();
    let wall_report = distribution(wall)?;
    let gpu_report = distribution(gpu)?;
    let wall_p95 = wall_report["p95"]
        .as_f64()
        .ok_or("causal wall p95 is unavailable")?;
    let wall_mean = wall_report["mean"]
        .as_f64()
        .ok_or("causal wall mean is unavailable")?;
    let stage_profile = stage_profile(run)?;
    Ok(json!({
        "run": run.strip_prefix(root).unwrap_or(run),
        "source-digest": source,
        "matches-current-source": source.as_deref() == Some(current_digest),
        "hardware": "apple-metal",
        "samples": samples,
        "diagnostic-samples": 0,
        "provisional": true,
        "completed-rounds": samples,
        "expected-rounds": expected_rounds,
        "truth": {
            "transaction-kind": "causal-hnet-bo",
            "window-tokens": window_tokens,
            "scored-tokens": scored_tokens,
            "evaluations-per-round": 1,
            "complete-candidate-transaction": true,
        },
        "latency": {
            "wall-ms": wall_report,
            "gpu-ms": gpu_report,
        },
        "target-round-ms": target_ms,
        "p95-target-met": wall_p95 <= target_ms,
        "strict-target-met": false,
        "required-p95-speedup": wall_p95 / target_ms,
        "throughput": {
            "mean-scored-tokens-per-second": scored_tokens as f64 * 1000.0 / wall_mean,
            "p95-scored-tokens-per-second": scored_tokens as f64 * 1000.0 / wall_p95,
        },
        "stage-profile": stage_profile,
        "stage-boundary": "provisional completed-round log rows; unavailable as a recordable baseline",
    }))
}

fn log_field<'a>(fields: &'a str, name: &str) -> Option<&'a str> {
    fields.split_whitespace().find_map(|entry| {
        entry
            .strip_prefix(name)
            .and_then(|value| value.strip_prefix('='))
    })
}

fn causal_measure(
    root: &Path,
    run: &Path,
    source: Option<String>,
    current_digest: &str,
    target_ms: f64,
) -> Result<Value, String> {
    let result = std::fs::read_to_string(run.join("result.toml"))
        .map_err(|error| format!("causal result is unavailable: {error}"))?;
    if !result
        .lines()
        .any(|line| line.trim() == "status = \"completed\"")
    {
        return Err("performance evidence requires a completed causal run".into());
    }
    let file = File::open(run.join("optimization.jsonl")).map_err(|error| error.to_string())?;
    let rows = BufReader::new(file)
        .lines()
        .map(|line| {
            blckbaux_artifact::json::from_str::<Value>(&line.map_err(|error| error.to_string())?)
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if rows.is_empty() {
        return Err("causal performance evidence has no rounds".into());
    }
    let metrics = [
        ("wall-ms", vec_path(&rows, &["round_wall_ms"], 1.0)),
        ("proposal-ms", vec_path(&rows, &["ask_ms"], 1.0)),
        ("evaluation-ms", vec_path(&rows, &["candidate_gpu_ms"], 1.0)),
        ("gpu-ms", vec_path(&rows, &["round_gpu_ms"], 1.0)),
        ("tell-ms", vec_path(&rows, &["tell_ms"], 1.0)),
    ];
    let latency = metrics
        .into_iter()
        .map(|(name, values)| Ok((name, distribution(values?)?)))
        .collect::<Result<blckbaux_artifact::json::Map, String>>()?;
    let wall = latency
        .get("wall-ms")
        .ok_or("causal wall latency is unavailable")?;
    let wall_p95 = wall["p95"]
        .as_f64()
        .ok_or("causal wall p95 is unavailable")?;
    let wall_mean = wall["mean"]
        .as_f64()
        .ok_or("causal wall mean is unavailable")?;
    let window_tokens = rows[0]["window_tokens"]
        .as_u64()
        .or_else(|| rows[0]["sequence_tokens"].as_u64())
        .ok_or("causal trace lacks window_tokens")?;
    let scored_tokens = rows[0]["scored_tokens"]
        .as_u64()
        .ok_or("causal trace lacks scored_tokens")?;
    if rows.iter().any(|row| {
        row["window_tokens"]
            .as_u64()
            .or_else(|| row["sequence_tokens"].as_u64())
            != Some(window_tokens)
            || row["scored_tokens"].as_u64() != Some(scored_tokens)
    }) {
        return Err("causal trace changes token geometry between rounds".into());
    }
    let stage_profile = stage_profile(run)?;
    Ok(json!({
        "run": run.strip_prefix(root).unwrap_or(run),
        "source-digest": source,
        "matches-current-source": source.as_deref() == Some(current_digest),
        "hardware": "apple-metal",
        "samples": rows.len(),
        "diagnostic-samples": 0,
        "truth": {
            "transaction-kind": "causal-hnet-bo",
            "window-tokens": window_tokens,
            "scored-tokens": scored_tokens,
            "evaluations-per-round": 1,
            "complete-candidate-transaction": true,
        },
        "latency": Value::from(latency),
        "initialization-samples": rows.iter().filter(|row| row["initializing"] == true).count(),
        "guided-samples": rows.iter().filter(|row| row["initializing"] == false).count(),
        "target-round-ms": target_ms,
        "p95-target-met": wall_p95 <= target_ms,
        "strict-target-met": rows.iter().all(|row| row["round_wall_ms"].as_f64().is_some_and(|wall| wall <= target_ms)),
        "required-p95-speedup": wall_p95 / target_ms,
        "throughput": {
            "mean-scored-tokens-per-second": scored_tokens as f64 * 1000.0 / wall_mean,
            "p95-scored-tokens-per-second": scored_tokens as f64 * 1000.0 / wall_p95,
        },
        "stage-profile": stage_profile,
        "stage-boundary": "one proposal, one causal likelihood evaluation, posterior update, and commit or rejection",
    }))
}

fn stage_profile(run: &Path) -> Result<Value, String> {
    let path = run.join("run.log");
    if !path.is_file() {
        return Ok(Value::null());
    }
    let text = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
    stage_text(&text)
}

fn stage_text(text: &str) -> Result<Value, String> {
    let mut gpu = Vec::new();
    let mut rows = Vec::new();
    let mut stages = BTreeMap::<String, Vec<f64>>::new();
    for line in text.lines() {
        let Some((_, encoded)) = line.split_once("scorer.stage ") else {
            continue;
        };
        let sample: Value =
            blckbaux_artifact::json::from_str(encoded).map_err(|error| error.to_string())?;
        gpu.push(
            sample["gpu_ms"]
                .as_f64()
                .ok_or("scorer stage sample lacks gpu_ms")?,
        );
        rows.push(
            sample["rows"]
                .as_u64()
                .ok_or("scorer stage sample lacks rows")?,
        );
        let fields = sample["stage_ms"]
            .as_map()
            .ok_or("scorer stage sample lacks stage_ms")?;
        for (name, value) in fields {
            let name = name
                .kind()
                .as_str()
                .ok_or("scorer stage name is not a string")?;
            stages.entry(name.to_owned()).or_default().push(
                value
                    .as_f64()
                    .ok_or("scorer stage duration is not a number")?,
            );
        }
    }
    if gpu.is_empty() {
        return Ok(Value::null());
    }
    let rows_per_sample = rows
        .iter()
        .all(|sample_rows| *sample_rows == rows[0])
        .then_some(rows[0]);
    let mut stage_reports = blckbaux_artifact::json::Map::new();
    for (name, values) in stages {
        stage_reports.insert(Value::from(name), distribution(values)?);
    }
    Ok(json!({
        "samples": gpu.len(),
        "rows-per-sample": rows_per_sample,
        "profiled-gpu-ms": distribution(gpu)?,
        "stage-ms": Value::from(stage_reports),
        "boundary": "one profiled scorer chunk; stage means are not complete-round sums",
    }))
}

fn metal_measure(
    root: &Path,
    run: &Path,
    result: Value,
    source: Option<String>,
    current_digest: &str,
    target_ms: f64,
) -> Result<Value, String> {
    if result["status"] != "completed" {
        return Err("performance evidence requires a completed run".into());
    }
    let file = File::open(run.join("optimization.jsonl")).map_err(|error| error.to_string())?;
    let rows = BufReader::new(file)
        .lines()
        .map(|line| {
            blckbaux_artifact::json::from_str::<Value>(&line.map_err(|error| error.to_string())?)
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let operational = rows
        .iter()
        .filter(|row| row["pool_rank_audit"].is_null())
        .cloned()
        .collect::<Vec<_>>();
    if operational.is_empty() {
        return Err("performance evidence has no non-diagnostic rounds".into());
    }
    let last = rows.last().ok_or("optimization trace is empty")?;
    let generation = &last["generation"];
    let tokens = &last["response"]["tokens"];
    let repaired = generation["repaired_tokens"]
        .as_u64()
        .or_else(|| (generation["repair_batches"].as_u64() == Some(0)).then_some(0));
    let drafted = generation["draft"]
        .as_seq()
        .map(|drafts| {
            drafts
                .iter()
                .filter_map(|draft| draft["generated_tokens"].as_u64())
                .sum::<u64>()
        })
        .unwrap_or(0);
    let metrics = [
        ("wall-ms", vec_path(&operational, &["wall_seconds"], 1000.0)),
        (
            "proposal-ms",
            vec_path(&operational, &["generation", "proposal_seconds"], 1000.0),
        ),
        (
            "evaluation-ms",
            vec_path(&operational, &["generation", "evaluation_seconds"], 1000.0),
        ),
        (
            "gpu-ms",
            vec_path(&operational, &["generation", "rollout_gpu_seconds"], 1000.0),
        ),
        (
            "reward-ms",
            vec_path(&operational, &["generation", "reward_seconds"], 1000.0),
        ),
        (
            "tell-ms",
            vec_path(&operational, &["generation", "tell_seconds"], 1000.0),
        ),
    ];
    let distributions = metrics
        .into_iter()
        .map(|(name, values)| Ok((name, distribution(values?)?)))
        .collect::<Result<blckbaux_artifact::json::Map, String>>()?;
    let wall_p95 = distributions
        .get("wall-ms")
        .and_then(|value| value["p95"].as_f64())
        .ok_or("wall latency lacks p95")?;
    let experiment_regression = experiment_regression(run, &result, &rows)?;
    Ok(json!({
        "run": run.strip_prefix(root).unwrap_or(run),
        "source-digest": source,
        "matches-current-source": source.as_deref() == Some(current_digest),
        "hardware": "apple-metal",
        "samples": operational.len(),
        "diagnostic-samples": rows.len() - operational.len(),
        "truth": {
            "context-tokens": tokens["context"].clone(),
            "prompt-tokens": tokens["prompt"].clone(),
            "drafted-tokens": drafted,
            "evaluated-tokens": generation["evaluated_tokens"].clone(),
            "evaluated-core-positions": generation["evaluated_positions"].clone(),
            "scored-tokens": generation["scored_tokens"].clone(),
            "repaired-tokens": repaired,
            "generated-tokens": tokens["generated"].clone(),
            "committed-tokens": tokens["committed"].clone(),
            "repair-batches": generation["repair_batches"].clone(),
            "exact-per-token-generation": result["exact_per_token_generation"].clone(),
            "direct-token-decisions": result["direct_token_decisions_per_candidate"].clone(),
            "hash-expanded-tokens": result["hash_expanded_tokens_per_candidate"].clone(),
        },
        "experiment-regression": experiment_regression,
        "latency": Value::from(distributions),
        "initialization-samples": operational.iter().filter(|row| row["phase"] == "initialization").count(),
        "guided-samples": operational.iter().filter(|row| row["phase"] == "guided").count(),
        "pool-rank-audit": pool_rank(&rows),
        "target-round-ms": target_ms,
        "p95-target-met": wall_p95 <= target_ms,
        "strict-target-met": operational.iter().all(|row| row["wall_seconds"].as_f64().is_some_and(|seconds| seconds * 1000.0 <= target_ms)),
        "required-p95-speedup": wall_p95 / target_ms,
    }))
}

fn cuda_measure(
    root: &Path,
    run: &Path,
    result: Value,
    source: Option<String>,
    current_digest: &str,
    target_ms: f64,
) -> Result<Value, String> {
    if result["schema"] == "blckbaux.cuda.hnet-bo-round.v7" {
        return cuda_transaction(root, run, result, source, current_digest, target_ms);
    }
    if result["schema"] != "blckbaux.cuda.hnet-bo-round.v2" || result["full_bo_loop"] != true {
        return Err("CUDA performance evidence requires a complete H-Net BO round".into());
    }
    let evaluation_ms = result["candidate"]["total_device_ms"]
        .as_f64()
        .ok_or("CUDA result lacks candidate stage timing")?;
    let metrics = [
        ("wall-ms", result["wall_ms"].as_f64()),
        ("proposal-ms", result["proposal"]["device_ms"].as_f64()),
        ("evaluation-ms", Some(evaluation_ms)),
        ("gpu-ms", result["device_ms"].as_f64()),
        ("tell-ms", result["tell"]["wall_ms"].as_f64()),
    ];
    let latency = metrics
        .into_iter()
        .map(|(name, value)| {
            let value = value.ok_or_else(|| format!("CUDA result lacks {name}"))?;
            Ok((name, distribution(vec![value])?))
        })
        .collect::<Result<blckbaux_artifact::json::Map, String>>()?;
    let wall = result["wall_ms"].as_f64().unwrap();
    Ok(json!({
        "run": run.strip_prefix(root).unwrap_or(run),
        "source-digest": source,
        "matches-current-source": source.as_deref() == Some(current_digest),
        "hardware": result["hardware"].clone(),
        "samples": 1,
        "truth": {
            "context-tokens": result["context_window_tokens"].clone(),
            "prompt-tokens": result["prompt_tokens"].clone(),
            "drafted-tokens": result["drafted_tokens"].clone(),
            "evaluated-tokens": result["evaluated_positions"].clone(),
            "repaired-tokens": result["repaired_tokens"].clone(),
            "generated-tokens": result["generated_tokens"].clone(),
            "committed-tokens": result["committed_tokens"].clone(),
            "repair-batches": result["repair_waves"].clone(),
            "exact-per-token-generation": result["exact_per_token_generation"].clone(),
            "direct-token-decisions": result["direct_token_decisions"].clone(),
            "hash-expanded-tokens": result["hash_expanded_tokens"].clone(),
        },
        "latency": Value::from(latency),
        "target-round-ms": target_ms,
        "p95-target-met": wall <= target_ms,
        "strict-target-met": wall <= target_ms,
        "required-p95-speedup": wall / target_ms,
        "stage-boundary": "CUDA scoring is fused into candidate evaluation; one Modal invocation contributes one sample",
    }))
}

fn cuda_transaction(
    root: &Path,
    run: &Path,
    result: Value,
    source: Option<String>,
    current_digest: &str,
    target_ms: f64,
) -> Result<Value, String> {
    if result["full_bo_loop"] != true
        || result["full_generation_loop"] != true
        || result["full_scoring_loop"] != true
        || result["complete_candidate_transaction"] != true
    {
        return Err(
            "CUDA performance evidence requires a complete paired H-Net BO transaction".into(),
        );
    }
    let file = File::open(run.join("rounds.jsonl")).map_err(|error| error.to_string())?;
    let rows = BufReader::new(file)
        .lines()
        .map(|line| {
            blckbaux_artifact::json::from_str::<Value>(&line.map_err(|error| error.to_string())?)
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let expected = result["bo"]["rounds"]
        .as_u64()
        .ok_or("CUDA result lacks bo.rounds")?;
    if rows.len() as u64 != expected || rows.is_empty() {
        return Err(format!(
            "CUDA round trace contains {} rows, expected {expected}",
            rows.len()
        ));
    }
    let metric = |key: &str| {
        rows.iter()
            .map(|row| {
                row[key]
                    .as_f64()
                    .ok_or_else(|| format!("CUDA round lacks {key}"))
            })
            .collect::<Result<Vec<_>, String>>()
    };
    let wall = metric("wall_ms")?;
    let gpu = metric("device_ms")?;
    let host_overhead = wall
        .iter()
        .zip(&gpu)
        .map(|(wall, gpu)| (wall - gpu).max(0.0))
        .collect::<Vec<_>>();
    let metrics = [
        ("wall-ms", wall),
        ("proposal-ms", metric("proposal_device_ms")?),
        (
            "incumbent-evaluation-ms",
            metric("incumbent_evaluation_device_ms")?,
        ),
        (
            "candidate-evaluation-ms",
            metric("candidate_evaluation_device_ms")?,
        ),
        ("evaluation-ms", metric("evaluation_device_ms")?),
        ("gpu-ms", gpu),
        ("restore-ms", metric("restore_device_ms")?),
        ("host-overhead-ms", host_overhead),
    ];
    let latency = metrics
        .into_iter()
        .map(|(name, values)| Ok((name, distribution(values)?)))
        .collect::<Result<blckbaux_artifact::json::Map, String>>()?;
    let wall_p95 = latency
        .get("wall-ms")
        .and_then(|value| value["p95"].as_f64())
        .ok_or("CUDA wall latency lacks p95")?;
    let context = result["context_window_tokens"]
        .as_u64()
        .ok_or("CUDA result lacks context_window_tokens")?;
    let prompt = result["prompt_tokens"]
        .as_u64()
        .ok_or("CUDA result lacks prompt_tokens")?;
    let generated = context
        .checked_sub(prompt)
        .ok_or("CUDA prompt exceeds context")?;
    Ok(json!({
        "run": run.strip_prefix(root).unwrap_or(run),
        "source-digest": source,
        "matches-current-source": source.as_deref() == Some(current_digest),
        "hardware": result["hardware"].clone(),
        "samples": rows.len(),
        "truth": {
            "transaction-kind": "paired-hnet-bo",
            "context-tokens": context,
            "prompt-tokens": prompt,
            "generated-tokens": generated,
            "evaluations-per-round": 2,
            "evaluated-tokens": result["evaluated_tokens"].clone(),
            "committed-tokens": result["committed_tokens"].clone(),
            "complete-candidate-transaction": true,
            "same-condition-pair": result["training_conditions"]["paired_candidate_incumbent"].clone(),
            "incumbent-remeasured": result["training_conditions"]["incumbent_remeasured_each_round"].clone(),
        },
        "latency": Value::from(latency),
        "initialization-samples": rows.iter().filter(|row| row["phase"] == "initialization").count(),
        "guided-samples": rows.iter().filter(|row| row["phase"] == "turbo_enn").count(),
        "target-round-ms": target_ms,
        "p95-target-met": wall_p95 <= target_ms,
        "strict-target-met": rows.iter().all(|row| row["wall_ms"].as_f64().is_some_and(|wall| wall <= target_ms)),
        "required-p95-speedup": wall_p95 / target_ms,
        "stage-boundary": "each sample includes proposal materialization, same-condition incumbent evaluation, candidate evaluation, posterior/reliability update, and commit or restore",
    }))
}

fn pool_rank(rows: &[Value]) -> Value {
    let audits = rows
        .iter()
        .map(|row| &row["pool_rank_audit"])
        .filter(|audit| {
            matches!(
                audit["schema"].as_str(),
                Some("blckbaux.pool_rank_audit.v1" | "blckbaux.pool_rank_audit.v2")
            )
        })
        .collect::<Vec<_>>();
    if audits.is_empty() {
        return Value::null();
    }
    let mean = |field: &str| {
        let values = audits
            .iter()
            .filter_map(|audit| audit[field].as_f64())
            .collect::<Vec<_>>();
        (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
    };
    let selected_top1 = audits
        .iter()
        .filter(|audit| audit["selected_rank"].as_u64() == Some(1))
        .count();
    let oracle_positive = audits
        .iter()
        .filter(|audit| {
            audit["best_improvement"]
                .as_f64()
                .is_some_and(|value| value > 0.0)
        })
        .count();
    let selected_positive = audits
        .iter()
        .filter(|audit| {
            audit["selected_improvement"]
                .as_f64()
                .is_some_and(|value| value > 0.0)
        })
        .count();
    let missed_positive = audits
        .iter()
        .filter(|audit| {
            audit["best_improvement"]
                .as_f64()
                .is_some_and(|value| value > 0.0)
                && audit["selected_improvement"]
                    .as_f64()
                    .is_some_and(|value| value <= 0.0)
        })
        .count();
    let correlation = |field: &str| {
        let pairs = audits
            .iter()
            .filter_map(|audit| audit["candidates"].as_seq())
            .flatten()
            .filter_map(|candidate| {
                Some((
                    candidate[field].as_f64()?,
                    candidate["improvement"].as_f64()?,
                ))
            })
            .collect::<Vec<_>>();
        pearson(&pairs)
    };
    let response_audit = audits
        .iter()
        .any(|audit| audit["schema"] == "blckbaux.pool_rank_audit.v2");
    let mean_field = if response_audit {
        "response_mean"
    } else {
        "predicted_mean"
    };
    let error_field = if response_audit {
        "response_standard_error"
    } else {
        "predicted_standard_error"
    };
    json!({
        "points": audits.len(),
        "pool-size": 4,
        "selected-top1": selected_top1,
        "selected-top1-rate": selected_top1 as f64 / audits.len() as f64,
        "random-top1-rate": 0.25,
        "mean-selected-rank": mean("selected_rank"),
        "mean-selection-regret": mean("selection_regret"),
        "mean-selected-improvement": mean("selected_improvement"),
        "mean-best-improvement": mean("best_improvement"),
        "oracle-positive": oracle_positive,
        "selected-positive": selected_positive,
        "missed-positive": missed_positive,
        "score-improvement-correlation": correlation("acquisition_score"),
        "ranking-model": response_audit.then_some("historical-linear-operator-response-v1"),
        "response-mean-improvement-correlation": correlation(mean_field),
        "response-uncertainty-improvement-correlation": correlation(error_field),
        "mean-policy": counterfactual(&audits, 0.0),
        "ucb-one-policy": counterfactual(&audits, 1.0),
        "lcb-one-policy": counterfactual(&audits, -1.0),
    })
}

fn counterfactual(audits: &[&Value], beta: f64) -> Value {
    let choices = audits
        .iter()
        .filter_map(|audit| {
            let candidates = audit["candidates"].as_seq()?;
            let response_audit = audit["schema"] == "blckbaux.pool_rank_audit.v2";
            let mean_field = if response_audit {
                "response_mean"
            } else {
                "predicted_mean"
            };
            let error_field = if response_audit {
                "response_standard_error"
            } else {
                "predicted_standard_error"
            };
            let selected = candidates
                .iter()
                .enumerate()
                .filter_map(|(index, candidate)| {
                    Some((
                        index,
                        candidate[mean_field].as_f64()? + beta * candidate[error_field].as_f64()?,
                    ))
                })
                .max_by(|left, right| left.1.total_cmp(&right.1))?
                .0;
            Some((
                selected,
                candidates[selected]["improvement"].as_f64()?,
                audit["best"].as_u64()? as usize,
            ))
        })
        .collect::<Vec<_>>();
    if choices.is_empty() {
        return Value::null();
    }
    json!({
        "points": choices.len(),
        "top1": choices.iter().filter(|choice| choice.0 == choice.2).count(),
        "positive": choices.iter().filter(|choice| choice.1 > 0.0).count(),
        "mean-improvement": choices.iter().map(|choice| choice.1).sum::<f64>() / choices.len() as f64,
    })
}

fn pearson(values: &[(f64, f64)]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let count = values.len() as f64;
    let mean_x = values.iter().map(|pair| pair.0).sum::<f64>() / count;
    let mean_y = values.iter().map(|pair| pair.1).sum::<f64>() / count;
    let covariance = values
        .iter()
        .map(|pair| (pair.0 - mean_x) * (pair.1 - mean_y))
        .sum::<f64>();
    let variance_x = values
        .iter()
        .map(|pair| (pair.0 - mean_x).powi(2))
        .sum::<f64>();
    let variance_y = values
        .iter()
        .map(|pair| (pair.1 - mean_y).powi(2))
        .sum::<f64>();
    let scale = (variance_x * variance_y).sqrt();
    (scale > 0.0).then_some(covariance / scale)
}

fn vec_path(rows: &[Value], path: &[&str], scale: f64) -> Result<Vec<f64>, String> {
    rows.iter()
        .map(|row| {
            path.iter()
                .fold(row, |value, key| &value[*key])
                .as_f64()
                .map(|value| value * scale)
                .ok_or_else(|| format!("controller trace lacks {}", path.join(".")))
        })
        .collect()
}

fn distribution(mut values: Vec<f64>) -> Result<Value, String> {
    if values.is_empty()
        || values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err("latency distribution requires finite nonnegative samples".into());
    }
    values.sort_by(f64::total_cmp);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let p95 = values[((values.len() * 95).div_ceil(100)).saturating_sub(1)];
    Ok(json!({
        "count": values.len(),
        "mean": mean,
        "p95": p95,
        "max": values[values.len() - 1],
    }))
}

fn experiment_regression(run: &Path, result: &Value, rows: &[Value]) -> Result<Value, String> {
    let first = rows.first().ok_or("optimization trace is empty")?;
    let response = &first["response"];
    let paired = &first["generation"]["paired_generation"];
    let journal_accepted = first["acceptance"]["accepted"]
        .as_bool()
        .ok_or("controller trace lacks acceptance.accepted")?;
    let response_accepted = response["accepted"]
        .as_bool()
        .ok_or("controller trace lacks response.accepted")?;
    if journal_accepted != response_accepted {
        return Err("controller trace disagrees on the accepted decision".into());
    }
    let round = first["round"]
        .as_u64()
        .ok_or("controller trace lacks round")?;
    let candidate_path = run.join(format!("round-{round:04}/candidate/generated.txt"));
    let incumbent_path = run.join(format!("round-{round:04}/incumbent/generated.txt"));
    let candidate_nll = -paired["candidate_reward"]
        .as_f64()
        .ok_or("controller trace lacks paired candidate NLL")?;
    let incumbent_nll = -paired["incumbent_reward"]
        .as_f64()
        .ok_or("controller trace lacks paired incumbent NLL")?;
    Ok(json!({
        "parameters": result["parameters"].clone(),
        "round": round,
        "evaluated-tokens": response["tokens"]["evaluated"].clone(),
        "candidate-objective": response["candidate"]["mean"].clone(),
        "incumbent-objective": response["baseline"]["mean"].clone(),
        "candidate-nll": candidate_nll,
        "incumbent-nll": incumbent_nll,
        "accepted": response_accepted,
        "decoded-output": {
            "identity": "sha256-decoded-utf8",
            "candidate": sha256_file(&candidate_path)?,
            "incumbent": sha256_file(&incumbent_path)?,
        },
    }))
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path)
        .map_err(|error| format!("decoded output {} is unavailable: {error}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn record(
    root: &Path,
    config: &Path,
    backend: PerformanceBackend,
    run: &Path,
    output: &Path,
) -> Result<(), String> {
    let certificate = certificate(root, config, backend, Some(run))?;
    if certificate["measurement"]["provisional"] == true {
        return Err("an incomplete run cannot be recorded as a performance baseline".into());
    }
    if certificate["measurement"]["matches-current-source"] != true {
        return Err(
            "run predates the current hot-path source; execute a short matching replay".into(),
        );
    }
    if certificate["measurement"]["matches-workload"] != true {
        return Err("run truth labels do not match the certified workload".into());
    }
    let output = root.join(output);
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    std::fs::write(
        &output,
        format!(
            "{}\n",
            pretty_string(&certificate).map_err(|error| error.to_string())?
        ),
    )
    .map_err(|error| error.to_string())?;
    println!("recorded {}", output.display());
    Ok(())
}

fn gate(
    root: &Path,
    config: &Path,
    backend: PerformanceBackend,
    run: Option<&Path>,
    baseline: &Path,
    tolerance_percent: u32,
) -> Result<(), String> {
    if tolerance_percent > 100 {
        return Err("performance tolerance percent must be in 0..=100".into());
    }
    let tolerance = f64::from(tolerance_percent) / 100.0;
    let baseline_path = root.join(baseline);
    let baseline: Value = blckbaux_artifact::json::from_reader(
        File::open(&baseline_path).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    if baseline["schema"] != SCHEMA {
        return Err("unsupported performance baseline schema".into());
    }
    let experiment_baseline = if backend == PerformanceBackend::Metal {
        let path = root.join(REGRESSION_BASELINE);
        let value: Value = blckbaux_artifact::json::from_reader(
            File::open(&path).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        if value["schema"] != REGRESSION_SCHEMA {
            return Err("unsupported experiment regression baseline schema".into());
        }
        Some(value)
    } else {
        None
    };
    let current = certificate(root, config, backend, run)?;
    let hot_path_changed =
        baseline["static-contract"]["source-digest"] != current["static-contract"]["source-digest"];
    let schedule_changed = baseline["static-contract"]["lowering"]["kernels"]
        != current["static-contract"]["lowering"]["kernels"];
    let workload_changed =
        baseline["static-contract"]["workload"] != current["static-contract"]["workload"];
    if workload_changed {
        return Err(
            "performance workload differs from the baseline; record a separate baseline".into(),
        );
    }
    let mut failures = Vec::new();
    for field in [
        "contraction-flops",
        "pisa-state-bytes",
        "executed-layer-visits",
    ] {
        let old = baseline["static-contract"]["architecture"][field]
            .as_u64()
            .ok_or_else(|| format!("baseline lacks architecture.{field}"))?;
        let new = current["static-contract"]["architecture"][field]
            .as_u64()
            .ok_or_else(|| format!("certificate lacks architecture.{field}"))?;
        if new > old {
            failures.push(format!("{field} increased {old} -> {new}"));
        }
    }
    let old_missing = baseline["static-contract"]["lowering"]["missing-count"]
        .as_u64()
        .ok_or("baseline lacks lowering.missing-count")?;
    let new_missing = current["static-contract"]["lowering"]["missing-count"]
        .as_u64()
        .ok_or("certificate lacks lowering.missing-count")?;
    if new_missing > old_missing {
        failures.push(format!(
            "missing backend lowerings increased {old_missing} -> {new_missing}"
        ));
    }
    let replay_required = hot_path_changed || schedule_changed || !failures.is_empty();
    if replay_required {
        if current["measurement"].is_null() {
            return Err(format!(
                "hot path changed; a matching short replay is required{}",
                if failures.is_empty() {
                    String::new()
                } else {
                    format!(": {}", failures.join("; "))
                }
            ));
        }
        if current["measurement"]["matches-current-source"] != true {
            return Err("measured replay does not match the current hot-path source".into());
        }
        if current["measurement"]["provisional"] == true {
            return Err("performance gates require a completed run".into());
        }
        if current["measurement"]["matches-workload"] != true {
            return Err("measured replay truth labels do not match the baseline workload".into());
        }
        if let Some(experiment_baseline) = &experiment_baseline {
            compare_experiment(experiment_baseline, &current, &mut failures)?;
        }
        compare_latency(&baseline, &current, tolerance, &mut failures)?;
    }
    let passed = failures.is_empty();
    let report = json!({
        "schema": "blckbaux.performance-gate.v1",
        "baseline": baseline_path.strip_prefix(root).unwrap_or(&baseline_path),
        "passed": passed,
        "hot-path-changed": hot_path_changed,
        "schedule-changed": schedule_changed,
        "measured-replay-required": replay_required,
        "tolerance": tolerance,
        "failures": failures,
        "p95-target-met": current["measurement"]["p95-target-met"].clone(),
        "strict-target-met": current["measurement"]["strict-target-met"].clone(),
    });
    print(&report)?;
    if passed {
        Ok(())
    } else {
        Err("performance regression gate failed".into())
    }
}

fn compare_experiment(
    expected: &Value,
    current: &Value,
    failures: &mut Vec<String>,
) -> Result<(), String> {
    if expected["config"] != current["config"] {
        failures.push("experiment config changed".into());
    }
    if expected["model"] != current["static-contract"]["model"] {
        failures.push("experiment model changed".into());
    }
    if expected["truth"] != current["measurement"]["truth"] {
        failures.push("truth labels changed".into());
    }
    let actual = &current["measurement"]["experiment-regression"];
    if actual.is_null() {
        return Err("measured replay lacks experiment-regression evidence".into());
    }
    for field in [
        "parameters",
        "round",
        "evaluated-tokens",
        "candidate-objective",
        "incumbent-objective",
        "candidate-nll",
        "incumbent-nll",
        "accepted",
    ] {
        if expected[field] != actual[field] {
            failures.push(format!(
                "experiment {field} changed {:?} -> {:?}",
                expected[field], actual[field]
            ));
        }
    }
    for role in ["identity", "candidate", "incumbent"] {
        if expected["decoded-output"][role] != actual["decoded-output"][role] {
            failures.push(format!("experiment decoded output {role} changed"));
        }
    }
    Ok(())
}

fn compare_latency(
    baseline: &Value,
    current: &Value,
    tolerance: f64,
    failures: &mut Vec<String>,
) -> Result<(), String> {
    for metric in ["wall-ms", "proposal-ms", "evaluation-ms", "gpu-ms"] {
        let Some(old) = baseline["measurement"]["latency"][metric]["p95"].as_f64() else {
            continue;
        };
        let new = current["measurement"]["latency"][metric]["p95"]
            .as_f64()
            .ok_or_else(|| format!("replay lacks measurement.latency.{metric}.p95"))?;
        let limit = old * (1.0 + tolerance) + NOISE_MS;
        if new > limit {
            failures.push(format!("{metric} p95 {new:.3} exceeds {limit:.3}"));
        }
    }
    Ok(())
}

fn schedule(backend: PerformanceBackend) -> BackendSchedule {
    match backend {
        PerformanceBackend::Metal => BackendSchedule::hnet_metal(),
        PerformanceBackend::CudaOxide => BackendSchedule::hnet_cuda(),
    }
}

fn backend_kind(value: PerformanceBackend) -> Backend {
    match value {
        PerformanceBackend::Metal => Backend::Metal,
        PerformanceBackend::CudaOxide => Backend::CudaOxide,
    }
}

fn compiler_name(compiler: blckbaux::experimental::Compiler) -> &'static str {
    match compiler {
        blckbaux::experimental::Compiler::Native => "native",
        blckbaux::experimental::Compiler::CudaOxide => "cuda-oxide",
        blckbaux::experimental::Compiler::Tvm => "tvm",
    }
}

fn print(value: &Value) -> Result<(), String> {
    println!(
        "{}",
        pretty_string(value).map_err(|error| error.to_string())?
    );
    Ok(())
}

fn summary(value: &Value) -> Result<(), String> {
    let contract = &value["static-contract"];
    let workload = &contract["workload"];
    let lowering = &contract["lowering"];
    let goal = &contract["goal"];
    let measurement = &value["measurement"];
    println!(
        "performance {} · {} · {} context · {:.0}ms target · {} token/s target",
        contract["backend"].as_str().unwrap_or("unknown-backend"),
        contract["model"].as_str().unwrap_or("unknown-model"),
        workload["retained-context-tokens"].as_u64().unwrap_or(0),
        workload["target-round-ms"].as_f64().unwrap_or(0.0),
        decimal(&workload["target-tokens-per-second"]),
    );
    println!(
        "static lowerings: {} kernels · {} missing · complete={}",
        lowering["kernel-count"].as_u64().unwrap_or(0),
        lowering["missing-count"].as_u64().unwrap_or(0),
        goal["lowering-complete"].as_bool().unwrap_or(false),
    );
    let mut blockers = Vec::new();
    if goal["context-met"] != true {
        blockers.push(format!(
            "workload has {} tokens; {} required",
            workload["retained-context-tokens"].as_u64().unwrap_or(0),
            goal["maximum-context-tokens"].as_u64().unwrap_or(0),
        ));
    }
    if goal["lowering-complete"] != true {
        blockers.push(format!(
            "{} semantic operations have no backend lowering",
            lowering["missing-count"].as_u64().unwrap_or(0)
        ));
    }
    if let Some(boundary) = contract["counting-boundary"].as_str() {
        blockers.push(format!("static work account is partial: {boundary}"));
    }
    if measurement.is_null() {
        println!("measurement: missing");
        blockers.push("no completed matching-source run was supplied".to_owned());
    } else {
        println!(
            "measurement: {} · {} operational + {} diagnostic samples · source={} · max-workload={} · provisional={}",
            measurement["hardware"]
                .as_str()
                .unwrap_or("unknown-hardware"),
            measurement["samples"].as_u64().unwrap_or(0),
            measurement["diagnostic-samples"].as_u64().unwrap_or(0),
            verdict(&measurement["matches-current-source"]),
            verdict(&measurement["matches-workload"]),
            if measurement["provisional"] == true {
                "yes"
            } else {
                "no"
            },
        );
        let wall = &measurement["latency"]["wall-ms"];
        println!(
            "wall ms: mean={} · p95={} · max={} · target={} · required speedup={}x",
            decimal(&wall["mean"]),
            decimal(&wall["p95"]),
            decimal(&wall["max"]),
            verdict(&measurement["p95-target-met"]),
            decimal(&measurement["required-p95-speedup"]),
        );
        if !measurement["throughput"].is_null() {
            println!(
                "throughput token/s: mean={} · p95={}",
                decimal(&measurement["throughput"]["mean-scored-tokens-per-second"]),
                decimal(&measurement["throughput"]["p95-scored-tokens-per-second"]),
            );
        }
        let profile = &measurement["stage-profile"];
        if let Some(stage_map) = profile["stage-ms"].as_map() {
            let mut stages = stage_map
                .iter()
                .filter_map(|(name, report)| {
                    Some((name.kind().as_str()?, report["mean"].as_f64()?))
                })
                .collect::<Vec<_>>();
            stages.sort_by(|left, right| right.1.total_cmp(&left.1));
            println!(
                "profiled scorer chunk: {} rows · gpu mean={}ms · hot stages={}",
                profile["rows-per-sample"].as_u64().unwrap_or(0),
                decimal(&profile["profiled-gpu-ms"]["mean"]),
                stages
                    .into_iter()
                    .take(5)
                    .map(|(name, mean)| format!("{name}={mean:.3}ms"))
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
        let audit = &measurement["pool-rank-audit"];
        if !audit.is_null() {
            println!(
                "pool audit: {} points · top1={}/{} ({}) · mean rank={} · corr score/response-mean/response-se={}/{}/{} · positive={}/{} · missed={} · mean regret={} · selected={} · oracle={}",
                audit["points"].as_u64().unwrap_or(0),
                audit["selected-top1"].as_u64().unwrap_or(0),
                audit["points"].as_u64().unwrap_or(0),
                decimal(&audit["selected-top1-rate"]),
                decimal(&audit["mean-selected-rank"]),
                decimal(&audit["score-improvement-correlation"]),
                decimal(&audit["response-mean-improvement-correlation"]),
                decimal(&audit["response-uncertainty-improvement-correlation"]),
                audit["selected-positive"].as_u64().unwrap_or(0),
                audit["oracle-positive"].as_u64().unwrap_or(0),
                audit["missed-positive"].as_u64().unwrap_or(0),
                scientific(&audit["mean-selection-regret"]),
                scientific(&audit["mean-selected-improvement"]),
                scientific(&audit["mean-best-improvement"]),
            );
            println!(
                "policy replay mean/ucb1/lcb1: top1={}/{}/{} · improvement={}/{}/{}",
                audit["mean-policy"]["top1"].as_u64().unwrap_or(0),
                audit["ucb-one-policy"]["top1"].as_u64().unwrap_or(0),
                audit["lcb-one-policy"]["top1"].as_u64().unwrap_or(0),
                scientific(&audit["mean-policy"]["mean-improvement"]),
                scientific(&audit["ucb-one-policy"]["mean-improvement"]),
                scientific(&audit["lcb-one-policy"]["mean-improvement"]),
            );
        }
        if measurement["matches-current-source"] != true {
            blockers.push("measurement predates the current hot-path source".to_owned());
        }
        if measurement["provisional"] == true {
            blockers.push(format!(
                "run stopped after {} of {} rounds; evidence cannot be recorded as a baseline",
                measurement["completed-rounds"].as_u64().unwrap_or(0),
                measurement["expected-rounds"].as_u64().unwrap_or(0),
            ));
        }
        if measurement["matches-workload"] != true {
            blockers
                .push("measurement does not exercise the configured maximum geometry".to_owned());
        }
        if measurement["p95-target-met"] != true {
            blockers.push("measured p95 misses the transaction target".to_owned());
        }
    }
    blockers.push("no matched BLCKBAUX-versus-Muon outcome is represented".to_owned());
    println!("blockers:");
    for blocker in blockers {
        println!("- {blocker}");
    }
    println!("full evidence: rerun with --json");
    Ok(())
}

fn decimal(value: &Value) -> String {
    value
        .as_f64()
        .map(|number| format!("{number:.3}"))
        .unwrap_or_else(|| "unknown".to_owned())
}

fn scientific(value: &Value) -> String {
    value
        .as_f64()
        .map(|number| format!("{number:.6e}"))
        .unwrap_or_else(|| "unknown".to_owned())
}

fn verdict(value: &Value) -> &'static str {
    match value.as_bool() {
        Some(true) => "yes",
        Some(false) => "no",
        None => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p95_tail() {
        let values = (1..=100).map(f64::from).collect();
        let report = distribution(values).unwrap();
        assert_eq!(report["p95"], 95.0);
        assert_eq!(report["max"], 100.0);
    }

    #[test]
    fn pool_quality() {
        let rows = [json!({
            "pool_rank_audit": {
                "schema": "blckbaux.pool_rank_audit.v2",
                "selected_rank": 1,
                "selected_improvement": 4.0,
                "best_improvement": 4.0,
                "selection_regret": 0.0,
                "best": 3,
                "candidates": [
                    {"predicted_mean": 11.0, "predicted_standard_error": 5.0, "response_mean": 1.0, "response_standard_error": 4.0, "acquisition_score": 1.0, "improvement": 1.0},
                    {"predicted_mean": 12.0, "predicted_standard_error": 4.0, "response_mean": 2.0, "response_standard_error": 3.0, "acquisition_score": 2.0, "improvement": 2.0},
                    {"predicted_mean": 13.0, "predicted_standard_error": 3.0, "response_mean": 3.0, "response_standard_error": 2.0, "acquisition_score": 3.0, "improvement": 3.0},
                    {"predicted_mean": 14.0, "predicted_standard_error": 2.0, "response_mean": 4.0, "response_standard_error": 1.0, "acquisition_score": 4.0, "improvement": 4.0},
                ],
            }
        })];
        let report = pool_rank(&rows);
        assert_eq!(report["selected-top1"], 1);
        assert_eq!(report["mean-policy"]["top1"], 1);
        assert_eq!(report["score-improvement-correlation"], 1.0);
        assert_eq!(report["response-mean-improvement-correlation"], 1.0);
    }

    #[test]
    fn cuda_geometry() {
        let measurement = json!({
            "truth": {
                "transaction-kind": "paired-hnet-bo",
                "context-tokens": 1_048_576,
                "prompt-tokens": 524_288,
                "generated-tokens": 524_288,
                "evaluations-per-round": 2,
                "evaluated-tokens": 134_217_728,
                "committed-tokens": 18_874_368,
                "complete-candidate-transaction": true,
                "same-condition-pair": true,
            }
        });
        assert!(truth_matches(&measurement, 1_048_576));
    }

    #[test]
    fn window_geometry() {
        let measurement = json!({
            "truth": {
                "transaction-kind": "causal-hnet-bo",
                "window-tokens": 1_048_576,
                "scored-tokens": 1_048_513,
                "evaluations-per-round": 1,
                "complete-candidate-transaction": true,
            }
        });
        assert!(truth_matches(&measurement, 1_048_576));
        assert!(!truth_matches(&measurement, 65_536));
    }

    #[test]
    fn causal_fields() {
        let fields = "round=721 total=1000 scored_tokens=1048513 gpu_ms=1107.184 wall_ms=1267.110";
        assert_eq!(log_field(fields, "round"), Some("721"));
        assert_eq!(log_field(fields, "scored_tokens"), Some("1048513"));
        assert_eq!(log_field(fields, "wall_ms"), Some("1267.110"));
        assert_eq!(log_field(fields, "missing"), None);
    }

    #[test]
    fn scorer_profile() {
        let text = concat!(
            "time scorer.stage {\"rows\":8192,\"gpu_ms\":500.0,\"stage_ms\":{\"gate\":120.0,\"pisa_attention\":140.0}}\n",
            "time scorer.stage {\"rows\":8192,\"gpu_ms\":480.0,\"stage_ms\":{\"gate\":110.0,\"pisa_attention\":130.0}}\n",
        );
        let profile = stage_text(text).unwrap();
        assert_eq!(profile["samples"], 2);
        assert_eq!(profile["rows-per-sample"], 8192);
        assert_eq!(profile["profiled-gpu-ms"]["mean"], 490.0);
        assert_eq!(profile["stage-ms"]["gate"]["mean"], 115.0);
        assert_eq!(profile["stage-ms"]["pisa_attention"]["mean"], 135.0);
    }

    fn semantic_fixture(candidate_nll: f64, candidate_digest: &str) -> Value {
        json!({
            "measurement": {
                "truth": {"context-tokens": 1_048_576, "evaluated-tokens": 871_296},
                "experiment-regression": {
                    "parameters": 1_177_756_440,
                    "round": 1,
                    "evaluated-tokens": 871_296,
                    "candidate-objective": -12.495894,
                    "incumbent-objective": -12.48408,
                    "candidate-nll": candidate_nll,
                    "incumbent-nll": 12.483136,
                    "accepted": false,
                    "decoded-output": {
                        "identity": "sha256-decoded-utf8",
                        "candidate": candidate_digest,
                        "incumbent": "incumbent-stable",
                    },
                },
            },
        })
    }

    #[test]
    fn gate_drift() {
        let replay = semantic_fixture(12.494953, "candidate-stable");
        let baseline = json!({
            "truth": replay["measurement"]["truth"].clone(),
            "parameters": 1_177_756_440,
            "round": 1,
            "evaluated-tokens": 871_296,
            "candidate-objective": -12.495894,
            "incumbent-objective": -12.48408,
            "candidate-nll": 12.494953,
            "incumbent-nll": 12.483136,
            "accepted": false,
            "decoded-output": {
                "identity": "sha256-decoded-utf8",
                "candidate": "candidate-stable",
                "incumbent": "incumbent-stable",
            },
        });
        let current = semantic_fixture(12.498162, "candidate-changed");
        let mut failures = Vec::new();
        compare_experiment(&baseline, &current, &mut failures).unwrap();
        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("candidate-nll changed"))
        );
        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("decoded output candidate changed"))
        );
    }

    #[test]
    fn gate_replay() {
        let current = semantic_fixture(12.494953, "candidate-stable");
        let mut baseline = current["measurement"]["experiment-regression"].clone();
        baseline["truth"] = current["measurement"]["truth"].clone();
        let mut failures = Vec::new();
        compare_experiment(&baseline, &current, &mut failures).unwrap();
        assert!(failures.is_empty(), "{failures:?}");
    }
}
