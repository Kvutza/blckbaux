use std::path::{Path, PathBuf};
use std::process::Command;

use blckbaux_tooling::build::BuildSystem;

struct HnetJob<'a> {
    root: &'a Path,
    dataset: &'a Path,
    validation: Option<&'a Path>,
    prompt: u32,
    interval: Option<u32>,
    model_seed: u64,
    generated: u32,
    temperature: String,
    proposal_seed: u64,
    acquisition: &'a str,
    acquisition_seed: u64,
    evaluation_seed: u64,
    validation_seed: u64,
    radius: String,
    radius_min: String,
    radius_max: String,
    arms: u32,
    candidates: u32,
    beta: String,
    rounds: u32,
    stage_samples: u32,
    reliability: String,
    surrogate: String,
    window: Option<u32>,
    reference_seed: u64,
    trust_shape: &'a str,
    output: &'a Path,
}

fn launch_modal(job: &HnetJob<'_>) -> Result<std::process::ExitStatus, String> {
    let mut command = Command::new(job.root.join("tools/buck2w"));
    command
        .current_dir(job.root)
        .env("BLCKBAUX_CUDA_DATASET", job.dataset)
        .env_remove("BLCKBAUX_CUDA_VALIDATION_DATASET")
        .args([
            "run",
            "//rust/crates/modal-runner:blckbaux-modal",
            "--",
            "hnet",
        ])
        .arg("--model-seed")
        .arg(job.model_seed.to_string())
        .arg("--generated")
        .arg(job.generated.to_string())
        .arg("--prompt")
        .arg(job.prompt.to_string())
        .arg("--visits")
        .arg("2")
        .arg("--temperature")
        .arg(&job.temperature)
        .arg("--seed")
        .arg(job.proposal_seed.to_string())
        .arg("--radius")
        .arg(&job.radius)
        .arg("--radius-min")
        .arg(&job.radius_min)
        .arg("--radius-max")
        .arg(&job.radius_max)
        .arg("--arms")
        .arg(job.arms.to_string())
        .arg("--candidates")
        .arg(job.candidates.to_string())
        .arg("--beta")
        .arg(&job.beta)
        .arg("--rounds")
        .arg(job.rounds.to_string())
        .arg("--stage-samples")
        .arg(job.stage_samples.to_string())
        .arg("--acquisition")
        .arg(job.acquisition)
        .arg("--acquisition-seed")
        .arg(job.acquisition_seed.to_string())
        .arg("--evaluation-seed")
        .arg(job.evaluation_seed.to_string())
        .arg("--validation-seed")
        .arg(job.validation_seed.to_string())
        .arg("--reliability")
        .arg(&job.reliability)
        .arg("--surrogate")
        .arg(&job.surrogate)
        .arg("--output")
        .arg(job.output);
    command.arg("--trust-shape").arg(job.trust_shape);
    if let (Some(validation), Some(interval)) = (job.validation, job.interval) {
        command
            .env("BLCKBAUX_CUDA_VALIDATION_DATASET", validation)
            .arg("--validation-interval")
            .arg(interval.to_string());
    }
    if let Some(window) = job.window {
        command
            .arg("--window-tokens")
            .arg(window.to_string())
            .arg("--reference-seed")
            .arg(job.reference_seed.to_string());
    }
    command.status().map_err(|error| error.to_string())
}

fn launch_local(job: &HnetJob<'_>) -> Result<std::process::ExitStatus, String> {
    let build = Command::new(job.root.join("tools/cuda-run"))
        .current_dir(job.root)
        .env_remove("RUSTUP_TOOLCHAIN")
        .arg("build")
        .status()
        .map_err(|error| format!("build local CUDA experiment: {error}"))?;
    if !build.success() {
        return Ok(build);
    }
    let mut command = Command::new(job.root.join("target/release/blckbaux-cuda"));
    command
        .current_dir(job.root)
        .arg("hnet-bo-bench")
        .arg(job.model_seed.to_string())
        .arg(job.dataset)
        .arg(job.prompt.to_string())
        .arg(job.output)
        .arg(job.generated.to_string())
        .arg("2")
        .arg(&job.temperature)
        .arg(job.proposal_seed.to_string())
        .arg(&job.radius)
        .arg(job.rounds.to_string())
        .arg(job.acquisition)
        .arg(job.acquisition_seed.to_string())
        .arg(job.evaluation_seed.to_string())
        .arg(job.arms.to_string())
        .arg(job.candidates.to_string())
        .arg(&job.beta)
        .arg(&job.radius_min)
        .arg(&job.radius_max)
        .arg(job.validation_seed.to_string())
        .arg(job.stage_samples.to_string());
    if let (Some(validation), Some(interval)) = (job.validation, job.interval) {
        command
            .arg(validation)
            .arg(job.prompt.to_string())
            .arg(interval.to_string());
    }
    command
        .arg("--reliability")
        .arg(&job.reliability)
        .arg("--surrogate")
        .arg(&job.surrogate);
    command.arg("--trust-shape").arg(job.trust_shape);
    if let Some(window) = job.window {
        command
            .arg("--window-tokens")
            .arg(window.to_string())
            .arg("--reference-seed")
            .arg(job.reference_seed.to_string());
    }
    command
        .status()
        .map_err(|error| format!("launch local CUDA experiment: {error}"))
}

pub(crate) fn run(root: &Path, path: &Path, out: &Path) -> Result<(), String> {
    let source_digest =
        blckbaux::experimental::perf_digest(root, blckbaux::experimental::Backend::CudaOxide)?;
    let config_path = absolute(root, path);
    let (source, _) = blckbaux::config::load_tune(&config_path)?;
    let resolved;
    let execution_path = if source.dataset().is_none() {
        resolved = crate::experiment::resolve(root, &config_path, None, BuildSystem::from_env())?;
        resolved.as_path()
    } else {
        config_path.as_path()
    };
    let (run, _) = blckbaux::config::load_tune(execution_path)?;
    run.validate_experiment()?;
    if run.model != Some(blckbaux::ModelArchitecture::FbtPisa1HnetMhc4V1) {
        return Err("CUDA TOML execution currently requires model=fbt-pisa1-hnet-mhc4-v1".into());
    }
    let (prompt, generated, temperature, window) = objective_shape(&run)?;
    let dataset = run
        .dataset()
        .ok_or("CUDA H-Net execution requires data.train")?;
    let validation = run.validation_dataset.as_deref();
    if let Some(validation) = validation {
        if dataset.canonicalize().map_err(|error| error.to_string())?
            == validation
                .canonicalize()
                .map_err(|error| error.to_string())?
        {
            return Err("CUDA H-Net training and validation corpora must be separate files".into());
        }
    }
    let interval = validation.map(|_| run.validation_interval.unwrap_or(run.rounds()));
    let policy = run.resident_enn(run.acquisition_seed())?;
    check_spec(&run, &policy)?;
    let output = absolute(root, out);
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let (acquisition, beta) = match run
        .acquisition
        .unwrap_or(blckbaux::AcquisitionConfig::UCB { beta: 2.0 })
    {
        blckbaux::AcquisitionConfig::UCB { beta } => ("ucb", beta as f32),
        blckbaux::AcquisitionConfig::Thompson => ("thompson", policy.ask.acquisition.beta),
        blckbaux::AcquisitionConfig::Random | blckbaux::AcquisitionConfig::Pareto => {
            return Err("CUDA H-Net resident search requires UCB or Thompson acquisition".into());
        }
    };
    let length = run.length();
    let job = HnetJob {
        root,
        dataset,
        validation,
        prompt,
        interval,
        model_seed: run.model_seed(),
        generated,
        temperature: temperature.to_string(),
        proposal_seed: run.proposal_seed(),
        acquisition,
        acquisition_seed: run.acquisition_seed(),
        evaluation_seed: run.sample_seeded(0),
        validation_seed: run.validation_seed(),
        radius: length.length_init.to_string(),
        radius_min: length.length_min.to_string(),
        radius_max: length.length_max.to_string(),
        arms: policy.pool.arms(),
        candidates: policy.pool.count(),
        beta: beta.to_string(),
        rounds: run.rounds(),
        window,
        reference_seed: run.reference_seed(),
        trust_shape: run
            .trust_region_shape
            .unwrap_or(blckbaux::config::TrustRegionShape::TensorFamilyStatic)
            .name(),
        stage_samples: run.scorer_stage_samples.unwrap_or(0),
        surrogate: blckbaux_artifact::json::to_string(&policy.surrogate_spec()?)
            .map_err(|error| error.to_string())?,
        reliability: blckbaux_artifact::json::to_string(
            &run.reliability_controller()
                .ok_or("CUDA H-Net requires reliability control")?
                .validate()?,
        )
        .map_err(|error| error.to_string())?,
        output: &output,
    };
    let status = if cfg!(target_os = "linux") {
        launch_local(&job)
    } else {
        launch_modal(&job)
    };
    let status = status?;
    if status.success() {
        std::fs::copy(&config_path, output.join("experiment.toml"))
            .map_err(|error| error.to_string())?;
        std::fs::write(
            output.join("performance-source.sha256"),
            format!("{source_digest}\n"),
        )
        .map_err(|error| error.to_string())?;
        Ok(())
    } else {
        Err(format!("CUDA H-Net execution exited with {status}"))
    }
}

/// Resolve objective geometry without fabricating a prompt/output split.
fn objective_shape(
    run: &blckbaux::config::RunConfig,
) -> Result<(u32, u32, f32, Option<u32>), String> {
    if let Some(window) = run.window_tokens {
        if run.generation.is_some() {
            return Err("causal corpus windows cannot also specify generation".into());
        }
        blckbaux_semantics::hnet::CausalPlan::window(window as usize, 16_384)?;
        // The positional prompt cap is unused by the corpus loader. One patch
        // is the explicit causal boundary, not an invented training prompt.
        return Ok((64, window, 0.0, Some(window)));
    }
    let generation = run
        .generation
        .as_ref()
        .ok_or("CUDA H-Net requires a causal window or generation settings")?;
    let prompt = generation
        .corpus_prompt_tokens
        .ok_or("CUDA H-Net generation requires corpus-prompt-tokens")?;
    let generated = generation.max_tokens;
    if prompt == 0 || prompt % 64 != 0 || generated == 0 || generated % 64 != 0 {
        return Err("H-Net generation requires aligned nonempty prompt/output patches".into());
    }
    Ok((prompt, generated, generation.temperature, None))
}

fn check_spec(
    run: &blckbaux::config::RunConfig,
    policy: &blckbaux::ResidentEnnConfig,
) -> Result<(), String> {
    blckbaux_semantics::hnet::ScaleShape::parse(
        run.trust_region_shape
            .unwrap_or(blckbaux::config::TrustRegionShape::TensorFamilyStatic)
            .name(),
    )?;
    if policy.proposal_method != blckbaux::procedural_pool::ProposalMethod::Additive
        || run.proposal_kernel() != blckbaux::ProposalKernel::Gaussian
        || policy.distance_scaling != blckbaux::DistanceScaling::Global
        || policy.history_geometry != blckbaux::config::HistoryGeometry::Latent
        || run.reliability_controller().is_none()
    {
        return Err("CUDA H-Net does not implement every resolved optimizer field; refusing a partial TOML projection".into());
    }
    Ok(())
}

fn absolute(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        root.join(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surrogate_projection() {
        let spec = blckbaux::config::ExperimentSpec::parse(
            r#"
version = 4
experiment = "pretrain"
model = "fbt-pisa1-hnet-mhc4-v1"
[data]
recipe = "stack-v3-rust-go-v1"
[objective]
function = "cross-entropy"
sampling = "natural"
window-tokens = 1048576
[execution]
rounds = 1000
round-target-ms = 1000
[optimizer]
algorithm = "turbo-enn"
[optimizer.candidate-generation]
generator = "independent"
arms = 1
candidates = 1
[optimizer.candidate-generation.perturbation]
distribution = "gaussian"
[optimizer.surrogate]
estimator = "enn"
scaling = "global"
neighbors = 10
neighbor-rule = "fixed"
epistemic-scale = 0.05
aleatoric-scale = 0.05
fit = "leave-one-out"
fit-candidates = 30
fit-samples = 10
[optimizer.acquisition]
function = "thompson"
[optimizer.trust-region]
policy = "reliability"
initial = 0.02
min = 0.01
max = 0.2
failure-tolerance = 4
"#,
        )
        .unwrap();
        let run = spec.compile().unwrap();
        let policy = run.resident_enn(73).unwrap();
        assert_eq!(
            objective_shape(&run).unwrap(),
            (64, 1_048_576, 0.0, Some(1_048_576))
        );
        check_spec(&run, &policy).unwrap();
        let settings = policy.surrogate_spec().unwrap();
        assert_eq!(policy.pool.count(), 1);
        assert_eq!(settings.neighbors, 10);
        assert_eq!(settings.fit_candidates, 30);
        assert_eq!(settings.fit_samples, 10);
        assert!(settings.fit);
        assert!(!settings.adaptive);
        let json = blckbaux_artifact::json::to_string(&settings).unwrap();
        let decoded: blckbaux::SurrogateSpec = blckbaux_artifact::json::from_str(&json).unwrap();
        assert_eq!(decoded, settings);
    }
}
