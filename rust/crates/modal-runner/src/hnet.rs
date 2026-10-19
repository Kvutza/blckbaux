use super::{Result, make_sandbox, pull_artifacts, source_tar, stream_exec, upload_source};
use modal_rs::{ModalClient, SandboxExecOptions};
use std::io;
use std::path::{Path, PathBuf};

pub(super) struct Config {
    output: PathBuf,
    model_seed: u64,
    prompt: u32,
    generated: u32,
    visits: u32,
    temperature: f32,
    seed: u64,
    acquisition: String,
    acquisition_seed: Option<u64>,
    evaluation_seed: Option<u64>,
    validation_seed: Option<u64>,
    radius: f32,
    radius_min: f32,
    radius_max: f32,
    arms: u32,
    candidates: u32,
    beta: f32,
    rounds: u32,
    stage_samples: u32,
    validation_interval: Option<u32>,
    reliability: blckbaux::ReliabilityPolicy,
    surrogate: blckbaux_semantics::optimizer::SurrogateSpec,
    window: Option<u32>,
    reference_seed: Option<u64>,
    trust_shape: Option<String>,
}

pub(super) fn parse<I: Iterator<Item = std::ffi::OsString>>(mut args: I) -> Result<Config> {
    let mut config = Config {
        output: "results/cuda/hnet-t4".into(),
        model_seed: 0,
        prompt: 524_288,
        generated: 1_048_576,
        visits: 2,
        temperature: 1.0,
        seed: 17,
        acquisition: "thompson".into(),
        acquisition_seed: None,
        evaluation_seed: None,
        validation_seed: None,
        radius: 0.001,
        radius_min: 0.001,
        radius_max: 2.0,
        arms: 1,
        candidates: 4,
        beta: 1.0,
        rounds: 128,
        stage_samples: 0,
        validation_interval: None,
        window: None,
        reference_seed: None,
        trust_shape: None,
        reliability: Default::default(),
        surrogate: blckbaux_semantics::optimizer::SurrogateSpec {
            fit: true,
            ..Default::default()
        },
    };
    while let Some(flag) = args.next() {
        let value = args.next().ok_or_else(|| {
            io::Error::other(format!("missing value for {}", flag.to_string_lossy()))
        })?;
        match flag.to_str() {
            Some("--output") => config.output = value.into(),
            Some("--model-seed") => config.model_seed = value.to_string_lossy().parse()?,
            Some("--prompt") => config.prompt = value.to_string_lossy().parse()?,
            Some("--generated") => config.generated = value.to_string_lossy().parse()?,
            Some("--visits") => config.visits = value.to_string_lossy().parse()?,
            Some("--temperature") => config.temperature = value.to_string_lossy().parse()?,
            Some("--seed") => config.seed = value.to_string_lossy().parse()?,
            Some("--trust-shape") => config.trust_shape = Some(value.to_string_lossy().into_owned()),
            Some("--window-tokens") => config.window = Some(value.to_string_lossy().parse()?),
            Some("--reference-seed") => config.reference_seed = Some(value.to_string_lossy().parse()?),
            Some("--acquisition") => config.acquisition = value.to_string_lossy().into_owned(),
            Some("--acquisition-seed") => {
                config.acquisition_seed = Some(value.to_string_lossy().parse()?)
            }
            Some("--evaluation-seed") => {
                config.evaluation_seed = Some(value.to_string_lossy().parse()?)
            }
            Some("--validation-seed") => {
                config.validation_seed = Some(value.to_string_lossy().parse()?)
            }
            Some("--radius") => config.radius = value.to_string_lossy().parse()?,
            Some("--radius-min") => config.radius_min = value.to_string_lossy().parse()?,
            Some("--radius-max") => config.radius_max = value.to_string_lossy().parse()?,
            Some("--arms") => config.arms = value.to_string_lossy().parse()?,
            Some("--candidates") => config.candidates = value.to_string_lossy().parse()?,
            Some("--beta") => config.beta = value.to_string_lossy().parse()?,
            Some("--rounds") => config.rounds = value.to_string_lossy().parse()?,
            Some("--stage-samples") => {
                config.stage_samples = value.to_string_lossy().parse()?
            }
            Some("--validation-interval") => {
                config.validation_interval = Some(value.to_string_lossy().parse()?)
            }
            Some("--reliability") => {
                config.reliability = blckbaux_artifact::json::from_str::<blckbaux::ReliabilityPolicy>(
                    &value.to_string_lossy(),
                )?.validate()?;
            }
            Some("--surrogate") => {
                config.surrogate = blckbaux_artifact::json::from_str::<blckbaux_semantics::optimizer::SurrogateSpec>(
                    &value.to_string_lossy(),
                )?.validate()?;
            }
            _ => return Err(io::Error::other("usage: blckbaux-modal hnet --output DIR --model-seed N --prompt N --generated N --visits N --temperature F --seed N --acquisition thompson|ucb --acquisition-seed N --evaluation-seed N --validation-seed N --radius F --radius-min F --radius-max F --arms N --candidates N --beta F --rounds N --stage-samples N [--validation-interval N] [--window-tokens N --reference-seed N] [--reliability JSON] [--surrogate JSON] [--trust-shape scalar|tensor-family-static]").into()),
        }
    }
    if config.prompt == 0
        || config.prompt % 64 != 0
        || config.generated == 0
        || (config.window.is_none() && config.generated % 64 != 0)
        || config.window.is_some_and(|window| {
            !(64..=1_048_576).contains(&window)
                || config.generated != window
                || config.temperature != 0.0
        })
        || (config.reference_seed.is_some() && config.window.is_none())
        || config.visits == 0
        || !config.temperature.is_finite()
        || config.temperature < 0.0
        || !config.radius.is_finite()
        || config.radius <= 0.0
        || !config.radius_min.is_finite()
        || !config.radius_max.is_finite()
        || config.radius_min <= 0.0
        || config.radius_min > config.radius
        || config.radius > config.radius_max
        || config.arms == 0
        || !matches!(config.candidates, 1 | 4)
        || !config.candidates.is_multiple_of(config.arms)
        || !config.beta.is_finite()
        || config.beta < 0.0
        || config.rounds < 128
        || !matches!(config.acquisition.as_str(), "thompson" | "ucb")
        || config.validation_interval == Some(0)
        || config
            .trust_shape
            .as_deref()
            .is_some_and(|name| !matches!(name, "scalar" | "tensor-family-static"))
    {
        return Err(io::Error::other("invalid H-Net objective parameters").into());
    }
    Ok(config)
}

fn command(config: &Config) -> String {
    let Config {
        generated,
        model_seed,
        prompt,
        visits,
        temperature,
        seed,
        acquisition,
        acquisition_seed,
        evaluation_seed,
        validation_seed,
        radius,
        radius_min,
        radius_max,
        arms,
        candidates,
        beta,
        rounds,
        stage_samples,
        validation_interval,
        ..
    } = config;
    let acquisition_seed = (*acquisition_seed).unwrap_or(*seed);
    let evaluation_seed = (*evaluation_seed).unwrap_or(*seed);
    let validation_seed = (*validation_seed).unwrap_or(evaluation_seed);
    // Only validated numeric policy fields reach this serialization. Quote the
    // complete argument independently of shell parsing and preserve each field.
    let reliability = blckbaux_artifact::json::to_string(&config.reliability)
        .expect("validated numeric reliability policy is serializable")
        .replace('\'', "'\\''");
    let surrogate = blckbaux_artifact::json::to_string(&config.surrogate)
        .expect("validated surrogate settings serialize")
        .replace('\'', "'\\''");
    let validation = (*validation_interval).map_or_else(String::new, |interval| {
        format!(" /tmp/blckbaux-corpus/validation/data.blckbauxptn {prompt} {interval}")
    });
    let trust_shape = config
        .trust_shape
        .as_deref()
        .unwrap_or(if config.window.is_some() {
            "tensor-family-static"
        } else {
            "scalar"
        });
    let causal = config.window.map_or_else(String::new, |window| {
        let reference = config.reference_seed.unwrap_or(evaluation_seed);
        format!(" --window-tokens {window} --reference-seed {reference}")
    });
    let profile = if std::env::var_os("BLCKBAUX_CUDA_PROFILE").is_some() {
        "export BLCKBAUX_CUDA_PROFILE=1; "
    } else {
        ""
    };
    let shadow = if std::env::var_os("BLCKBAUX_CUDA_PISA_SHADOW").is_some() {
        "export BLCKBAUX_CUDA_PISA_SHADOW=1; "
    } else {
        ""
    };
    let compare = if std::env::var_os("BLCKBAUX_CUDA_PISA_COMPARE").is_some() {
        "export BLCKBAUX_CUDA_PISA_COMPARE=1; "
    } else {
        ""
    };
    let fused_readout = if std::env::var_os("BLCKBAUX_CUDA_FUSED_READOUT").is_some() {
        "export BLCKBAUX_CUDA_FUSED_READOUT=1; "
    } else {
        ""
    };
    format!(
        "set -euo pipefail; cd /opt/blckbaux; \
         {profile}\
         {shadow}\
         {compare}\
         {fused_readout}\
         CARGO_HOME=/cache/cargo CARGO_TARGET_DIR=/cache/target ./tools/cuda-run build; \
         rm -rf /tmp/blckbaux-generation; \
         /cache/target/release/blckbaux-cuda hnet-bo-bench \
         {model_seed} /tmp/blckbaux-corpus/train/data.blckbauxptn \
         {prompt} /tmp/blckbaux-generation \
         {generated} {visits} {temperature} {seed} {radius} {rounds} \
         {acquisition} {acquisition_seed} {evaluation_seed} \
         {arms} {candidates} {beta} {radius_min} {radius_max} {validation_seed} \
         {stage_samples}{validation} --reliability '{reliability}' --surrogate '{surrogate}' --trust-shape {trust_shape}{causal}"
    )
}

pub(super) async fn run(
    client: &mut ModalClient,
    app: &modal_rs::App,
    config: Config,
) -> Result<()> {
    let root = std::env::current_dir()?;
    let digest =
        blckbaux::experimental::perf_digest(&root, blckbaux::experimental::Backend::CudaOxide)?;
    let dataset = std::env::var_os("BLCKBAUX_CUDA_DATASET")
        .ok_or("H-Net T4 execution requires BLCKBAUX_CUDA_DATASET")?;
    let dataset = PathBuf::from(dataset);
    let train = blckbaux_artifact::pretrain::CorpusBundle::new(&dataset)?;
    let validation = if config.validation_interval.is_some() {
        let path = std::env::var_os("BLCKBAUX_CUDA_VALIDATION_DATASET")
            .ok_or("validation interval requires BLCKBAUX_CUDA_VALIDATION_DATASET")?;
        Some(blckbaux_artifact::pretrain::CorpusBundle::new(Path::new(
            &path,
        ))?)
    } else {
        None
    };
    if config.output.exists() {
        return Err("H-Net output already exists; choose a fresh result directory".into());
    }
    let archive = source_tar()?;
    let sandbox = make_sandbox(client, app).await?;
    let result: Result<(Vec<u8>, Vec<u8>, Vec<u8>, Option<Vec<u8>>)> = async {
        upload_source(&sandbox, client, &archive).await?;
        upload_corpus(&sandbox, client, &train, "train").await?;
        if let Some(validation) = &validation {
            upload_corpus(&sandbox, client, validation, "validation").await?;
        }
        let status = stream_exec(
            &sandbox,
            client,
            SandboxExecOptions::new(vec!["bash", "-lc", &command(&config)]).with_timeout(3_500),
        )
        .await?;
        if !status.is_success() {
            return Err(
                io::Error::other(format!("CUDA H-Net generation failed: {status:?}")).into(),
            );
        }
        let (record, tokens) = pull_artifacts(&sandbox, client, &config.output, None).await?;
        let rounds = pull_log(&sandbox, client, "/tmp/blckbaux-generation/rounds.jsonl").await?;
        let validation = if config.validation_interval.is_some() {
            Some(
                pull_log(
                    &sandbox,
                    client,
                    "/tmp/blckbaux-generation/validation.jsonl",
                )
                .await?,
            )
        } else {
            None
        };
        Ok((record, tokens, rounds, validation))
    }
    .await;
    let stopped = sandbox.terminate(client).await;
    let (record, tokens, rounds, validation) = result?;
    stopped?;
    save(
        &config.output,
        &record,
        &tokens,
        &rounds,
        validation.as_deref(),
        &digest,
    )?;
    print!("{}", String::from_utf8_lossy(&record));
    println!("HNET_T4 ok=true output={}", config.output.display());
    Ok(())
}

/// Upload only the immutable files selected by the corpus relocation plan.
/// Relative destinations are generated by blckbaux-artifact, not shell input paths.
async fn upload_corpus(
    sandbox: &modal_rs::Sandbox,
    client: &mut ModalClient,
    bundle: &blckbaux_artifact::pretrain::CorpusBundle,
    name: &str,
) -> Result<()> {
    for (source, relative) in &bundle.files {
        upload_file(sandbox, client, name, relative, std::fs::read(source)?).await?;
    }
    if let Some(manifest) = &bundle.manifest {
        upload_file(
            sandbox,
            client,
            name,
            Path::new("data.blckbauxptn"),
            manifest.clone(),
        )
        .await?;
    }
    Ok(())
}

async fn upload_file(
    sandbox: &modal_rs::Sandbox,
    client: &mut ModalClient,
    name: &str,
    relative: &Path,
    bytes: Vec<u8>,
) -> Result<()> {
    if !matches!(name, "train" | "validation")
        || relative
            .components()
            .any(|part| !matches!(part, std::path::Component::Normal(_)))
        || !relative
            .to_string_lossy()
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"-./".contains(&byte))
    {
        return Err("unsafe corpus relocation destination".into());
    }
    let path = Path::new("/tmp/blckbaux-corpus").join(name).join(relative);
    let command = format!(
        "mkdir -p '{}'; cat > '{}'",
        path.parent().ok_or("upload has no parent")?.display(),
        path.display()
    );
    let upload = sandbox
        .exec(
            client,
            SandboxExecOptions::new(vec!["bash", "-lc", &command])
                .with_stdin(bytes)
                .with_timeout(60),
        )
        .await?;
    if !upload.exit_status.is_success() {
        return Err("H-Net corpus upload failed".into());
    }
    Ok(())
}

async fn pull_log(
    sandbox: &modal_rs::Sandbox,
    client: &mut ModalClient,
    path: &str,
) -> Result<Vec<u8>> {
    let result = sandbox
        .exec(
            client,
            SandboxExecOptions::new(vec!["cat", path]).with_timeout(30),
        )
        .await?;
    if !result.exit_status.is_success() {
        return Err(io::Error::other(format!("CUDA H-Net log download failed: {path}")).into());
    }
    result
        .stdout
        .ok_or_else(|| io::Error::other(format!("CUDA H-Net log was empty: {path}")))
        .map_err(Into::into)
}

fn save(
    output: &Path,
    record: &[u8],
    tokens: &[u8],
    rounds: &[u8],
    validation: Option<&[u8]>,
    digest: &str,
) -> Result<()> {
    std::fs::create_dir_all(output)?;
    std::fs::write(output.join("result.json"), record)?;
    std::fs::write(output.join("tokens.json"), tokens)?;
    std::fs::write(output.join("rounds.jsonl"), rounds)?;
    if let Some(validation) = validation {
        std::fs::write(output.join("validation.jsonl"), validation)?;
    }
    std::fs::write(
        output.join("performance-source.sha256"),
        format!("{digest}\n"),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_command() {
        let config = parse(
            [
                "--window-tokens",
                "1048576",
                "--generated",
                "1048576",
                "--temperature",
                "0",
                "--rounds",
                "1000",
                "--candidates",
                "1",
                "--reference-seed",
                "137",
                "--trust-shape",
                "tensor-family-static",
                "--validation-interval",
                "1000",
            ]
            .into_iter()
            .map(std::ffi::OsString::from),
        )
        .unwrap();
        let command = command(&config);
        assert!(command.contains("/tmp/blckbaux-corpus/train/data.blckbauxptn"));
        assert!(command.contains("/tmp/blckbaux-corpus/validation/data.blckbauxptn"));
        assert!(command.contains("--window-tokens 1048576 --reference-seed 137"));
        assert!(command.contains("--trust-shape tensor-family-static"));
        for args in [
            vec!["--window-tokens", "1048576", "--temperature", "1"],
            vec!["--window-tokens", "0"],
            vec!["--reference-seed", "137"],
            vec!["--trust-shape", "unknown"],
        ] {
            assert!(parse(args.into_iter().map(std::ffi::OsString::from)).is_err());
        }
    }

    #[test]
    fn resolved_policy() {
        let spec = blckbaux_semantics::optimizer::SurrogateSpec {
            neighbors: 10,
            fit_candidates: 30,
            fit_samples: 10,
            fit: true,
            adaptive: true,
            ..Default::default()
        };
        let json = blckbaux_artifact::json::to_string(&spec).unwrap();
        let args = [
            "--candidates",
            "1",
            "--rounds",
            "1000",
            "--acquisition",
            "thompson",
            "--surrogate",
            &json,
        ];
        let config = parse(args.into_iter().map(std::ffi::OsString::from)).unwrap();
        assert_eq!(config.surrogate, spec);
        assert_eq!(config.candidates, 1);
        let command = command(&config);
        assert!(command.contains(&format!("--surrogate '{json}'")));
        assert!(command.contains("--reliability '"));
        assert!(
            parse(
                ["--candidates", "2"]
                    .into_iter()
                    .map(std::ffi::OsString::from)
            )
            .is_err()
        );
        let wrong = json.replace("\"neighbors\"", "\"neighbours\"");
        assert!(
            parse(
                ["--surrogate", &wrong]
                    .into_iter()
                    .map(std::ffi::OsString::from)
            )
            .is_err()
        );
    }
}
