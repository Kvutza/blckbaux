use super::VOCAB;
use super::bo::{MAX_HISTORY, NeighborValueEstimator, PoolShape, RadiusController, SurrogateSpec};
use crate::model::{FbtModel, ModelOutput};
use crate::{CudaResult, cuda_error, read_prefix, target_name};
use blckbaux_cuda_device::{ObjectiveShape, ScoreStats};
use cuda_core::{DeviceBuffer, LaunchConfig1D};
use half::f16;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

#[derive(Clone)]
struct HnetTask {
    prompt: Vec<u32>,
    targets: Vec<u32>,
    corpus: Option<blckbaux_artifact::pretrain::CorpusBatch>,
}

impl HnetTask {
    fn causal(batch: blckbaux_artifact::pretrain::CorpusBatch) -> Self {
        Self {
            prompt: Vec::new(),
            targets: Vec::new(),
            corpus: Some(batch),
        }
    }
    fn input(&self) -> &[u32] {
        self.corpus
            .as_ref()
            .map_or(self.prompt.as_slice(), |c| c.batch.tokens.as_slice())
    }
    fn targets(&self) -> &[u32] {
        self.corpus.as_ref().map_or(self.targets.as_slice(), |c| {
            &c.batch.labels[blckbaux_semantics::hnet::PATCH - 1..]
        })
    }
    fn capacity(&self) -> CudaResult<usize> {
        if let Some(corpus) = &self.corpus {
            Ok(corpus.batch.tokens.len())
        } else {
            self.prompt
                .len()
                .checked_add(self.targets.len())
                .ok_or_else(|| "CUDA H-Net context overflow".into())
        }
    }
    fn configure(&self, model: &mut FbtModel) -> CudaResult<()> {
        if let Some(corpus) = &self.corpus {
            model.configure_causal(&corpus.batch)
        } else {
            model.configure_hnet(&self.prompt, self.targets.len(), Some(&self.targets))
        }
    }
    fn sample_seed(&self, fallback: u64) -> u64 {
        self.corpus.as_ref().map_or(fallback, |c| {
            blckbaux_semantics::random::splitmix64(c.batch.offset as u64)
        })
    }
    fn weight(&self) -> u32 {
        self.corpus.as_ref().map_or(1, |c| c.weight)
    }
}

enum HnetSource {
    Rows(Vec<HnetTask>),
    Corpus {
        data: blckbaux_artifact::pretrain::PretrainDataset,
        prompt: usize,
        output: usize,
        seed: u64,
    },
    Causal {
        data: blckbaux_artifact::pretrain::CorpusSet,
        window: usize,
        seed: u64,
    },
}

impl HnetSource {
    fn causal(input: &str, window: usize, seed: u64, validation: bool) -> CudaResult<Self> {
        let path = Path::new(input);
        let data = if validation {
            blckbaux_artifact::pretrain::CorpusSet::load_eval(path)?
        } else {
            blckbaux_artifact::pretrain::CorpusSet::load(path)?
        };
        Ok(Self::Causal { data, window, seed })
    }
    fn load(input: &str, target: &str, output: usize, seed: u64) -> CudaResult<Self> {
        if input.ends_with(".blckbauxptn") {
            let prompt = target
                .parse::<usize>()
                .map_err(|error| format!("invalid H-Net corpus prompt cap: {error}"))?;
            return Ok(Self::Corpus {
                data: blckbaux_artifact::pretrain::PretrainDataset::load(Path::new(input))?,
                prompt,
                output,
                seed,
            });
        }
        Ok(Self::Rows(hnet_tasks(input, target)?))
    }

    fn task(&self, round: usize) -> CudaResult<HnetTask> {
        match self {
            Self::Rows(tasks) => {
                let index = if tasks.len() == 1 { 0 } else { round };
                tasks
                    .get(index)
                    .cloned()
                    .ok_or_else(|| "CUDA H-Net training task schedule ended early".into())
            }
            Self::Corpus {
                data,
                prompt,
                output,
                seed,
            } => {
                let round = u32::try_from(round).map_err(|_| "CUDA H-Net round exceeds u32")?;
                let (prompt, targets, _) = data.window(*prompt, *output, *seed, round)?;
                Ok(HnetTask {
                    prompt,
                    targets,
                    corpus: None,
                })
            }
            Self::Causal { data, window, seed } => data
                .causal_batch(*window, *seed, u32::try_from(round).map_err(cuda_error)?)
                .map(HnetTask::causal),
        }
    }

    fn fixed(&self) -> bool {
        matches!(self, Self::Rows(tasks) if tasks.len() == 1)
    }

    fn count(&self, rounds: usize) -> usize {
        match self {
            Self::Rows(tasks) => tasks.len(),
            Self::Corpus { .. } | Self::Causal { .. } => rounds,
        }
    }

    fn capacity(&self) -> CudaResult<usize> {
        match self {
            Self::Rows(tasks) => tasks
                .iter()
                .map(HnetTask::capacity)
                .collect::<CudaResult<Vec<_>>>()?
                .into_iter()
                .max()
                .ok_or_else(|| String::from("CUDA H-Net BO requires a training task")),
            Self::Corpus { prompt, output, .. } => prompt
                .checked_add(*output)
                .ok_or_else(|| String::from("CUDA H-Net BO context overflow")),
            Self::Causal { window, .. } => Ok(*window),
        }
    }

    fn validation(&self) -> CudaResult<Vec<HnetTask>> {
        const TASKS: usize = 32;
        match self {
            Self::Rows(tasks) => Ok(tasks.clone()),
            Self::Causal { data, window, seed } => data
                .causal_batches(*window, *seed)
                .map(|batches| batches.into_iter().map(HnetTask::causal).collect()),
            Self::Corpus {
                data,
                prompt,
                output,
                seed,
            } => {
                let mut tasks = Vec::with_capacity(TASKS);
                let mut offsets = std::collections::HashSet::with_capacity(TASKS);
                for round in 0..data.sequences() {
                    let Some((task_prompt, targets, offset)) =
                        data.natural_task(*prompt, *output, *seed, round)?
                    else {
                        break;
                    };
                    if offsets.insert(offset) {
                        tasks.push(HnetTask {
                            prompt: task_prompt,
                            targets,
                            corpus: None,
                        });
                    }
                    if tasks.len() == TASKS {
                        return Ok(tasks);
                    }
                }
                if !tasks.is_empty() {
                    return Err(format!(
                        "CUDA H-Net validation corpus yielded only {} distinct natural tasks; {TASKS} required",
                        tasks.len()
                    ));
                }
                let count = TASKS.min(data.sequences() as usize);
                let total = prompt
                    .checked_add(*output)
                    .ok_or("CUDA H-Net validation token range overflow")?
                    .min(blckbaux_artifact::pretrain::CONTEXT as usize);
                let prompt_len = (*prompt).min(total);
                if prompt_len == total {
                    return Err("CUDA H-Net validation context leaves no target tokens".into());
                }
                for index in 0..count {
                    let source = data.sequence(index as u32)?;
                    tasks.push(HnetTask {
                        prompt: source[..prompt_len].to_vec(),
                        targets: source[prompt_len..total].to_vec(),
                        corpus: None,
                    });
                }
                Ok(tasks)
            }
        }
    }
}

fn hnet_token_rows(path: &str) -> CudaResult<Vec<Vec<u32>>> {
    let value: blckbaux_artifact::json::Value =
        blckbaux_artifact::json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let rows = value
        .as_seq()
        .ok_or_else(|| format!("H-Net token file {path} must contain a JSON array"))?;
    if rows.first().is_some_and(|row| row.is_seq()) {
        blckbaux_artifact::json::from_value(value).map_err(|e| e.to_string())
    } else {
        let row = blckbaux_artifact::json::from_value(value).map_err(|e| e.to_string())?;
        Ok(vec![row])
    }
}

fn hnet_tasks(prompt_path: &str, target_path: &str) -> CudaResult<Vec<HnetTask>> {
    let prompts = hnet_token_rows(prompt_path)?;
    let targets = hnet_token_rows(target_path)?;
    if prompts.len() != targets.len() || prompts.is_empty() {
        return Err(
            "H-Net prompt and target files must contain the same positive row count".into(),
        );
    }
    Ok(prompts
        .into_iter()
        .zip(targets)
        .map(|(prompt, targets)| HnetTask {
            prompt,
            targets,
            corpus: None,
        })
        .collect())
}

fn validate_hnet_task(task: &HnetTask, generated_cap: usize) -> CudaResult<()> {
    if let Some(corpus) = &task.corpus {
        let batch = &corpus.batch;
        blckbaux_semantics::hnet::CausalPlan::new(
            batch.tokens.len(),
            batch.labels.len(),
            &batch.loss_mask,
            usize::MAX,
        )?;
        if batch.tokens.len() > generated_cap
            || batch
                .tokens
                .iter()
                .chain(&batch.labels)
                .any(|&t| t >= VOCAB as u32)
        {
            return Err("CUDA causal H-Net batch exceeds its window or vocabulary".into());
        }
        return Ok(());
    }
    if task.prompt.is_empty()
        || task.prompt.len() % 64 != 0
        || task.targets.is_empty()
        || task.targets.len() % 64 != 0
        || task.targets.len() > generated_cap
        || task
            .prompt
            .iter()
            .chain(&task.targets)
            .any(|&token| token >= VOCAB as u32)
    {
        return Err("CUDA H-Net BO tasks require vocabulary-valid, nonempty 64-token prompt/target patches within GENERATED_TOKENS".into());
    }
    Ok(())
}

const fn hnet_learning_checkpoint(round: usize) -> bool {
    (round <= 512 && round.is_power_of_two()) || (round >= 1000 && round % 1000 == 0)
}

fn hnet_condition_seed(root: u64, round: usize) -> CudaResult<u64> {
    let round = u64::try_from(round).map_err(|_| "CUDA H-Net round exceeds u64")?;
    Ok(super::bo::splitmix64(
        root ^ round.wrapping_add(1) ^ 0x636f_7270_7573_2d78,
    ))
}

fn validate_hnet_tasks(
    model: &mut FbtModel,
    tasks: &[HnetTask],
    visits: usize,
    temperature: f32,
    evaluation_seed: u64,
    round: usize,
    phase: &str,
    progress: &mut BufWriter<fs::File>,
) -> CudaResult<blckbaux_artifact::json::Value> {
    let started = std::time::Instant::now();
    let mut rewards = Vec::with_capacity(tasks.len());
    let mut nlls = Vec::with_capacity(tasks.len());
    let mut sources = Vec::with_capacity(tasks.len());
    let mut weighted_nll = 0.0_f64;
    let mut weighted_bits = 0.0_f64;
    let mut total_weight = 0_u64;
    let mut evaluated_tokens = 0_usize;
    let mut device_ms = 0.0_f32;
    for task in tasks {
        task.configure(model)?;
        let evaluation = model.evaluate_hnet(
            visits,
            temperature,
            task.sample_seed(evaluation_seed),
            false,
        )?;
        let stats = evaluation
            .stats
            .ok_or("CUDA H-Net held-out task was not scored")?;
        rewards.push(stats.reward);
        nlls.push(stats.mean_nll);
        let weight = task.weight();
        total_weight += u64::from(weight);
        weighted_nll += f64::from(stats.mean_nll) * f64::from(weight);
        let estimate = evaluation
            .estimate
            .ok_or("CUDA held-out likelihood omitted its estimate")?;
        weighted_bits += estimate.bits / task.targets().len() as f64 * f64::from(weight);
        evaluated_tokens += task.targets().len();
        sources.push(blckbaux_artifact::json::json!({
            "source": task.corpus.as_ref().map_or("task", |c| c.source.as_str()),
            "weight": weight, "nll": stats.mean_nll,
            "bits_per_token": estimate.bits / task.targets().len() as f64,
            "corpus_offset": task.corpus.as_ref().map(|c| c.batch.offset),
            "scored_tokens": task.targets().len(),
        }));
        device_ms += evaluation.device_ms;
    }
    if total_weight == 0 {
        return Err("CUDA held-out likelihood has no positive source weight".into());
    }
    let mean_nll = weighted_nll / total_weight as f64;
    let mean_reward = -mean_nll;
    let record = blckbaux_artifact::json::json!({
        "schema": "blckbaux.cuda.hnet-validation.v1",
        "phase": phase,
        "round": round,
        "tasks": tasks.len(),
        "reward": mean_reward,
        "mean_nll": mean_nll,
        "bits_per_token": weighted_bits / total_weight as f64,
        "sources": sources,
        "evaluated_tokens": evaluated_tokens,
        "aggregation": "weighted-mean",
        "task_rewards": rewards,
        "task_nlls": nlls,
        "device_ms": device_ms,
        "wall_ms": started.elapsed().as_secs_f64() * 1000.0,
        "used_for_acceptance": false,
        "excluded_from_round_timing": true
    });
    writeln!(
        progress,
        "{}",
        blckbaux_artifact::json::to_string(&record).map_err(|e| e.to_string())?
    )
    .map_err(|e| format!("write CUDA H-Net validation log: {e}"))?;
    progress
        .flush()
        .map_err(|e| format!("flush CUDA H-Net validation log: {e}"))?;
    Ok(record)
}

pub fn check() -> CudaResult<()> {
    crate::precision::check()?;
    crate::hnet_probe::check()?;
    super::routing::check()?;
    let mut model = FbtModel::fixture()?;
    score_check(&model)?;
    hnet_check(&model)?;
    let tokens = vec![0; 4096];
    for (visits, value) in [(1, 0x4600), (2, 0x4800), (4, 0x4a00)] {
        let result = model.run(&tokens, visits, 0.0, 17)?;
        if result.tokens.iter().any(|&token| token != 0)
            || result.hidden.iter().any(|&v| v != 0x3c00)
            || result.stream_probe.iter().any(|&v| v != value)
        {
            return Err(format!(
                "full model fixture mismatch at {visits} recurrent visits; probe={:?}",
                &result.stream_probe[..8]
            ));
        }
        report(&result);
    }
    let sampled = model.run(&tokens, 2, 0.8, 17)?;
    let expected = model.generate(&[0; 128], 2, 0.8, 17, 1)?;
    let unrolled = model.generate(&[0; 128], 2, 0.8, 17, 2)?;
    if expected.tokens[128..] != sampled.tokens[127..4095]
        || expected.tokens != unrolled.tokens
        || expected.tokens[128..]
            .iter()
            .all(|&token| token == expected.tokens[128])
    {
        return Err(
            "accepted-prefix generation changed with unrolling or stochastic sampling collapsed"
                .into(),
        );
    }
    println!(
        "GENERATION_PARITY ok=true prompt=128 generated=3968 temperature=0.8 unroll=1,2 passes={}/{}",
        expected.passes, unrolled.passes
    );
    println!(
        "MODEL_PARITY ok=true layers=5 experts=625 shared=1 streams=4 recurrence=1,2,4 rope=true readout=true"
    );
    model.check_causal()?;
    Ok(())
}

/// Capture full initialized weights and real forward outputs, without BO
/// updates. Allocation dimensions match the shared finite case explicitly.
pub fn initialized_check(path: &Path) -> CudaResult<()> {
    use blckbaux_semantics::hnet::probe::initialized as case;
    let mut model = FbtModel::initialize_hnet(case::SEED, case::ROWS, 4096)?;
    let trace = model.initialized_trace()?;
    fs::write(
        path,
        blckbaux_artifact::json::to_vec(&trace).map_err(cuda_error)?,
    )
    .map_err(cuda_error)?;
    println!(
        "INITIALIZED_FORWARD backend=cuda seed={} parameters={} rows={} targets={} output={}",
        case::SEED,
        blckbaux_semantics::shape::HNET_PARAMS,
        case::ROWS,
        case::TARGETS,
        path.display()
    );
    Ok(())
}

fn hnet_check(model: &FbtModel) -> CudaResult<()> {
    let macros = [3_u32, 7];
    let (tokens, _) = model.expand_patches(&macros)?;
    let mut expected = Vec::with_capacity(128);
    for macro_token in macros {
        let mut previous = macro_token;
        expected.push(previous);
        for intra in 1..64_u32 {
            let mut mixed = (u64::from(previous) << 32)
                ^ (u64::from(macro_token) << 16)
                ^ (u64::from(intra) << 8)
                ^ 0x3c00;
            mixed = mixed.wrapping_add(0x9e3779b97f4a7c15);
            mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d049bb133111eb);
            mixed ^= mixed >> 31;
            previous = (mixed % VOCAB as u64) as u32;
            expected.push(previous);
        }
    }
    if tokens != expected {
        return Err("CUDA H-Net patch expansion mismatch".into());
    }
    println!("HNET_PARITY ok=true patches=2 patch-size=64 tokens=128");
    Ok(())
}

fn score_check(model: &FbtModel) -> CudaResult<()> {
    const ROWS: usize = 2;

    let mut logits = vec![f16::from_f32(-2.0).to_bits(); ROWS * VOCAB];
    let targets = [3_u32, 7];
    for (row, target) in targets.iter().enumerate() {
        logits[row * VOCAB + *target as usize] = f16::from_f32(2.0).to_bits();
    }
    let generated = [3_u32, 9];
    let logits = DeviceBuffer::from_host(&model.stream, &logits).map_err(cuda_error)?;
    let targets = DeviceBuffer::from_host(&model.stream, &targets).map_err(cuda_error)?;
    let generated = DeviceBuffer::from_host(&model.stream, &generated).map_err(cuda_error)?;
    let mut losses = DeviceBuffer::zeroed(&model.stream, ROWS).map_err(cuda_error)?;
    let mut output = DeviceBuffer::zeroed(&model.stream, 1).map_err(cuda_error)?;
    let launch = model
        .module
        .prepare_score(LaunchConfig1D::new(ROWS as u32, 256, 0))
        .map_err(cuda_error)?;
    model
        .module
        .score(
            &model.stream,
            &launch,
            &logits,
            &targets,
            &mut losses,
            ROWS as u32,
            0,
            0,
        )
        .map_err(cuda_error)?;
    let launch = model
        .module
        .prepare_objective(LaunchConfig1D::new(1, 256, 0))
        .map_err(cuda_error)?;
    model
        .module
        .objective(
            &model.stream,
            &launch,
            &losses,
            &generated,
            &targets,
            &mut output,
            ObjectiveShape {
                losses: ROWS as u32,
                tokens: ROWS as u32,
                patch: 1,
            },
        )
        .map_err(cuda_error)?;
    model.context.check_err().map_err(cuda_error)?;
    let loss = (2.0_f32.exp() + (VOCAB - 1) as f32 * (-2.0_f32).exp()).ln() - 2.0;
    let stats: Vec<ScoreStats> = read_prefix(&output, &model.stream, 1)?;
    let stats = stats[0];
    if (stats.mean_nll - loss).abs() > 0.002
        || (stats.maximum_nll - loss).abs() > 0.002
        || (stats.reward + loss).abs() > 0.002
        || stats.positional_accuracy != 0.5
        || stats.tokens != 2
        || stats.mismatches != 1
    {
        return Err(format!("CUDA generation scoring mismatch: {stats:?}"));
    }
    println!(
        "SCORING_PARITY ok=true rows={} reward={:.6} accuracy={:.3}",
        stats.tokens, stats.reward, stats.positional_accuracy
    );
    Ok(())
}

pub fn run(args: &[String]) -> CudaResult<()> {
    if args.len() < 3 || args.len() > 7 {
        return Err("usage: ./blckbaux cuda model CHECKPOINT TOKENS_JSON OUTPUT_DIR [CONTEXT=4096] [VISITS=2] [TEMPERATURE=0.8] [SEED=17]".into());
    }
    let tokens: Vec<u32> =
        blckbaux_artifact::json::from_slice(&fs::read(&args[1]).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let context = arg(args, 3, 4096_usize)?;
    let visits = arg(args, 4, 2_usize)?;
    let temperature = arg(args, 5, 0.8_f32)?;
    let seed = arg(args, 6, 17_u64)?;
    let output = Path::new(&args[2]);
    fs::create_dir(output).map_err(|e| format!("create fresh output directory: {e}"))?;
    let mut model = FbtModel::open(Path::new(&args[0]), tokens.len(), context)?;
    let result = model.run(&tokens, visits, temperature, seed)?;
    fs::write(
        output.join("tokens.json"),
        blckbaux_artifact::json::to_string(&result.tokens).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let hidden = result
        .hidden
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    fs::write(output.join("hidden.f16"), hidden).map_err(|e| e.to_string())?;
    let record = blckbaux_artifact::json::json!({ "stage": "cuda-full-model-forward", "checkpoint": args[0], "rows": tokens.len(), "context": context, "physical-layers": 5, "layer-visits": result.visits, "temperature": temperature, "seed": seed, "device-ms": result.device_ms, "wall-ms": result.wall_ms });
    fs::write(
        output.join("result.json"),
        blckbaux_artifact::json::pretty_string(&record).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    report(&result);
    Ok(())
}
pub fn generate(args: &[String]) -> CudaResult<()> {
    if args.len() < 3 || args.len() > 9 {
        return Err("usage: ./blckbaux cuda generate CHECKPOINT PROMPT_JSON OUTPUT_DIR [CONTEXT=4096] [VISITS=2] [TEMPERATURE=0.8] [SEED=17] [UNROLL=2] [CHUNK=0]".into());
    }
    let prompt: Vec<u32> =
        blckbaux_artifact::json::from_slice(&fs::read(&args[1]).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let context = arg(args, 3, 4096_usize)?;
    let visits = arg(args, 4, 2_usize)?;
    let temperature = arg(args, 5, 0.8_f32)?;
    let seed = arg(args, 6, 17_u64)?;
    let unroll = arg(args, 7, 2_usize)?;
    let chunk = arg(args, 8, 0_usize)?;
    let output = Path::new(&args[2]);
    let mut model = if chunk == 0 {
        FbtModel::open(Path::new(&args[0]), context, context)?
    } else {
        FbtModel::open_streamed(Path::new(&args[0]), context, chunk)?
    };
    let (result, evaluated) = if chunk == 0 {
        let result = model.generate(&prompt, visits, temperature, seed, unroll)?;
        let evaluated = context * result.passes;
        (result, evaluated)
    } else {
        model.generate_streamed(&prompt, visits, temperature, seed, unroll, chunk)?
    };
    fs::create_dir(output).map_err(|e| format!("create fresh output directory: {e}"))?;
    fs::write(
        output.join("tokens.json"),
        blckbaux_artifact::json::to_string(&result.tokens).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let record = blckbaux_artifact::json::json!({ "stage": "cuda-accepted-prefix-generation", "checkpoint": args[0], "context": context, "prompt-tokens": result.prompt_tokens, "drafted-tokens": 0, "evaluated-positions": evaluated, "repaired-tokens": context - result.prompt_tokens, "generated-tokens": context - result.prompt_tokens, "committed-tokens": context - result.prompt_tokens, "physical-layers": 5, "layer-visits": result.visits, "temperature": temperature, "seed": seed, "unroll": unroll, "chunk-rows": chunk, "passes": result.passes, "device-ms": result.device_ms, "wall-ms": result.wall_ms, "full-bo-loop": false });
    fs::write(
        output.join("result.json"),
        blckbaux_artifact::json::pretty_string(&record).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!(
        "GENERATE prompt={} generated={} passes={} executions={} device_ms={:.3} wall_ms={:.3}",
        result.prompt_tokens,
        context - result.prompt_tokens,
        result.passes,
        result.visits,
        result.device_ms,
        result.wall_ms
    );
    Ok(())
}

pub fn hnet_bench(args: &[String]) -> CudaResult<()> {
    if args.len() < 4 || args.len() > 7 {
        return Err("usage: hnet-bench MODEL_SEED PROMPT_JSON OUTPUT_DIR GENERATED_TOKENS [VISITS=2] [TEMPERATURE=1.0] [SEED=17]".into());
    }
    let model_seed = args[0]
        .parse::<u64>()
        .map_err(|error| format!("invalid H-Net model seed: {error}"))?;
    let prompt: Vec<u32> =
        blckbaux_artifact::json::from_slice(&fs::read(&args[1]).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let generated = arg(args, 3, 1_048_576_usize)?;
    let visits = arg(args, 4, 2_usize)?;
    let temperature = arg(args, 5, 1.0_f32)?;
    let seed = arg(args, 6, 17_u64)?;
    if prompt.is_empty()
        || prompt.len() % 64 != 0
        || generated == 0
        || generated % 64 != 0
        || prompt.iter().any(|&token| token >= VOCAB as u32)
    {
        return Err(
            "H-Net prompt and output must contain vocabulary-valid, nonempty 64-token patches"
                .into(),
        );
    }
    let core_positions = prompt
        .len()
        .checked_add(generated)
        .ok_or("H-Net context overflow")?
        .div_ceil(64);
    let core_rows = core_positions.next_power_of_two().max(64);
    let core_context = core_rows.max(4096);
    if core_context > 65_536 {
        return Err("H-Net coarse context exceeds the resident CUDA limit".into());
    }
    let output = Path::new(&args[2]);
    fs::create_dir(output).map_err(|e| format!("create fresh output directory: {e}"))?;
    let wall = std::time::Instant::now();
    let mut model = FbtModel::initialize_hnet(model_seed, core_rows, core_context)?;
    model.configure_hnet(&prompt, generated, None)?;
    let result = model.evaluate_hnet(
        visits,
        temperature,
        seed,
        std::env::var_os("BLCKBAUX_CUDA_PROFILE").is_some(),
    )?;
    let tokens = model.hnet_tokens()?;
    let throughput = generated as f64 * 1000.0 / f64::from(result.device_ms);
    let architecture_weights = model.weights.format == "hnet-mhc4";
    let record = blckbaux_artifact::json::json!({
        "schema": "blckbaux.cuda.hnet-generation.v1",
        "target": target_name(&model.context)?,
        "hardware": model.context.device_name().map_err(cuda_error)?,
        "model_seed": model_seed,
        "checkpoint_format": model.weights.format,
        "execution_architecture": "hnet-mhc4",
        "architecture_weights_match": architecture_weights,
        "patch_size": 64,
        "context_window_tokens": prompt.len() + generated,
        "prompt_tokens": prompt.len(),
        "drafted_tokens": 0,
        "core_context_positions": core_context,
        "evaluated_core_positions": core_rows,
        "core_active_positions": core_positions,
        "evaluated_tokens": generated,
        "repaired_tokens": 0,
        "generated_tokens": generated,
        "committed_tokens": generated,
        "forward_passes": 1,
        "layer_visits_per_wave": result.visits,
        "temperature": temperature,
        "seed": seed,
        "core_device_ms": result.core_ms,
        "generation_device_ms": result.generation_ms,
        "objective_device_ms": result.objective_ms,
        "device_ms": result.device_ms,
        "generated_tokens_per_device_second": throughput,
        "wall_ms": wall.elapsed().as_secs_f64() * 1000.0,
        "exact_within_patch_causal": true,
        "generation_method": "learned-hnet-64-wave",
        "full_generation_loop": true,
        "full_bo_loop": false
    });
    fs::write(
        output.join("tokens.json"),
        blckbaux_artifact::json::to_string(&tokens).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let json = blckbaux_artifact::json::pretty_string(&record).map_err(|e| e.to_string())?;
    fs::write(output.join("result.json"), &json).map_err(|e| e.to_string())?;
    println!("{json}");
    Ok(())
}

/// Run the resident Gaussian ENN experiment with the shared host controller.
/// Optional reliability and surrogate JSON carry the resolved TOML settings.
pub fn hnet_bo_bench(args: &[String]) -> CudaResult<()> {
    let mut args = args;
    let mut reliability = blckbaux_semantics::optimizer::ReliabilityPolicy::default();
    let mut window_tokens = None;
    let mut reference_seed = None;
    let mut trust_shape = None;
    let mut surrogate = SurrogateSpec {
        fit: true,
        ..Default::default()
    };
    while args.len() >= 2 {
        let flag = args[args.len() - 2].as_str();
        let value = &args[args.len() - 1];
        match flag {
            "--trust-shape" => {
                trust_shape = Some(blckbaux_semantics::hnet::ScaleShape::parse(value)?);
            }
            "--window-tokens" => {
                window_tokens = Some(
                    value
                        .parse::<usize>()
                        .map_err(|error| format!("invalid causal window: {error}"))?,
                )
            }
            "--reference-seed" => {
                reference_seed = Some(
                    value
                        .parse::<u64>()
                        .map_err(|error| format!("invalid corpus seed: {error}"))?,
                )
            }
            "--reliability" => {
                reliability = blckbaux_artifact::json::from_str::<
                    blckbaux_semantics::optimizer::ReliabilityPolicy,
                >(value)
                .map_err(|error| format!("invalid reliability policy: {error}"))?
                .validate()?
            }
            "--surrogate" => {
                surrogate = blckbaux_artifact::json::from_str::<SurrogateSpec>(value)
                    .map_err(|error| format!("invalid surrogate settings: {error}"))?
                    .validate()?
            }
            _ => break,
        }
        args = &args[..args.len() - 2];
    }
    if args.len() < 5 || !matches!(args.len(), 5..=13 | 16 | 19 | 20 | 22 | 23) {
        return Err("usage: hnet-bo-bench MODEL_SEED TRAIN_INPUT TARGET_INPUT OUTPUT_DIR GENERATED_TOKENS [VISITS=2] [TEMPERATURE=1.0] [PROPOSAL_SEED=17] [RADIUS=0.001] [ROUNDS=128] [ACQUISITION=ucb] [ACQUISITION_SEED=PROPOSAL_SEED] [EVALUATION_SEED=PROPOSAL_SEED] [ARMS=1 CANDIDATES=4 UCB_BETA=1 RADIUS_MIN=RADIUS RADIUS_MAX=2 VALIDATION_SEED=EVALUATION_SEED] [STAGE_SAMPLES=0] [VALIDATION_INPUT VALIDATION_TARGET VALIDATION_INTERVAL] [--reliability JSON] [--surrogate JSON] [--window-tokens N --reference-seed N] [--trust-shape scalar|tensor-family-static] (for an BLCKBAUXPTN input, TARGET_INPUT is the prompt cap; JSON inputs may be one flat task or ordered task rows)".into());
    }
    let model_seed = args[0]
        .parse::<u64>()
        .map_err(|error| format!("invalid H-Net model seed: {error}"))?;
    let generated = arg(args, 4, 1_048_576_usize)?;
    let visits = arg(args, 5, 2_usize)?;
    let temperature = arg(args, 6, 1.0_f32)?;
    let proposal_seed = arg(args, 7, 17_u64)?;
    let radius = arg(args, 8, 0.001_f32)?;
    let bo_rounds = arg(args, 9, 128_usize)?;
    let acquisition_seed = arg(args, 11, proposal_seed)?;
    let evaluation_seed = arg(args, 12, proposal_seed)?;
    let semantic_args = matches!(args.len(), 19 | 20 | 22 | 23);
    let arms = if semantic_args {
        arg(args, 13, 1_usize)?
    } else {
        1
    };
    let candidates = if semantic_args {
        arg(args, 14, 4_usize)?
    } else {
        4
    };
    let beta = if semantic_args {
        arg(args, 15, 1.0_f32)?
    } else {
        1.0
    };
    let radius_min = if semantic_args {
        arg(args, 16, radius)?
    } else {
        radius
    };
    let radius_max = if semantic_args {
        arg(args, 17, 2.0_f32)?
    } else {
        2.0
    };
    let validation_seed = if semantic_args {
        arg(args, 18, evaluation_seed)?
    } else {
        evaluation_seed
    };
    let stage_samples = if matches!(args.len(), 20 | 23) {
        arg(args, 19, 0_usize)?
    } else {
        0
    };
    let reference_seed = reference_seed.unwrap_or(evaluation_seed);
    let trust_shape = trust_shape.unwrap_or(if window_tokens.is_some() {
        blckbaux_semantics::hnet::ScaleShape::TensorFamilyStatic
    } else {
        blckbaux_semantics::hnet::ScaleShape::Scalar
    });
    let training = if let Some(window) = window_tokens {
        HnetSource::causal(&args[1], window, reference_seed, false)?
    } else {
        HnetSource::load(&args[1], &args[2], generated, evaluation_seed)?
    };
    let pool_shape = PoolShape::new(arms, candidates)?;
    if !matches!(pool_shape.count(), 1 | 4) {
        return Err("CUDA H-Net resident search requires one or four candidates".into());
    }
    let acquisition = args.get(10).map(String::as_str).unwrap_or("ucb");
    if !matches!(acquisition, "ucb" | "thompson") {
        return Err("CUDA H-Net operator-BO requires UCB or Thompson acquisition".into());
    }
    let validation_start = match args.len() {
        16 => Some(13),
        22 => Some(19),
        23 => Some(20),
        _ => None,
    };
    let validation = if let Some(start) = validation_start {
        if let Some(window) = window_tokens {
            HnetSource::causal(&args[start], window, reference_seed, true)?.validation()?
        } else {
            HnetSource::load(&args[start], &args[start + 1], generated, validation_seed)?
                .validation()?
        }
    } else {
        Vec::new()
    };
    let validation_interval = if let Some(start) = validation_start {
        Some(
            args[start + 2]
                .parse::<usize>()
                .map_err(|error| format!("invalid H-Net validation interval: {error}"))?,
        )
    } else {
        None
    };
    if generated == 0
        || (window_tokens.is_none() && generated % 64 != 0)
        || window_tokens.is_some_and(|window| {
            window < 64 || window > 1_048_576 || generated != window || temperature != 0.0
        })
        || !radius.is_finite()
        || radius <= 0.0
        || bo_rounds < 128
        || validation_interval == Some(0)
    {
        return Err(
            "CUDA H-Net BO requires valid 64-token prompt/target patches and a positive radius"
                .into(),
        );
    }
    let primary = training.task(0)?;
    for task in std::iter::once(&primary).chain(&validation) {
        validate_hnet_task(task, generated)?;
    }
    if let HnetSource::Rows(tasks) = &training
        && tasks.len() != 1
        && tasks.len() < bo_rounds
    {
        return Err(
            "CUDA H-Net rotating training inputs require one pre-materialized task per BO round"
                .into(),
        );
    }
    if validation
        .iter()
        .any(|heldout| primary.input() == heldout.input() && primary.targets() == heldout.targets())
    {
        return Err(
            "CUDA H-Net held-out validation tasks must be disjoint from training tasks".into(),
        );
    }
    let core_positions = std::iter::once(training.capacity()?)
        .chain(
            validation
                .iter()
                .map(HnetTask::capacity)
                .collect::<CudaResult<Vec<_>>>()?,
        )
        .max()
        .ok_or("CUDA H-Net BO requires a training task")?
        .div_ceil(64);
    let core_rows = core_positions
        .next_power_of_two()
        .max(if window_tokens.is_some() { 128 } else { 64 });
    let core_context = core_rows.max(4096);
    if core_context > 65_536 {
        return Err("CUDA H-Net BO coarse context exceeds the resident CUDA limit".into());
    }
    let output = Path::new(&args[3]);
    fs::create_dir(output).map_err(|e| format!("create fresh output directory: {e}"))?;
    let progress_file = fs::File::create(output.join("rounds.jsonl"))
        .map_err(|e| format!("create round log: {e}"))?;
    let mut progress = BufWriter::new(progress_file);
    let mut validation_progress = if validation.is_empty() {
        None
    } else {
        Some(BufWriter::new(
            fs::File::create(output.join("validation.jsonl"))
                .map_err(|e| format!("create validation log: {e}"))?,
        ))
    };
    let mut model = FbtModel::initialize_hnet(model_seed, core_rows, core_context)?;
    model.weights.set_hnet_shape(trust_shape, &model.stream)?;
    let mut backup = model.weights.backup(&model.stream)?;
    let physical_parameters = model.weights.physical_elements();
    let logical_parameters = physical_parameters - model.weights.embedding.len() as u64;
    primary.configure(&mut model)?;

    // The incumbent measurement initializes BO history and is excluded from
    // the timed round, matching the Metal experiment's timing contract.
    let base = model.evaluate_hnet(
        visits,
        temperature,
        primary.sample_seed(evaluation_seed),
        false,
    )?;
    let base_stats = base.stats.ok_or("CUDA H-Net baseline was not scored")?;
    let base_patch_nlls = base
        .patch_nlls
        .as_ref()
        .ok_or("CUDA H-Net baseline patch losses were not returned")?
        .clone();
    let mut current_incumbent_stats = base_stats;
    let mut current_incumbent_patch_nlls = base_patch_nlls;
    model
        .weights
        .reset_neighbor_history(&backup, &model.module, &model.stream)?;

    let mut validation_records = Vec::new();
    if let Some(validation_progress) = validation_progress.as_mut() {
        validation_records.push(validate_hnet_tasks(
            &mut model,
            &validation,
            visits,
            temperature,
            evaluation_seed,
            0,
            "initial",
            validation_progress,
        )?);
    }

    let mut round_wall_ms = 0.0_f64;
    let mut proposal_ms = 0.0;
    let mut observations = Vec::with_capacity(bo_rounds);
    let mut stage_profiles = Vec::with_capacity(stage_samples.min(bo_rounds));
    let mut best_reward = base_stats.reward;
    let mut accepted_rounds = 0_usize;
    let mut candidate = None;
    let mut stats = None;
    let tokens;
    let mut evaluation_device_ms = 0.0_f32;
    let mut incumbent_device_ms = 0.0_f32;
    let mut restore_ms = 0.0_f32;
    let mut candidate_evaluated_tokens = 0_usize;
    let mut incumbent_evaluated_tokens = 0_usize;
    let mut committed_tokens = 0_usize;
    let mut total_observations = 1_usize;
    let mut controller = RadiusController::configured(radius, radius_min, radius_max, reliability)?;
    // Paired Metal search is relative: the measured initial incumbent is the
    // zero origin, and accepted response means/variances accumulate from it.
    let mut history = NeighborValueEstimator::with_spec(0.0, 0.0, acquisition_seed, surrogate)?;
    let mut reuse = super::bo::SeedReuse::default();
    // A single fixed training condition keeps every H-Net buffer address and
    // every scalar launch argument stable. Retain that state so CUDA graphs
    // captured by the initial evaluation can be replayed by both members of
    // every Metal-equivalent incumbent/candidate pair. Rotating or validation
    // conditions still reconfigure explicitly.
    let retain_fixed_hnet_state = training.fixed() && validation.is_empty();
    for round in 0..bo_rounds {
        let iteration_wall = std::time::Instant::now();
        let task_index = if training.fixed() { 0 } else { round };
        let task = training.task(round)?;
        validate_hnet_task(&task, generated)?;
        let condition_seed = hnet_condition_seed(evaluation_seed, round)?;
        let round_evaluation_seed = task.sample_seed(if training.fixed() {
            evaluation_seed
        } else {
            super::bo::splitmix64(condition_seed ^ 0x7361_6d70_6c65_2d78)
        });
        if !retain_fixed_hnet_state {
            task.configure(&mut model)?;
        }
        // Metal measures the incumbent under the same round condition before
        // evaluating the candidate. When conditions are stationary (fixed training
        // tasks), re-evaluating the unchanged incumbent burns ~950 ms on device
        // and doubles query depth to d=2. We cache the exact incumbent stats and
        // patch losses, eliminating redundant incumbent execution (d=1).
        let (incumbent_stats, incumbent_patch_nlls, round_incumbent_device_ms) = if training.fixed()
        {
            (
                current_incumbent_stats,
                current_incumbent_patch_nlls.clone(),
                0.0_f32,
            )
        } else {
            let incumbent =
                model.evaluate_hnet(visits, temperature, round_evaluation_seed, false)?;
            let dev_ms = incumbent.device_ms;
            let stats = incumbent
                .stats
                .ok_or("CUDA H-Net round incumbent was not scored")?;
            let patch_nlls = incumbent
                .patch_nlls
                .ok_or("CUDA H-Net round incumbent patch losses were not returned")?;
            incumbent_evaluated_tokens = incumbent_evaluated_tokens
                .checked_add(task.targets().len())
                .ok_or("CUDA H-Net incumbent token count overflow")?;
            (stats, patch_nlls, dev_ms)
        };
        incumbent_device_ms += round_incumbent_device_ms;
        if window_tokens.is_some() {
            model.capture_hnet_pair()?;
        }
        let rewarming = total_observations > history.len();
        let round_radius = controller.radius();
        let (root, reused) = if window_tokens.is_some() {
            reuse.select(proposal_seed, round as u64)?
        } else {
            (
                proposal_seed
                    .checked_add(round as u64)
                    .ok_or("CUDA H-Net proposal seed overflow")?,
                false,
            )
        };
        let seeds = (0..pool_shape.count())
            .map(|candidate| pool_shape.seed(root, candidate))
            .collect::<Vec<_>>();
        let geometry = model.weights.neighbor_pool_distances(
            &backup,
            &model.module,
            &model.stream,
            round_radius,
            &seeds,
            history.len(),
        )?;
        let incumbent_prediction = history.incumbent_prediction()?;
        let round_acquisition_seed = acquisition_seed
            .checked_add(round as u64)
            .ok_or("CUDA H-Net acquisition seed overflow")?;
        let draws = history
            .identities()
            .iter()
            .map(|&identity| {
                blckbaux_semantics::random::normal_metric(round_acquisition_seed, identity, 0)
                    as f32
            })
            .collect::<Vec<_>>();
        let pool = geometry
            .iter()
            .map(|row| {
                history.acquire(
                    row,
                    beta,
                    (acquisition == "thompson").then_some(draws.as_slice()),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let modeled_selection = pool
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.score
                    .total_cmp(&right.score)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
            .ok_or("CUDA H-Net candidate pool is empty")?;
        let initializing = history.len() < surrogate.neighbors;
        let selected = if initializing {
            round % pool.len()
        } else {
            modeled_selection
        };
        let candidate_prediction = pool[selected];
        let acquisition_score = if initializing {
            0.0
        } else {
            candidate_prediction.score
        };
        let predicted_improvement =
            f64::from(candidate_prediction.mean) - f64::from(incumbent_prediction.mean);
        let proposal_round_ms = model.weights.materialize_gaussian(
            &backup,
            &model.module,
            &model.stream,
            round_radius,
            seeds[selected],
        )?;
        proposal_ms += proposal_round_ms;
        let mut evaluated =
            model.evaluate_hnet(visits, temperature, round_evaluation_seed, false)?;
        evaluation_device_ms += evaluated.device_ms;
        candidate_evaluated_tokens = candidate_evaluated_tokens
            .checked_add(task.targets().len())
            .ok_or("CUDA H-Net candidate token count overflow")?;
        let evaluated_stats = evaluated
            .stats
            .ok_or("CUDA H-Net candidate was not scored")?;
        let candidate_patch_nlls = evaluated
            .patch_nlls
            .as_ref()
            .ok_or("CUDA H-Net candidate patch losses were not returned")?;
        if candidate_patch_nlls.len() != incumbent_patch_nlls.len()
            || candidate_patch_nlls.is_empty()
        {
            return Err("CUDA H-Net paired patch response shape changed".into());
        }
        let (patch_responses, paired_mean, paired_variance) = if window_tokens.is_some() {
            let estimate = evaluated
                .estimate
                .ok_or("CUDA causal objective omitted its paired estimate")?;
            (
                evaluated
                    .patch_gains
                    .take()
                    .ok_or("CUDA causal objective omitted patch responses")?,
                estimate
                    .improvement
                    .ok_or("CUDA causal objective omitted paired gain")?,
                estimate
                    .paired_variance
                    .ok_or("CUDA causal objective omitted paired variance")?,
            )
        } else {
            let patch_responses = incumbent_patch_nlls
                .iter()
                .zip(candidate_patch_nlls)
                .map(|(incumbent, candidate)| incumbent - candidate)
                .collect::<Vec<_>>();
            let paired_mean = patch_responses.iter().sum::<f32>() / patch_responses.len() as f32;
            let paired_variance = if patch_responses.len() > 1 {
                patch_responses
                    .iter()
                    .map(|value| (value - paired_mean).powi(2))
                    .sum::<f32>()
                    / ((patch_responses.len() - 1) * patch_responses.len()) as f32
            } else {
                0.0
            };
            (patch_responses, paired_mean, paired_variance)
        };
        let response_radius =
            super::bo::response_rms(Some(&patch_responses), f64::from(round_radius));
        let optimizer_candidate = history.base_value() + paired_mean;
        let optimizer_variance = history.base_variance() + paired_variance;
        let prediction_variance = f64::from(candidate_prediction.standard_error).powi(2)
            + f64::from(incumbent_prediction.standard_error).powi(2);
        let decision = if initializing {
            let threshold = if rewarming {
                2.0 * (f64::from(optimizer_variance) + f64::from(history.base_variance())).sqrt()
            } else {
                0.0
            };
            // Metal warm-up compares the stored FP32 outcome with the stored
            // incumbent, after relative-reward addition has rounded to FP32.
            controller.initial_evidence(
                f64::from(optimizer_candidate) - f64::from(history.base_value()),
                predicted_improvement,
                threshold,
            )?
        } else {
            controller.observe_evidence(
                super::bo::ReliabilityEvidence {
                    observed_improvement: f64::from(paired_mean),
                    observation_variance: f64::from(paired_variance),
                    predicted_improvement,
                    prediction_variance,
                    response_radius,
                    nominal_radius: f64::from(round_radius),
                    accepted: false,
                },
                true,
            )?
        };
        let accepted = decision.accepted;
        if window_tokens.is_some() {
            reuse.observe(root, accepted, initializing);
        }
        model
            .weights
            .archive_neighbor(selected, history.next_slot()?, &model.stream)?;
        history.append(
            &geometry[selected],
            optimizer_candidate,
            optimizer_variance,
            accepted,
        )?;
        let restore_round_ms = if accepted {
            accepted_rounds += 1;
            best_reward = evaluated_stats.reward;
            committed_tokens = committed_tokens
                .checked_add(task.targets().len())
                .ok_or("CUDA H-Net committed token count overflow")?;
            let elapsed = model.weights.commit(&mut backup, &model.stream)?;
            model.commit_hnet_index(true);
            if training.fixed() {
                current_incumbent_stats = evaluated_stats;
                current_incumbent_patch_nlls = candidate_patch_nlls.to_vec();
            }
            elapsed
        } else {
            let elapsed = model.weights.restore(&backup, &model.stream)?;
            model.commit_hnet_index(false);
            elapsed
        };
        restore_ms += restore_round_ms;
        total_observations += 1;
        let round_device_ms =
            proposal_round_ms + round_incumbent_device_ms + evaluated.device_ms + restore_round_ms;
        let timed_wall_ms = iteration_wall.elapsed().as_secs_f64() * 1000.0;
        if let Some(sample) = stage_sample(round, bo_rounds, stage_samples) {
            let profiled = model.evaluate_hnet(visits, temperature, round_evaluation_seed, true)?;
            let core = profiled
                .core_profile
                .as_ref()
                .ok_or("CUDA H-Net stage sample omitted its core profile")?;
            let generation = profiled
                .generation_profile
                .as_ref()
                .ok_or("CUDA H-Net stage sample omitted its generation profile")?;
            stage_profiles.push(blckbaux_artifact::json::json!({
                "sample": sample,
                "round": round,
                "state": if accepted { "accepted_candidate" } else { "restored_incumbent" },
                "core": {
                    "encode_ms": core.embed_ms,
                    "mhc_ms": core.mhc_ms,
                    "qkv_ms": core.qkv_ms,
                    "index_ms": core.index_ms,
                    "pisa_ms": core.pisa_ms,
                    "output_ms": core.output_ms,
                    "attention_ms": core.attention_ms,
                    "moe_router_ms": core.moe_router_ms,
                    "moe_route_ms": core.moe_route_ms,
                    "moe_experts_ms": core.moe_experts_ms,
                    "moe_combine_ms": core.moe_combine_ms,
                    "moe_ms": core.moe_ms,
                    "total_ms": core.total_ms
                },
                "generation": {
                    "init_ms": generation.init_ms,
                    "gemm_ms": generation.gemm_ms,
                    "project_ms": generation.project_ms,
                    "reduce_ms": generation.reduce_ms,
                    "total_ms": generation.total_ms
                },
                "device_ms": profiled.device_ms,
                "wall_ms": profiled.wall_ms
            }));
        }
        let observation = blckbaux_artifact::json::json!({
            "round": round,
            "condition": {
                "task": task_index,
                "seed": condition_seed,
                "evaluation_seed": round_evaluation_seed,
                "input_tokens": task.input().len(),
                "target_tokens": task.targets().len(),
                "corpus_offset": task.corpus.as_ref().map(|c| c.batch.offset),
                "source": task.corpus.as_ref().map(|c| c.source.as_str()),
                "causal": window_tokens.is_some()
            },
            "phase": if initializing { "initialization" } else { "operator_bo" },
            "radius": round_radius,
            "next_radius": decision.next_radius,
            "proposal_root": root,
            "reused_root": reused,
            "proposal_seed": seeds[selected],
            "selected": selected,
            "acquisition_score": acquisition_score,
            "predicted_mean": candidate_prediction.mean,
            "predicted_standard_error": candidate_prediction.standard_error,
            "incumbent_mean": incumbent_prediction.mean,
            "incumbent_standard_error": incumbent_prediction.standard_error,
            "reward": evaluated_stats.reward,
            "incumbent_condition_reward": incumbent_stats.reward,
            "incumbent_condition_nll": incumbent_stats.mean_nll,
            "optimizer_reward": optimizer_candidate,
            "optimizer_variance": optimizer_variance,
            "paired_patch_improvement": paired_mean,
            "paired_patch_variance": paired_variance,
            "paired_patch_channels": patch_responses.len(),
            "mean_nll": evaluated_stats.mean_nll,
            "last_accepted_condition_reward": best_reward,
            "optimizer_incumbent": history.base_value(),
            "accepted": accepted,
            "observed_improvement": decision.improvement,
            "acceptance_threshold": decision.threshold,
            "predicted_improvement": decision.predicted_improvement,
            "reliability_action": decision.action.name(),
            "standardized_error": decision.standardized_error,
            "sharpness_ratio": decision.sharpness_ratio,
            "agreement_ratio": decision.agreement_ratio,
            "response_radius": response_radius,
            "accepted_rounds": accepted_rounds,
            "history": history.len(),
            "acceptance_rate": accepted_rounds as f64 / (round + 1) as f64,
            "proposal_device_ms": proposal_round_ms,
            "incumbent_evaluation_device_ms": round_incumbent_device_ms,
            "candidate_evaluation_device_ms": evaluated.device_ms,
            "evaluation_device_ms": round_incumbent_device_ms + evaluated.device_ms,
            "restore_device_ms": restore_round_ms,
            "device_ms": round_device_ms,
            "evaluation_wall_ms": evaluated.wall_ms,
            "wall_ms": timed_wall_ms,
            "candidate_evaluated_tokens": candidate_evaluated_tokens,
            "incumbent_evaluated_tokens": incumbent_evaluated_tokens,
            "evaluated_tokens": candidate_evaluated_tokens + incumbent_evaluated_tokens,
            "committed_tokens": committed_tokens
        });
        writeln!(
            progress,
            "{}",
            blckbaux_artifact::json::to_string(&observation).map_err(|e| e.to_string())?
        )
        .map_err(|e| format!("write round log: {e}"))?;
        progress
            .flush()
            .map_err(|e| format!("flush round log: {e}"))?;
        observations.push(observation);
        candidate = Some(evaluated);
        stats = Some(evaluated_stats);
        round_wall_ms += timed_wall_ms;
        let completed = round + 1;
        let scheduled_validation = validation_interval
            .is_some_and(|interval| completed % interval == 0)
            || (window_tokens.is_none() && hnet_learning_checkpoint(completed));
        if completed < bo_rounds
            && scheduled_validation
            && let Some(validation_progress) = validation_progress.as_mut()
        {
            validation_records.push(validate_hnet_tasks(
                &mut model,
                &validation,
                visits,
                temperature,
                evaluation_seed,
                completed,
                "checkpoint",
                validation_progress,
            )?);
        }
    }
    let candidate = candidate.ok_or("CUDA H-Net BO produced no candidate")?;
    let stats = stats.ok_or("CUDA H-Net BO produced no candidate score")?;
    if let Some(validation_progress) = validation_progress.as_mut() {
        validation_records.push(validate_hnet_tasks(
            &mut model,
            &validation,
            visits,
            temperature,
            evaluation_seed,
            bo_rounds,
            "final",
            validation_progress,
        )?);
    }
    // The round evaluator leaves token buffers from the last proposal even
    // when that proposal was rejected. Re-evaluate the restored incumbent so
    // tokens.json has explicit best-model provenance, outside round timing.
    let validation_wall = std::time::Instant::now();
    primary.configure(&mut model)?;
    let final_incumbent = model.evaluate_hnet(
        visits,
        temperature,
        primary.sample_seed(evaluation_seed),
        false,
    )?;
    let final_stats = final_incumbent
        .stats
        .ok_or("CUDA H-Net final incumbent was not scored")?;
    if training.fixed() && (final_stats.reward - best_reward).abs() > 1.0e-5 {
        return Err(format!(
            "CUDA H-Net final incumbent reward {} differs from optimizer best {best_reward}",
            final_stats.reward
        )
        .into());
    }
    tokens = model.hnet_tokens()?;
    let validation_wall_ms = validation_wall.elapsed().as_secs_f64() * 1000.0;
    let index_audit = if std::env::var_os("BLCKBAUX_CUDA_PISA_COMPARE").is_some() {
        model.force_full_hnet_index()?;
        let exact = model.evaluate_hnet(visits, temperature, evaluation_seed, false)?;
        let exact_stats = exact
            .stats
            .ok_or("CUDA H-Net exact index audit was not scored")?;
        let (support_recall, exact_rows, compared_rows) = model.hnet_index_overlap(exact.visits)?;
        Some((
            exact,
            exact_stats,
            support_recall,
            exact_rows,
            compared_rows,
        ))
    } else {
        None
    };
    let tell_wall = std::time::Instant::now();
    let improved = accepted_rounds > 0;
    let tell_ms = tell_wall.elapsed().as_secs_f64() * 1000.0;
    let device_ms = proposal_ms + incumbent_device_ms + evaluation_device_ms + restore_ms;
    let evaluated_total = candidate_evaluated_tokens
        .checked_add(incumbent_evaluated_tokens)
        .ok_or("CUDA H-Net evaluated-token count overflow")?;
    let record = blckbaux_artifact::json::json!({
        "schema": "blckbaux.cuda.hnet-bo-round.v7",
        "target": target_name(&model.context)?,
        "hardware": model.context.device_name().map_err(cuda_error)?,
        "model_seed": model_seed,
        "checkpoint_format": model.weights.format,
        "execution_architecture": "hnet-mhc4",
        "architecture_weights_match": model.weights.format == "hnet-mhc4",
        "phase": "optimization",
        "bo": {
            "rounds": bo_rounds,
            "observations": observations,
            "accepted_rounds": accepted_rounds,
            "optimizer_incumbent": history.base_value(),
            "optimizer_variance": history.base_variance(),
            "acquisition": acquisition,
            "ucb_beta": beta,
            "surrogate": "neighbor-value-v1",
            "initial_observations": surrogate.neighbors,
            "history": history.len(),
            "total_observations": total_observations,
            "history_capacity": MAX_HISTORY
        },
        "training_conditions": {
            "tasks": training.count(bo_rounds),
            "schedule": if training.fixed() { "fixed" } else { "metal-corpus-window" },
            "paired_candidate_incumbent": true,
            "incumbent_remeasured_each_round": !training.fixed(),
            "fixed_hnet_state_reused": retain_fixed_hnet_state
        },
        "validation": {
            "configured": !validation.is_empty(),
            "tasks": validation.len(),
            "interval": validation_interval,
            "learning_checkpoints": window_tokens.is_none(),
            "records": validation_records,
            "used_for_acceptance": false,
            "excluded_from_round_timing": true
        },
        "proposal": {
            "trust_region_shape": match trust_shape { blckbaux_semantics::hnet::ScaleShape::Scalar => "scalar", blckbaux_semantics::hnet::ScaleShape::TensorFamilyStatic => "tensor-family-static" },
            "method": "independent",
            "distribution": "gaussian",
            "full_coordinate_innovation": true,
            "arms": arms,
            "candidates": candidates,
            "candidates_per_arm": candidates / arms,
            "radius_initial": radius,
            "radius_final": controller.radius(),
            "radius_min": radius_min,
            "radius_max": radius_max,
            "device_ms": proposal_ms,
            "logical_parameter_coordinates": logical_parameters,
            "materialized_physical_elements": physical_parameters
        },
        "objective": if window_tokens.is_some() { "causal-corpus-cross-entropy" } else { "conditional-target-cross-entropy" },
        "context_window_tokens": primary.capacity()?,
        "prompt_tokens": if window_tokens.is_some() { 0 } else { primary.input().len() },
        "drafted_tokens": 0,
        "core_context_positions": core_context,
        "evaluated_core_positions": core_rows,
        "core_active_positions": core_positions,
        "evaluated_tokens": evaluated_total,
        "candidate_evaluated_tokens": candidate_evaluated_tokens,
        "incumbent_evaluated_tokens": incumbent_evaluated_tokens,
        "repaired_tokens": 0,
        "generated_tokens": if window_tokens.is_some() { 0 } else { evaluated_total },
        "prediction_tokens": if window_tokens.is_some() { evaluated_total } else { 0 },
        "committed_tokens": committed_tokens,
        "scored_tokens": evaluated_total,
        "forward_passes": bo_rounds * if training.fixed() { 1 } else { 2 },
        "temperature": temperature,
        "seeds": {
            "proposal": proposal_seed,
            "acquisition": acquisition_seed,
            "evaluation": evaluation_seed,
            "corpus": reference_seed
        },
        "readout_schedule": "hierarchical-8x6",
        "incumbent": {
            "reward": base_stats.reward,
            "mean_nll": base_stats.mean_nll,
            "core_device_ms": base.core_ms,
            "generation_device_ms": base.generation_ms,
            "objective_device_ms": base.objective_ms,
            "device_ms": base.device_ms,
            "evaluation_wall_ms": base.wall_ms,
            "timed_in_round": false
        },
        "candidate": {
            "reward": stats.reward,
            "mean_nll": stats.mean_nll,
            "maximum_nll": stats.maximum_nll,
            "positional_accuracy": stats.positional_accuracy,
            "mismatches": stats.mismatches,
            "patch_nll_octiles": stats.patch_nll_octiles,
            "core_device_ms": candidate.core_ms,
            "generation_device_ms": candidate.generation_ms,
            "objective_reduce_device_ms": candidate.objective_ms,
            "evaluation_wall_ms": candidate.wall_ms,
            "total_device_ms": candidate.device_ms
        },
        "final_incumbent_validation": {
            "reward": final_stats.reward,
            "mean_nll": final_stats.mean_nll,
            "device_ms": final_incumbent.device_ms,
            "wall_ms": validation_wall_ms,
            "tokens_provenance": "recomputed-from-final-accepted-incumbent",
            "excluded_from_round_timing": true
        },
        "profile": (!stage_profiles.is_empty()).then(|| blckbaux_artifact::json::json!({
            "requested_samples": stage_samples,
            "recorded_samples": stage_profiles.len(),
            "samples": stage_profiles,
            "excluded_from_round_timing": true
        })),
        "index_audit": index_audit.as_ref().map(|(exact, exact_stats, support_recall, exact_rows, compared_rows)| blckbaux_artifact::json::json!({
            "shadow_support_recall": support_recall,
            "exact_support_rows": exact_rows,
            "compared_rows": compared_rows,
            "shadow_reward": stats.reward,
            "full_reward": exact_stats.reward,
            "absolute_reward_error": (stats.reward - exact_stats.reward).abs(),
            "full_core_device_ms": exact.core_ms,
            "full_generation_device_ms": exact.generation_ms,
            "full_total_device_ms": exact.device_ms,
            "excluded_from_timed_transaction": true
        })),
        "tell": {
            "accepted": improved,
            "device_restore_ms": restore_ms,
            "wall_ms": tell_ms,
            "history_after": history.len(),
            "total_observations": total_observations,
            "trust_region_length": controller.radius()
        },
        "incumbent_evaluation_device_ms": incumbent_device_ms,
        "candidate_evaluation_device_ms": evaluation_device_ms,
        "device_ms": device_ms,
        "wall_ms": round_wall_ms,
        "rounds_per_device_second": bo_rounds as f64 * 1000.0 / f64::from(device_ms),
        "full_generation_loop": window_tokens.is_none(),
        "full_scoring_loop": true,
        "full_bo_loop": true,
        "complete_candidate_transaction": true
    });
    fs::write(
        output.join("tokens.json"),
        blckbaux_artifact::json::to_string(&tokens).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let json = blckbaux_artifact::json::pretty_string(&record).map_err(|e| e.to_string())?;
    fs::write(output.join("result.json"), &json).map_err(|e| e.to_string())?;
    println!("{json}");
    Ok(())
}

pub fn bench(args: &[String]) -> CudaResult<()> {
    if args.len() < 2 || args.len() > 10 {
        return Err("usage: generation-bench CHECKPOINT OUTPUT_DIR [CONTEXT=4096] [PROMPT=128] [VISITS=2] [TEMPERATURE=0.8] [SEED=17] [UNROLL=2] [CHUNK=0] [PROMPT_JSON]".into());
    }
    let context = arg(args, 2, 4096_usize)?;
    let prompt_tokens = arg(args, 3, 128_usize)?;
    let visits = arg(args, 4, 2_usize)?;
    let temperature = arg(args, 5, 0.8_f32)?;
    let seed = arg(args, 6, 17_u64)?;
    let unroll = arg(args, 7, 2_usize)?;
    let chunk = arg(args, 8, 0_usize)?;
    if prompt_tokens == 0 || prompt_tokens >= context {
        return Err("generation benchmark prompt must be nonempty and shorter than context".into());
    }
    let prompt: Vec<u32> = if let Some(path) = args.get(9) {
        blckbaux_artifact::json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?
    } else {
        vec![0; prompt_tokens]
    };
    if prompt.len() != prompt_tokens || prompt.iter().any(|&t| t >= VOCAB as u32) {
        return Err("benchmark prompt file must match the declared count and vocabulary".into());
    }
    let output = Path::new(&args[1]);
    fs::create_dir(output).map_err(|e| format!("create fresh output directory: {e}"))?;
    if std::env::var_os("BLCKBAUX_SYNTH_GEMM").is_some() {
        fs::write(output.join("gemm.json"), crate::gemm::measure()?).map_err(|e| e.to_string())?;
    }
    let precision_cases = crate::precision::check()?;
    let loaded_at = std::time::Instant::now();
    let mut model = if chunk == 0 {
        FbtModel::open(Path::new(&args[0]), context, context)?
    } else {
        FbtModel::open_streamed(Path::new(&args[0]), context, chunk)?
    };
    let profile = if chunk == 0 {
        Some(model.profile_wave(&vec![0; context], visits, temperature, seed)?)
    } else {
        None
    };
    let (result, evaluated) = if chunk == 0 {
        let result = model.generate(&prompt, visits, temperature, seed, unroll)?;
        let evaluated = context * result.passes;
        (result, evaluated)
    } else {
        model.generate_streamed(&prompt, visits, temperature, seed, unroll, chunk)?
    };
    let mut comparisons = Vec::new();
    let load_and_generation_ms = loaded_at.elapsed().as_secs_f64() * 1000.0;
    if context > 65536 && chunk != 0 {
        let other_chunk = if chunk == 512 { 1024 } else { 512 };
        let mut reference = FbtModel::open_streamed(Path::new(&args[0]), context, other_chunk)?;
        let (other, other_evaluated) =
            reference.generate_streamed(&prompt, visits, temperature, seed, unroll, other_chunk)?;
        if other.tokens != result.tokens || other.frontiers != result.frontiers {
            return Err("million-context generation changed with chunk size".into());
        }
        comparisons.push(blckbaux_artifact::json::json!({
            "case": "chunk-invariance", "reference_chunk_rows": other_chunk,
            "reference_device_ms": other.device_ms,
            "reference_evaluated_positions": other_evaluated,
            "token_parity": true, "frontier_parity": true,
            "packed_reference": false
        }));
    }
    if chunk != 0 && context <= 65536 {
        let mut reference = FbtModel::open(Path::new(&args[0]), context, context)?;
        let packed = reference.generate(&prompt, visits, temperature, seed, unroll)?;
        if packed.tokens != result.tokens || packed.frontiers != result.frontiers {
            return Err("streamed generation differs from packed reference".into());
        }
        comparisons.push(blckbaux_artifact::json::json!({
            "case": "selected-chunk", "chunk_rows": chunk,
            "packed_device_ms": packed.device_ms,
            "streamed_device_ms": result.device_ms,
            "packed_evaluated_positions": context * packed.passes,
            "streamed_evaluated_positions": evaluated,
            "token_parity": true, "frontier_parity": true
        }));
    }
    if context == 4096 && chunk == 0 {
        let prompt = vec![0; prompt_tokens];
        let varied = (0..context)
            .map(|i| ((i * 7919 + 17) % VOCAB) as u32)
            .collect::<Vec<_>>();
        let mut streamed = FbtModel::open(Path::new(&args[0]), context, context)?;
        for (case, input, chunk, case_seed, case_unroll) in [
            ("zero-512", prompt.as_slice(), 512, seed, unroll),
            ("zero-2048", prompt.as_slice(), 2048, seed, unroll),
            (
                "varied-512",
                &varied[..prompt_tokens],
                512,
                seed.wrapping_add(1),
                2,
            ),
        ] {
            let packed = model.generate(input, visits, temperature, case_seed, case_unroll)?;
            let (cached, evaluated) = streamed.generate_streamed(
                input,
                visits,
                temperature,
                case_seed,
                case_unroll,
                chunk,
            )?;
            if packed.tokens != cached.tokens || packed.frontiers != cached.frontiers {
                let mismatch = packed
                    .tokens
                    .iter()
                    .zip(&cached.tokens)
                    .position(|(a, b)| a != b);
                return Err(format!(
                    "streamed parity failed: case={case} first_token={mismatch:?} packed_frontiers={:?} cached_frontiers={:?}",
                    packed.frontiers, cached.frontiers
                ));
            }
            comparisons.push(blckbaux_artifact::json::json!({
                "case": case,
                "chunk_rows": chunk,
                "seed": case_seed,
                "unroll": case_unroll,
                "packed_device_ms": packed.device_ms,
                "streamed_device_ms": cached.device_ms,
                "streamed_wall_ms": cached.wall_ms,
                "packed_evaluated_positions": context * packed.passes,
                "streamed_evaluated_positions": evaluated,
                "committed_tokens": context - input.len(),
                "token_parity": true,
                "frontier_parity": true
            }));
            println!(
                "STREAMED_PARITY case={case} chunk={chunk} evaluated={evaluated} packed_ms={:.3} streamed_ms={:.3}",
                packed.device_ms, cached.device_ms
            );
        }
    }
    let generated = context - prompt_tokens;
    let record = blckbaux_artifact::json::json!({
        "schema": "blckbaux.cuda.generation.v1",
        "fp16_narrowing_cases": precision_cases,
        "fp16_widening_patterns": 65536,
        "synth_gemm_tile_k": std::env::var("BLCKBAUX_SYNTH_GEMM").ok(),
        "synth_scope": std::env::var_os("BLCKBAUX_SYNTH_GEMM").map(|_| "readout_only"),
        "streamed_comparison": comparisons,
        "chunk_rows": chunk,
        "repair_chunk_rows": if chunk == 0 { 0 } else { super::cached::repair_chunk(chunk)? },
        "logit_scratch_bytes": model.scratch.logits.len() * 2,
        "readout_chunk_rows": model.scratch.logits.len() / VOCAB,
        "target": target_name(&model.context)?,
        "hardware": model.context.device_name().map_err(crate::cuda_error)?,
        "prompt_source": if args.get(9).is_some() { "token_file" } else { "zero_token_fixture" },
        "checkpoint": args[0],
        "context_tokens": context,
        "prompt_tokens": prompt_tokens,
        "drafted_tokens": 0,
        "evaluated_positions": evaluated,
        "repaired_tokens": generated,
        "generated_tokens": generated,
        "committed_tokens": generated,
        "repair_waves": result.passes,
        "frontiers": result.frontiers,
        "layer_visits_per_wave": result.visits,
        "temperature": temperature,
        "seed": seed,
        "unroll": unroll,
        "device_ms": result.device_ms,
        "wall_ms": result.wall_ms,
        "first_wave_ms": result.first_wave_ms,
        "checkpoint_load_and_generation_ms": load_and_generation_ms,
        "first_wave_chunk_profiles": result.chunk_profiles.iter().map(|(first, p)| blckbaux_artifact::json::json!({
            "first_position": first, "rows": chunk,
            "embed_ms": p.embed_ms, "mhc_ms": p.mhc_ms, "qkv_ms": p.qkv_ms,
            "index_ms": p.index_ms, "pisa_ms": p.pisa_ms, "output_ms": p.output_ms,
            "moe_ms": p.moe_ms, "readout_ms": p.readout_ms, "total_ms": p.total_ms
        })).collect::<Vec<_>>(),
        "wave_profile": profile.map(|profile| blckbaux_artifact::json::json!({
            "embed_ms": profile.embed_ms,
            "mhc_ms": profile.mhc_ms,
            "qkv_ms": profile.qkv_ms,
            "index_ms": profile.index_ms,
            "pisa_ms": profile.pisa_ms,
            "output_ms": profile.output_ms,
            "attention_ms": profile.attention_ms,
            "moe_router_ms": profile.moe_router_ms,
            "moe_route_ms": profile.moe_route_ms,
            "moe_experts_ms": profile.moe_experts_ms,
            "moe_combine_ms": profile.moe_combine_ms,
            "moe_ms": profile.moe_ms,
            "readout_ms": profile.readout_ms,
            "total_ms": profile.total_ms
        })),
        "full_generation_loop": true,
        "full_bo_loop": false
    });
    let json = blckbaux_artifact::json::pretty_string(&record).map_err(|e| e.to_string())?;
    fs::write(output.join("result.json"), &json).map_err(|e| e.to_string())?;
    fs::write(
        output.join("tokens.json"),
        blckbaux_artifact::json::to_string(&result.tokens).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!("{json}");
    Ok(())
}

fn arg<T: std::str::FromStr>(args: &[String], index: usize, default: T) -> CudaResult<T> {
    args.get(index).map_or(Ok(default), |value| {
        value
            .parse()
            .map_err(|_| format!("invalid argument {index}: {value}"))
    })
}

fn stage_sample(round: usize, rounds: usize, samples: usize) -> Option<usize> {
    let samples = samples.min(rounds);
    match samples {
        0 => None,
        1 => (round == 0).then_some(0),
        _ => (0..samples).find(|&sample| sample * (rounds - 1) / (samples - 1) == round),
    }
}
fn report(result: &ModelOutput) {
    println!(
        "MODEL rows={} executions={} device_ms={:.3} wall_ms={:.3}",
        result.tokens.len(),
        result.visits,
        result.device_ms,
        result.wall_ms
    );
}
