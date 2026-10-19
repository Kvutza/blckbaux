//! Immutable packed token streams for long-context pretraining experiments.

use std::fs::File;
use std::path::Path;

mod corpus;
pub use corpus::{CorpusBatch, CorpusBundle, CorpusSet};

const MAGIC: &[u8; 8] = b"BLCKPTN1";
const LEGACY_MAGIC: &[u8; 8] = b"ENNXPTN1";
const HEADER_BYTES: usize = 24;
pub const VOCAB: u32 = 8192;
pub const CONTEXT: u32 = 4096;
const MAX_TOKENS: u32 = 1_048_576;

#[derive(Debug)]
pub struct PretrainDataset {
    sequences: u32,
    tokens: Vec<u32>,
    trace: Option<TraceIndex>,
    code: Option<CodeIndex>,
}

/// One fixed corpus condition for causal next-token likelihood.
///
/// The model receives `tokens`; `labels` are the immediately following corpus
/// tokens.  Candidate parameters never alter either sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CausalBatch {
    pub tokens: Vec<u32>,
    pub labels: Vec<u32>,
    pub loss_mask: Vec<u8>,
    pub offset: usize,
}

#[derive(Debug)]
struct TraceIndex {
    episodes: Vec<(usize, usize)>,
    repository_starts: Vec<usize>,
    actions: Vec<TraceAction>,
    pad: u32,
}

#[derive(Debug, Clone, Copy)]
struct TraceAction {
    episode: usize,
    start: usize,
    end: usize,
}

#[derive(Debug)]
struct CodeIndex {
    actions: Vec<CodeAction>,
    pad: u32,
}

#[derive(Debug, Clone, Copy)]
struct CodeAction {
    repository_start: usize,
    start: usize,
    end: usize,
}

impl PretrainDataset {
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|error| format!("pretraining dataset {}: {error}", path.display()))?;
        if bytes.len() < HEADER_BYTES || !has_magic(&bytes) {
            return Err("pretraining dataset has an invalid BLCKPTN1 header".into());
        }
        let field = |offset| {
            u32::from_le_bytes(
                bytes[offset..offset + 4]
                    .try_into()
                    .expect("four-byte field"),
            )
        };
        let vocab = field(8);
        let context = field(12);
        let sequences = field(16);
        let reserved = field(20);
        if vocab == 0 || context != CONTEXT || sequences < 2 || sequences % 2 != 0 {
            return Err(format!(
                "pretraining dataset requires nonzero vocab, context={CONTEXT}, and a positive even sequence count; found vocab={vocab}, context={context}, sequences={sequences}"
            ));
        }
        if reserved != 0 {
            return Err("pretraining dataset reserved header field must be zero".into());
        }
        let elements = usize::try_from(u64::from(sequences) * u64::from(context))
            .map_err(|_| "pretraining dataset size does not fit this machine")?;
        let stride = if vocab <= 65536 { 2 } else { 4 };
        let expected = HEADER_BYTES
            .checked_add(
                elements
                    .checked_mul(stride)
                    .ok_or("pretraining token byte overflow")?,
            )
            .ok_or("pretraining dataset byte size overflow")?;
        if bytes.len() != expected {
            return Err(format!(
                "pretraining dataset has {} bytes, expected {expected}",
                bytes.len()
            ));
        }
        let tokens = if vocab <= 65536 {
            bytes[HEADER_BYTES..]
                .chunks_exact(2)
                .map(|val| u32::from(u16::from_le_bytes([val[0], val[1]])))
                .collect::<Vec<_>>()
        } else {
            bytes[HEADER_BYTES..]
                .chunks_exact(4)
                .map(|val| u32::from_le_bytes([val[0], val[1], val[2], val[3]]))
                .collect::<Vec<_>>()
        };
        if let Some((index, token)) = tokens
            .iter()
            .copied()
            .enumerate()
            .find(|(_, token)| *token >= vocab)
        {
            return Err(format!(
                "pretraining token {token} at offset {index} exceeds the vocabulary"
            ));
        }
        let trace = TraceIndex::load(path, &tokens)?;
        let code = CodeIndex::load(path, &tokens)?;
        Ok(Self {
            sequences,
            tokens,
            trace,
            code,
        })
    }

    pub fn batches(&self) -> u32 {
        self.sequences / 2
    }

    pub fn sequences(&self) -> u32 {
        self.sequences
    }

    pub fn tokens(&self) -> usize {
        self.tokens.len()
    }

    pub fn prefix(&self, count: usize) -> Result<&[u32], String> {
        self.range(0, count)
    }

    pub fn range(&self, start: usize, count: usize) -> Result<&[u32], String> {
        let end = start
            .checked_add(count)
            .ok_or("pretraining token range overflow")?;
        self.tokens.get(start..end).ok_or_else(|| {
            format!(
                "context workload needs token positions {start}..{end}, dataset contains {}",
                self.tokens.len()
            )
        })
    }

    /// Select a contiguous, non-wrapping corpus sequence and construct its
    /// shifted next-token targets.  Origins remain aligned to the corpus'
    /// immutable 4096-token records, but no runtime packing or repetition is
    /// performed.
    pub fn causal_batch(
        &self,
        sequence_tokens: usize,
        seed: u64,
        round: u32,
    ) -> Result<CausalBatch, String> {
        if sequence_tokens == 0 || sequence_tokens > MAX_TOKENS as usize {
            return Err(format!(
                "causal sequence length must be in 1..={MAX_TOKENS}"
            ));
        }
        let required = sequence_tokens
            .checked_add(1)
            .ok_or("causal sequence length overflow")?;
        if required > self.tokens.len() {
            return Err(format!(
                "causal sequence needs {required} contiguous corpus tokens, dataset contains {}",
                self.tokens.len()
            ));
        }
        let last = self.tokens.len() - required;
        let origins = last / CONTEXT as usize + 1;
        let origin = splitmix64(seed ^ u64::from(round).wrapping_mul(0x9e37_79b9_7f4a_7c15))
            as usize
            % origins;
        let offset = origin * CONTEXT as usize;
        let source = &self.tokens[offset..offset + required];
        Ok(CausalBatch {
            tokens: source[..sequence_tokens].to_vec(),
            labels: source[1..].to_vec(),
            loss_mask: vec![1; sequence_tokens],
            offset,
        })
    }

    /// Return one contiguous corpus window on the circular token stream.
    /// The requested window may cross the physical end once, but it may not
    /// repeat the corpus. This lets long-context rounds cover every possible
    /// sequence-aligned origin without manufacturing repeated-token proxies.
    pub fn circular(&self, start: usize, count: usize) -> Result<Vec<u32>, String> {
        if count > self.tokens.len() {
            return Err(format!(
                "context workload needs {count} source tokens, dataset contains {}",
                self.tokens.len()
            ));
        }
        if self.tokens.is_empty() {
            return Err("pretraining dataset is empty".into());
        }
        let start = start % self.tokens.len();
        let first = count.min(self.tokens.len() - start);
        let mut window = Vec::with_capacity(count);
        window.extend_from_slice(&self.tokens[start..start + first]);
        window.extend_from_slice(&self.tokens[..count - first]);
        Ok(window)
    }

    pub fn sequence(&self, index: u32) -> Result<&[u32], String> {
        if index >= self.sequences {
            return Err(format!(
                "pretraining sequence {index} exceeds the {} available sequences",
                self.sequences
            ));
        }
        let start = index as usize * CONTEXT as usize;
        Ok(&self.tokens[start..start + CONTEXT as usize])
    }

    /// Sample a natural corpus continuation within the complete context budget.
    ///
    /// Trace corpora yield complete assistant actions. Source corpora begin at
    /// a file-content boundary and continue over the packed repository stream.
    pub fn natural_task(
        &self,
        prompt_cap: usize,
        output_cap: usize,
        seed: u64,
        round: u32,
    ) -> Result<Option<(Vec<u32>, Vec<u32>, usize)>, String> {
        if let Some(trace) = &self.trace {
            return trace.task(&self.tokens, prompt_cap, output_cap, seed, round);
        }
        if let Some(code) = &self.code {
            return code.task(&self.tokens, prompt_cap, output_cap, seed, round);
        }
        Ok(None)
    }

    pub fn window(
        &self,
        prompt_cap: usize,
        output_cap: usize,
        seed: u64,
        round: u32,
    ) -> Result<(Vec<u32>, Vec<u32>, usize), String> {
        if let Some(task) = self.natural_task(prompt_cap, output_cap, seed, round)? {
            return Ok(task);
        }
        let total = prompt_cap
            .checked_add(output_cap)
            .ok_or("pretraining corpus window overflow")?;
        if total > self.tokens() {
            return Err(format!(
                "one pretraining condition needs {total} source tokens, corpus contains {}",
                self.tokens()
            ));
        }
        let sequences = u64::from(self.sequences());
        let origin = splitmix64(seed) % sequences;
        let mut stride = splitmix64(seed ^ 0x7769_6e64_6f77_2d78) % sequences;
        stride = stride.max(1);
        while gcd(stride, sequences) != 1 {
            stride = (stride + 1) % sequences;
            stride = stride.max(1);
        }
        let sequence = (origin + u64::from(round) % sequences * stride) % sequences;
        let offset = usize::try_from(sequence)
            .map_err(|_| "pretraining sequence index exceeds usize")?
            .checked_mul(CONTEXT as usize)
            .ok_or("pretraining corpus offset overflow")?;
        let source = self.circular(offset, total)?;
        Ok((
            source[..prompt_cap].to_vec(),
            source[prompt_cap..].to_vec(),
            offset,
        ))
    }

    pub fn batch(&self, index: u32) -> Result<&[u32], String> {
        if index >= self.batches() {
            return Err(format!(
                "pretraining batch {index} exceeds the {} available batches",
                self.batches()
            ));
        }
        let elements = 2 * CONTEXT as usize;
        let start = index as usize * elements;
        Ok(&self.tokens[start..start + elements])
    }
}

fn has_magic(bytes: &[u8]) -> bool {
    bytes.starts_with(MAGIC) || bytes.starts_with(LEGACY_MAGIC)
}

impl TraceIndex {
    fn task(
        &self,
        tokens: &[u32],
        prompt_cap: usize,
        output_cap: usize,
        seed: u64,
        round: u32,
    ) -> Result<Option<(Vec<u32>, Vec<u32>, usize)>, String> {
        let context = MAX_TOKENS as usize;
        let eligible = self
            .actions
            .iter()
            .filter(|action| {
                let episode_start = self.episodes[action.episode].0;
                let output = action.end - action.start;
                let prompt = (action.start - episode_start).next_multiple_of(64);
                output >= 32
                    && output <= output_cap
                    && prompt <= prompt_cap
                    && prompt + output <= context
            })
            .collect::<Vec<_>>();
        if eligible.is_empty() {
            return Err(format!(
                "trace corpus has no complete assistant action within prompt cap {prompt_cap}, output cap {output_cap}, and context {context}"
            ));
        }
        let draw = splitmix64(seed ^ u64::from(round).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let action = eligible[draw as usize % eligible.len()];
        let mut prompt_start = self.episodes[action.episode].0;
        for &(start, end) in self.episodes[self.repository_starts[action.episode]..=action.episode]
            .iter()
            .rev()
        {
            let prompt = (action.start - start).next_multiple_of(64);
            if prompt <= prompt_cap
                && prompt + action.end - action.start <= context
                && (end <= action.start || start == self.episodes[action.episode].0)
            {
                prompt_start = start;
            }
        }
        let source = &tokens[prompt_start..action.start];
        let padding = source.len().next_multiple_of(64) - source.len();
        let mut prompt = Vec::with_capacity(source.len() + padding);
        prompt.resize(padding, self.pad);
        prompt.extend_from_slice(source);
        Ok(Some((
            prompt,
            tokens[action.start..action.end].to_vec(),
            action.start,
        )))
    }
    fn load(path: &Path, tokens: &[u32]) -> Result<Option<Self>, String> {
        let manifest = path
            .parent()
            .ok_or("pretraining dataset has no parent directory")?
            .join("manifest.json");
        if !manifest.is_file() {
            return Ok(None);
        }
        let document: crate::json::Value =
            crate::json::from_reader(File::open(&manifest).map_err(|error| error.to_string())?)
                .map_err(|error| {
                    format!("invalid corpus manifest {}: {error}", manifest.display())
                })?;
        if document["recipe"]["serialization"].as_str() != Some("role_tool_channel_v4") {
            return Ok(None);
        }
        let marker = |name: &str| -> Result<u32, String> {
            document["tokenizer"]["special_tokens"][name]
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| format!("trace corpus manifest has no {name} token"))
        };
        let episode = marker("<|episode|>")?;
        let repository = marker("<|repository|>")?;
        let bos = marker("<bos>")?;
        let turn = marker("<|turn|>")?;
        let assistant = marker("<|assistant|>")?;
        let endturn = marker("<|endturn|>")?;
        let pad = marker("<|pad|>")?;
        let markers = tokens
            .iter()
            .enumerate()
            .filter_map(|(index, token)| (*token == episode).then_some(index))
            .collect::<Vec<_>>();
        // The flat corpus quota may truncate its final episode. Consecutive
        // episode markers prove that every span retained here is complete.
        let episodes = markers
            .windows(2)
            .map(|pair| {
                let start =
                    pair[0].saturating_sub(usize::from(pair[0] > 0 && tokens[pair[0] - 1] == bos));
                let end =
                    pair[1].saturating_sub(usize::from(pair[1] > 0 && tokens[pair[1] - 1] == bos));
                (start, end)
            })
            .collect::<Vec<_>>();
        let mut repository_starts = Vec::with_capacity(episodes.len());
        let mut repository_start = 0;
        for (index, &(start, _)) in episodes.iter().enumerate() {
            let episode = start + usize::from(tokens[start] == bos);
            if tokens.get(episode + 1) == Some(&repository) {
                repository_start = index;
            }
            repository_starts.push(repository_start);
        }
        let mut actions = Vec::new();
        for (episode_index, &(start, end)) in episodes.iter().enumerate() {
            let mut cursor = start;
            while cursor + 1 < end {
                if tokens[cursor] == turn && tokens[cursor + 1] == assistant {
                    if let Some(offset) = tokens[cursor + 2..end]
                        .iter()
                        .position(|token| *token == endturn)
                    {
                        actions.push(TraceAction {
                            episode: episode_index,
                            start: cursor,
                            end: cursor + 2 + offset + 1,
                        });
                    }
                }
                cursor += 1;
            }
        }
        if episodes.is_empty() || actions.is_empty() {
            return Err("trace corpus contains no complete assistant actions".into());
        }
        Ok(Some(Self {
            episodes,
            repository_starts,
            actions,
            pad,
        }))
    }
}

impl CodeIndex {
    fn load(path: &Path, tokens: &[u32]) -> Result<Option<Self>, String> {
        let manifest = path
            .parent()
            .ok_or("pretraining dataset has no parent directory")?
            .join("manifest.json");
        if !manifest.is_file() {
            return Ok(None);
        }
        let document: crate::json::Value =
            crate::json::from_reader(File::open(&manifest).map_err(|error| error.to_string())?)
                .map_err(|error| {
                    format!("invalid corpus manifest {}: {error}", manifest.display())
                })?;
        if document["recipe"]["serialization"].as_str() != Some("repository_file_v1") {
            return Ok(None);
        }
        let marker = |name: &str| -> Result<u32, String> {
            document["tokenizer"]["special_tokens"][name]
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| format!("code corpus manifest has no {name} token"))
        };
        let repository = marker("<|repository|>")?;
        let bos = marker("<bos>")?;
        let file = marker("<|file|>")?;
        let content = marker("<|content|>")?;
        let endfile = marker("<|endfile|>")?;
        let pad = marker("<|pad|>")?;
        let mut repository_start = None;
        let mut actions = Vec::new();
        let mut cursor = 0;
        while cursor < tokens.len() {
            if tokens[cursor] == repository {
                repository_start = Some(
                    cursor.saturating_sub(usize::from(cursor > 0 && tokens[cursor - 1] == bos)),
                );
            } else if tokens[cursor] == file {
                let Some(start_offset) = tokens[cursor + 1..]
                    .iter()
                    .position(|token| *token == content)
                else {
                    break;
                };
                let start = cursor + 1 + start_offset + 1;
                let Some(end_offset) = tokens[start..].iter().position(|token| *token == endfile)
                else {
                    break;
                };
                let end = start + end_offset + 1;
                let repository_start = repository_start
                    .ok_or("code corpus file appeared before its repository marker")?;
                actions.push(CodeAction {
                    repository_start,
                    start,
                    end,
                });
                cursor = end;
                continue;
            }
            cursor += 1;
        }
        if actions.is_empty() {
            return Err("code corpus contains no complete files".into());
        }
        Ok(Some(Self { actions, pad }))
    }

    fn task(
        &self,
        tokens: &[u32],
        prompt_cap: usize,
        output_cap: usize,
        seed: u64,
        round: u32,
    ) -> Result<Option<(Vec<u32>, Vec<u32>, usize)>, String> {
        let context = MAX_TOKENS as usize;
        let budget = context.min(prompt_cap.saturating_add(output_cap));
        let eligible = self
            .actions
            .iter()
            .filter(|action| {
                let prompt = (action.start - action.repository_start).next_multiple_of(64);
                prompt <= prompt_cap
                    && prompt < budget
                    && output_cap.min(budget - prompt) >= 32
                    && action.end > action.start
            })
            .collect::<Vec<_>>();
        if eligible.is_empty() {
            return Err(format!(
                "code corpus has no file boundary within the {budget}-token context budget"
            ));
        }
        let draw = splitmix64(seed ^ u64::from(round).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let action = eligible[draw as usize % eligible.len()];
        let source = &tokens[action.repository_start..action.start];
        let padding = source.len().next_multiple_of(64) - source.len();
        let mut prompt = Vec::with_capacity(source.len() + padding);
        prompt.resize(padding, self.pad);
        prompt.extend_from_slice(source);
        let output = output_cap.min(budget - prompt.len());
        Ok(Some((
            prompt,
            circular_tokens(tokens, action.start, output)?,
            action.start,
        )))
    }
}

fn circular_tokens(tokens: &[u32], start: usize, count: usize) -> Result<Vec<u32>, String> {
    if tokens.is_empty() || count > tokens.len() {
        return Err(format!(
            "packed continuation needs {count} tokens, corpus contains {}",
            tokens.len()
        ));
    }
    let start = start % tokens.len();
    let first = count.min(tokens.len() - start);
    let mut output = Vec::with_capacity(count);
    output.extend_from_slice(&tokens[start..start + first]);
    output.extend_from_slice(&tokens[..count - first]);
    Ok(output)
}

/// Load one contiguous prefix from an BLCKPTN1 corpus for an accelerator run.
pub fn load_tokens(path: &Path, count: usize) -> Result<Vec<u32>, String> {
    Ok(PretrainDataset::load(path)?.prefix(count)?.to_vec())
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

fn splitmix64(value: u64) -> u64 {
    let value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut mixed = value;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^ (mixed >> 31)
}

#[cfg(test)]
mod tests {
    use super::{CONTEXT, CodeAction, CodeIndex, HEADER_BYTES, MAGIC, PretrainDataset, VOCAB};

    fn document(sequences: u32) -> Vec<u8> {
        let elements = sequences as usize * CONTEXT as usize;
        let mut bytes = Vec::with_capacity(HEADER_BYTES + elements * 2);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&VOCAB.to_le_bytes());
        bytes.extend_from_slice(&CONTEXT.to_le_bytes());
        bytes.extend_from_slice(&sequences.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        for index in 0..elements {
            bytes.extend_from_slice(&((index as u16) % VOCAB as u16).to_le_bytes());
        }
        bytes
    }

    #[test]
    fn fixed_batches() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("pretrain.bin");
        std::fs::write(&path, document(4)).unwrap();
        let data = PretrainDataset::load(&path).unwrap();
        assert_eq!(data.batches(), 2);
        assert_eq!(data.sequences(), 4);
        assert_eq!(data.sequence(3).unwrap().len(), CONTEXT as usize);
        assert!(data.sequence(4).is_err());
        assert_eq!(data.batch(1).unwrap().len(), 2 * CONTEXT as usize);
        assert!(data.batch(2).is_err());
    }

    #[test]
    fn causal_batches() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("pretrain.bin");
        std::fs::write(&path, document(4)).unwrap();
        let data = PretrainDataset::load(&path).unwrap();
        let batch = data.causal_batch(4096, 17, 3).unwrap();
        assert_eq!(batch.tokens.len(), 4096);
        assert_eq!(batch.labels.len(), 4096);
        assert_eq!(batch.loss_mask, vec![1; 4096]);
        assert_eq!(batch.offset % CONTEXT as usize, 0);
        assert_eq!(&batch.tokens[1..], &batch.labels[..4095]);
        assert_eq!(batch.labels[4095], data.tokens[batch.offset + 4096]);
        assert_eq!(batch, data.causal_batch(4096, 17, 3).unwrap());
    }

    #[test]
    fn nonwrapping_batches() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("pretrain.bin");
        std::fs::write(&path, document(2)).unwrap();
        let data = PretrainDataset::load(&path).unwrap();
        assert!(data.causal_batch(data.tokens() + 1, 0, 0).is_err());
    }

    #[test]
    fn invalid_tokens() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("pretrain.bin");
        let mut bytes = document(2);
        bytes[HEADER_BYTES..HEADER_BYTES + 2].copy_from_slice(&(VOCAB as u16).to_le_bytes());
        std::fs::write(&path, bytes).unwrap();
        assert!(
            PretrainDataset::load(&path)
                .unwrap_err()
                .contains("exceeds the vocabulary")
        );
    }

    #[test]
    fn wide_batches() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("wide.bin");
        let vocab: u32 = 256_000;
        let sequences: u32 = 4;
        let elements = sequences as usize * CONTEXT as usize;
        let mut bytes = Vec::with_capacity(HEADER_BYTES + elements * 4);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&vocab.to_le_bytes());
        bytes.extend_from_slice(&CONTEXT.to_le_bytes());
        bytes.extend_from_slice(&sequences.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        for index in 0..elements {
            bytes.extend_from_slice(&((index as u32) % vocab).to_le_bytes());
        }
        std::fs::write(&path, bytes).unwrap();
        let data = PretrainDataset::load(&path).unwrap();
        assert_eq!(data.batches(), 2);
        assert_eq!(data.sequences(), 4);
        assert_eq!(data.sequence(3).unwrap().len(), CONTEXT as usize);
        assert_eq!(data.batch(1).unwrap().len(), 2 * CONTEXT as usize);
    }

    #[test]
    fn code_caps() {
        let tokens = (0..1024).collect::<Vec<_>>();
        let code = CodeIndex {
            actions: vec![CodeAction {
                repository_start: 0,
                start: 128,
                end: 256,
            }],
            pad: 0,
        };
        let (prompt, output, _) = code.task(&tokens, 512, 256, 17, 0).unwrap().unwrap();
        assert_eq!(prompt.len(), 128);
        assert_eq!(output.len(), 256);
    }
}
