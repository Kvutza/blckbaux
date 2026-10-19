use super::VOCAB;
use crate::{CudaResult, copy_prefix, cuda_error, read_prefix, timing_event};
use blckbaux_artifact::json::Value;
use blckbaux_cuda_device::{NeighborPoint, fbt_model};
use blckbaux_semantics::tensor::key as tensor_key;
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};
use deser::Deserialize;
use half::f16;
use memmap2::MmapOptions;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

#[derive(Deserialize)]
struct Header {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

pub(super) struct Site {
    pub predictor: DeviceBuffer<u16>,
    pub bias: DeviceBuffer<u16>,
    pub control: DeviceBuffer<u16>,
    pub norm: DeviceBuffer<u16>,
}
pub(super) struct Layer {
    pub qkv: DeviceBuffer<u16>,
    pub output: DeviceBuffer<u16>,
    pub router: DeviceBuffer<u16>,
    pub gate: DeviceBuffer<u16>,
    pub down: DeviceBuffer<u16>,
    pub attention: Site,
    pub moe: Site,
}
pub(crate) struct DiffusionWeights {
    pub mask: DeviceBuffer<u16>,
    pub index: DeviceBuffer<u16>,
}

pub(crate) struct HnetWeights {
    pub mask: DeviceBuffer<u16>,
    pub positions: DeviceBuffer<u16>,
}

pub(crate) struct Weights {
    pub format: &'static str,
    pub diffusion: Option<DiffusionWeights>,
    pub hnet: Option<HnetWeights>,
    pub embedding: DeviceBuffer<u16>,
    pub readout: DeviceBuffer<u16>,
    pub final_norm: DeviceBuffer<u16>,
    pub layers: Vec<Layer>,
    proposal: Vec<ProposalSpec>,
    gaussian_random: Option<GaussianRandom>,
    neighbor_sampler: Option<NeighborSampler>,
}

struct ProposalSpec {
    elements: usize,
    block_len: u32,
    scales: DeviceBuffer<f32>,
    host_scales: Vec<f32>,
    tensor_id: u64,
    logical_start: u64,
    transpose_columns: u32,
}

struct NeighborRange {
    buffer: usize,
    start: u32,
    count: u32,
}

struct NeighborSampler {
    points: DeviceBuffer<NeighborPoint>,
    seeds: DeviceBuffer<u64>,
    pool: DeviceBuffer<u16>,
    history: DeviceBuffer<u16>,
    distances: DeviceBuffer<f32>,
    ranges: Vec<NeighborRange>,
    total: u32,
}

struct GaussianRandom {
    thresholds: DeviceBuffer<u32>,
    widths: DeviceBuffer<f32>,
    densities: DeviceBuffer<f32>,
    status: Option<ProposalBuffers>,
}

struct ProposalBuffers {
    partials: DeviceBuffer<u32>,
    summary: DeviceBuffer<u32>,
}

pub(super) struct WeightBackup {
    buffers: Vec<DeviceBuffer<u16>>,
}

struct Checkpoint {
    map: memmap2::Mmap,
    start: usize,
    headers: BTreeMap<String, Header>,
    format: &'static str,
}

/// Storage-only fixture through the real full-row materialization adapter.
/// Three differently sized tensors exercise partial CTAs, status-buffer reuse,
/// invalid tail coordinates, unchanged rows, rollback and subsequent recovery.
/// This is not a model forward pass or an optimizer learning experiment.
pub fn proposal_check() -> CudaResult<()> {
    let context = cuda_core::CudaContext::new(0).map_err(cuda_error)?;
    let stream = context.default_stream();
    // SAFETY: generated bindings load their matching device module.
    let module = unsafe { fbt_model::load(&context) }.map_err(cuda_error)?;
    let lengths = [1025, 33, 257];
    let upload = |count| DeviceBuffer::from_host(&stream, &vec![0x3c00; count]).map_err(cuda_error);
    let mut specs = Vec::new();
    for (index, &elements) in lengths.iter().enumerate() {
        let host_scales = vec![0.25; elements];
        specs.push(ProposalSpec {
            elements,
            block_len: 1,
            scales: DeviceBuffer::from_host(&stream, &host_scales).map_err(cuda_error)?,
            host_scales,
            tensor_id: index as u64 + 1,
            logical_start: 0,
            transpose_columns: 0,
        });
    }
    let mut weights = Weights {
        format: "proposal-fixture",
        diffusion: None,
        hnet: None,
        embedding: upload(lengths[0])?,
        readout: upload(lengths[1])?,
        final_norm: upload(lengths[2])?,
        layers: Vec::new(),
        proposal: specs,
        gaussian_random: Some(GaussianRandom::new(&stream)?),
        neighbor_sampler: None,
    };
    let backup = weights.backup(&stream)?;
    // copy_prefix's safe slice interface also applies to pinned memory, whose
    // DMA is genuinely asynchronous unless the helper waits for completion.
    let mut staging = cuda_core::PinnedHostBuffer::from_slice(&context, &vec![0.5_f32; lengths[0]])
        .map_err(cuda_error)?;
    copy_prefix(&weights.proposal[0].scales, staging.as_slice(), &stream)?;
    staging.as_mut_slice().fill(0.75);
    if read_prefix(&weights.proposal[0].scales, &stream, lengths[0])? != vec![0.5; lengths[0]] {
        return Err("borrowed pinned upload outlived its source borrow".into());
    }
    copy_prefix(
        &weights.proposal[0].scales,
        &vec![0.25; lengths[0]],
        &stream,
    )?;
    let clean = |weights: &Weights| -> CudaResult<()> {
        let mut result = Ok(());
        weights.visit(|buffer| {
            if result.is_ok() {
                result = read_prefix(buffer, &stream, buffer.len()).and_then(|values| {
                    if values.iter().any(|&v| v != 0x3c00) {
                        Err("rejected proposal did not restore the complete incumbent".into())
                    } else {
                        Ok(())
                    }
                });
            }
        });
        result
    };
    if weights
        .materialize_gaussian(&backup, &module, &stream, 0.0, 17)
        .is_ok()
    {
        return Err("proposal fixture accepted zero radius".into());
    }
    weights.materialize_gaussian(&backup, &module, &stream, 0.01, 17)?;
    let status = weights
        .gaussian_random
        .as_ref()
        .and_then(|g| g.status.as_ref())
        .ok_or("proposal fixture omitted status")?;
    let actual = read_prefix(&status.summary, &stream, lengths.len())?;
    if actual != vec![blckbaux_semantics::proposal::CHANGED; lengths.len()] {
        return Err(format!(
            "valid proposal reported incorrect flags: {actual:?}"
        ));
    }
    // Overflow only the final coordinate of the first tensor. The other
    // tensors have smaller status ranges; their reuse must not erase this flag.
    let mut scales = vec![0.25; lengths[0]];
    *scales.last_mut().ok_or("empty fixture tensor")? = f32::MAX;
    copy_prefix(&weights.proposal[0].scales, &scales, &stream)?;
    let result = weights.materialize_gaussian(&backup, &module, &stream, 0.01, 17);
    if !matches!(result, Err(ref error) if error == "selected proposal is nonfinite") {
        return Err(format!(
            "invalid tail proposal was not rejected: {result:?}"
        ));
    }
    clean(&weights)?;
    copy_prefix(
        &weights.proposal[0].scales,
        &vec![0.25; lengths[0]],
        &stream,
    )?;
    weights.materialize_gaussian(&backup, &module, &stream, 0.01, 17)?;
    // Ordinary nonzero Gaussian draws at this radius round away at the unit
    // incumbent. Materialization must not inject a one-ULP replacement update.
    let result = weights.materialize_gaussian(&backup, &module, &stream, 1.0e-20, 17);
    if !matches!(result, Err(ref error) if error == "selected proposal did not change") {
        return Err(format!(
            "rounded-away Gaussian proposal was not rejected: {result:?}"
        ));
    }
    clean(&weights)?;
    // Force the fixture's normal draws to zero without altering production
    // tables or kernels; all candidates then equal the finite incumbent.
    let random = weights
        .gaussian_random
        .as_ref()
        .ok_or("proposal fixture lost RNG state")?;
    copy_prefix(&random.thresholds, &[u32::MAX; 256], &stream)?;
    copy_prefix(&random.widths, &[0.0; 256], &stream)?;
    let result = weights.materialize_gaussian(&backup, &module, &stream, 0.01, 17);
    if !matches!(result, Err(ref error) if error == "selected proposal did not change") {
        return Err(format!("unchanged proposal was not rejected: {result:?}"));
    }
    clean(&weights)?;
    let random = weights
        .gaussian_random
        .as_ref()
        .ok_or("proposal fixture lost RNG state")?;
    copy_prefix(
        &random.thresholds,
        &blckbaux_semantics::gaussian::ZIGGURAT.thresholds,
        &stream,
    )?;
    copy_prefix(
        &random.widths,
        &blckbaux_semantics::gaussian::ZIGGURAT.widths,
        &stream,
    )?;
    weights.materialize_gaussian(&backup, &module, &stream, 0.01, 17)?;
    context.check_err().map_err(cuda_error)?;
    println!(
        "PROPOSAL_VALIDITY ok=true tensors=1025,33,257 valid=true invalid_tail=rejected unchanged=rejected rollback=exact recovery=true"
    );
    Ok(())
}

impl Checkpoint {
    fn open(path: &Path) -> CudaResult<Self> {
        let file = File::open(path).map_err(cuda_error)?;
        // SAFETY: read-only mapping retained for every tensor upload.
        let map = unsafe { MmapOptions::new().map(&file) }.map_err(cuda_error)?;
        if map.len() < 8 {
            return Err("truncated Safetensors prefix".into());
        }
        let len = usize::try_from(u64::from_le_bytes(map[..8].try_into().unwrap()))
            .map_err(cuda_error)?;
        let start = len.checked_add(8).ok_or("checkpoint header overflow")?;
        if start > map.len() {
            return Err("truncated Safetensors header".into());
        }
        let mut root: BTreeMap<String, Value> =
            blckbaux_artifact::json::from_slice(&map[8..start]).map_err(cuda_error)?;
        let metadata: BTreeMap<String, String> = blckbaux_artifact::json::from_value(
            root.remove("__metadata__")
                .ok_or("checkpoint metadata missing")?,
        )
        .map_err(cuda_error)?;
        let format = match metadata.get("format").map(String::as_str) {
            Some("blckbaux.fbt-pisa1-looped-mhc4-rope.v1") => "looped-mhc4",
            Some("blckbaux.fbt-pisa1-diffusion-mhc4-rope.v1") => "diffusion-mhc4",
            Some("blckbaux.fbt-pisa1-hnet-mhc4-rope.v1") => "hnet-mhc4",
            _ => {
                return Err(
                    "checkpoint must be a looped, diffusion, or H-Net mHC RoPE model".into(),
                );
            }
        };
        for (name, expected) in [
            ("position_encoding", "rope"),
            ("rope_base", "10000"),
            ("rope_dimensions", "64"),
        ] {
            if metadata.get(name).map(String::as_str) != Some(expected) {
                return Err(format!("checkpoint {name} must be {expected}"));
            }
        }
        let mut headers = BTreeMap::new();
        let mut ranges = Vec::new();
        for (name, value) in root {
            let header: Header = blckbaux_artifact::json::from_value(value).map_err(cuda_error)?;
            let count = header
                .shape
                .iter()
                .try_fold(1_usize, |n, d| n.checked_mul(*d))
                .ok_or("tensor shape overflow")?;
            let [a, b] = header.data_offsets;
            if header.dtype != "F16"
                || a > b
                || count.checked_mul(2) != Some(b - a)
                || start.checked_add(b).is_none_or(|end| end > map.len())
            {
                return Err(format!("invalid F16 tensor {name}"));
            }
            ranges.push((a, b));
            headers.insert(name, header);
        }
        ranges.sort_unstable();
        if ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err("overlapping checkpoint tensors".into());
        }
        Ok(Self {
            map,
            start,
            headers,
            format,
        })
    }
    fn tensor(&self, name: &str, shape: &[usize], layer: Option<usize>) -> CudaResult<Vec<u16>> {
        let header = self
            .headers
            .get(name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        if header.shape != shape {
            return Err(format!(
                "tensor {name} shape {:?}, expected {shape:?}",
                header.shape
            ));
        }
        let [mut a, mut b] = header.data_offsets;
        if let Some(layer) = layer {
            let stride = (b - a) / 5;
            a += layer * stride;
            b = a + stride;
        }
        Ok(self.map[self.start + a..self.start + b]
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect())
    }
}

use blckbaux_semantics::random::splitmix64 as splitmix;

fn response_block_key(tensor_id: u64, start: u64, length: u64) -> u64 {
    splitmix(tensor_id ^ splitmix(start) ^ splitmix(length ^ 0xa076_1d64_78bd_642f))
}

impl NeighborSampler {
    fn new(specs: &[ProposalSpec], stream: &CudaStream) -> CudaResult<Self> {
        // The packed CUDA embedding is a second physical view of Metal's one
        // logical embedding/readout tensor.  It is materialized, but only the
        // row-major readout participates in neighbour geometry.
        let dimensions = specs
            .iter()
            .filter(|spec| spec.transpose_columns == 0)
            .try_fold(0_usize, |sum, spec| sum.checked_add(spec.elements))
            .ok_or("CUDA neighbour dimensions overflow")?;
        let mut points = Vec::new();
        let mut ranges = Vec::new();
        for (buffer, spec) in specs.iter().enumerate() {
            if spec.transpose_columns != 0 {
                continue;
            }
            let range_start = u32::try_from(points.len()).map_err(cuda_error)?;
            let block_len = spec.block_len as usize;
            for block in 0..spec.host_scales.len() {
                let count = (8192usize.saturating_mul(block_len) / dimensions)
                    .max(8)
                    .min(block_len);
                let key = response_block_key(
                    spec.tensor_id,
                    spec.logical_start + (block * block_len) as u64,
                    block_len as u64,
                );
                let scale = spec.host_scales[block];
                let weight = 1.0 / (block_len as f32 * scale * scale);
                for stratum in 0..count {
                    let start = stratum * block_len / count;
                    let end = (stratum + 1) * block_len / count;
                    let element = start
                        + splitmix(key ^ (stratum as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
                            as usize
                            % (end - start);
                    points.push(NeighborPoint {
                        physical: (block * block_len + element) as u64,
                        key,
                        element: element as u64,
                        metric_weight: (end - start) as f32 * weight,
                        scale,
                    });
                }
            }
            let count = u32::try_from(points.len()).map_err(cuda_error)? - range_start;
            ranges.push(NeighborRange {
                buffer,
                start: range_start,
                count,
            });
        }
        let total = u32::try_from(points.len()).map_err(cuda_error)?;
        if total == 0 {
            return Err("CUDA neighbour quadrature is empty".into());
        }
        let total_usize = total as usize;
        Ok(Self {
            points: DeviceBuffer::from_host(stream, &points).map_err(cuda_error)?,
            seeds: DeviceBuffer::zeroed(stream, 4).map_err(cuda_error)?,
            pool: DeviceBuffer::zeroed(stream, 4 * total_usize).map_err(cuda_error)?,
            history: DeviceBuffer::zeroed(stream, 128 * total_usize).map_err(cuda_error)?,
            distances: DeviceBuffer::zeroed(stream, 4 * 128).map_err(cuda_error)?,
            ranges,
            total,
        })
    }
}

fn spherical(
    random: &mut StdRng,
    matrices: usize,
    input: usize,
    output: usize,
    column_norm: f32,
) -> Vec<u16> {
    let elements = matrices
        .checked_mul(input)
        .and_then(|v| v.checked_mul(output))
        .expect("fixed model tensor dimensions must fit usize");
    let mut values = vec![0; elements];
    blckbaux_semantics::hnet::columns(
        &mut values,
        input,
        output,
        column_norm,
        || StandardNormal.sample(random),
        |v| f16::from_f32(v).to_bits(),
    )
    .expect("fixed H-Net Gaussian column initialization");
    values
}

fn mhc_bias(layers: usize, coefficients: usize) -> Vec<u16> {
    let mut values = vec![0; layers * coefficients];
    let input_bias = f16::from_f32(-3.0_f32.ln()).to_bits();
    for layer in 0..layers {
        values[layer * coefficients..layer * coefficients + 4].fill(input_bias);
    }
    values
}

use blckbaux_semantics::hnet::Initialization as HnetScales;

impl GaussianRandom {
    fn new(stream: &CudaStream) -> CudaResult<Self> {
        let tables = &blckbaux_semantics::gaussian::ZIGGURAT;
        Ok(Self {
            thresholds: DeviceBuffer::from_host(stream, &tables.thresholds).map_err(cuda_error)?,
            widths: DeviceBuffer::from_host(stream, &tables.widths).map_err(cuda_error)?,
            densities: DeviceBuffer::from_host(stream, &tables.densities).map_err(cuda_error)?,
            status: None,
        })
    }

    fn prepare(&mut self, specs: &[ProposalSpec], stream: &CudaStream) -> CudaResult<()> {
        let blocks = specs
            .iter()
            .map(|spec| spec.elements.div_ceil(256))
            .max()
            .filter(|&n| n != 0)
            .ok_or("CUDA proposal inventory is empty")?;
        u32::try_from(blocks).map_err(cuda_error)?;
        u32::try_from(specs.len()).map_err(cuda_error)?;
        if self.status.as_ref().is_none_or(|status| {
            status.partials.len() < blocks || status.summary.len() < specs.len()
        }) {
            self.status = Some(ProposalBuffers {
                partials: DeviceBuffer::zeroed(stream, blocks).map_err(cuda_error)?,
                summary: DeviceBuffer::zeroed(stream, specs.len()).map_err(cuda_error)?,
            });
        }
        Ok(())
    }
}

fn hnet_proposal_specs(
    weights: &Weights,
    stream: &CudaStream,
    policy: HnetScales,
    shape: blckbaux_semantics::hnet::ScaleShape,
) -> CudaResult<Vec<ProposalSpec>> {
    // Order matches visit_mut. The tied embedding is a physical transpose,
    // not another logical parameter family.
    let mut inventory = vec![
        (
            "embedding_readout",
            weights.embedding.len(),
            512 * VOCAB,
            0,
            512,
        ),
        (
            "embedding_readout",
            weights.readout.len(),
            512 * VOCAB,
            0,
            0,
        ),
        ("final_norm", weights.final_norm.len(), 512, 0, 0),
    ];
    for (index, layer) in weights.layers.iter().enumerate() {
        for (name, elements, block) in [
            ("qkv", layer.qkv.len(), 512 * 640),
            ("attention_output", layer.output.len(), 512 * 512),
            ("router", layer.router.len(), 512 * 625),
            ("expert_gate_up", layer.gate.len(), 512 * 432),
            ("expert_down", layer.down.len(), 216 * 512),
            (
                "mhc_attention_predictor",
                layer.attention.predictor.len(),
                2048 * 24,
            ),
            ("mhc_attention_bias", layer.attention.bias.len(), 24),
            ("mhc_attention_control", layer.attention.control.len(), 4),
            ("attention_norm", layer.attention.norm.len(), 512),
            ("mhc_moe_predictor", layer.moe.predictor.len(), 2048 * 24),
            ("mhc_moe_bias", layer.moe.bias.len(), 24),
            ("mhc_moe_control", layer.moe.control.len(), 4),
            ("ffn_norm", layer.moe.norm.len(), 512),
        ] {
            inventory.push((name, elements, block, index * elements, 0));
        }
    }
    let hnet = weights
        .hnet
        .as_ref()
        .ok_or("procedural H-Net weights are missing local tensors")?;
    inventory.push(("mask_embed", hnet.mask.len(), 512, 0, 0));
    inventory.push(("index_query", hnet.positions.len(), 64 * 512, 0, 0));
    inventory
        .into_iter()
        .map(|(name, elements, block_len, start, transpose)| {
            if block_len == 0 || elements % block_len != 0 {
                return Err("invalid procedural H-Net proposal geometry".into());
            }
            let host_scales =
                vec![policy.proposal_scale(name)? * shape.multiplier(name); elements / block_len];
            Ok(ProposalSpec {
                elements,
                block_len: block_len as u32,
                scales: DeviceBuffer::from_host(stream, &host_scales).map_err(cuda_error)?,
                host_scales,
                tensor_id: tensor_key(name),
                logical_start: start as u64,
                transpose_columns: transpose as u32,
            })
        })
        .collect()
}

impl Weights {
    /// Hash every logical parameter in packed-family order, independently of
    /// this backend's per-layer allocation order and tied physical transpose.
    pub(super) fn logical_hashes(
        &self,
        stream: &CudaStream,
    ) -> CudaResult<Vec<blckbaux_artifact::forward::WeightHash>> {
        use blckbaux_artifact::forward::WordHash;
        use blckbaux_semantics::hnet::probe::initialized::FAMILIES;
        let mut hashes = FAMILIES
            .iter()
            .map(|family| (tensor_key(family.name), WordHash::new(family.name)))
            .collect::<BTreeMap<_, _>>();
        let mut index = 0;
        let mut result = Ok(());
        self.visit(|buffer| {
            if result.is_err() {
                return;
            }
            result = (|| {
                let spec = self
                    .proposal
                    .get(index)
                    .ok_or("CUDA digest inventory is incomplete")?;
                index += 1;
                if spec.transpose_columns != 0 {
                    return Ok(());
                }
                let hash = hashes
                    .get_mut(&spec.tensor_id)
                    .ok_or("CUDA digest found an unknown parameter family")?;
                if spec.logical_start != hash.elements() || spec.elements != buffer.len() {
                    return Err(
                        "CUDA logical parameter order or size differs from its descriptor".into(),
                    );
                }
                hash.update(&read_prefix(buffer, stream, buffer.len())?)
            })();
        });
        result?;
        if index != self.proposal.len() {
            return Err("CUDA digest omitted a physical parameter buffer".into());
        }
        let mut output = hashes
            .into_values()
            .map(WordHash::finish)
            .collect::<Vec<_>>();
        output.sort_by(|a, b| a.name.cmp(&b.name));
        if output.iter().map(|value| value.elements).sum::<u64>()
            != blckbaux_semantics::shape::HNET_PARAMS
        {
            return Err("CUDA logical parameter count differs from the H-Net contract".into());
        }
        // Verify the actual tied alias independently of the packing helper.
        let readout = read_prefix(&self.readout, stream, self.readout.len())?;
        let embedding = read_prefix(&self.embedding, stream, self.embedding.len())?;
        for token in 0..VOCAB {
            for dimension in 0..512 {
                if embedding[token * 512 + dimension] != readout[dimension * VOCAB + token] {
                    return Err(format!(
                        "CUDA tied embedding differs at token {token}, dimension {dimension}"
                    ));
                }
            }
        }
        Ok(output)
    }

    pub fn initialize_hnet(seed: u64, stream: &CudaStream) -> CudaResult<Self> {
        use blckbaux_semantics::shape as dimensions;
        const LAYERS: usize = dimensions::HNET_LAYERS as usize;
        const WIDTH: usize = dimensions::HNET_WIDTH as usize;
        const ROUTED: usize = dimensions::ROUTED_COUNT as usize;
        const EXPERTS: usize = dimensions::EXPERT_COUNT as usize;
        const GATE: usize = dimensions::GATE_WIDTH as usize;
        const DOWN: usize = dimensions::EXPERT_WIDTH as usize;
        const MHC_INPUT: usize = dimensions::MHC_INPUT as usize;
        const MHC_COEFFICIENTS: usize = dimensions::MHC_COEFS as usize;
        let policy = HnetScales::new();
        let residual_gain = policy.residual_gain;
        let router_norm = policy.router_column_norm;
        let predictor_control = policy.predictor_control;
        let transport_lambda = policy.transport_lambda;
        let down_norm = policy.down_norm();
        let random = |domain| StdRng::seed_from_u64(splitmix(seed ^ domain));
        let mut router_rng = random(0x6a09_e667);
        let mut qkv_rng = random(0xbb67_ae85);
        let mut output_rng = random(0x3c6e_f372);
        let mut gate_rng = random(0xa54f_f53a);
        let mut down_rng = random(0x510e_527f);
        let mut attention_predictor_rng = random(0x428a_2f98);
        let mut moe_predictor_rng = random(0x7137_4491);
        let upload = |values: &[u16]| DeviceBuffer::from_host(stream, values).map_err(cuda_error);
        let site = |rng: &mut StdRng| -> CudaResult<Site> {
            Ok(Site {
                predictor: upload(&spherical(rng, 1, MHC_INPUT, MHC_COEFFICIENTS, 1.0))?,
                bias: upload(&mhc_bias(1, MHC_COEFFICIENTS))?,
                control: upload(&[
                    f16::from_f32(predictor_control).to_bits(),
                    f16::from_f32(predictor_control).to_bits(),
                    f16::from_f32(predictor_control).to_bits(),
                    f16::from_f32(transport_lambda).to_bits(),
                ])?,
                norm: upload(&vec![0x3c00; WIDTH])?,
            })
        };
        let mut layers = Vec::with_capacity(LAYERS);
        for _ in 0..LAYERS {
            layers.push(Layer {
                qkv: upload(&spherical(&mut qkv_rng, 1, WIDTH, 640, 1.0))?,
                output: upload(&spherical(&mut output_rng, 1, WIDTH, WIDTH, residual_gain))?,
                router: upload(&spherical(&mut router_rng, 1, WIDTH, ROUTED, router_norm))?,
                gate: upload(&spherical(&mut gate_rng, EXPERTS, WIDTH, GATE, 1.0))?,
                down: upload(&spherical(&mut down_rng, EXPERTS, DOWN, WIDTH, down_norm))?,
                attention: site(&mut attention_predictor_rng)?,
                moe: site(&mut moe_predictor_rng)?,
            });
        }
        let mut readout_rng = random(0x9b05_688c);
        let readout_values = spherical(&mut readout_rng, 1, WIDTH, VOCAB, 1.0);
        let embedding = DeviceBuffer::from_host(
            stream,
            &crate::prefill::pack_embedding(&readout_values, WIDTH, VOCAB),
        )
        .map_err(cuda_error)?;
        let readout = upload(&readout_values)?;
        let mut mask_rng = random(0x4d41_534b);
        let mut position_rng = random(0x494e_4458);
        let hnet = Some(HnetWeights {
            mask: upload(&spherical(&mut mask_rng, 1, WIDTH, 1, 1.0))?,
            positions: upload(&spherical(&mut position_rng, 1, 64, WIDTH, 1.0))?,
        });
        let mut weights = Self {
            format: "hnet-mhc4",
            diffusion: None,
            hnet,
            embedding,
            readout,
            final_norm: upload(&vec![0x3c00; WIDTH])?,
            layers,
            proposal: Vec::new(),
            gaussian_random: Some(GaussianRandom::new(stream)?),
            neighbor_sampler: None,
        };
        weights.proposal = hnet_proposal_specs(
            &weights,
            stream,
            policy,
            blckbaux_semantics::hnet::ScaleShape::Scalar,
        )?;
        weights.neighbor_sampler = Some(NeighborSampler::new(&weights.proposal, stream)?);
        Ok(weights)
    }

    /// Resolve scales before creating the initial observation or weight backup.
    pub(super) fn set_hnet_shape(
        &mut self,
        shape: blckbaux_semantics::hnet::ScaleShape,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        self.proposal = hnet_proposal_specs(self, stream, HnetScales::new(), shape)?;
        self.neighbor_sampler = Some(NeighborSampler::new(&self.proposal, stream)?);
        Ok(())
    }

    pub fn open(path: &Path, stream: &CudaStream) -> CudaResult<Self> {
        let checkpoint = Checkpoint::open(path)?;
        let upload = |name: &str, shape: &[usize], layer| {
            DeviceBuffer::from_host(stream, &checkpoint.tensor(name, shape, layer)?)
                .map_err(cuda_error)
        };
        let readout = checkpoint.tensor("embedding_readout", &[512, VOCAB], None)?;
        let embedding = DeviceBuffer::from_host(
            stream,
            &crate::prefill::pack_embedding(&readout, 512, VOCAB),
        )
        .map_err(cuda_error)?;
        let readout = DeviceBuffer::from_host(stream, &readout).map_err(cuda_error)?;
        let final_norm = upload("final_norm", &[512], None)?;
        let mut layers = Vec::with_capacity(5);
        for layer in 0..5 {
            let site = |attention| -> CudaResult<Site> {
                let prefix = if attention {
                    "mhc_attention"
                } else {
                    "mhc_moe"
                };
                Ok(Site {
                    predictor: upload(&format!("{prefix}_predictor"), &[5, 2048, 24], Some(layer))?,
                    bias: upload(&format!("{prefix}_bias"), &[5, 24], Some(layer))?,
                    control: upload(&format!("{prefix}_control"), &[5, 4], Some(layer))?,
                    norm: upload(
                        if attention {
                            "attention_norm"
                        } else {
                            "ffn_norm"
                        },
                        &[5, 512],
                        Some(layer),
                    )?,
                })
            };
            layers.push(Layer {
                qkv: upload("qkv", &[5, 512, 640], Some(layer))?,
                output: upload("attention_output", &[5, 512, 512], Some(layer))?,
                router: upload("router", &[5, 512, 625], Some(layer))?,
                gate: upload("expert_gate_up", &[5, 626, 512, 432], Some(layer))?,
                down: upload("expert_down", &[5, 626, 216, 512], Some(layer))?,
                attention: site(true)?,
                moe: site(false)?,
            });
        }
        let diffusion = if checkpoint.format == "diffusion-mhc4" {
            Some(DiffusionWeights {
                mask: upload("mask_embed", &[512], None)?,
                index: upload("index_query", &[64, 64], None)?,
            })
        } else {
            None
        };
        let hnet = if checkpoint.format == "hnet-mhc4" {
            Some(HnetWeights {
                mask: upload("mask_embed", &[512], None)?,
                positions: upload("index_query", &[64, 512], None)?,
            })
        } else {
            None
        };
        let proposal = proposal_specs(&checkpoint, stream)?;
        let neighbor_sampler = (checkpoint.format == "hnet-mhc4")
            .then(|| NeighborSampler::new(&proposal, stream))
            .transpose()?;
        Ok(Self {
            format: checkpoint.format,
            diffusion,
            hnet,
            embedding,
            readout,
            final_norm,
            layers,
            proposal,
            gaussian_random: (checkpoint.format == "hnet-mhc4")
                .then(|| GaussianRandom::new(stream))
                .transpose()?,
            neighbor_sampler,
        })
    }
}

impl Weights {
    pub(super) fn fixture(stream: &CudaStream) -> CudaResult<Self> {
        let upload =
            |value, len| DeviceBuffer::from_host(stream, &vec![value; len]).map_err(cuda_error);
        let mut layers = Vec::new();
        for _ in 0..5 {
            let site = || -> CudaResult<Site> {
                Ok(Site {
                    predictor: upload(0, 2048 * 24)?,
                    bias: upload(0, 24)?,
                    control: upload(0, 4)?,
                    norm: upload(0x3c00, 512)?,
                })
            };
            layers.push(Layer {
                qkv: upload(0x1800, 512 * 640)?,
                output: upload(0x1800, 512 * 512)?,
                router: upload(0, 512 * 625)?,
                gate: upload(0, 626 * 512 * 432)?,
                down: upload(0, 626 * 216 * 512)?,
                attention: site()?,
                moe: site()?,
            });
        }
        Ok(Self {
            format: "fixture",
            diffusion: None,
            hnet: None,
            embedding: upload(0x3c00, VOCAB * 512)?,
            readout: upload(0x3c00, 512 * VOCAB)?,
            final_norm: upload(0x3c00, 512)?,
            layers,
            proposal: Vec::new(),
            gaussian_random: None,
            neighbor_sampler: None,
        })
    }
}

fn rms(values: &[u16], floor: f32) -> f32 {
    let sum = values
        .iter()
        .map(|&bits| {
            let value = f16::from_bits(bits).to_f32() as f64;
            value * value
        })
        .sum::<f64>();
    (sum / values.len() as f64).sqrt().max(f64::from(floor)) as f32
}

fn proposal_specs(checkpoint: &Checkpoint, stream: &CudaStream) -> CudaResult<Vec<ProposalSpec>> {
    let mut specs = Vec::new();
    let mut push = |name: &str,
                    _family: u32,
                    shape: &[usize],
                    layer: Option<usize>,
                    block_len: usize,
                    transpose_columns: usize,
                    floor: f32|
     -> CudaResult<()> {
        let values = checkpoint.tensor(name, shape, layer)?;
        if block_len == 0 || values.len() % block_len != 0 || u32::try_from(block_len).is_err() {
            return Err(format!("invalid CUDA proposal blocks for {name}"));
        }
        let scales = values
            .chunks_exact(block_len)
            .map(|block| rms(block, floor))
            .collect::<Vec<_>>();
        specs.push(ProposalSpec {
            elements: values.len(),
            block_len: block_len as u32,
            scales: DeviceBuffer::from_host(stream, &scales).map_err(cuda_error)?,
            host_scales: scales,
            tensor_id: tensor_key(name),
            logical_start: layer.map_or(0, |layer| layer * values.len()) as u64,
            transpose_columns: transpose_columns as u32,
        });
        Ok(())
    };
    let tied = 512 * VOCAB;
    push(
        "embedding_readout",
        7,
        &[512, VOCAB],
        None,
        tied,
        512,
        1.0e-6,
    )?;
    push("embedding_readout", 7, &[512, VOCAB], None, tied, 0, 1.0e-6)?;
    push("final_norm", 14, &[512], None, 512, 0, 1.0e-6)?;
    for layer in 0..5 {
        push("qkv", 1, &[5, 512, 640], Some(layer), 512 * 640, 0, 1.0e-6)?;
        push(
            "attention_output",
            2,
            &[5, 512, 512],
            Some(layer),
            512 * 512,
            0,
            1.0e-6,
        )?;
        push(
            "router",
            0,
            &[5, 512, 625],
            Some(layer),
            512 * 625,
            0,
            1.0e-6,
        )?;
        push(
            "expert_gate_up",
            3,
            &[5, 626, 512, 432],
            Some(layer),
            512 * 432,
            0,
            1.0e-6,
        )?;
        push(
            "expert_down",
            4,
            &[5, 626, 216, 512],
            Some(layer),
            216 * 512,
            0,
            1.0e-6,
        )?;
        for (prefix, predictor_family, bias_family, control_family, norm_family, floor) in [
            ("mhc_attention", 8, 9, 10, 5, 1.0e-6),
            ("mhc_moe", 11, 12, 13, 6, 1.0e-6),
        ] {
            push(
                &format!("{prefix}_predictor"),
                predictor_family,
                &[5, 2048, 24],
                Some(layer),
                2048 * 24,
                0,
                floor,
            )?;
            push(
                &format!("{prefix}_bias"),
                bias_family,
                &[5, 24],
                Some(layer),
                24,
                0,
                floor,
            )?;
            push(
                &format!("{prefix}_control"),
                control_family,
                &[5, 4],
                Some(layer),
                4,
                0,
                0.02,
            )?;
            push(
                if prefix == "mhc_attention" {
                    "attention_norm"
                } else {
                    "ffn_norm"
                },
                norm_family,
                &[5, 512],
                Some(layer),
                512,
                0,
                floor,
            )?;
        }
    }
    if checkpoint.format == "diffusion-mhc4" {
        push("mask_embed", 15, &[512], None, 512, 0, 1.0e-6)?;
        push("index_query", 16, &[64, 64], None, 4096, 0, 1.0e-6)?;
    }
    if checkpoint.format == "hnet-mhc4" {
        push("mask_embed", 15, &[512], None, 512, 0, 1.0e-6)?;
        push("index_query", 16, &[64, 512], None, 64 * 512, 0, 1.0e-6)?;
    }
    Ok(specs)
}

fn copy_buffer(
    destination: &DeviceBuffer<u16>,
    source: &DeviceBuffer<u16>,
    stream: &CudaStream,
) -> CudaResult<()> {
    if destination.len() != source.len() {
        return Err("CUDA weight backup shape mismatch".into());
    }
    unsafe {
        cuda_core::simt::memory::memcpy_dtod_async(
            destination.cu_deviceptr(),
            source.cu_deviceptr(),
            source
                .len()
                .checked_mul(std::mem::size_of::<u16>())
                .ok_or("CUDA weight backup byte overflow")?,
            stream.cu_stream(),
        )
        .map_err(cuda_error)
    }
}

impl Weights {
    fn visit(&self, mut visit: impl FnMut(&DeviceBuffer<u16>)) {
        visit(&self.embedding);
        visit(&self.readout);
        visit(&self.final_norm);
        for layer in &self.layers {
            visit(&layer.qkv);
            visit(&layer.output);
            visit(&layer.router);
            visit(&layer.gate);
            visit(&layer.down);
            for site in [&layer.attention, &layer.moe] {
                visit(&site.predictor);
                visit(&site.bias);
                visit(&site.control);
                visit(&site.norm);
            }
        }
        if let Some(diffusion) = &self.diffusion {
            visit(&diffusion.mask);
            visit(&diffusion.index);
        }
        if let Some(hnet) = &self.hnet {
            visit(&hnet.mask);
            visit(&hnet.positions);
        }
    }

    fn visit_mut(&mut self, mut visit: impl FnMut(&mut DeviceBuffer<u16>)) {
        visit(&mut self.embedding);
        visit(&mut self.readout);
        visit(&mut self.final_norm);
        for layer in &mut self.layers {
            visit(&mut layer.qkv);
            visit(&mut layer.output);
            visit(&mut layer.router);
            visit(&mut layer.gate);
            visit(&mut layer.down);
            for site in [&mut layer.attention, &mut layer.moe] {
                visit(&mut site.predictor);
                visit(&mut site.bias);
                visit(&mut site.control);
                visit(&mut site.norm);
            }
        }
        if let Some(diffusion) = &mut self.diffusion {
            visit(&mut diffusion.mask);
            visit(&mut diffusion.index);
        }
        if let Some(hnet) = &mut self.hnet {
            visit(&mut hnet.mask);
            visit(&mut hnet.positions);
        }
    }

    pub(super) fn backup(&self, stream: &CudaStream) -> CudaResult<WeightBackup> {
        let mut result: CudaResult<()> = Ok(());
        let mut buffers = Vec::with_capacity(self.proposal.len());
        self.visit(|source| {
            if result.is_err() {
                return;
            }
            result = (|| {
                let backup = DeviceBuffer::zeroed(stream, source.len()).map_err(cuda_error)?;
                copy_buffer(&backup, source, stream)?;
                buffers.push(backup);
                Ok(())
            })();
        });
        result?;
        if buffers.len() != self.proposal.len() {
            return Err("CUDA proposal inventory does not cover every model buffer".into());
        }
        Ok(WeightBackup { buffers })
    }

    pub(super) fn restore(
        &mut self,
        backup: &WeightBackup,
        stream: &CudaStream,
    ) -> CudaResult<f32> {
        let start = timing_event(stream)?;
        let mut index = 0;
        let mut result: CudaResult<()> = Ok(());
        self.visit_mut(|destination| {
            if result.is_err() {
                return;
            }
            result = backup
                .buffers
                .get(index)
                .ok_or_else(|| "CUDA weight backup is incomplete".to_string())
                .and_then(|source| copy_buffer(destination, source, stream));
            index += 1;
        });
        result?;
        if index != backup.buffers.len() {
            return Err("CUDA weight backup contains extra buffers".into());
        }
        let end = timing_event(stream)?;
        start.elapsed_ms(&end).map_err(cuda_error)
    }

    pub(super) fn commit(&self, backup: &mut WeightBackup, stream: &CudaStream) -> CudaResult<f32> {
        let start = timing_event(stream)?;
        let mut index = 0;
        let mut result: CudaResult<()> = Ok(());
        self.visit(|source| {
            if result.is_err() {
                return;
            }
            result = backup
                .buffers
                .get(index)
                .ok_or_else(|| "CUDA weight backup is incomplete".to_string())
                .and_then(|destination| copy_buffer(destination, source, stream));
            index += 1;
        });
        result?;
        if index != backup.buffers.len() {
            return Err("CUDA weight backup contains extra buffers".into());
        }
        let end = timing_event(stream)?;
        start.elapsed_ms(&end).map_err(cuda_error)
    }

    pub(super) fn reset_neighbor_history(
        &mut self,
        backup: &WeightBackup,
        module: &fbt_model::LoadedModule,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        let sampler = self
            .neighbor_sampler
            .as_mut()
            .ok_or("CUDA H-Net neighbour sampler is missing")?;
        for range in &sampler.ranges {
            let base = backup
                .buffers
                .get(range.buffer)
                .ok_or("CUDA neighbour base inventory is incomplete")?;
            let launch = module
                .prepare_hnet_neighbor_capture(LaunchConfig1D::new(
                    range.count.div_ceil(256),
                    256,
                    0,
                ))
                .map_err(cuda_error)?;
            module
                .hnet_neighbor_capture(
                    stream,
                    &launch,
                    base,
                    &sampler.points,
                    &mut sampler.history,
                    range.start,
                    range.count,
                    sampler.total,
                    0,
                )
                .map_err(cuda_error)?;
        }
        Ok(())
    }

    pub(super) fn neighbor_pool_distances(
        &mut self,
        backup: &WeightBackup,
        module: &fbt_model::LoadedModule,
        stream: &CudaStream,
        radius: f32,
        seeds: &[u64],
        history_count: usize,
    ) -> CudaResult<Vec<Vec<f32>>> {
        if history_count == 0
            || history_count > super::bo::MAX_HISTORY
            || !matches!(seeds.len(), 1 | 4)
        {
            return Err("CUDA H-Net neighbour history must be in 1..=128".into());
        }
        let random = self
            .gaussian_random
            .as_ref()
            .ok_or("CUDA Gaussian random state is missing")?;
        let sampler = self
            .neighbor_sampler
            .as_mut()
            .ok_or("CUDA H-Net neighbour sampler is missing")?;
        copy_prefix(&sampler.seeds, seeds, stream)?;
        for range in &sampler.ranges {
            let base = backup
                .buffers
                .get(range.buffer)
                .ok_or("CUDA neighbour base inventory is incomplete")?;
            let launch = module
                .prepare_hnet_neighbor_pool(LaunchConfig1D::new(range.count.div_ceil(256), 256, 0))
                .map_err(cuda_error)?;
            module
                .hnet_neighbor_pool(
                    stream,
                    &launch,
                    base,
                    &sampler.points,
                    &sampler.seeds,
                    &random.thresholds,
                    &random.widths,
                    &random.densities,
                    &mut sampler.pool,
                    range.start,
                    range.count,
                    sampler.total,
                    radius,
                    seeds.len() as u32,
                )
                .map_err(cuda_error)?;
        }
        let pairs = u32::try_from(seeds.len() * history_count).map_err(cuda_error)?;
        let launch = module
            .prepare_hnet_neighbor_distances(LaunchConfig1D::new(pairs, 256, 0))
            .map_err(cuda_error)?;
        module
            .hnet_neighbor_distances(
                stream,
                &launch,
                &sampler.pool,
                &sampler.history,
                &sampler.points,
                &mut sampler.distances,
                sampler.total,
                history_count as u32,
            )
            .map_err(cuda_error)?;
        let values = read_prefix(&sampler.distances, stream, seeds.len() * history_count)?;
        Ok((0..seeds.len())
            .map(|candidate| {
                values[candidate * history_count..(candidate + 1) * history_count].to_vec()
            })
            .collect())
    }

    pub(super) fn archive_neighbor(
        &mut self,
        candidate: usize,
        row: usize,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        if candidate >= 4 || row >= 128 {
            return Err("CUDA neighbour archive index is out of range".into());
        }
        let sampler = self
            .neighbor_sampler
            .as_mut()
            .ok_or("CUDA H-Net neighbour sampler is missing")?;
        let bytes = sampler.total as usize * std::mem::size_of::<u16>();
        unsafe {
            cuda_core::simt::memory::memcpy_dtod_async(
                sampler.history.cu_deviceptr() + (row * bytes) as u64,
                sampler.pool.cu_deviceptr() + (candidate * bytes) as u64,
                bytes,
                stream.cu_stream(),
            )
            .map_err(cuda_error)
        }
    }

    pub(super) fn materialize_gaussian(
        &mut self,
        backup: &WeightBackup,
        module: &fbt_model::LoadedModule,
        stream: &CudaStream,
        radius: f32,
        seed: u64,
    ) -> CudaResult<f32> {
        if !radius.is_finite() || radius <= 0.0 || backup.buffers.len() != self.proposal.len() {
            return Err("CUDA Gaussian proposal is invalid".into());
        }
        let mut random = self
            .gaussian_random
            .take()
            .ok_or("CUDA Gaussian random state is missing")?;
        let specs = std::mem::take(&mut self.proposal);
        let result = (|| -> CudaResult<f32> {
            random.prepare(&specs, stream)?;
            let status = random
                .status
                .as_mut()
                .ok_or("CUDA proposal status is missing")?;
            let summary_launch = module
                .prepare_proposal_status(LaunchConfig1D::new(1, 256, 0))
                .map_err(cuda_error)?;
            let start = timing_event(stream)?;
            let mut index = 0;
            let mut submitted: CudaResult<()> = Ok(());
            self.visit_mut(|candidate| {
                if submitted.is_err() {
                    return;
                }
                submitted = (|| {
                    let base = backup
                        .buffers
                        .get(index)
                        .ok_or("CUDA proposal base is incomplete")?;
                    let spec = specs
                        .get(index)
                        .ok_or("CUDA proposal scale inventory is incomplete")?;
                    if candidate.len() != spec.elements || base.len() != spec.elements {
                        return Err("CUDA proposal buffer shape differs from its inventory".into());
                    }
                    let elements = u64::try_from(candidate.len()).map_err(cuda_error)?;
                    let blocks =
                        u32::try_from(candidate.len().div_ceil(256)).map_err(cuda_error)?;
                    let launch = module
                        .prepare_gaussian_fp16(LaunchConfig1D::new(blocks, 256, 0))
                        .map_err(cuda_error)?;
                    module
                        .gaussian_fp16(
                            stream,
                            &launch,
                            base,
                            &spec.scales,
                            &random.thresholds,
                            &random.widths,
                            &random.densities,
                            candidate,
                            &mut status.partials,
                            elements,
                            spec.block_len,
                            radius,
                            spec.tensor_id,
                            spec.logical_start,
                            spec.transpose_columns,
                            seed,
                        )
                        .map_err(cuda_error)?;
                    module
                        .proposal_status(
                            stream,
                            &summary_launch,
                            &status.partials,
                            &mut status.summary,
                            blocks,
                            u32::try_from(index).map_err(cuda_error)?,
                        )
                        .map_err(cuda_error)?;
                    index += 1;
                    Ok(())
                })();
            });
            submitted?;
            if index != specs.len() {
                return Err("CUDA proposal did not visit every model buffer".into());
            }
            let end = timing_event(stream)?;
            let flags = read_prefix(&status.summary, stream, specs.len())?
                .into_iter()
                .fold(0, |flags, next| flags | next);
            blckbaux_semantics::proposal::validate(flags, false)?;
            start.elapsed_ms(&end).map_err(cuda_error)
        })();
        // A rejected proposal must not strand the host inventory/RNG or leave
        // invalid candidate weights available to a later evaluation.
        self.proposal = specs;
        self.gaussian_random = Some(random);
        if let Err(error) = result {
            self.restore(backup, stream).map_err(|restore| {
                format!("{error}; restoring incumbent also failed: {restore}")
            })?;
            return Err(error);
        }
        result
    }

    pub(super) fn materialize_basis0(
        &mut self,
        backup: &WeightBackup,
        module: &fbt_model::LoadedModule,
        stream: &CudaStream,
        radius: f32,
    ) -> CudaResult<f32> {
        if !radius.is_finite() || radius <= 0.0 || backup.buffers.len() != self.proposal.len() {
            return Err("CUDA polynomial-threshold proposal is invalid".into());
        }
        let proposal = std::mem::take(&mut self.proposal);
        let mut index = 0;
        let mut result: CudaResult<()> = Ok(());
        let start = timing_event(stream)?;
        self.visit_mut(|candidate| {
            if result.is_err() {
                return;
            }
            result = (|| {
                let base = backup
                    .buffers
                    .get(index)
                    .ok_or("CUDA proposal base is incomplete")?;
                let spec = proposal
                    .get(index)
                    .ok_or("CUDA proposal scale inventory is incomplete")?;
                let elements = u64::try_from(candidate.len()).map_err(cuda_error)?;
                let blocks = u32::try_from(candidate.len().div_ceil(256)).map_err(cuda_error)?;
                let launch = module
                    .prepare_threshold_basis0_fp16(LaunchConfig1D::new(blocks, 256, 0))
                    .map_err(cuda_error)?;
                module
                    .threshold_basis0_fp16(
                        stream,
                        &launch,
                        base,
                        &spec.scales,
                        candidate,
                        elements,
                        spec.block_len,
                        radius,
                    )
                    .map_err(cuda_error)?;
                index += 1;
                Ok(())
            })();
        });
        self.proposal = proposal;
        result?;
        let end = timing_event(stream)?;
        if index != self.proposal.len() {
            return Err("CUDA proposal did not visit every model buffer".into());
        }
        start.elapsed_ms(&end).map_err(cuda_error)
    }

    pub(super) fn physical_elements(&self) -> u64 {
        let mut elements = 0_u64;
        self.visit(|buffer| elements += buffer.len() as u64);
        elements
    }
}
