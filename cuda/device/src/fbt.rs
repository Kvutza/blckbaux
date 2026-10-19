use cuda_device::{
    DisjointSlice, SharedArray, cuda_module, kernel, launch_bounds, launch_contract, ptx_asm,
    thread, warp,
};

use super::dense_mix64;

pub const VOCAB: u32 = blckbaux_semantics::hnet::VOCABULARY as u32;
pub use blckbaux_semantics::hnet::{EncodeShape as HnetEncodeShape, ShardStats, WaveShape};
use blckbaux_semantics::hnet::{ObjectiveStats, PairShape};

pub const FBT_THREADS: u32 = 256;
const TILE: u32 = 16;
const PISA_SCORES: usize = 512;
const PISA_WIDTH: usize = 64;
const PISA_CANDIDATES: usize = 16;
const PISA_SELECTED: usize = 8;
const PISA_GROUP_HEADS: usize = 4;
const PISA_HEAD_GROUPS: u32 = 8 / PISA_GROUP_HEADS as u32;
const PISA_HEAD_SCORES: usize = PISA_SCORES * PISA_GROUP_HEADS;
const PISA_QUERIES: usize = PISA_WIDTH * PISA_GROUP_HEADS;
const PISA_VALUES: usize = 256 * PISA_GROUP_HEADS;
const ROUTE_THREADS: u32 = 256;

/// OR-reduce the two publication flags, with all 256 lanes participating.
#[inline(always)]
fn block_flags(flags: u32) -> u32 {
    static mut FLAGS: SharedArray<u32, 8> = SharedArray::UNINIT;
    let shared = unsafe { SharedArray::as_raw_mut_ptr(&raw mut FLAGS) };
    let tid = thread::threadIdx_x();
    let flags = u32::from(warp::any(
        flags & blckbaux_semantics::proposal::INVALID != 0,
    )) * blckbaux_semantics::proposal::INVALID
        | u32::from(warp::any(
            flags & blckbaux_semantics::proposal::CHANGED != 0,
        )) * blckbaux_semantics::proposal::CHANGED;
    if tid & 31 == 0 {
        unsafe {
            shared.add((tid >> 5) as usize).write(flags);
        }
    }
    thread::sync_threads();
    let mut result = 0;
    if tid == 0 {
        for warp in 0..8 {
            result |= unsafe { shared.add(warp).read() };
        }
    }
    result
}

/// One deterministic coordinate in stratified neighbour quadrature.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct NeighborPoint {
    pub physical: u64,
    pub key: u64,
    pub element: u64,
    pub metric_weight: f32,
    pub scale: f32,
}

pub type HnetNeighborPoint = NeighborPoint;

// SAFETY: repr(C), with only DeviceCopy scalar fields.
unsafe impl cuda_core::DeviceCopy for NeighborPoint {}

#[inline(always)]
fn gaussian_mix32(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

#[inline(always)]
fn gaussian_bits(seed: u64, key: u64, element: u64, attempt: u32, stream: u32) -> u32 {
    let domain = seed as u32
        ^ ((seed >> 32) as u32).wrapping_mul(0xc2b2_ae35)
        ^ (key as u32).wrapping_mul(0x27d4_eb2f)
        ^ ((key >> 32) as u32).wrapping_mul(0x1656_67b1);
    let counter = (element as u32).wrapping_mul(0x9e37_79b9)
        ^ ((element >> 32) as u32).wrapping_mul(0x85eb_ca6b);
    gaussian_mix32(
        domain ^ counter ^ attempt.wrapping_mul(0xd3a2_646c) ^ stream.wrapping_mul(0xfd70_46c5),
    )
}

#[inline(always)]
fn gaussian_uniform(bits: u32) -> f32 {
    ((((bits >> 8) + 1) as f32) * 0.000_000_059_604_645).min(0.999_999_94)
}

#[inline(always)]
fn gaussian_normal(
    seed: u64,
    key: u64,
    element: u64,
    thresholds: &[u32],
    widths: &[f32],
    densities: &[f32],
) -> f32 {
    let mut attempt = 0_u32;
    loop {
        let bits = gaussian_bits(seed, key, element, attempt, 0);
        let signed = bits as i32;
        let layer = (bits & 255) as usize;
        let magnitude = if signed < 0 {
            (!bits).wrapping_add(1)
        } else {
            bits
        };
        let sample = signed as f32 * widths[layer];
        if magnitude < thresholds[layer] {
            return sample;
        }
        if layer == 0 {
            let mut tail = 0_u32;
            loop {
                let x = -gaussian_uniform(gaussian_bits(
                    seed,
                    key,
                    element,
                    attempt.wrapping_add(tail),
                    1,
                ))
                .ln()
                    / 3.654_152_9;
                let y = -gaussian_uniform(gaussian_bits(
                    seed,
                    key,
                    element,
                    attempt.wrapping_add(tail),
                    2,
                ))
                .ln();
                if 2.0 * y >= x * x {
                    return if signed < 0 {
                        -3.654_152_9 - x
                    } else {
                        3.654_152_9 + x
                    };
                }
                tail = tail.wrapping_add(1);
            }
        }
        let uniform = gaussian_uniform(gaussian_bits(seed, key, element, attempt, 1));
        let density = densities[layer] + uniform * (densities[layer - 1] - densities[layer]);
        if density < (-0.5 * sample * sample).exp() {
            return sample;
        }
        attempt = attempt.wrapping_add(1);
    }
}

#[inline(always)]
fn gaussian_block_key(tensor_id: u64, start: u64, length: u64) -> u64 {
    dense_mix64(tensor_id ^ dense_mix64(start) ^ dense_mix64(length ^ 0xa076_1d64_78bd_642f))
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FbtShape {
    pub rows: u32,
    pub width: u32,
    pub vocab: u32,
    pub epsilon_bits: u32,
}

// SAFETY: FbtShape is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for FbtShape {}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ScoreStats {
    pub reward: f32,
    pub mean_nll: f32,
    pub maximum_nll: f32,
    pub positional_accuracy: f32,
    pub tokens: u32,
    pub mismatches: u32,
    pub patch_nll_octiles: [f32; 8],
}

// SAFETY: ScoreStats is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for ScoreStats {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ObjectiveShape {
    pub losses: u32,
    pub tokens: u32,
    pub patch: u32,
}

// SAFETY: ObjectiveShape is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for ObjectiveShape {}

impl FbtShape {
    pub fn new(rows: u32, width: u32, vocab: u32, epsilon: f32) -> Result<Self, &'static str> {
        if rows == 0 || width == 0 || vocab == 0 || !epsilon.is_finite() || epsilon <= 0.0 {
            return Err("FBT CUDA shape requires positive finite dimensions and epsilon");
        }
        Ok(Self {
            rows,
            width,
            vocab,
            epsilon_bits: epsilon.to_bits(),
        })
    }

    #[inline(always)]
    pub fn epsilon(self) -> f32 {
        f32::from_bits(self.epsilon_bits)
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulShape {
    pub rows: u32,
    pub columns: u32,
    pub inner: u32,
    pub input_stride: u32,
    pub weight_stride: u32,
    pub output_stride: u32,
    pub input_offset: u64,
    pub weight_offset: u64,
    pub output_offset: u64,
}

// SAFETY: MatmulShape is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for MatmulShape {}

impl MatmulShape {
    pub fn validate(self) -> Result<(), &'static str> {
        if self.rows == 0 || self.columns == 0 || self.inner == 0 {
            return Err("FBT CUDA matmul dimensions must be positive");
        }
        if self.input_stride < self.inner
            || self.weight_stride < self.columns
            || self.output_stride < self.columns
        {
            return Err("FBT CUDA matmul strides do not cover their logical rows");
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PisaShape {
    pub rows: u32,
    pub context: u32,
    pub heads: u32,
    pub head_width: u32,
    pub selected: u32,
    pub block_tokens: u32,
    pub query_start: u32,
    pub query_stride: u32,
    pub kv_stride: u32,
    pub key_offset: u32,
    pub visibility: u32,
}

// SAFETY: PisaShape is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for PisaShape {}

impl PisaShape {
    pub fn fbt(rows: u32, context: u32) -> Result<Self, &'static str> {
        let shape = Self {
            rows,
            context,
            heads: 8,
            head_width: 64,
            selected: 8,
            block_tokens: 64,
            query_start: 0,
            query_stride: 640,
            kv_stride: 640,
            key_offset: 512,
            visibility: 0,
        };
        shape.validate()?;
        Ok(shape)
    }

    pub fn validate(self) -> Result<(), &'static str> {
        if self.rows == 0
            || self.context == 0
            || self.heads == 0
            || self.head_width == 0
            || self.selected == 0
            || self.block_tokens == 0
        {
            return Err("FBT CUDA PISA shape has zero dimensions");
        }
        let leaves = self.context / self.block_tokens;
        let cached = self.query_stride == 512 && self.kv_stride == 128 && self.key_offset == 0;
        let packed = self.query_stride == 640 && self.kv_stride == 640 && self.key_offset == 512;
        if (!cached && !packed)
            || (packed && (self.rows % self.context != 0 || self.query_start != 0))
            || (cached
                && (self.rows > 4096
                    || self
                        .query_start
                        .checked_add(self.rows)
                        .is_none_or(|end| end > self.context)))
            || self.head_width as usize != PISA_WIDTH
            || self.heads != 8
            || self.block_tokens != 64
            || self.selected as usize != PISA_SELECTED
            || self.context % self.block_tokens != 0
            || !leaves.is_power_of_two()
            || !(64..=32768).contains(&leaves)
            || (self.visibility != 0
                && (!self.visibility.is_power_of_two() || !(128..=4096).contains(&self.visibility)))
            || self.selected as usize * self.block_tokens as usize > PISA_SCORES
        {
            return Err("FBT CUDA PISA shape is invalid");
        }
        Ok(())
    }

    pub fn cached(rows: u32, context: u32, start: u32) -> Result<Self, &'static str> {
        let shape = Self {
            rows,
            context,
            heads: 8,
            head_width: 64,
            selected: 8,
            block_tokens: 64,
            query_start: start,
            query_stride: 512,
            kv_stride: 128,
            key_offset: 0,
            visibility: 0,
        };
        shape.validate()?;
        Ok(shape)
    }

    #[inline(always)]
    pub fn query_width(self) -> u32 {
        self.heads * self.head_width
    }

    #[inline(always)]
    pub fn qkv_width(self) -> u32 {
        self.query_width() + 2 * self.head_width
    }

    #[inline(always)]
    pub fn leaves(self) -> u32 {
        self.context / self.block_tokens
    }

    #[inline(always)]
    pub fn nodes(self) -> u32 {
        2 * self.leaves() - 1
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouteShape {
    pub rows: u32,
    pub experts: u32,
    pub top_k: u32,
}

// SAFETY: RouteShape is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for RouteShape {}

impl RouteShape {
    pub fn fbt(rows: u32) -> Result<Self, &'static str> {
        if rows == 0 {
            return Err("FBT CUDA routing requires at least one row");
        }
        Ok(Self {
            rows,
            experts: 128,
            top_k: 3,
        })
    }

    pub fn validate(self) -> Result<(), &'static str> {
        if self.rows == 0 || self.experts < 4 || self.experts > 625 || self.top_k != 3 {
            return Err("FBT CUDA routing shape is invalid");
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeShape {
    pub width: u32,
    pub expert_width: u32,
    pub activation_stride: u32,
    pub row_tiles: u32,
}

// SAFETY: MoeShape is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for MoeShape {}

impl MoeShape {
    pub fn fbt() -> Self {
        Self {
            width: 512,
            expert_width: 216,
            activation_stride: 224,
            row_tiles: 8,
        }
    }

    pub fn validate(self) -> Result<(), &'static str> {
        if self.width == 0
            || self.expert_width == 0
            || self.activation_stride < self.expert_width
            || self.row_tiles == 0
        {
            return Err("FBT CUDA MoE shape is invalid");
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoutedTile {
    pub expert: u32,
    pub first_row: u32,
    pub valid_rows: u32,
    pub reserved: u32,
}

// SAFETY: RoutedTile is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for RoutedTile {}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeedbackShape {
    pub rows: u32,
    pub width: u32,
    pub context: u32,
    pub token_epsilon_bits: u32,
    pub fused_epsilon_bits: u32,
    pub token_unit_rms: u32,
    pub fused_unit_rms: u32,
}

// SAFETY: FeedbackShape is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for FeedbackShape {}

impl FeedbackShape {
    pub fn fbt(rows: u32, context: u32) -> Result<Self, &'static str> {
        let shape = Self {
            rows,
            width: 512,
            context,
            token_epsilon_bits: 1.0e-5_f32.to_bits(),
            fused_epsilon_bits: 1.0e-5_f32.to_bits(),
            token_unit_rms: 1,
            fused_unit_rms: 1,
        };
        shape.validate()?;
        Ok(shape)
    }

    pub fn validate(self) -> Result<(), &'static str> {
        let token = f32::from_bits(self.token_epsilon_bits);
        let fused = f32::from_bits(self.fused_epsilon_bits);
        if self.rows == 0
            || self.width == 0
            || self.context == 0
            || self.rows % self.context != 0
            || !token.is_finite()
            || token <= 0.0
            || !fused.is_finite()
            || fused <= 0.0
            || self.token_unit_rms > 1
            || self.fused_unit_rms > 1
        {
            return Err("FBT CUDA feedback shape is invalid");
        }
        Ok(())
    }

    #[inline(always)]
    fn token_epsilon(self) -> f32 {
        f32::from_bits(self.token_epsilon_bits)
    }

    #[inline(always)]
    fn fused_epsilon(self) -> f32 {
        f32::from_bits(self.fused_epsilon_bits)
    }
}

#[inline(always)]
pub(super) fn half(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from(bits & 0x7c00);
    let mantissa = u32::from(bits & 0x03ff);
    if exponent == 0x7c00 {
        let payload = if mantissa == 0 {
            0
        } else {
            0x0040_0000 | (mantissa << 13)
        };
        return f32::from_bits(sign | 0x7f80_0000 | payload);
    }
    // Native conversion preserves signed zero and gradual underflow. Keep the
    // explicit exceptional case above to preserve the existing NaN payloads.
    f16::from_bits(bits) as f32
}

#[inline(always)]
pub(super) fn half_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits & 0x8000_0000) >> 16;
    let exponent = bits & 0x7f80_0000;
    let mantissa = bits & 0x007f_ffff;
    if exponent == 0x7f80_0000 {
        let nan = if mantissa == 0 { 0 } else { 0x0200 };
        return (sign | 0x7c00 | nan | (mantissa >> 13)) as u16;
    }
    (value as f16).to_bits()
}

#[inline(always)]
fn silu(value: f32) -> f32 {
    let e = (-value.abs()).exp();
    let sigmoid = if value >= 0.0 {
        1.0 / (1.0 + e)
    } else {
        e / (1.0 + e)
    };
    value * sigmoid
}

#[inline(always)]
fn sigmoid(value: f32) -> f32 {
    let e = (-value.abs()).exp();
    if value >= 0.0 {
        1.0 / (1.0 + e)
    } else {
        e / (1.0 + e)
    }
}

#[inline(always)]
fn contains_leaf(node: u32, level: u32, leaf: u32) -> bool {
    let begin = node << level;
    leaf >= begin && leaf < begin + (1 << level)
}

#[inline(always)]
pub(super) unsafe fn reduce_sum(values: *mut f32) -> f32 {
    let tid = thread::threadIdx_x() + thread::threadIdx_y() * thread::blockDim_x();
    thread::sync_threads();
    let mut stride = FBT_THREADS / 2;
    while stride > 0 {
        if tid < stride {
            unsafe {
                let value =
                    values.add(tid as usize).read() + values.add((tid + stride) as usize).read();
                values.add(tid as usize).write(value);
            }
        }
        thread::sync_threads();
        stride /= 2;
    }
    unsafe { values.read() }
}

#[inline(always)]
pub(super) unsafe fn reduce_max(values: *mut f32) -> f32 {
    let tid = thread::threadIdx_x() + thread::threadIdx_y() * thread::blockDim_x();
    thread::sync_threads();
    let mut stride = FBT_THREADS / 2;
    while stride > 0 {
        if tid < stride {
            unsafe {
                let left = values.add(tid as usize).read();
                let right = values.add((tid + stride) as usize).read();
                values.add(tid as usize).write(left.max(right));
            }
        }
        thread::sync_threads();
        stride /= 2;
    }
    unsafe { values.read() }
}

#[cuda_module]
pub mod fbt_model {
    use super::*;

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn convert_check(
        input: &[f32],
        mut narrowed: DisjointSlice<u16>,
        mut widened: DisjointSlice<u32>,
        count: u32,
    ) {
        let i = thread::blockIdx_x() * 256 + thread::threadIdx_x();
        if i < count {
            unsafe {
                *narrowed.get_unchecked_mut(i as usize) = half_bits(input[i as usize]);
                *widened.get_unchecked_mut(i as usize) = half(i as u16).to_bits();
            }
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn rope(mut qkv: DisjointSlice<u16>, table: &[f32], rows: u32, context: u32, first: u32) {
        let index = thread::blockIdx_x() * 256 + thread::threadIdx_x();
        if index >= rows * 9 * 32 {
            return;
        }
        let row = index / (9 * 32);
        let pair = index % 32;
        let head = (index / 32) % 9;
        let phase = (((first + row) % context) * 32 + pair) as usize * 2;
        let offset = (row * 640 + head * 64 + pair) as usize;
        let x = half(unsafe { *qkv.get_unchecked_mut(offset) });
        let y = half(unsafe { *qkv.get_unchecked_mut(offset + 32) });
        unsafe {
            *qkv.get_unchecked_mut(offset) =
                half_bits((-y).mul_add(table[phase + 1], x * table[phase]));
            *qkv.get_unchecked_mut(offset + 32) =
                half_bits(x.mul_add(table[phase + 1], y * table[phase]));
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn mhc_replicate(input: &[u16], mut streams: DisjointSlice<u16>, rows: u32) {
        let index = thread::blockIdx_x() * 256 + thread::threadIdx_x();
        if index >= rows * 512 {
            return;
        }
        let base = (index / 512) * 2048 + index % 512;
        for stream in 0..4 {
            unsafe {
                *streams.get_unchecked_mut((base + stream * 512) as usize) = input[index as usize];
            }
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn mhc_predict(
        streams: &[u16],
        predictor: &[u16],
        bias: &[u16],
        control: &[u16],
        mut coefficients: DisjointSlice<f32>,
        rows: u32,
        parallel: u32,
    ) {
        static mut RAW: SharedArray<f32, 24> = SharedArray::UNINIT;
        static mut SCALE: SharedArray<f32, 1> = SharedArray::UNINIT;
        let row = thread::blockIdx_x();
        if row >= rows {
            return;
        }
        let tid = thread::threadIdx_x();
        let lane = tid % 32;
        let group = tid / 32;
        let raw = unsafe { SharedArray::as_raw_mut_ptr(&raw mut RAW) };
        let scale = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCALE) };
        let active = control[0] != 0 || control[1] != 0 || control[2] != 0;
        let mut squared = 0.0_f32;
        let mut dots = [0.0_f32; 3];
        if active {
            let mut input = lane;
            while input < 2048 {
                let value = half(streams[(row * 2048 + input) as usize]);
                if group == 0 {
                    squared = value.mul_add(value, squared);
                }
                for item in 0..3 {
                    let output = group * 3 + item;
                    let category = if output < 4 {
                        0
                    } else if output < 20 {
                        1
                    } else {
                        2
                    };
                    if control[category] != 0 {
                        dots[item as usize] = value.mul_add(
                            half(predictor[(input * 24 + output) as usize]),
                            dots[item as usize],
                        );
                    }
                }
                input += 32;
            }
        }
        let squared = warp::reduce_sum_f32(squared);
        if tid == 0 {
            unsafe {
                scale.write(if active {
                    (squared / 2048.0 + 1.0e-5).sqrt().recip()
                } else {
                    1.0
                });
            }
        }
        thread::sync_threads();
        for item in 0..3 {
            let sum = warp::reduce_sum_f32(dots[item as usize]);
            let output = group * 3 + item;
            let category = if output < 4 {
                0
            } else if output < 20 {
                1
            } else {
                2
            };
            if lane == 0 {
                unsafe {
                    raw.add(output as usize).write(
                        half(bias[output as usize]) + half(control[category]) * sum * scale.read(),
                    );
                }
            }
        }
        thread::sync_threads();
        if parallel != 0 {
            // All 32 lanes participate in every shuffle. Lanes 16..31 repeat
            // the first 16 entries; only lanes 0..15 publish the transport.
            // Preserve the serial reference's addition order and 20 iterations.
            if tid >= 32 {
                return;
            }
            let base = row as usize * 24;
            if tid == 0 {
                let mut total = 0.0;
                for index in 0..4 {
                    let v = sigmoid(unsafe { raw.add(index).read() });
                    unsafe {
                        *coefficients.get_unchecked_mut(base + index) = v;
                    }
                    total += v;
                }
                for index in 0..4 {
                    unsafe {
                        *coefficients.get_unchecked_mut(base + index) /= total.max(1.0e-20);
                        *coefficients.get_unchecked_mut(base + 20 + index) =
                            2.0 * sigmoid(raw.add(20 + index).read());
                    }
                }
            }
            let entry = (tid % 16) as usize;
            let r = entry / 4;
            let c = entry % 4;
            let mut maximum = f32::NEG_INFINITY;
            for index in 0..16 {
                maximum = maximum.max(unsafe { raw.add(4 + index).read() });
            }
            let mut value = (unsafe { raw.add(4 + entry).read() } - maximum)
                .max(-80.0)
                .exp();
            for _ in 0..20 {
                let mut sum = 0.0;
                for column in 0..4 {
                    sum += warp::shuffle_f32_sync(u32::MAX, value, (r * 4 + column) as u32);
                }
                value /= sum.max(1.0e-20);
                let mut sum = 0.0;
                for row in 0..4 {
                    sum += warp::shuffle_f32_sync(u32::MAX, value, (row * 4 + c) as u32);
                }
                value /= sum.max(1.0e-20);
            }
            if tid < 16 {
                let lambda = half(control[3]).clamp(0.0, 1.0);
                unsafe {
                    *coefficients.get_unchecked_mut(base + 4 + entry) =
                        (1.0 - lambda) * if r == c { 1.0 } else { 0.0 } + lambda * value;
                }
            }
            return;
        }
        if tid != 0 {
            return;
        }
        let base = row as usize * 24;
        let mut total = 0.0;
        for index in 0..4 {
            let v = sigmoid(unsafe { raw.add(index).read() });
            unsafe {
                *coefficients.get_unchecked_mut(base + index) = v;
            }
            total += v;
        }
        for index in 0..4 {
            unsafe {
                *coefficients.get_unchecked_mut(base + index) /= total.max(1.0e-20);
            }
        }
        let mut transport = [0.0_f32; 16];
        let mut maximum = f32::NEG_INFINITY;
        for index in 0..16 {
            maximum = maximum.max(unsafe { raw.add(4 + index).read() });
        }
        for index in 0..16 {
            transport[index] = (unsafe { raw.add(4 + index).read() } - maximum)
                .max(-80.0)
                .exp();
        }
        for _ in 0..20 {
            for r in 0..4 {
                let mut sum = 0.0;
                for c in 0..4 {
                    sum += transport[r * 4 + c];
                }
                for c in 0..4 {
                    transport[r * 4 + c] /= sum.max(1.0e-20);
                }
            }
            for c in 0..4 {
                let mut sum = 0.0;
                for r in 0..4 {
                    sum += transport[r * 4 + c];
                }
                for r in 0..4 {
                    transport[r * 4 + c] /= sum.max(1.0e-20);
                }
            }
        }
        let lambda = half(control[3]).clamp(0.0, 1.0);
        for r in 0..4 {
            for c in 0..4 {
                unsafe {
                    *coefficients.get_unchecked_mut(base + 4 + r * 4 + c) = (1.0 - lambda)
                        * if r == c { 1.0 } else { 0.0 }
                        + lambda * transport[r * 4 + c];
                }
            }
        }
        for index in 0..4 {
            unsafe {
                *coefficients.get_unchecked_mut(base + 20 + index) =
                    2.0 * sigmoid(raw.add(20 + index).read());
            }
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn mhc_mix(
        streams: &[u16],
        coefficients: &[f32],
        weight: &[u16],
        mut output: DisjointSlice<u16>,
        rows: u32,
        mean: u32,
    ) {
        static mut SUMS: SharedArray<f32, 256> = SharedArray::UNINIT;
        let row = thread::blockIdx_x();
        if row >= rows {
            return;
        }
        let tid = thread::threadIdx_x();
        let mut values = [0.0_f32; 2];
        let mut squared = 0.0;
        for item in 0..2 {
            let column = tid + item * 256;
            let mut value = 0.0_f32;
            for stream in 0..4 {
                let coefficient = if mean == 1 {
                    0.25
                } else {
                    coefficients[(row * 24 + stream) as usize]
                };
                value = coefficient.mul_add(
                    half(streams[(row * 2048 + stream * 512 + column) as usize]),
                    value,
                );
            }
            values[item as usize] = half(half_bits(value));
            squared = values[item as usize].mul_add(values[item as usize], squared);
        }
        let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
        unsafe {
            sums.add(tid as usize).write(squared);
        }
        let scale = (unsafe { reduce_sum(sums) } / 512.0 + 1.0e-5)
            .sqrt()
            .recip();
        for item in 0..2 {
            let column = tid + item * 256;
            unsafe {
                *output.get_unchecked_mut((row * 512 + column) as usize) =
                    half_bits(values[item as usize] * scale * half(weight[column as usize]));
            }
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn mhc_update(
        streams: &[u16],
        branch: &[u16],
        coefficients: &[f32],
        mut output: DisjointSlice<u16>,
        rows: u32,
    ) {
        let index = thread::blockIdx_x() * 256 + thread::threadIdx_x();
        if index >= rows * 2048 {
            return;
        }
        let row = index / 2048;
        let stream = (index % 2048) / 512;
        let column = index % 512;
        let mut value = coefficients[(row * 24 + 20 + stream) as usize]
            * half(branch[(row * 512 + column) as usize]);
        for source in 0..4 {
            value = coefficients[(row * 24 + 4 + stream * 4 + source) as usize].mul_add(
                half(streams[(row * 2048 + source * 512 + column) as usize]),
                value,
            );
        }
        unsafe {
            *output.get_unchecked_mut(index as usize) = half_bits(value);
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn route_counts(
        experts: &[u32],
        mut counts: DisjointSlice<u32>,
        mut prefixes: DisjointSlice<u32>,
        assignments: u32,
    ) {
        static mut SUMS: SharedArray<f32, 256> = SharedArray::UNINIT;
        let expert = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
        let chunks = assignments.div_ceil(1024);
        let mut total = 0;
        for chunk in 0..chunks {
            if tid == 0 {
                unsafe {
                    *prefixes.get_unchecked_mut((expert * chunks + chunk) as usize) = total;
                }
            }
            let mut count = 0.0;
            for item in 0..4 {
                let index = chunk * 1024 + tid + item * 256;
                if index < assignments && experts[index as usize] == expert {
                    count += 1.0;
                }
            }
            unsafe {
                sums.add(tid as usize).write(count);
            }
            total += unsafe { reduce_sum(sums) } as u32;
            thread::sync_threads();
        }
        if tid == 0 {
            unsafe {
                *counts.get_unchecked_mut(expert as usize) = total;
            }
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_bounds(32)]
    #[launch_contract(domain = 1, block = (32, 1, 1), dynamic_shared = 0)]
    pub fn route_layout(
        counts: &[u32],
        mut offsets: DisjointSlice<u32>,
        mut tiles: DisjointSlice<RoutedTile>,
        experts: u32,
        tile_capacity: u32,
    ) {
        if thread::threadIdx_x() != 0 {
            return;
        }
        let mut start = 0;
        let mut tile = 0;
        for expert in 0..experts {
            unsafe {
                *offsets.get_unchecked_mut(expert as usize) = start;
            }
            let count = counts[expert as usize];
            let mut row = 0;
            while row < count {
                unsafe {
                    *tiles.get_unchecked_mut(tile as usize) = RoutedTile {
                        expert,
                        first_row: start + row,
                        valid_rows: (count - row).min(64),
                        reserved: 0,
                    };
                }
                row += 64;
                tile += 1;
            }
            start += count;
        }
        while tile < tile_capacity {
            unsafe {
                *tiles.get_unchecked_mut(tile as usize) = RoutedTile {
                    expert: 0,
                    first_row: 0,
                    valid_rows: 0,
                    reserved: 0,
                };
            }
            tile += 1;
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn route_pack(
        input: &[u16],
        experts: &[u32],
        offsets: &[u32],
        prefixes: &[u32],
        mut mapping: DisjointSlice<u32>,
        mut packed: DisjointSlice<u16>,
        assignments: u32,
    ) {
        static mut SUMS: SharedArray<f32, 256> = SharedArray::UNINIT;
        let assignment = thread::blockIdx_x();
        if assignment >= assignments {
            return;
        }
        let tid = thread::threadIdx_x();
        let expert = experts[assignment as usize];
        let chunk = assignment / 1024;
        let mut index = chunk * 1024 + tid;
        let mut rank = 0.0;
        while index < assignment {
            if experts[index as usize] == expert {
                rank += 1.0;
            }
            index += 256;
        }
        let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
        unsafe {
            sums.add(tid as usize).write(rank);
        }
        let target = offsets[expert as usize]
            + prefixes[(expert * assignments.div_ceil(1024) + chunk) as usize]
            + unsafe { reduce_sum(sums) } as u32;
        if tid == 0 {
            unsafe {
                *mapping.get_unchecked_mut(assignment as usize) = target;
            }
        }
        for item in 0..2 {
            let column = tid + item * 256;
            unsafe {
                *packed.get_unchecked_mut((target * 512 + column) as usize) =
                    input[(assignment / 3 * 512 + column) as usize];
            }
        }
    }

    /// One block reduces the first mismatch, commits that correction, and
    /// refreshes the remaining draft. The frontier advances monotonically.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn verify_prefix(
        predicted: &[u32],
        mut tokens: DisjointSlice<u32>,
        mut frontier: DisjointSlice<u32>,
        rows: u32,
    ) {
        static mut FIRST: SharedArray<u32, 256> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x();
        let first = unsafe { SharedArray::as_raw_mut_ptr(&raw mut FIRST) };
        let start = unsafe { frontier.as_mut_ptr().read() };
        if start > rows {
            return;
        }
        let data = tokens.as_mut_ptr();
        let mut mismatch = rows;
        let mut position = start + tid;
        while position < rows {
            if unsafe { data.add(position as usize).read() } != predicted[(position - 1) as usize] {
                mismatch = mismatch.min(position);
            }
            position += 256;
        }
        unsafe {
            first.add(tid as usize).write(mismatch);
        }
        thread::sync_threads();
        let mut stride = 128;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    first.add(tid as usize).write(
                        first
                            .add(tid as usize)
                            .read()
                            .min(first.add((tid + stride) as usize).read()),
                    );
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        let mismatch = unsafe { first.read() };
        position = mismatch + tid;
        while position < rows {
            unsafe {
                *tokens.get_unchecked_mut(position as usize) =
                    predicted[(position - 1) as usize].min(VOCAB - 1);
            }
            position += 256;
        }
        thread::sync_threads();
        if tid == 0 {
            unsafe {
                *frontier.get_unchecked_mut(0) = if mismatch == rows {
                    rows
                } else if predicted[(mismatch - 1) as usize] < VOCAB {
                    mismatch + 1
                } else {
                    rows + 1
                };
            }
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn sample(
        logits: &[u16],
        mut tokens: DisjointSlice<u32>,
        rows: u32,
        temperature: f32,
        seed: u64,
        first_row: u32,
    ) {
        static mut SCORES: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut IDS: SharedArray<u32, 256> = SharedArray::UNINIT;
        let row = thread::blockIdx_x();
        if row >= rows {
            return;
        }
        let tid = thread::threadIdx_x();
        let scores = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCORES) };
        let maximum = if temperature > 0.0 {
            let mut maximum = f32::NEG_INFINITY;
            let mut token = tid;
            while token < VOCAB {
                let value = half(logits[row as usize * VOCAB as usize + token as usize]);
                if value.is_finite() {
                    maximum = maximum.max(value);
                }
                token += 256;
            }
            unsafe {
                scores.add(tid as usize).write(maximum);
            }
            let maximum = unsafe { reduce_max(scores) };
            thread::sync_threads();
            maximum
        } else {
            0.0
        };
        let mut best = f32::NEG_INFINITY;
        let mut selected = u32::MAX;
        let mut token = tid;
        while token < VOCAB {
            let logit = half(logits[row as usize * VOCAB as usize + token as usize]);
            if logit.is_finite() {
                let mut score = logit;
                if temperature > 0.0 {
                    let mut hash = seed ^ (u64::from(first_row + row) << 32) ^ u64::from(token);
                    hash = (hash ^ (hash >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                    hash = (hash ^ (hash >> 27)).wrapping_mul(0x94d049bb133111eb);
                    hash ^= hash >> 31;
                    let uniform = ((hash >> 41) as f32 + 0.5) / 8388608.0;
                    score = (logit - maximum) / temperature - (-uniform.ln()).ln();
                }
                if score > best || (score == best && token < selected) {
                    best = score;
                    selected = token;
                }
            }
            token += 256;
        }

        let ids = unsafe { SharedArray::as_raw_mut_ptr(&raw mut IDS) };
        unsafe {
            scores.add(tid as usize).write(best);
            ids.add(tid as usize).write(selected);
        }
        thread::sync_threads();
        let mut stride = 128;
        while stride > 0 {
            if tid < stride {
                let right = tid + stride;
                let a = unsafe { scores.add(tid as usize).read() };
                let b = unsafe { scores.add(right as usize).read() };
                let x = unsafe { ids.add(tid as usize).read() };
                let y = unsafe { ids.add(right as usize).read() };
                if b > a || (b == a && y < x) {
                    unsafe {
                        scores.add(tid as usize).write(b);
                        ids.add(tid as usize).write(y);
                    }
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if tid == 0 {
            unsafe {
                *tokens.get_unchecked_mut((first_row + row) as usize) = ids.read();
            }
        }
    }

    /// Expands one macro token into a deterministic 64-token patch. Each CUDA
    /// thread owns a patch, preserving the Metal H-Net recurrence exactly.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn expand_patches(
        macros: &[u32],
        readout: &[u16],
        mut tokens: DisjointSlice<u32>,
        patches: u32,
        first_patch: u32,
    ) {
        let mut patch = thread::threadIdx_x() + thread::blockIdx_x() * 256;
        let stride = thread::gridDim_x() * 256;
        while patch < patches {
            let macro_token = macros[(first_patch + patch) as usize].min(VOCAB - 1);
            let base = patch * 64;
            unsafe { *tokens.get_unchecked_mut(base as usize) = macro_token };
            let mut previous = macro_token;
            let mut intra = 1_u32;
            while intra < 64 {
                let dimension = (intra * 17) % 512;
                let weight = readout[dimension as usize * VOCAB as usize + previous as usize];
                let mut mixed = (u64::from(previous) << 32)
                    ^ (u64::from(macro_token) << 16)
                    ^ (u64::from(intra) << 8)
                    ^ u64::from(weight);
                mixed = mixed.wrapping_add(0x9e3779b97f4a7c15);
                mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d049bb133111eb);
                mixed ^= mixed >> 31;
                previous = (mixed % u64::from(VOCAB)) as u32;
                unsafe { *tokens.get_unchecked_mut((base + intra) as usize) = previous };
                intra += 1;
            }
            patch += stride;
        }
    }

    /// Form the coarse H-Net sequence from all observed fine tokens. Future
    /// patches contain u32::MAX and therefore receive the learned mask vector.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn hnet_encode(
        readout: &[u16],
        tokens: &[u32],
        positions: &[u16],
        mask: &[u16],
        mut output: DisjointSlice<u16>,
        shape: HnetEncodeShape,
    ) {
        let index = thread::blockIdx_x() * 256 + thread::threadIdx_x();
        if index >= shape.rows * shape.width {
            return;
        }
        let row = index / shape.width;
        let dimension = index % shape.width;
        let first = row * shape.patch;
        let mut value = 0.0_f32;
        // Match the encoder operation for partial/internal masks. A masked
        // first token alone does not imply that the whole patch is masked.
        let mut count = 0_u32;
        let mut intra = 0_u32;
        while intra < shape.patch {
            let token = tokens[(first + intra) as usize];
            if token < shape.vocabulary {
                value = half(positions[(intra * shape.width + dimension) as usize]).mul_add(
                    half(readout[dimension as usize * shape.vocabulary as usize + token as usize]),
                    value,
                );
                count += 1;
            }
            intra += 1;
        }
        unsafe {
            *output.get_unchecked_mut(index as usize) = if count == 0 {
                mask[dimension as usize]
            } else {
                half_bits(value * (shape.patch as f32 / count as f32).sqrt())
            };
        }
    }

    /// Build the normalized eight-way vocabulary hierarchy used by the Metal
    /// H-Net readout. Every output coordinate is independent.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn hnet_hierarchy_centroids(
        readout: &[u16],
        mut centroids: DisjointSlice<u16>,
        shape: WaveShape,
    ) {
        let gid = thread::blockIdx_x() * 256 + thread::threadIdx_x();
        let dimensions = 64_u32;
        let nodes = 37_448_u32;
        if gid >= nodes * dimensions {
            return;
        }
        let node = gid / dimensions;
        let dimension = gid % dimensions;
        let (offset, leaves) = if node < 8 {
            (0, 32_768)
        } else if node < 72 {
            (8, 4_096)
        } else if node < 584 {
            (72, 512)
        } else if node < 4_680 {
            (584, 64)
        } else {
            (4_680, 8)
        };
        let first = (node - offset) * leaves;
        let mut value = 0.0_f32;
        let mut token = 0_u32;
        while token < leaves {
            value += half(readout[(dimension * VOCAB + first + token) as usize]);
            token += 1;
        }
        unsafe {
            *centroids.get_unchecked_mut(gid as usize) =
                half_bits(value * (leaves as f32).sqrt().recip());
        }
    }

    /// Generate and score one 64-token patch from the same six-level H-Net
    /// hierarchy as Metal. Two warps advance sampled and reference prefixes.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn hnet_hierarchical_expand(
        hidden: &[u16],
        qkv: &[u16],
        positions: &[u16],
        readout: &[u16],
        centroids: &[u16],
        seeds: &[u64],
        expected: &[u32],
        mut output: DisjointSlice<u32>,
        mut losses: DisjointSlice<f32>,
        shape: WaveShape,
    ) {
        static mut QUERY: SharedArray<u16, 128> = SharedArray::UNINIT;
        static mut BASE: SharedArray<f32, 64> = SharedArray::UNINIT;
        static mut PREFIX: SharedArray<f32, 128> = SharedArray::UNINIT;
        static mut LOGITS: SharedArray<f32, 16> = SharedArray::UNINIT;
        static mut SAMPLED: SharedArray<f32, 8> = SharedArray::UNINIT;
        static mut SELECTED: SharedArray<u32, 1> = SharedArray::UNINIT;
        static mut TOKEN_LOSS: SharedArray<f32, 1> = SharedArray::UNINIT;

        let block = thread::blockIdx_x();
        let first = block * 64;
        if first >= shape.tokens {
            return;
        }
        let tid = thread::threadIdx_x();
        let branch = tid >> 5;
        let lane = tid & 31;
        let query = unsafe { SharedArray::as_raw_mut_ptr(&raw mut QUERY) };
        let base = unsafe { SharedArray::as_raw_mut_ptr(&raw mut BASE) };
        let prefix = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PREFIX) };
        let logits = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LOGITS) };
        let sampled = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SAMPLED) };
        let selected = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SELECTED) };
        let token_loss = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TOKEN_LOSS) };

        let mut dimension = lane;
        while dimension < 64 {
            if branch == 0 {
                let mut value = 0.0_f32;
                let mut inner = 0_u32;
                while inner < 512 {
                    value = half(hidden[((shape.proposal_start + block) * 512 + inner) as usize])
                        .mul_add(half(qkv[(inner * 640 + dimension) as usize]), value);
                    inner += 1;
                }
                unsafe { base.add(dimension as usize).write(value * 0.0441941738) };
            }
            unsafe { prefix.add((branch * 64 + dimension) as usize).write(0.0) };
            dimension += 32;
        }
        thread::sync_threads();

        let mut intra = 0_u32;
        while intra < 64 {
            let position = first + intra;
            if position >= shape.tokens {
                break;
            }
            dimension = lane;
            while dimension < 64 {
                unsafe {
                    query
                        .add((branch * 64 + dimension) as usize)
                        .write(half_bits(
                            base.add(dimension as usize).read()
                                + half(positions[(intra * 512 + dimension) as usize])
                                + prefix.add((branch * 64 + dimension) as usize).read(),
                        ));
                }
                dimension += 32;
            }
            if tid == 0 {
                unsafe {
                    selected.write(0);
                    token_loss.write(0.0);
                }
            }
            thread::sync_threads();
            let target = if shape.score != 0 {
                expected[position as usize]
            } else {
                0
            };
            let child = lane & 7;
            let mut level = 0_u32;
            while level < 6 {
                if lane < 8 {
                    let parent = if branch != 0 {
                        target >> ((6 - level) * 3)
                    } else {
                        unsafe { selected.read() }
                    };
                    let index = parent * 8 + child;
                    let mut value = 0.0_f32;
                    dimension = 0;
                    while dimension < 64 {
                        let weight = if level < 5 {
                            let offset = match level {
                                0 => 0,
                                1 => 8,
                                2 => 72,
                                3 => 584,
                                _ => 4_680,
                            };
                            half(centroids[((offset + index) * 64 + dimension) as usize])
                        } else {
                            half(readout[(dimension * VOCAB + index) as usize])
                        };
                        value =
                            half(unsafe { query.add((branch * 64 + dimension) as usize).read() })
                                .mul_add(weight, value);
                        dimension += 1;
                    }
                    unsafe { logits.add((branch * 8 + child) as usize).write(value) };
                    if branch == 0 {
                        let mut choice = value;
                        if shape.temperature > 0.0 {
                            let key = (u64::from(level) << 56) ^ u64::from(index);
                            let mut bits = seeds[0] ^ (u64::from(position) << 32) ^ key;
                            bits = bits.wrapping_add(0x9e3779b97f4a7c15);
                            bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                            bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d049bb133111eb);
                            bits ^= bits >> 31;
                            let uniform = ((bits >> 41) as u32 as f32 + 0.5) / 8_388_608.0;
                            choice -= shape.temperature * (-uniform.ln()).ln();
                        }
                        unsafe { sampled.add(child as usize).write(choice) };
                    }
                }
                thread::sync_threads();
                if tid == 0 {
                    let mut choice = 0_u32;
                    let mut child_index = 1_u32;
                    while child_index < 8 {
                        unsafe {
                            if sampled.add(child_index as usize).read()
                                > sampled.add(choice as usize).read()
                            {
                                choice = child_index;
                            }
                        }
                        child_index += 1;
                    }
                    unsafe { selected.write(selected.read() * 8 + choice) };
                    if shape.score != 0 {
                        let mut maximum = unsafe { logits.add(8).read() };
                        child_index = 1;
                        while child_index < 8 {
                            maximum = maximum
                                .max(unsafe { logits.add((8 + child_index) as usize).read() });
                            child_index += 1;
                        }
                        let mut sum = 0.0_f32;
                        child_index = 0;
                        while child_index < 8 {
                            sum += (unsafe { logits.add((8 + child_index) as usize).read() }
                                - maximum)
                                .exp();
                            child_index += 1;
                        }
                        let target_child = (target >> ((5 - level) * 3)) & 7;
                        unsafe {
                            token_loss.write(
                                token_loss.read() + maximum + sum.ln()
                                    - logits.add((8 + target_child) as usize).read(),
                            )
                        };
                    }
                }
                thread::sync_threads();
                level += 1;
            }
            if tid == 0 {
                unsafe {
                    *output.get_unchecked_mut(position as usize) = selected.read();
                    if shape.score != 0 {
                        *losses.get_unchecked_mut(position as usize) = token_loss.read();
                    }
                }
            }
            thread::sync_threads();
            dimension = lane;
            while dimension < 64 {
                let token = if branch == 0 {
                    unsafe { selected.read() }
                } else {
                    target
                };
                let coordinate = half(positions[(intra * 512 + dimension) as usize]);
                let shifted = coordinate + 1.0986123;
                let e = (-shifted.abs()).exp();
                let retention = if shifted >= 0.0 {
                    1.0 / (1.0 + e)
                } else {
                    e / (1.0 + e)
                };
                let scale = (1.0 - retention * retention).max(1.0e-6).sqrt();
                let slot = (branch * 64 + dimension) as usize;
                unsafe {
                    prefix.add(slot).write(retention.mul_add(
                        prefix.add(slot).read(),
                        scale * half(readout[(dimension * VOCAB + token) as usize]),
                    ));
                }
                dimension += 32;
            }
            thread::sync_threads();
            intra += 1;
        }
    }

    /// Project each coarse state into the first 64 query coordinates and seed
    /// the causal within-patch recurrence.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn hnet_wave_init(
        hidden: &[u16],
        qkv: &[u16],
        positions: &[u16],
        mut base: DisjointSlice<f32>,
        mut prefix: DisjointSlice<f32>,
        mut query: DisjointSlice<u16>,
        shape: WaveShape,
    ) {
        let index = thread::blockIdx_x() * 256 + thread::threadIdx_x();
        if index >= shape.blocks * 64 {
            return;
        }
        let block = index / 64;
        let dimension = index % 64;
        let mut value = 0.0_f32;
        let mut inner = 0_u32;
        while inner < 512 {
            value = half(hidden[((shape.proposal_start + block) * 512 + inner) as usize])
                .mul_add(half(qkv[(inner * 640 + dimension) as usize]), value);
            inner += 1;
        }
        let value = value * 0.0441941738_f32;
        unsafe {
            *base.get_unchecked_mut(index as usize) = value;
            *prefix.get_unchecked_mut(index as usize) = 0.0;
            *query.get_unchecked_mut(index as usize) =
                half_bits(value + half(positions[dimension as usize]));
        }
    }

    /// Evaluate one 1,024-token vocabulary shard after the Turing tensor-core
    /// projection. A max pass followed by a stable exponential-sum pass uses
    /// one exponential per logit while preserving exact FP32 softmax scoring.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn hnet_wave_project(
        logits: &[f32],
        seeds: &[u64],
        expected: &[u32],
        mut partials: DisjointSlice<ShardStats>,
        shape: WaveShape,
    ) {
        static mut MAXIMA: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut TOTALS: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut TARGETS: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut BEST: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut IDS: SharedArray<u32, 256> = SharedArray::UNINIT;
        let group = thread::blockIdx_x();
        let shard = group % shape.shards;
        let block = group / shape.shards;
        if block >= shape.blocks {
            return;
        }
        let tid = thread::threadIdx_x();
        let position = block * 64 + shape.intra;
        if position >= shape.tokens {
            return;
        }
        let first_token = shard * 1024 + tid;
        let expected_token = expected[position as usize];
        let mut local_maximum = f32::NEG_INFINITY;
        let mut local_target = f32::NEG_INFINITY;
        let mut local_best = f32::NEG_INFINITY;
        let mut local_id = first_token;
        let mut item = 0_u32;
        while item < 4 {
            let token = first_token + item * 256;
            let logit = logits[block as usize * VOCAB as usize + token as usize];
            local_maximum = local_maximum.max(logit);
            if shape.score != 0 && expected_token == token {
                local_target = logit;
            }
            let mut sample = logit;
            if shape.temperature > 0.0 && shape.temperature != 1.0 {
                let mut bits = seeds[0] ^ (u64::from(position) << 32) ^ u64::from(token);
                bits = bits.wrapping_add(0x9e3779b97f4a7c15);
                bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d049bb133111eb);
                bits ^= bits >> 31;
                let uniform = ((bits >> 41) as f32 + 0.5) / 8388608.0;
                sample -= shape.temperature * (-uniform.ln()).ln();
            }
            if sample > local_best || (sample == local_best && token < local_id) {
                local_best = sample;
                local_id = token;
            }
            item += 1;
        }
        let maxima = unsafe { SharedArray::as_raw_mut_ptr(&raw mut MAXIMA) };
        let totals = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TOTALS) };
        let targets = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TARGETS) };
        let best = unsafe { SharedArray::as_raw_mut_ptr(&raw mut BEST) };
        let ids = unsafe { SharedArray::as_raw_mut_ptr(&raw mut IDS) };
        unsafe {
            maxima.add(tid as usize).write(local_maximum);
            targets.add(tid as usize).write(local_target);
            best.add(tid as usize).write(local_best);
            ids.add(tid as usize).write(local_id);
        }
        thread::sync_threads();
        let mut stride = 128_u32;
        while stride > 0 {
            if tid < stride {
                let right = tid + stride;
                unsafe {
                    maxima.add(tid as usize).write(
                        maxima
                            .add(tid as usize)
                            .read()
                            .max(maxima.add(right as usize).read()),
                    );
                    let left_target = targets.add(tid as usize).read();
                    let right_target = targets.add(right as usize).read();
                    targets
                        .add(tid as usize)
                        .write(left_target.max(right_target));
                    let left_best = best.add(tid as usize).read();
                    let right_best = best.add(right as usize).read();
                    let left_id = ids.add(tid as usize).read();
                    let right_id = ids.add(right as usize).read();
                    if right_best > left_best || (right_best == left_best && right_id < left_id) {
                        best.add(tid as usize).write(right_best);
                        ids.add(tid as usize).write(right_id);
                    }
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        let maximum = unsafe { maxima.read() };
        let mut local_sum = 0.0_f32;
        let weighted_reservoir = shape.temperature == 1.0;
        let mut local_id = first_token;
        item = 0;
        while item < 4 {
            let token = first_token + item * 256;
            let weight = (logits[block as usize * VOCAB as usize + token as usize] - maximum).exp();
            let next_sum = local_sum + weight;
            if weighted_reservoir {
                let mut bits =
                    seeds[0] ^ (u64::from(position) << 32) ^ u64::from(token) ^ 0x243f6a8885a308d3;
                bits = bits.wrapping_add(0x9e3779b97f4a7c15);
                bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d049bb133111eb);
                bits ^= bits >> 31;
                let uniform = ((bits >> 41) as f32 + 0.5) / 8388608.0;
                if uniform * next_sum < weight {
                    local_id = token;
                }
            }
            local_sum = next_sum;
            item += 1;
        }
        unsafe {
            totals.add(tid as usize).write(local_sum);
            if weighted_reservoir {
                ids.add(tid as usize).write(local_id);
            }
        }
        thread::sync_threads();
        stride = 128;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    let left_sum = totals.add(tid as usize).read();
                    let right_sum = totals.add((tid + stride) as usize).read();
                    let combined = left_sum + right_sum;
                    if weighted_reservoir && right_sum > 0.0 {
                        let mut bits = seeds[0]
                            ^ (u64::from(position) << 32)
                            ^ (u64::from(stride) << 16)
                            ^ u64::from(tid)
                            ^ 0x13198a2e03707344;
                        bits = bits.wrapping_add(0x9e3779b97f4a7c15);
                        bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                        bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d049bb133111eb);
                        bits ^= bits >> 31;
                        let uniform = ((bits >> 41) as f32 + 0.5) / 8388608.0;
                        if uniform * combined < right_sum {
                            ids.add(tid as usize)
                                .write(ids.add((tid + stride) as usize).read());
                        }
                    }
                    totals.add(tid as usize).write(combined);
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if tid == 0 {
            unsafe {
                *partials
                    .get_unchecked_mut(block as usize * shape.shards as usize + shard as usize) =
                    ShardStats {
                        maximum: maxima.read(),
                        sum: totals.read(),
                        target: targets.read(),
                        best: best.read(),
                        token: ids.read(),
                        reserved: 0,
                    };
            }
        }
    }

    /// Stable log-sum-exp merge and exact weighted reservoir composition.
    #[inline(always)]
    fn merge_shards(
        mut left: ShardStats,
        right: ShardStats,
        temperature: f32,
        seed: u64,
    ) -> ShardStats {
        let maximum = left.maximum.max(right.maximum);
        let left_sum = left.sum * (left.maximum - maximum).exp();
        let right_sum = right.sum * (right.maximum - maximum).exp();
        let combined = left_sum + right_sum;
        if temperature == 1.0 && right_sum > 0.0 {
            let mut bits = seed ^ 0xa4093822299f31d0;
            bits = bits.wrapping_add(0x9e3779b97f4a7c15);
            bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d049bb133111eb);
            bits ^= bits >> 31;
            let uniform = ((bits >> 41) as f32 + 0.5) / 8388608.0;
            if uniform * combined < right_sum {
                left.token = right.token;
            }
        }
        left.sum = combined;
        left.maximum = maximum;
        left.target = left.target.max(right.target);
        if temperature != 1.0
            && (right.best > left.best || (right.best == left.best && right.token < left.token))
        {
            left.best = right.best;
            left.token = right.token;
        }
        left
    }

    /// Merge the vocabulary shards, publish one sampled token and exact NLL,
    /// and advance the 64-dimensional causal recurrence.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn hnet_wave_reduce(
        partials: &[ShardStats],
        base: &[f32],
        mut prefix: DisjointSlice<f32>,
        mut query: DisjointSlice<u16>,
        positions: &[u16],
        readout: &[u16],
        seeds: &[u64],
        mut output: DisjointSlice<u32>,
        mut losses: DisjointSlice<f32>,
        shape: WaveShape,
    ) {
        static mut CHOSEN: SharedArray<u32, 1> = SharedArray::UNINIT;
        let block = thread::blockIdx_x();
        if block >= shape.blocks {
            return;
        }
        if block * 64 + shape.intra >= shape.tokens {
            return;
        }
        let tid = thread::threadIdx_x();
        let chosen = unsafe { SharedArray::as_raw_mut_ptr(&raw mut CHOSEN) };
        // Each lane merges a strided subset, followed by a balanced block
        // reduction. Work per lane grows with shards / 64, not shards.
        static mut MERGED: SharedArray<ShardStats, 64> = SharedArray::UNINIT;
        let merged = unsafe { SharedArray::as_raw_mut_ptr(&raw mut MERGED) };
        let row = block as usize * shape.shards as usize;
        let position = block * 64 + shape.intra;
        let mut stats = partials[row + tid as usize];
        let mut shard = tid + 64;
        while shard < shape.shards {
            stats = merge_shards(
                stats,
                partials[row + shard as usize],
                shape.temperature,
                seeds[0] ^ (u64::from(position) << 32) ^ u64::from(shard),
            );
            shard += 64;
        }
        unsafe { merged.add(tid as usize).write(stats) };
        thread::sync_threads();
        let mut stride = 32_u32;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    let left = merged.add(tid as usize).read();
                    let right = merged.add((tid + stride) as usize).read();
                    merged.add(tid as usize).write(merge_shards(
                        left,
                        right,
                        shape.temperature,
                        seeds[0]
                            ^ (u64::from(position) << 32)
                            ^ (u64::from(stride) << 16)
                            ^ u64::from(tid)
                            ^ 0x452821e638d01377,
                    ));
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if tid == 0 {
            let stats = unsafe { merged.read() };
            unsafe {
                chosen.write(stats.token);
                *output.get_unchecked_mut((block * 64 + shape.intra) as usize) = stats.token;
                if shape.score != 0 {
                    *losses.get_unchecked_mut((block * 64 + shape.intra) as usize) =
                        stats.maximum + stats.sum.ln() - stats.target;
                }
            }
        }
        thread::sync_threads();
        if shape.intra + 1 < 64 {
            let index = block * 64 + tid;
            let token = unsafe { chosen.read() };
            let previous = unsafe { *prefix.get_unchecked_mut(index as usize) };
            let coordinate = half(positions[(shape.intra * 512 + tid) as usize]);
            let retention = sigmoid(coordinate + 1.098_612_3);
            let input_scale = (1.0 - retention * retention).max(1.0e-6).sqrt();
            let value = retention.mul_add(
                previous,
                input_scale * half(readout[tid as usize * VOCAB as usize + token as usize]),
            );
            unsafe {
                *prefix.get_unchecked_mut(index as usize) = value;
                *query.get_unchecked_mut(index as usize) = half_bits(
                    base[index as usize]
                        + half(positions[((shape.intra + 1) * 512 + tid) as usize])
                        + value,
                );
            }
        }
    }

    /// Materialize the first polynomial-threshold basis direction into FP16
    /// candidate weights. Basis zero is the constant positive Walsh feature,
    /// matching the initialization round used by the Metal optimizer.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn threshold_basis0_fp16(
        base: &[u16],
        scales: &[f32],
        mut candidate: DisjointSlice<u16>,
        elements: u64,
        block_len: u32,
        radius: f32,
    ) {
        let mut index = u64::from(thread::blockIdx_x()) * 256 + u64::from(thread::threadIdx_x());
        let stride = u64::from(thread::gridDim_x()) * 256;
        while index < elements {
            let block = index / u64::from(block_len);
            let value = half(base[index as usize]) + scales[block as usize] * radius;
            unsafe { *candidate.get_unchecked_mut(index as usize) = half_bits(value) };
            index += stride;
        }
    }

    /// Materialize an independent dense Gaussian proposal in CUDA FP16 weights.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn gaussian_fp16(
        base: &[u16],
        scales: &[f32],
        thresholds: &[u32],
        widths: &[f32],
        densities: &[f32],
        mut candidate: DisjointSlice<u16>,
        mut status: DisjointSlice<u32>,
        elements: u64,
        block_len: u32,
        radius: f32,
        tensor_id: u64,
        logical_start: u64,
        transpose_columns: u32,
        seed: u64,
    ) {
        let mut index = u64::from(thread::blockIdx_x()) * 256 + u64::from(thread::threadIdx_x());
        let stride = u64::from(thread::gridDim_x()) * 256;
        let mut flags = 0;
        while index < elements {
            let logical = if transpose_columns == 0 {
                index
            } else {
                let rows = elements / u64::from(transpose_columns);
                (index % u64::from(transpose_columns)) * rows + index / u64::from(transpose_columns)
            };
            let block = logical / u64::from(block_len);
            let element = logical % u64::from(block_len);
            let key = gaussian_block_key(
                tensor_id,
                logical_start + block * u64::from(block_len),
                u64::from(block_len),
            );
            let direction = gaussian_normal(
                seed ^ 0x8ebc_6af0_9c88_c6e3,
                key,
                element,
                thresholds,
                widths,
                densities,
            );
            let delta = scales[block as usize] * radius * direction;
            let cand_bits = half_bits(half(base[index as usize]) + delta);
            flags |= blckbaux_semantics::proposal::fp16_flags(base[index as usize], cand_bits);
            unsafe { *candidate.get_unchecked_mut(index as usize) = cand_bits };
            index += stride;
        }
        let flags = block_flags(flags);
        if thread::threadIdx_x() == 0 {
            unsafe {
                *status.get_unchecked_mut(thread::blockIdx_x() as usize) = flags;
            }
        }
    }

    /// Summarize one complete tensor on device; only one word per tensor
    /// crosses to the host, rather than a full parameter row or block array.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn proposal_status(
        status: &[u32],
        mut summary: DisjointSlice<u32>,
        blocks: u32,
        tensor: u32,
    ) {
        let mut index = thread::threadIdx_x();
        let mut flags = 0;
        while index < blocks {
            flags |= status[index as usize];
            index += 256;
        }
        let flags = block_flags(flags);
        if thread::threadIdx_x() == 0 {
            unsafe {
                *summary.get_unchecked_mut(tensor as usize) = flags;
            }
        }
    }

    /// Materialize only the configured shadow descriptors used by ENN geometry.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn hnet_neighbor_pool(
        base: &[u16],
        points: &[HnetNeighborPoint],
        seeds: &[u64],
        thresholds: &[u32],
        widths: &[f32],
        densities: &[f32],
        mut pool: DisjointSlice<u16>,
        point_start: u32,
        point_count: u32,
        total_points: u32,
        radius: f32,
        candidates: u32,
    ) {
        let local = thread::blockIdx_x() * 256 + thread::threadIdx_x();
        if local >= point_count {
            return;
        }
        let index = point_start + local;
        let point = points[index as usize];
        let value = half(base[point.physical as usize]);
        let mut candidate = 0_u32;
        while candidate < candidates {
            let noise = gaussian_normal(
                seeds[candidate as usize] ^ 0x8ebc_6af0_9c88_c6e3,
                point.key,
                point.element,
                thresholds,
                widths,
                densities,
            );
            unsafe {
                *pool.get_unchecked_mut((candidate * total_points + index) as usize) =
                    half_bits(value + point.scale * radius * noise);
            }
            candidate += 1;
        }
    }

    /// Capture the current incumbent at the sampled Metal metric coordinates.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn hnet_neighbor_capture(
        base: &[u16],
        points: &[HnetNeighborPoint],
        mut history: DisjointSlice<u16>,
        point_start: u32,
        point_count: u32,
        total_points: u32,
        row: u32,
    ) {
        let local = thread::blockIdx_x() * 256 + thread::threadIdx_x();
        if local >= point_count {
            return;
        }
        let index = point_start + local;
        let point = points[index as usize];
        unsafe {
            *history.get_unchecked_mut((row * total_points + index) as usize) =
                base[point.physical as usize];
        }
    }

    /// Reduce the four sampled descriptors against every retained history row.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn hnet_neighbor_distances(
        pool: &[u16],
        history: &[u16],
        points: &[HnetNeighborPoint],
        mut distances: DisjointSlice<f32>,
        total_points: u32,
        history_count: u32,
    ) {
        static mut SUMS: SharedArray<f32, 256> = SharedArray::UNINIT;
        let pair = thread::blockIdx_x();
        let pairs = 4 * history_count;
        if pair >= pairs {
            return;
        }
        let candidate = pair / history_count;
        let row = pair % history_count;
        let tid = thread::threadIdx_x();
        let mut sum = 0.0_f32;
        let mut index = tid;
        while index < total_points {
            let point = points[index as usize];
            let delta = half(pool[(candidate * total_points + index) as usize])
                - half(history[(row * total_points + index) as usize]);
            sum += delta * delta * point.metric_weight;
            index += 256;
        }
        let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
        unsafe { sums.add(tid as usize).write(sum) };
        thread::sync_threads();
        let mut step = 128;
        while step > 0 {
            if tid < step {
                unsafe {
                    sums.add(tid as usize).write(
                        sums.add(tid as usize).read() + sums.add((tid + step) as usize).read(),
                    );
                }
            }
            thread::sync_threads();
            step >>= 1;
        }
        if tid == 0 {
            unsafe { *distances.get_unchecked_mut(pair as usize) = sums.read() };
        }
    }

    /// Reduce exactly one 64-token H-Net patch into one causal response channel.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn patch_mean_nll(losses: &[f32], mut patches: DisjointSlice<f32>, tokens: u32) {
        static mut SUMS: SharedArray<f32, 64> = SharedArray::UNINIT;
        let patch = thread::blockIdx_x();
        if patch >= tokens.div_ceil(64) {
            return;
        }
        let tid = thread::threadIdx_x();
        let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
        unsafe {
            sums.add(tid as usize).write(if patch * 64 + tid < tokens {
                losses[(patch * 64 + tid) as usize]
            } else {
                0.0
            });
        }
        thread::sync_threads();
        let mut step = 32;
        while step > 0 {
            if tid < step {
                unsafe {
                    let value =
                        sums.add(tid as usize).read() + sums.add((tid + step) as usize).read();
                    sums.add(tid as usize).write(value);
                }
            }
            thread::sync_threads();
            step >>= 1;
        }
        if tid == 0 {
            unsafe {
                *patches.get_unchecked_mut(patch as usize) =
                    sums.read() / (tokens - patch * 64).min(64) as f32
            };
        }
    }

    /// Computes one numerically stable target NLL per generated position.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn score(
        logits: &[u16],
        targets: &[u32],
        mut losses: DisjointSlice<f32>,
        rows: u32,
        target_offset: u32,
        loss_offset: u32,
    ) {
        static mut MAXIMA: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut TOTALS: SharedArray<f32, 256> = SharedArray::UNINIT;
        let row = thread::blockIdx_x();
        if row >= rows {
            return;
        }
        let tid = thread::threadIdx_x();
        let maxima = unsafe { SharedArray::as_raw_mut_ptr(&raw mut MAXIMA) };
        let totals = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TOTALS) };
        let mut local_maximum = f32::NEG_INFINITY;
        let mut local_total = 0.0;
        let mut token = tid;
        while token < VOCAB {
            let value = half(logits[row as usize * VOCAB as usize + token as usize]);
            let maximum = local_maximum.max(value);
            local_total = local_total * (local_maximum - maximum).exp() + (value - maximum).exp();
            local_maximum = maximum;
            token += 256;
        }
        unsafe {
            maxima.add(tid as usize).write(local_maximum);
            totals.add(tid as usize).write(local_total);
        }
        thread::sync_threads();
        let mut stride = 128;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    let left_maximum = maxima.add(tid as usize).read();
                    let right_maximum = maxima.add((tid + stride) as usize).read();
                    let maximum = left_maximum.max(right_maximum);
                    let total = totals.add(tid as usize).read() * (left_maximum - maximum).exp()
                        + totals.add((tid + stride) as usize).read()
                            * (right_maximum - maximum).exp();
                    maxima.add(tid as usize).write(maximum);
                    totals.add(tid as usize).write(total);
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if tid == 0 {
            let target = targets[(target_offset + row) as usize].min(VOCAB - 1);
            let target_logit = half(logits[row as usize * VOCAB as usize + target as usize]);
            let maximum = unsafe { maxima.read() };
            let normalizer = unsafe { totals.read() };
            unsafe {
                *losses.get_unchecked_mut((loss_offset + row) as usize) =
                    maximum + normalizer.ln() - target_logit;
            }
        }
    }

    /// Reduce token losses and same-condition token differences. Patch means
    /// use their actual lengths, including a partial final patch. These are
    /// the same raw moments consumed by the Metal host, not differences of
    /// separately rounded patch means.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn hnet_objective(
        losses: &[f32],
        generated: &[u32],
        expected: &[u32],
        mut output: DisjointSlice<ObjectiveStats>,
        baseline: &[f32],
        shape: PairShape,
        mut patches: DisjointSlice<f32>,
    ) {
        static mut FLOATS: SharedArray<f32, 64> = SharedArray::UNINIT;
        static mut COUNTS: SharedArray<u32, 24> = SharedArray::UNINIT;
        static mut OCTILES: SharedArray<f32, 8> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x();
        let lane = tid & 31;
        let group = tid >> 5;
        let floats = unsafe { SharedArray::as_raw_mut_ptr(&raw mut FLOATS) };
        let counts = unsafe { SharedArray::as_raw_mut_ptr(&raw mut COUNTS) };
        let octiles = unsafe { SharedArray::as_raw_mut_ptr(&raw mut OCTILES) };
        let mut values = [0.0_f32; 8];
        values[6] = f32::NEG_INFINITY;
        let mut flags = [0_u32, 0, u32::MAX];
        let mut position = tid;
        while position < shape.tokens {
            let loss = losses[position as usize];
            let equal = generated[position as usize] == expected[position as usize];
            values[0] += loss;
            if shape.paired != 0 {
                values[3] += baseline[position as usize] - loss;
            }
            values[6] = values[6].max(loss);
            if equal {
                flags[0] += 1;
            } else {
                flags[1] += 1;
                flags[2] = flags[2].min(position);
                values[7] += loss;
            }
            position += 256;
        }
        let batch_tokens = shape.batch_tokens.max(1);
        let batches = shape.tokens.div_ceil(batch_tokens);
        let mut batch = tid;
        while batch < batches {
            let begin = batch * batch_tokens;
            let end = (begin + batch_tokens).min(shape.tokens);
            let (mut loss, mut difference) = (0.0_f32, 0.0_f32);
            position = begin;
            while position < end {
                loss += losses[position as usize];
                if shape.paired != 0 {
                    difference += baseline[position as usize] - losses[position as usize];
                }
                position += 1;
            }
            let inverse = 1.0 / (end - begin) as f32;
            let centered = loss * inverse - 12.0;
            difference *= inverse;
            if shape.paired != 0 {
                unsafe { *patches.get_unchecked_mut(batch as usize) = difference };
            }
            values[1] += centered;
            values[2] += centered * centered;
            if shape.paired != 0 {
                values[4] += difference;
                values[5] += difference * difference;
            }
            batch += 256;
        }
        if tid < 8 {
            let begin = tid * batch_tokens / 8;
            let finish = (tid + 1) * batch_tokens / 8;
            let (mut sum, mut count, mut base) = (0.0_f32, 0_u32, 0_u32);
            while base < shape.tokens {
                position = base + begin;
                let end = (base + finish).min(shape.tokens);
                while position < end {
                    sum += losses[position as usize];
                    count += 1;
                    position += 1;
                }
                base += batch_tokens;
            }
            unsafe {
                octiles.add(tid as usize).write(if count == 0 {
                    f32::NAN
                } else {
                    sum / count as f32
                });
            }
        }
        let mut channel = 0;
        while channel < 8 {
            let reduced = if channel == 6 {
                warp::reduce_max_f32(values[channel])
            } else {
                warp::reduce_sum_f32(values[channel])
            };
            if lane == 0 {
                unsafe {
                    floats.add(channel * 8 + group as usize).write(reduced);
                }
            }
            channel += 1;
        }
        let mut offset = 16;
        while offset > 0 {
            flags[0] += warp::shuffle_xor(flags[0], offset);
            flags[1] += warp::shuffle_xor(flags[1], offset);
            flags[2] = flags[2].min(warp::shuffle_xor(flags[2], offset));
            offset >>= 1;
        }
        if lane == 0 {
            for channel in 0..3 {
                unsafe {
                    counts
                        .add(channel * 8 + group as usize)
                        .write(flags[channel]);
                }
            }
        }
        thread::sync_threads();
        if group == 0 {
            for channel in 0..8 {
                let value = if lane < 8 {
                    unsafe { floats.add(channel * 8 + lane as usize).read() }
                } else if channel == 6 {
                    f32::NEG_INFINITY
                } else {
                    0.0
                };
                values[channel] = if channel == 6 {
                    warp::reduce_max_f32(value)
                } else {
                    warp::reduce_sum_f32(value)
                };
            }
            for channel in 0..3 {
                flags[channel] = if lane < 8 {
                    unsafe { counts.add(channel * 8 + lane as usize).read() }
                } else if channel == 2 {
                    u32::MAX
                } else {
                    0
                };
            }
            offset = 16;
            while offset > 0 {
                flags[0] += warp::shuffle_xor(flags[0], offset);
                flags[1] += warp::shuffle_xor(flags[1], offset);
                flags[2] = flags[2].min(warp::shuffle_xor(flags[2], offset));
                offset >>= 1;
            }
            if lane == 0 {
                unsafe {
                    *output.get_unchecked_mut(0) = ObjectiveStats {
                        sum: values[0],
                        batch_centered_sum: values[1],
                        squared_batch_centered_sum: values[2],
                        difference_sum: values[3],
                        difference_batch_sum: values[4],
                        squared_difference_batch_sum: values[5],
                        maximum: values[6],
                        mismatch_sum: values[7],
                        batches,
                        matches: flags[0],
                        mismatches: flags[1],
                        first_mismatch: flags[2],
                        patch_nll_octiles: [
                            *octiles.add(0),
                            *octiles.add(1),
                            *octiles.add(2),
                            *octiles.add(3),
                            *octiles.add(4),
                            *octiles.add(5),
                            *octiles.add(6),
                            *octiles.add(7),
                        ],
                    };
                }
            }
        }
    }

    /// Reduces scored positions to the free-running generation objective and
    /// its accuracy diagnostics without returning logits to the host.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn objective(
        losses: &[f32],
        generated: &[u32],
        targets: &[u32],
        mut output: DisjointSlice<ScoreStats>,
        shape: ObjectiveShape,
    ) {
        static mut SUMS: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut MAXIMA: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut MATCHES: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut PATCH_NLL: SharedArray<f32, 8> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x();
        let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
        let maxima = unsafe { SharedArray::as_raw_mut_ptr(&raw mut MAXIMA) };
        let matches = unsafe { SharedArray::as_raw_mut_ptr(&raw mut MATCHES) };
        let patch_nll = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PATCH_NLL) };
        let mut sum = 0.0;
        let mut maximum = f32::NEG_INFINITY;
        let mut matched = 0.0;
        let mut row = tid;
        while row < shape.losses {
            let loss = losses[row as usize];
            sum += loss;
            maximum = maximum.max(loss);
            row += 256;
        }
        row = tid;
        while row < shape.tokens {
            if generated[row as usize] == targets[row as usize] {
                matched += 1.0;
            }
            row += 256;
        }
        if tid < 8 {
            let patch = shape.patch.max(1);
            let begin = tid * patch / 8;
            let finish = (tid + 1) * patch / 8;
            let mut patch_sum = 0.0;
            let mut patch_count = 0_u32;
            let mut base = 0_u32;
            while base < shape.losses {
                let end = (base + finish).min(shape.losses);
                let mut position = base + begin;
                while position < end {
                    patch_sum += losses[position as usize];
                    patch_count += 1;
                    position += 1;
                }
                base += patch;
            }
            unsafe {
                patch_nll.add(tid as usize).write(if patch_count == 0 {
                    f32::NAN
                } else {
                    patch_sum / patch_count as f32
                });
            }
        }
        thread::sync_threads();
        unsafe {
            sums.add(tid as usize).write(sum);
            maxima.add(tid as usize).write(maximum);
            matches.add(tid as usize).write(matched);
        }
        let sum = unsafe { reduce_sum(sums) };
        let maximum = unsafe { reduce_max(maxima) };
        let matched = unsafe { reduce_sum(matches) };
        if tid == 0 {
            let loss_count = shape.losses.max(1) as f32;
            let token_count = shape.tokens.max(1) as f32;
            let mean = sum / loss_count;
            unsafe {
                *output.get_unchecked_mut(0) = ScoreStats {
                    reward: -mean,
                    mean_nll: mean,
                    maximum_nll: maximum,
                    positional_accuracy: matched / token_count,
                    tokens: shape.tokens,
                    mismatches: shape.tokens - matched as u32,
                    patch_nll_octiles: [
                        *patch_nll.add(0),
                        *patch_nll.add(1),
                        *patch_nll.add(2),
                        *patch_nll.add(3),
                        *patch_nll.add(4),
                        *patch_nll.add(5),
                        *patch_nll.add(6),
                        *patch_nll.add(7),
                    ],
                };
            }
        }
    }

    /// Keeps the module PTX floor at 6.5 for Turing matrix instructions.
    /// CUDA-Oxide does not infer that floor from user-authored m16n8k8 PTX.
    #[kernel]
    #[launch_bounds(32)]
    pub fn turing_floor() {
        let mut c0 = 0_i32;
        let mut c1 = 0_i32;
        // SAFETY: this exported feature anchor is never launched. Its only
        // role is to make CUDA-Oxide select PTX 6.5 for the module.
        unsafe {
            ptx_asm!(
                "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};",
                inout("+r") c0,
                inout("+r") c1,
                in("r") 0_u32,
                in("r") 0_u32,
                options(register_only),
            );
        }
        let _ = (c0, c1);
    }

    /// Embedding lookup from a resident token-major weight packing. Packing
    /// happens once during executor construction so each output row is read
    /// contiguously during the measured loop.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(FBT_THREADS)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn embed_packed(
        weights: &[u16],
        tokens: &[u32],
        mut output: DisjointSlice<u16>,
        shape: FbtShape,
    ) {
        let row = thread::blockIdx_x();
        if row >= shape.rows {
            return;
        }
        let token = tokens[row as usize];
        if token >= shape.vocab {
            return;
        }
        let output_base = row * shape.width;
        let source_base = token * shape.width;
        let mut column = thread::threadIdx_x();
        while column < shape.width {
            let destination = (output_base + column) as usize;
            let source = (source_base + column) as usize;
            unsafe { *output.get_unchecked_mut(destination) = weights[source] };
            column += FBT_THREADS;
        }
    }

    #[kernel]
    #[launch_bounds(FBT_THREADS)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn residual(input: &[u16], branch: &[u16], mut output: DisjointSlice<u16>, elements: u32) {
        let mut index = thread::threadIdx_x() + thread::blockIdx_x() * FBT_THREADS;
        let stride = thread::gridDim_x() * FBT_THREADS;
        while index < elements {
            let value = half(input[index as usize]) + half(branch[index as usize]);
            unsafe { *output.get_unchecked_mut(index as usize) = half_bits(value) };
            index += stride;
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_bounds(FBT_THREADS)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn feedback_shift(
        history: &[u16],
        mut previous: DisjointSlice<u16>,
        mut fused: DisjointSlice<u32>,
        shape: FeedbackShape,
    ) {
        let row = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        if row >= shape.rows {
            return;
        }
        let active = row % shape.context != 0;
        if tid == 0 {
            unsafe { *fused.get_unchecked_mut(row as usize) = active as u32 };
        }
        let mut column = tid;
        while column < shape.width {
            let destination = (row * shape.width + column) as usize;
            let value = if active {
                history[((row - 1) * shape.width + column) as usize]
            } else {
                half_bits(0.0)
            };
            unsafe { *previous.get_unchecked_mut(destination) = value };
            column += FBT_THREADS;
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_bounds(FBT_THREADS)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn feedback_norm(input: &[u16], mut output: DisjointSlice<u16>, shape: FeedbackShape) {
        static mut REDUCTION: SharedArray<f32, 256> = SharedArray::UNINIT;
        let row = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        if row >= shape.rows {
            return;
        }
        let reduction = unsafe { SharedArray::as_raw_mut_ptr(&raw mut REDUCTION) };
        let base = row * shape.width;
        let mut sum = 0.0;
        let mut column = tid;
        while column < shape.width {
            let value = half(input[(base + column) as usize]);
            sum += value * value;
            column += FBT_THREADS;
        }
        unsafe { reduction.add(tid as usize).write(sum) };
        let scale = if shape.token_unit_rms != 0 {
            (unsafe { reduce_sum(reduction) } / shape.width as f32 + shape.token_epsilon())
                .sqrt()
                .recip()
        } else {
            thread::sync_threads();
            1.0
        };
        column = tid;
        while column < shape.width {
            let index = (base + column) as usize;
            unsafe { *output.get_unchecked_mut(index) = half_bits(half(input[index]) * scale) };
            column += FBT_THREADS;
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_bounds(FBT_THREADS)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn feedback_combine(
        value: &[u16],
        logit: &[u16],
        tokens: &[u16],
        fused: &[u32],
        mut output: DisjointSlice<u16>,
        shape: FeedbackShape,
    ) {
        static mut REDUCTION: SharedArray<f32, 256> = SharedArray::UNINIT;
        let row = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        if row >= shape.rows {
            return;
        }
        let base = row * shape.width;
        if fused[row as usize] == 0 {
            let mut column = tid;
            while column < shape.width {
                let index = (base + column) as usize;
                unsafe { *output.get_unchecked_mut(index) = tokens[index] };
                column += FBT_THREADS;
            }
            return;
        }
        let reduction = unsafe { SharedArray::as_raw_mut_ptr(&raw mut REDUCTION) };
        let mut sum = 0.0;
        let mut column = tid;
        while column < shape.width {
            let index = (base + column) as usize;
            let combined = half(value[index]) * sigmoid(half(logit[index]));
            sum += combined * combined;
            column += FBT_THREADS;
        }
        unsafe { reduction.add(tid as usize).write(sum) };
        let scale = if shape.fused_unit_rms != 0 {
            (unsafe { reduce_sum(reduction) } / shape.width as f32 + shape.fused_epsilon())
                .sqrt()
                .recip()
        } else {
            thread::sync_threads();
            1.0
        };
        column = tid;
        while column < shape.width {
            let index = (base + column) as usize;
            let combined = half(value[index]) * sigmoid(half(logit[index]));
            unsafe { *output.get_unchecked_mut(index) = half_bits(combined * scale) };
            column += FBT_THREADS;
        }
    }

    #[kernel]
    #[launch_bounds(FBT_THREADS)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn rms(input: &[u16], weight: &[u16], mut output: DisjointSlice<u16>, shape: FbtShape) {
        static mut REDUCTION: SharedArray<f32, 256> = SharedArray::UNINIT;
        let row = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        if row >= shape.rows {
            return;
        }
        let base = row as usize * shape.width as usize;
        let mut sum = 0.0;
        let mut column = tid;
        while column < shape.width {
            let value = half(input[base + column as usize]);
            sum += value * value;
            column += FBT_THREADS;
        }
        let reduction = unsafe { SharedArray::as_raw_mut_ptr(&raw mut REDUCTION) };
        unsafe { reduction.add(tid as usize).write(sum) };
        let total = unsafe { reduce_sum(reduction) };
        let scale = (total / shape.width as f32 + shape.epsilon())
            .sqrt()
            .recip();
        column = tid;
        while column < shape.width {
            let value = half(input[base + column as usize]) * scale * half(weight[column as usize]);
            unsafe { *output.get_unchecked_mut(base + column as usize) = half_bits(value) };
            column += FBT_THREADS;
        }
    }

    /// Portable tiled FP16 matmul. The backend schedule may replace this with
    /// an sm_75 tensor-core specialization without changing the program ABI.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(FBT_THREADS)]
    #[launch_contract(domain = 2, block = (16, 16, 1), dynamic_shared = 0)]
    pub fn matmul(
        input: &[u16],
        weight: &[u16],
        mut output: DisjointSlice<u16, thread::Runtime2DIndex>,
        shape: MatmulShape,
    ) {
        static mut LEFT: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut RIGHT: SharedArray<f32, 256> = SharedArray::UNINIT;
        let x = thread::threadIdx_x();
        let y = thread::threadIdx_y();
        let row = thread::blockIdx_y() * TILE + y;
        let column = thread::blockIdx_x() * TILE + x;
        let index = y * TILE + x;
        let left = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LEFT) };
        let right = unsafe { SharedArray::as_raw_mut_ptr(&raw mut RIGHT) };
        let mut value = 0.0;
        let mut start = 0;
        while start < shape.inner {
            let inner_x = start + x;
            let inner_y = start + y;
            let a = if row < shape.rows && inner_x < shape.inner {
                let offset = shape.input_offset
                    + u64::from(row) * u64::from(shape.input_stride)
                    + u64::from(inner_x);
                half(input[offset as usize])
            } else {
                0.0
            };
            let b = if inner_y < shape.inner && column < shape.columns {
                let offset = shape.weight_offset
                    + u64::from(inner_y) * u64::from(shape.weight_stride)
                    + u64::from(column);
                half(weight[offset as usize])
            } else {
                0.0
            };
            unsafe {
                left.add(index as usize).write(a);
                right.add(index as usize).write(b);
            }
            thread::sync_threads();
            let mut k = 0;
            while k < TILE && start + k < shape.inner {
                unsafe {
                    value += left.add((y * TILE + k) as usize).read()
                        * right.add((k * TILE + x) as usize).read();
                }
                k += 1;
            }
            thread::sync_threads();
            start += TILE;
        }
        if row < shape.rows && column < shape.columns {
            let offset = shape.output_offset
                + u64::from(row) * u64::from(shape.output_stride)
                + u64::from(column);
            unsafe { *output.get_unchecked_mut(offset as usize) = half_bits(value) };
        }
    }

    /// Turing tensor-core FP16 matmul. Eight warps share a 64x16 block tile;
    /// each warp owns 16x8 outputs and accumulates them in FP32.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(FBT_THREADS)]
    #[launch_contract(domain = 2, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn matmul_turing(
        input: &[u16],
        weight: &[u16],
        mut output: DisjointSlice<u16, thread::Runtime2DIndex>,
        shape: MatmulShape,
    ) {
        turing::<256>(
            input,
            weight,
            output.as_mut_ptr(),
            shape,
            thread::blockIdx_y() * 64,
        );
    }

    /// H-Net local readout on SM75 tensor cores. Unlike the generic matmul,
    /// this retains FP32 accumulators for exact online NLL and sampling.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(FBT_THREADS)]
    #[launch_contract(domain = 2, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn hnet_wave_gemm(
        input: &[u16],
        weight: &[u16],
        mut output: DisjointSlice<f32, thread::Runtime2DIndex>,
        shape: WaveShape,
    ) {
        turing_f32::<256>(
            input,
            weight,
            output.as_mut_ptr(),
            shape.blocks,
            thread::blockIdx_y() * 64,
        );
    }

    /// Fused H-Net readout for SM75. A CTA owns 64 coarse rows and a
    /// 128-token vocabulary shard. Queries and the finished logits stay in
    /// shared memory; only the sufficient statistics needed by the global
    /// shard merge reach device memory.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(FBT_THREADS)]
    #[launch_contract(domain = 2, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn hnet_wave_fused_stats(
        input: &[u16],
        weight: &[u16],
        seeds: &[u64],
        expected: &[u32],
        mut partials: DisjointSlice<ShardStats, thread::Runtime2DIndex>,
        shape: WaveShape,
    ) {
        static mut QUERY: SharedArray<u16, 4096> = SharedArray::UNINIT;
        static mut WEIGHT: SharedArray<u16, 1024> = SharedArray::UNINIT;
        static mut LOGITS: SharedArray<f32, 8192> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let shard = thread::blockIdx_x();
        let block_start = thread::blockIdx_y() * 64;
        let query = unsafe { SharedArray::as_raw_mut_ptr(&raw mut QUERY) };
        let weights = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WEIGHT) };
        let logits = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LOGITS) };

        let mut index = tid;
        while index < 4096 {
            let local_row = index >> 6;
            let dimension = index & 63;
            let block = block_start + local_row;
            unsafe {
                query.add(index as usize).write(if block < shape.blocks {
                    input[(block * 64 + dimension) as usize]
                } else {
                    0
                });
            }
            index += FBT_THREADS;
        }
        thread::sync_threads();

        let warp = tid >> 5;
        let lane = tid & 31;
        let group = lane >> 2;
        let pair = (lane & 3) << 1;
        let warp_row = warp >> 1;
        let warp_column = warp & 1;
        let local_row0 = warp_row * 16 + group;
        let local_row1 = local_row0 + 8;
        let local_column0 = warp_column * 8 + pair;
        let local_column1 = local_column0 + 1;
        let fragment_column = warp_column * 8 + group;
        let vocabulary_start = shard * 128;

        let mut panel = 0_u32;
        while panel < 8 {
            index = tid;
            while index < 1024 {
                let inner = index >> 4;
                let column = index & 15;
                let token = vocabulary_start + panel * 16 + column;
                unsafe {
                    weights
                        .add(index as usize)
                        .write(weight[inner as usize * VOCAB as usize + token as usize]);
                }
                index += FBT_THREADS;
            }
            thread::sync_threads();

            let mut c0 = 0.0_f32;
            let mut c1 = 0.0_f32;
            let mut c2 = 0.0_f32;
            let mut c3 = 0.0_f32;
            let mut start = 0_u32;
            while start < 64 {
                let a0_offset = local_row0 * 64 + start + pair;
                let a1_offset = local_row1 * 64 + start + pair;
                let b_offset = (start + pair) * 16 + fragment_column;
                let a0 = unsafe {
                    u32::from(query.add(a0_offset as usize).read())
                        | (u32::from(query.add((a0_offset + 1) as usize).read()) << 16)
                };
                let a1 = unsafe {
                    u32::from(query.add(a1_offset as usize).read())
                        | (u32::from(query.add((a1_offset + 1) as usize).read()) << 16)
                };
                let b = unsafe {
                    u32::from(weights.add(b_offset as usize).read())
                        | (u32::from(weights.add((b_offset + 16) as usize).read()) << 16)
                };
                unsafe {
                    ptx_asm!(
                        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};",
                        inout("+f") c0,
                        inout("+f") c1,
                        inout("+f") c2,
                        inout("+f") c3,
                        in("r") a0,
                        in("r") a1,
                        in("r") b,
                        options(register_only),
                    );
                }
                start += 8;
            }
            let panel_column = panel * 16;
            unsafe {
                logits
                    .add((local_row0 * 128 + panel_column + local_column0) as usize)
                    .write(c0);
                logits
                    .add((local_row0 * 128 + panel_column + local_column1) as usize)
                    .write(c1);
                logits
                    .add((local_row1 * 128 + panel_column + local_column0) as usize)
                    .write(c2);
                logits
                    .add((local_row1 * 128 + panel_column + local_column1) as usize)
                    .write(c3);
            }
            thread::sync_threads();
            panel += 1;
        }

        if tid < 64 {
            let block = block_start + tid;
            if block < shape.blocks && block * 64 + shape.intra < shape.tokens {
                let position = block * 64 + shape.intra;
                let expected_token = expected[position as usize];
                let row_offset = tid * 128;
                let mut maximum = f32::NEG_INFINITY;
                let mut target = f32::NEG_INFINITY;
                let mut best = f32::NEG_INFINITY;
                let mut selected = vocabulary_start;
                let mut column = 0_u32;
                while column < 128 {
                    let token = vocabulary_start + column;
                    let value = unsafe { logits.add((row_offset + column) as usize).read() };
                    maximum = maximum.max(value);
                    if shape.score != 0 && token == expected_token {
                        target = value;
                    }
                    if shape.temperature != 1.0 {
                        let mut sample = value;
                        if shape.temperature > 0.0 {
                            let mut bits =
                                seeds[0] ^ (u64::from(position) << 32) ^ u64::from(token);
                            bits = bits.wrapping_add(0x9e3779b97f4a7c15);
                            bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                            bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d049bb133111eb);
                            bits ^= bits >> 31;
                            let uniform = ((bits >> 41) as f32 + 0.5) / 8388608.0;
                            sample -= shape.temperature * (-uniform.ln()).ln();
                        }
                        if sample > best || (sample == best && token < selected) {
                            best = sample;
                            selected = token;
                        }
                    }
                    column += 1;
                }

                let weighted_reservoir = shape.temperature == 1.0;
                let mut sum = 0.0_f32;
                column = 0;
                while column < 128 {
                    let token = vocabulary_start + column;
                    let value = unsafe { logits.add((row_offset + column) as usize).read() };
                    let weight_value = (value - maximum).exp();
                    let next_sum = sum + weight_value;
                    if weighted_reservoir {
                        let mut bits = seeds[0]
                            ^ (u64::from(position) << 32)
                            ^ u64::from(token)
                            ^ 0x243f6a8885a308d3;
                        bits = bits.wrapping_add(0x9e3779b97f4a7c15);
                        bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                        bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d049bb133111eb);
                        bits ^= bits >> 31;
                        let uniform = ((bits >> 41) as f32 + 0.5) / 8388608.0;
                        if uniform * next_sum < weight_value {
                            selected = token;
                        }
                    }
                    sum = next_sum;
                    column += 1;
                }
                unsafe {
                    *partials.get_unchecked_mut(
                        block as usize * shape.shards as usize + shard as usize,
                    ) = ShardStats {
                        maximum,
                        sum,
                        target,
                        best,
                        token: selected,
                        reserved: 0,
                    };
                }
            }
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_bounds(FBT_THREADS)]
    #[launch_contract(domain = 2, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn routed_project(
        input: &[u16],
        weights: &[u16],
        routes: &[RoutedTile],
        mut output: DisjointSlice<u16, thread::Runtime2DIndex>,
        mode: u32,
    ) {
        let tile = thread::blockIdx_y();
        routed::<256>(input, weights, routes, output.as_mut_ptr(), mode, tile, 0);
    }

    #[kernel(unchecked_indexing)]
    #[launch_bounds(64)]
    #[launch_contract(domain = 2, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn routed_compact(
        input: &[u16],
        weights: &[u16],
        routes: &[RoutedTile],
        mut output: DisjointSlice<u16, thread::Runtime2DIndex>,
        mode: u32,
    ) {
        let block = thread::blockIdx_y();
        let tile = block / 4;
        let first = (block % 4) * 16;
        routed::<64>(
            input,
            weights,
            routes,
            output.as_mut_ptr(),
            mode,
            tile,
            first,
        );
    }

    #[inline(always)]
    fn routed<const THREADS: u32>(
        input: &[u16],
        weights: &[u16],
        routes: &[RoutedTile],
        output: *mut u16,
        mode: u32,
        tile: u32,
        first: u32,
    ) {
        let route = routes[tile as usize];
        if first >= route.valid_rows {
            return;
        }
        let (columns, inner, input_stride) = if mode == 0 {
            (432, 512, 512)
        } else {
            (512, 216, 224)
        };
        let expert = if route.reserved == 1 {
            0
        } else {
            route.expert + 1
        };
        let shape = MatmulShape {
            rows: route.valid_rows,
            columns,
            inner,
            input_stride,
            weight_stride: columns,
            output_stride: columns,
            input_offset: u64::from(route.first_row) * u64::from(input_stride),
            weight_offset: u64::from(expert) * u64::from(inner) * u64::from(columns),
            output_offset: u64::from(route.first_row) * u64::from(columns),
        };
        turing::<THREADS>(input, weights, output, shape, first);
    }

    #[kernel(unchecked_indexing)]
    #[launch_bounds(FBT_THREADS)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn routed_activate(projection: &[u16], mut output: DisjointSlice<u16>, rows: u32) {
        let index = thread::blockIdx_x() * FBT_THREADS + thread::threadIdx_x();
        if index >= rows * 216 {
            return;
        }
        let row = index / 216;
        let column = index % 216;
        let gate = half(projection[(row * 432 + column) as usize]);
        let up = half(projection[(row * 432 + 216 + column) as usize]);
        unsafe {
            *output.get_unchecked_mut((row * 224 + column) as usize) = half_bits(silu(gate) * up);
        }
    }

    #[inline(always)]
    fn turing<const THREADS: u32>(
        input: &[u16],
        weight: &[u16],
        output: *mut u16,
        shape: MatmulShape,
        block_row: u32,
    ) {
        static mut LEFT_HALF: SharedArray<u16, 512> = SharedArray::UNINIT;
        static mut RIGHT_HALF: SharedArray<u16, 128> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x();
        let warp = tid >> 5;
        let lane = tid & 31;
        let group = lane >> 2;
        let pair = (lane & 3) << 1;
        let warp_row = warp >> 1;
        let warp_column = warp & 1;

        let block_column = thread::blockIdx_x() * 16;
        let tile_row = block_row + warp_row * 16;
        let tile_column = block_column + warp_column * 8;
        let row0 = tile_row + group;
        let row1 = row0 + 8;
        let column0 = tile_column + pair;
        let column1 = column0 + 1;
        let mut c0 = 0.0_f32;
        let mut c1 = 0.0_f32;
        let mut c2 = 0.0_f32;
        let mut c3 = 0.0_f32;
        let left = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LEFT_HALF) };
        let right = unsafe { SharedArray::as_raw_mut_ptr(&raw mut RIGHT_HALF) };
        let mut start = 0;
        while start < shape.inner {
            let mut index = tid;
            while index < THREADS * 2 {
                let local_row = index >> 3;
                let local_inner = index & 7;
                let row = block_row + local_row;
                let inner = start + local_inner;
                if row < shape.rows && inner < shape.inner {
                    let offset = shape.input_offset
                        + u64::from(row) * u64::from(shape.input_stride)
                        + u64::from(inner);
                    unsafe { left.add(index as usize).write(input[offset as usize]) };
                } else {
                    unsafe { left.add(index as usize).write(0) };
                }
                index += THREADS;
            }
            let mut index = tid;
            while index < 128 {
                let local_inner = index >> 4;
                let local_column = index & 15;
                let inner = start + local_inner;
                let column = block_column + local_column;
                if inner < shape.inner && column < shape.columns {
                    let offset = shape.weight_offset
                        + u64::from(inner) * u64::from(shape.weight_stride)
                        + u64::from(column);
                    unsafe { right.add(index as usize).write(weight[offset as usize]) };
                } else {
                    unsafe { right.add(index as usize).write(0) };
                }
                index += THREADS;
            }
            thread::sync_threads();
            let local_row0 = warp_row * 16 + group;
            let local_row1 = local_row0 + 8;
            let local_column = warp_column * 8 + group;
            let a0_offset = local_row0 * 8 + pair;
            let a1_offset = local_row1 * 8 + pair;
            let b_offset = pair * 16 + local_column;
            let a0 = unsafe {
                u32::from(left.add(a0_offset as usize).read())
                    | (u32::from(left.add((a0_offset + 1) as usize).read()) << 16)
            };
            let a1 = unsafe {
                u32::from(left.add(a1_offset as usize).read())
                    | (u32::from(left.add((a1_offset + 1) as usize).read()) << 16)
            };
            let b = unsafe {
                u32::from(right.add(b_offset as usize).read())
                    | (u32::from(right.add((b_offset + 16) as usize).read()) << 16)
            };
            // SAFETY: every lane in the warp executes the same SM75 MMA with
            // fragments laid out according to PTX m16n8k8 row/column rules.
            unsafe {
                ptx_asm!(
                    "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};",
                    inout("+f") c0,
                    inout("+f") c1,
                    inout("+f") c2,
                    inout("+f") c3,
                    in("r") a0,
                    in("r") a1,
                    in("r") b,
                    options(register_only),
                );
            }
            thread::sync_threads();
            start += 8;
        }
        let store = |row: u32, column: u32, value: f32| {
            if row < shape.rows && column < shape.columns {
                let offset = shape.output_offset
                    + u64::from(row) * u64::from(shape.output_stride)
                    + u64::from(column);
                unsafe { output.add(offset as usize).write(half_bits(value)) };
            }
        };
        store(row0, column0, c0);
        store(row0, column1, c1);
        store(row1, column0, c2);
        store(row1, column1, c3);
    }

    #[inline(always)]
    fn turing_f32<const THREADS: u32>(
        input: &[u16],
        weight: &[u16],
        output: *mut f32,
        rows: u32,
        block_row: u32,
    ) {
        static mut LEFT_HALF_F32: SharedArray<u16, 512> = SharedArray::UNINIT;
        static mut RIGHT_HALF_F32: SharedArray<u16, 128> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x();
        let warp = tid >> 5;
        let lane = tid & 31;
        let group = lane >> 2;
        let pair = (lane & 3) << 1;
        let warp_row = warp >> 1;
        let warp_column = warp & 1;
        let block_column = thread::blockIdx_x() * 16;
        let tile_row = block_row + warp_row * 16;
        let tile_column = block_column + warp_column * 8;
        let row0 = tile_row + group;
        let row1 = row0 + 8;
        let column0 = tile_column + pair;
        let column1 = column0 + 1;
        let mut c0 = 0.0_f32;
        let mut c1 = 0.0_f32;
        let mut c2 = 0.0_f32;
        let mut c3 = 0.0_f32;
        let left = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LEFT_HALF_F32) };
        let right = unsafe { SharedArray::as_raw_mut_ptr(&raw mut RIGHT_HALF_F32) };
        let mut start = 0_u32;
        while start < 64 {
            let mut index = tid;
            while index < THREADS * 2 {
                let local_row = index >> 3;
                let local_inner = index & 7;
                let row = block_row + local_row;
                let inner = start + local_inner;
                unsafe {
                    left.add(index as usize).write(if row < rows {
                        input[(row * 64 + inner) as usize]
                    } else {
                        0
                    });
                }
                index += THREADS;
            }
            index = tid;
            while index < 128 {
                let local_inner = index >> 4;
                let local_column = index & 15;
                unsafe {
                    right.add(index as usize).write(
                        weight[(start + local_inner) as usize * VOCAB as usize
                            + (block_column + local_column) as usize],
                    );
                }
                index += THREADS;
            }
            thread::sync_threads();
            let local_row0 = warp_row * 16 + group;
            let local_row1 = local_row0 + 8;
            let local_column = warp_column * 8 + group;
            let a0_offset = local_row0 * 8 + pair;
            let a1_offset = local_row1 * 8 + pair;
            let b_offset = pair * 16 + local_column;
            let a0 = unsafe {
                u32::from(left.add(a0_offset as usize).read())
                    | (u32::from(left.add((a0_offset + 1) as usize).read()) << 16)
            };
            let a1 = unsafe {
                u32::from(left.add(a1_offset as usize).read())
                    | (u32::from(left.add((a1_offset + 1) as usize).read()) << 16)
            };
            let b = unsafe {
                u32::from(right.add(b_offset as usize).read())
                    | (u32::from(right.add((b_offset + 16) as usize).read()) << 16)
            };
            unsafe {
                ptx_asm!(
                    "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};",
                    inout("+f") c0,
                    inout("+f") c1,
                    inout("+f") c2,
                    inout("+f") c3,
                    in("r") a0,
                    in("r") a1,
                    in("r") b,
                    options(register_only),
                );
            }
            thread::sync_threads();
            start += 8;
        }
        let store = |row: u32, column: u32, value: f32| {
            if row < rows {
                unsafe {
                    output
                        .add(row as usize * VOCAB as usize + column as usize)
                        .write(value)
                };
            }
        };
        store(row0, column0, c0);
        store(row0, column1, c1);
        store(row1, column0, c2);
        store(row1, column1, c3);
    }

    #[kernel(unchecked_indexing)]
    #[launch_bounds(64)]
    #[launch_contract(domain = 2, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn pisaleaves(
        qkv: &[u16],
        mut pyramid: DisjointSlice<u16, thread::Runtime2DIndex>,
        shape: PisaShape,
    ) {
        let leaf = thread::blockIdx_x();
        let sequence = thread::blockIdx_y();
        let dim = thread::threadIdx_x();
        if leaf >= shape.leaves()
            || sequence >= shape.rows / shape.context
            || dim >= shape.head_width
        {
            return;
        }
        let first = sequence * shape.context + leaf * shape.block_tokens;
        let mut sum = 0.0;
        let mut token = 0;
        while token < shape.block_tokens {
            let offset = u64::from(first + token) * u64::from(shape.kv_stride)
                + u64::from(shape.key_offset + dim);
            sum += half(qkv[offset as usize]);
            token += 1;
        }
        let destination = (u64::from(sequence) * u64::from(shape.nodes()) + u64::from(leaf))
            * u64::from(shape.head_width)
            + u64::from(dim);
        unsafe {
            *pyramid.get_unchecked_mut(destination as usize) =
                half_bits(sum / shape.block_tokens as f32)
        };
    }

    #[kernel(unchecked_indexing)]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn cache_leaves(kv: &[u16], mut tree: DisjointSlice<u16>, first: u32, count: u32) {
        let group = thread::blockIdx_x();
        let dim = thread::threadIdx_x();
        if group >= count || dim >= 64 {
            return;
        }
        let leaf = first + group;
        let mut sum = 0.0;
        for token in 0..64 {
            sum += half(kv[((leaf * 64 + token) * 128 + dim) as usize]);
        }
        unsafe { *tree.get_unchecked_mut((leaf * 64 + dim) as usize) = half_bits(sum / 64.0) };
    }

    /// Separate the current chunk's queries from persistent absolute-position KV.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn cache_split(
        qkv: &[u16],
        mut queries: DisjointSlice<u16>,
        mut kv: DisjointSlice<u16>,
        rows: u32,
        first: u32,
    ) {
        let index = thread::blockIdx_x() * 256 + thread::threadIdx_x();
        if index >= rows * 640 {
            return;
        }
        let row = index / 640;
        let column = index % 640;
        unsafe {
            if column < 512 {
                *queries.get_unchecked_mut((row * 512 + column) as usize) = qkv[index as usize];
            } else {
                *kv.get_unchecked_mut(((first + row) * 128 + column - 512) as usize) =
                    qkv[index as usize];
            }
        }
    }

    /// A bounded chunk changes only a narrow path through the tree. Each lane
    /// owns one summary dimension at every level, avoiding inter-level launches.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn cache_ancestors(mut tree: DisjointSlice<u16>, leaves: u32, first: u32, end: u32) {
        let dim = thread::threadIdx_x();
        let data = tree.as_mut_ptr();
        let (mut first, mut end) = (first, end);
        let (mut child, mut count, mut parent) = (0, leaves, leaves);
        while count > 1 {
            first /= 2;
            end = end.div_ceil(2);
            for node in first..end {
                let left = ((child + 2 * node) * 64 + dim) as usize;
                let output = ((parent + node) * 64 + dim) as usize;
                // SAFETY: this lane owns dimension dim throughout the tree;
                // all preceding child levels have already been computed.
                unsafe {
                    let value =
                        0.5 * (half(data.add(left).read()) + half(data.add(left + 64).read()));
                    data.add(output).write(half_bits(value));
                }
            }
            child = parent;
            count /= 2;
            parent += count;
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn cache_parents(
        mut tree: DisjointSlice<u16>,
        child: u32,
        parent: u32,
        first: u32,
        count: u32,
    ) {
        let group = thread::blockIdx_x();
        let dim = thread::threadIdx_x();
        if group >= count || dim >= 64 {
            return;
        }
        let node = first + group;
        let left = ((child + 2 * node) * 64 + dim) as usize;
        let output = ((parent + node) * 64 + dim) as usize;
        let data = tree.as_mut_ptr();
        // SAFETY: the preceding stream dispatch completed this child level.
        let value =
            unsafe { 0.5 * (half(data.add(left).read()) + half(data.add(left + 64).read())) };
        unsafe { *tree.get_unchecked_mut(output) = half_bits(value) };
    }

    #[kernel(unchecked_indexing)]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn pisaupper(mut pyramid: DisjointSlice<u16>, shape: PisaShape) {
        let sequence = thread::blockIdx_x();
        let dim = thread::threadIdx_x();
        if sequence >= shape.rows / shape.context || dim >= shape.head_width {
            return;
        }
        let base = u64::from(sequence) * u64::from(shape.nodes()) * u64::from(shape.head_width);
        let mut child_offset = 0;
        let mut parent_offset = shape.leaves();
        let mut child_count = shape.leaves();
        while child_count > 1 {
            let parent_count = child_count / 2;
            let mut parent = 0;
            while parent < parent_count {
                let left = base
                    + u64::from(child_offset + 2 * parent) * u64::from(shape.head_width)
                    + u64::from(dim);
                let right = left + u64::from(shape.head_width);
                let destination = base
                    + u64::from(parent_offset + parent) * u64::from(shape.head_width)
                    + u64::from(dim);
                let data = pyramid.as_mut_ptr();
                // SAFETY: both children are in the previous, complete tree
                // level. The barrier below completes this level before any
                // thread reads it as a child in the next iteration.
                let value = unsafe {
                    0.5 * (half(data.add(left as usize).read())
                        + half(data.add(right as usize).read()))
                };
                unsafe { *pyramid.get_unchecked_mut(destination as usize) = half_bits(value) };
                parent += 1;
            }
            thread::sync_threads();
            child_offset = parent_offset;
            parent_offset += parent_count;
            child_count = parent_count;
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn pisa_select(
        qkv: &[u16],
        pyramid: &[u16],
        mut blocks: DisjointSlice<u32>,
        shape: PisaShape,
    ) {
        static mut QUERY: SharedArray<f32, PISA_WIDTH> = SharedArray::UNINIT;
        static mut SCORES: SharedArray<f32, PISA_CANDIDATES> = SharedArray::UNINIT;
        static mut CANDIDATES: SharedArray<u32, PISA_CANDIDATES> = SharedArray::UNINIT;
        static mut CHOSEN: SharedArray<u32, PISA_SELECTED> = SharedArray::UNINIT;
        static mut EXPANDED: SharedArray<u32, PISA_CANDIDATES> = SharedArray::UNINIT;
        let row = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        if row >= shape.rows {
            return;
        }
        let query = unsafe { SharedArray::as_raw_mut_ptr(&raw mut QUERY) };
        let scores = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCORES) };
        let candidates = unsafe { SharedArray::as_raw_mut_ptr(&raw mut CANDIDATES) };
        let chosen = unsafe { SharedArray::as_raw_mut_ptr(&raw mut CHOSEN) };
        let expanded = unsafe { SharedArray::as_raw_mut_ptr(&raw mut EXPANDED) };
        if tid < shape.head_width {
            let mut sum = 0.0;
            let mut head = 0;
            while head < shape.heads {
                let offset = u64::from(row) * u64::from(shape.query_stride)
                    + u64::from(head * shape.head_width + tid);
                sum += half(qkv[offset as usize]);
                head += 1;
            }
            unsafe { query.add(tid as usize).write(sum) };
        }
        if tid < PISA_CANDIDATES as u32 {
            unsafe { candidates.add(tid as usize).write(tid) };
        }
        thread::sync_threads();
        let absolute = shape.query_start + row % shape.context;
        let position = if shape.visibility == 0 {
            absolute
        } else {
            ((absolute / shape.visibility + 1) * shape.visibility).min(shape.context) - 1
        };
        let current = position / shape.block_tokens;
        let previous = if current == 0 { 0 } else { current - 1 };
        let sequence = row / shape.context;
        let mut level = shape.leaves().trailing_zeros() - 4;
        loop {
            let offset = 2 * shape.leaves() - (2 * shape.leaves() >> level);
            if tid < PISA_CANDIDATES as u32 {
                let node = unsafe { candidates.add(tid as usize).read() };
                let mut score = f32::NEG_INFINITY;
                if node != u32::MAX {
                    let span = 1 << level;
                    let last = (node + 1) * span - 1;
                    if contains_leaf(node, level, 0)
                        || contains_leaf(node, level, previous)
                        || contains_leaf(node, level, current)
                    {
                        score = f32::INFINITY;
                    } else if last < current {
                        score = 0.0;
                        let summary = (u64::from(sequence) * u64::from(shape.nodes())
                            + u64::from(offset + node))
                            * u64::from(shape.head_width);
                        let mut dim = 0;
                        while dim < shape.head_width {
                            score += unsafe { query.add(dim as usize).read() }
                                * half(pyramid[(summary + u64::from(dim)) as usize]);
                            dim += 1;
                        }
                    }
                }
                unsafe { scores.add(tid as usize).write(score) };
            }
            thread::sync_threads();
            if tid == 0 {
                let mut max_finite = f32::NEG_INFINITY;
                if level == 0 {
                    let mut candidate = 0;
                    while candidate < PISA_CANDIDATES as u32 {
                        let score = unsafe { scores.add(candidate as usize).read() };
                        if score.is_finite() {
                            max_finite = max_finite.max(score);
                        }
                        candidate += 1;
                    }
                }
                let mut slot = 0;
                while slot < shape.selected {
                    let mut best = u32::MAX;
                    let mut best_score = f32::NEG_INFINITY;
                    let mut candidate = 0;
                    while candidate < PISA_CANDIDATES as u32 {
                        let score = unsafe { scores.add(candidate as usize).read() };
                        let node = unsafe { candidates.add(candidate as usize).read() };
                        if score > best_score
                            || (score == best_score
                                && score > f32::NEG_INFINITY
                                && (best == u32::MAX
                                    || node < unsafe { candidates.add(best as usize).read() }))
                        {
                            best = candidate;
                            best_score = score;
                        }
                        candidate += 1;
                    }
                    let node = if best == u32::MAX
                        || (level == 0
                            && max_finite > f32::NEG_INFINITY
                            && best_score < max_finite - 12.0)
                    {
                        u32::MAX
                    } else {
                        unsafe { candidates.add(best as usize).read() }
                    };
                    unsafe { chosen.add(slot as usize).write(node) };
                    if best != u32::MAX {
                        unsafe { scores.add(best as usize).write(f32::NEG_INFINITY) };
                    }
                    slot += 1;
                }
                if level > 0 {
                    let mut slot = 0;
                    while slot < shape.selected {
                        let node = unsafe { chosen.add(slot as usize).read() };
                        unsafe {
                            expanded
                                .add((2 * slot) as usize)
                                .write(if node == u32::MAX { u32::MAX } else { 2 * node });
                            expanded
                                .add((2 * slot + 1) as usize)
                                .write(if node == u32::MAX {
                                    u32::MAX
                                } else {
                                    2 * node + 1
                                });
                        }
                        slot += 1;
                    }
                } else {
                    let mut slot = 0;
                    while slot < shape.selected {
                        unsafe {
                            *blocks.get_unchecked_mut((row * shape.selected + slot) as usize) =
                                chosen.add(slot as usize).read()
                        };
                        slot += 1;
                    }
                }
            }
            thread::sync_threads();
            if level == 0 {
                break;
            }
            if tid < PISA_CANDIDATES as u32 {
                unsafe {
                    candidates
                        .add(tid as usize)
                        .write(expanded.add(tid as usize).read())
                };
            }
            thread::sync_threads();
            level -= 1;
        }
    }

    /// Re-score the incumbent sparse support under the candidate operator.
    /// Seven sibling leaves provide local motion and one causal hash sample
    /// reserves 1/16 of the pre-indexer budget for exploration. The resulting
    /// top eight are exact with respect to this sixteen-leaf candidate set.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn pisa_shadow_refine(
        qkv: &[u16],
        pyramid: &[u16],
        incumbent: &[u32],
        mut blocks: DisjointSlice<u32>,
        visit_offset: u32,
        shape: PisaShape,
    ) {
        static mut QUERY: SharedArray<f32, PISA_WIDTH> = SharedArray::UNINIT;
        static mut SCORES: SharedArray<f32, PISA_CANDIDATES> = SharedArray::UNINIT;
        static mut CANDIDATES: SharedArray<u32, PISA_CANDIDATES> = SharedArray::UNINIT;
        let row = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        if row >= shape.rows {
            return;
        }
        let query = unsafe { SharedArray::as_raw_mut_ptr(&raw mut QUERY) };
        let scores = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCORES) };
        let candidates = unsafe { SharedArray::as_raw_mut_ptr(&raw mut CANDIDATES) };
        if tid < shape.head_width {
            let mut sum = 0.0;
            let mut head = 0;
            while head < shape.heads {
                let offset = u64::from(row) * u64::from(shape.query_stride)
                    + u64::from(head * shape.head_width + tid);
                sum += half(qkv[offset as usize]);
                head += 1;
            }
            unsafe { query.add(tid as usize).write(sum) };
        }
        let absolute = shape.query_start + row % shape.context;
        let position = if shape.visibility == 0 {
            absolute
        } else {
            ((absolute / shape.visibility + 1) * shape.visibility).min(shape.context) - 1
        };
        let current = position / shape.block_tokens;
        let previous = if current == 0 { 0 } else { current - 1 };
        let sequence = row / shape.context;
        if tid < PISA_CANDIDATES as u32 {
            let slot = tid % PISA_SELECTED as u32;
            let address = visit_offset + row * shape.selected + slot;
            let old = incumbent[address as usize];
            let candidate = if tid < PISA_SELECTED as u32 {
                old
            } else if tid + 1 < PISA_CANDIDATES as u32 {
                old ^ 1
            } else {
                let mut bits = u64::from(row) ^ (u64::from(visit_offset) << 32);
                bits = bits.wrapping_add(0x9e3779b97f4a7c15);
                bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d049bb133111eb);
                bits ^= bits >> 31;
                (bits as u32) % (current + 1)
            };
            unsafe { candidates.add(tid as usize).write(candidate) };
        }
        thread::sync_threads();
        if tid < PISA_CANDIDATES as u32 {
            let candidate = unsafe { candidates.add(tid as usize).read() };
            let mut duplicate = false;
            let mut earlier = 0;
            while earlier < tid {
                duplicate |= unsafe { candidates.add(earlier as usize).read() } == candidate;
                earlier += 1;
            }
            let mut score = f32::NEG_INFINITY;
            if !duplicate && candidate < shape.leaves() {
                if candidate == 0 || candidate == previous || candidate == current {
                    score = f32::INFINITY;
                } else if candidate < current {
                    score = 0.0;
                    let summary = (u64::from(sequence) * u64::from(shape.nodes())
                        + u64::from(candidate))
                        * u64::from(shape.head_width);
                    let mut dim = 0;
                    while dim < shape.head_width {
                        score += unsafe { query.add(dim as usize).read() }
                            * half(pyramid[(summary + u64::from(dim)) as usize]);
                        dim += 1;
                    }
                }
            }
            unsafe { scores.add(tid as usize).write(score) };
        }
        thread::sync_threads();
        if tid == 0 {
            let mut output = 0;
            while output < shape.selected {
                let mut best = u32::MAX;
                let mut best_score = f32::NEG_INFINITY;
                let mut candidate = 0;
                while candidate < PISA_CANDIDATES as u32 {
                    let score = unsafe { scores.add(candidate as usize).read() };
                    let leaf = unsafe { candidates.add(candidate as usize).read() };
                    if score > best_score
                        || (score == best_score
                            && score > f32::NEG_INFINITY
                            && (best == u32::MAX
                                || leaf < unsafe { candidates.add(best as usize).read() }))
                    {
                        best = candidate;
                        best_score = score;
                    }
                    candidate += 1;
                }
                let leaf = if best == u32::MAX {
                    current
                } else {
                    unsafe { candidates.add(best as usize).read() }
                };
                unsafe {
                    *blocks.get_unchecked_mut((row * shape.selected + output) as usize) = leaf;
                    if best != u32::MAX {
                        scores.add(best as usize).write(f32::NEG_INFINITY);
                    }
                }
                output += 1;
            }
        }
    }

    #[kernel(unchecked_indexing)]
    #[launch_bounds(ROUTE_THREADS)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn routetopk(
        scores_input: &[u16],
        mut experts_output: DisjointSlice<u32>,
        mut weights_output: DisjointSlice<u16>,
        mut margin_output: DisjointSlice<f32>,
        shape: RouteShape,
    ) {
        static mut SCORES: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut IDS: SharedArray<u32, 256> = SharedArray::UNINIT;
        static mut BEST_SCORES: SharedArray<f32, 4> = SharedArray::UNINIT;
        static mut BEST_IDS: SharedArray<u32, 4> = SharedArray::UNINIT;
        let row = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        if row >= shape.rows {
            return;
        }
        let scores = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCORES) };
        let ids = unsafe { SharedArray::as_raw_mut_ptr(&raw mut IDS) };
        let best_scores = unsafe { SharedArray::as_raw_mut_ptr(&raw mut BEST_SCORES) };
        let best_ids = unsafe { SharedArray::as_raw_mut_ptr(&raw mut BEST_IDS) };
        let mut score = f32::NEG_INFINITY;
        let mut id = tid;
        let mut expert = tid;
        while expert < shape.experts {
            let candidate = half(scores_input[(row * shape.experts + expert) as usize]);
            if candidate > score {
                score = candidate;
                id = expert;
            }
            expert += ROUTE_THREADS;
        }
        unsafe {
            scores.add(tid as usize).write(score);
            ids.add(tid as usize).write(id);
        }
        thread::sync_threads();
        let mut position = 0;
        while position < 4 {
            let mut stride = ROUTE_THREADS / 2;
            while stride > 0 {
                if tid < stride {
                    let right = tid + stride;
                    let left_score = unsafe { scores.add(tid as usize).read() };
                    let right_score = unsafe { scores.add(right as usize).read() };
                    let left_id = unsafe { ids.add(tid as usize).read() };
                    let right_id = unsafe { ids.add(right as usize).read() };
                    if right_score > left_score || (right_score == left_score && right_id < left_id)
                    {
                        unsafe {
                            scores.add(tid as usize).write(right_score);
                            ids.add(tid as usize).write(right_id);
                        }
                    }
                }
                thread::sync_threads();
                stride /= 2;
            }
            let selected = unsafe { ids.read() };
            if tid == 0 {
                unsafe {
                    best_scores.add(position as usize).write(scores.read());
                    best_ids.add(position as usize).write(selected);
                }
            }
            thread::sync_threads();
            let mut next_score = f32::NEG_INFINITY;
            let mut next_id = tid;
            let mut expert = tid;
            while expert < shape.experts {
                let mut excluded = expert == selected;
                let mut earlier = 0;
                while earlier < position {
                    excluded =
                        excluded || expert == unsafe { best_ids.add(earlier as usize).read() };
                    earlier += 1;
                }
                let candidate = half(scores_input[(row * shape.experts + expert) as usize]);
                if !excluded && candidate > next_score {
                    next_score = candidate;
                    next_id = expert;
                }
                expert += ROUTE_THREADS;
            }
            unsafe {
                scores.add(tid as usize).write(next_score);
                ids.add(tid as usize).write(next_id);
            }
            thread::sync_threads();
            position += 1;
        }
        if tid == 0 {
            let denominator = sigmoid(unsafe { best_scores.read() })
                + sigmoid(unsafe { best_scores.add(1).read() })
                + sigmoid(unsafe { best_scores.add(2).read() });
            let mut route = 0;
            while route < shape.top_k {
                let destination = (row * shape.top_k + route) as usize;
                unsafe {
                    *experts_output.get_unchecked_mut(destination) =
                        best_ids.add(route as usize).read();
                    *weights_output.get_unchecked_mut(destination) =
                        half_bits(sigmoid(best_scores.add(route as usize).read()) / denominator);
                }
                route += 1;
            }
            unsafe {
                *margin_output.get_unchecked_mut(row as usize) =
                    best_scores.add(2).read() - best_scores.add(3).read()
            };
        }
    }

    /// Fused selected-block PISA attention. One block owns one row/head and
    /// performs online-compatible FP32 softmax over the selected 64-token blocks.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(FBT_THREADS)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn pisa_attention(
        qkv: &[u16],
        kv: &[u16],
        blocks: &[u32],
        mut output: DisjointSlice<u16>,
        shape: PisaShape,
    ) {
        static mut SCORES: SharedArray<f32, PISA_SCORES> = SharedArray::UNINIT;
        static mut REDUCTION: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut QUERY: SharedArray<f32, PISA_WIDTH> = SharedArray::UNINIT;
        static mut BLOCKS: SharedArray<u32, PISA_SELECTED> = SharedArray::UNINIT;
        let group = thread::blockIdx_x();
        let row = group / shape.heads;
        let head = group % shape.heads;
        let tid = thread::threadIdx_x();
        if row >= shape.rows {
            return;
        }
        let query_width = shape.query_width();
        let kv_stride = shape.kv_stride;
        let sequence = row / shape.context;
        let absolute = shape.query_start + row % shape.context;
        let position = if shape.visibility == 0 {
            absolute
        } else {
            ((absolute / shape.visibility + 1) * shape.visibility).min(shape.context) - 1
        };
        let scores = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCORES) };
        let reduction = unsafe { SharedArray::as_raw_mut_ptr(&raw mut REDUCTION) };
        let query = unsafe { SharedArray::as_raw_mut_ptr(&raw mut QUERY) };
        let selected_blocks = unsafe { SharedArray::as_raw_mut_ptr(&raw mut BLOCKS) };
        let selected_tokens = shape.selected * shape.block_tokens;
        if tid < shape.head_width {
            let q = u64::from(row) * u64::from(shape.query_stride)
                + u64::from(head * shape.head_width + tid);
            unsafe { query.add(tid as usize).write(half(qkv[q as usize])) };
        }
        if tid < shape.selected {
            unsafe {
                selected_blocks
                    .add(tid as usize)
                    .write(blocks[(row * shape.selected + tid) as usize])
            };
        }
        thread::sync_threads();
        // A warp owns one selected token. Adjacent lanes load adjacent key
        // dimensions instead of 256 threads issuing strided scalar dot products.
        let lane = tid % 32;
        let warp_index = tid / 32;
        let mut local = warp_index;
        while local < selected_tokens {
            let slot = local / shape.block_tokens;
            let within = local % shape.block_tokens;
            let block = unsafe { selected_blocks.add(slot as usize).read() };
            let source_position = block.saturating_mul(shape.block_tokens) + within;
            let valid = block != u32::MAX && source_position <= position;
            let mut dot = 0.0_f32;
            if valid {
                let source = sequence * shape.context + source_position;
                for dimension in [lane, lane + 32] {
                    let key = (u64::from(source) * u64::from(kv_stride)
                        + u64::from(shape.key_offset + dimension))
                        as usize;
                    dot =
                        unsafe { query.add(dimension as usize).read() }.mul_add(half(kv[key]), dot);
                }
            }
            let dot = warp::reduce_sum_f32(dot);
            if lane == 0 {
                let score = if valid {
                    dot * 0.125
                } else {
                    f32::NEG_INFINITY
                };
                unsafe {
                    scores.add(local as usize).write(score);
                }
            }
            local += 8;
        }
        thread::sync_threads();
        let first = if tid < selected_tokens {
            unsafe { scores.add(tid as usize).read() }
        } else {
            f32::NEG_INFINITY
        };
        let second_index = tid + FBT_THREADS;
        let second = if second_index < selected_tokens {
            unsafe { scores.add(second_index as usize).read() }
        } else {
            f32::NEG_INFINITY
        };
        unsafe { reduction.add(tid as usize).write(first.max(second)) };
        let maximum = unsafe { reduce_max(reduction) };
        let mut subtotal = 0.0;
        for index in [tid, second_index] {
            if index < selected_tokens {
                let score = unsafe { scores.add(index as usize).read() };
                let probability = if score.is_finite() {
                    (score - maximum).exp()
                } else {
                    0.0
                };
                unsafe { scores.add(index as usize).write(probability) };
                subtotal += probability;
            }
        }
        unsafe { reduction.add(tid as usize).write(subtotal) };
        let total = unsafe { reduce_sum(reduction) }.max(f32::MIN_POSITIVE);
        // Four 64-lane groups split the value accumulation across token
        // positions, retaining adjacent value loads within each warp.
        let dim = tid % shape.head_width;
        let groups = FBT_THREADS / shape.head_width;
        let mut value = 0.0;
        let mut index = tid / shape.head_width;
        while index < selected_tokens {
            let slot = index / shape.block_tokens;
            let within = index % shape.block_tokens;
            let block = unsafe { selected_blocks.add(slot as usize).read() };
            let source_position = block.saturating_mul(shape.block_tokens) + within;
            if block != u32::MAX && source_position <= position {
                let source = sequence * shape.context + source_position;
                let v = (u64::from(source) * u64::from(kv_stride)
                    + u64::from(shape.key_offset + shape.head_width + dim))
                    as usize;
                value += unsafe { scores.add(index as usize).read() } * half(kv[v]);
            }
            index += groups;
        }
        thread::sync_threads();
        unsafe {
            reduction.add(tid as usize).write(value);
        }
        thread::sync_threads();
        if tid < shape.head_width {
            let mut sum = 0.0;
            for group in 0..groups {
                sum += unsafe {
                    reduction
                        .add((group * shape.head_width + tid) as usize)
                        .read()
                };
            }
            let destination = (u64::from(row) * u64::from(query_width)
                + u64::from(head * shape.head_width + tid)) as usize;
            unsafe { *output.get_unchecked_mut(destination) = half_bits(sum / total) };
        }
    }

    /// Selected-block MQA attention with a small query-head group per block.
    /// Groups reuse K/V while retaining occupancy on Turing GPUs.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(FBT_THREADS)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn pisa_multihead(
        qkv: &[u16],
        kv: &[u16],
        blocks: &[u32],
        mut output: DisjointSlice<u16>,
        shape: PisaShape,
    ) {
        static mut SCORES: SharedArray<f32, PISA_HEAD_SCORES> = SharedArray::UNINIT;
        static mut QUERIES: SharedArray<f32, PISA_QUERIES> = SharedArray::UNINIT;
        static mut VALUES: SharedArray<f32, PISA_VALUES> = SharedArray::UNINIT;
        static mut BLOCKS: SharedArray<u32, PISA_SELECTED> = SharedArray::UNINIT;
        static mut TOTALS: SharedArray<f32, PISA_GROUP_HEADS> = SharedArray::UNINIT;
        let group_index = thread::blockIdx_x() % PISA_HEAD_GROUPS;
        let row = thread::blockIdx_x() / PISA_HEAD_GROUPS;
        let head_base = group_index * PISA_GROUP_HEADS as u32;
        let tid = thread::threadIdx_x();
        if row >= shape.rows {
            return;
        }
        let lane = tid % 32;
        let warp_index = tid / 32;
        let scores = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCORES) };
        let queries = unsafe { SharedArray::as_raw_mut_ptr(&raw mut QUERIES) };
        let values = unsafe { SharedArray::as_raw_mut_ptr(&raw mut VALUES) };
        let selected_blocks = unsafe { SharedArray::as_raw_mut_ptr(&raw mut BLOCKS) };
        let totals = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TOTALS) };
        if tid < PISA_QUERIES as u32 {
            let index = head_base * shape.head_width + tid;
            if index < shape.heads * shape.head_width {
                let source = u64::from(row) * u64::from(shape.query_stride) + u64::from(index);
                unsafe { queries.add(tid as usize).write(half(qkv[source as usize])) };
            }
        }
        if tid < shape.selected {
            unsafe {
                selected_blocks
                    .add(tid as usize)
                    .write(blocks[(row * shape.selected + tid) as usize])
            };
        }
        thread::sync_threads();

        let selected_tokens = shape.selected * shape.block_tokens;
        let sequence = row / shape.context;
        let absolute = shape.query_start + row % shape.context;
        let position = if shape.visibility == 0 {
            absolute
        } else {
            ((absolute / shape.visibility + 1) * shape.visibility).min(shape.context) - 1
        };
        let mut local = warp_index;
        while local < selected_tokens {
            let slot = local / shape.block_tokens;
            let within = local % shape.block_tokens;
            let block = unsafe { selected_blocks.add(slot as usize).read() };
            let source_position = block.saturating_mul(shape.block_tokens) + within;
            let valid = block != u32::MAX && source_position <= position;
            let source = sequence * shape.context + source_position;
            let key0 = if valid {
                let index = u64::from(source) * u64::from(shape.kv_stride)
                    + u64::from(shape.key_offset + lane);
                half(kv[index as usize])
            } else {
                0.0
            };
            let key1 = if valid {
                let index = u64::from(source) * u64::from(shape.kv_stride)
                    + u64::from(shape.key_offset + lane + 32);
                half(kv[index as usize])
            } else {
                0.0
            };
            let mut head = 0;
            while head < PISA_GROUP_HEADS as u32 {
                let query = head * shape.head_width;
                let dot = unsafe {
                    queries.add((query + lane) as usize).read() * key0
                        + queries.add((query + lane + 32) as usize).read() * key1
                };
                let dot = warp::reduce_sum_f32(dot);
                if lane == 0 {
                    let score = if valid {
                        dot * 0.125
                    } else {
                        f32::NEG_INFINITY
                    };
                    unsafe {
                        scores
                            .add((head * selected_tokens + local) as usize)
                            .write(score)
                    };
                }
                head += 1;
            }
            local += 8;
        }
        thread::sync_threads();

        let head = warp_index;
        let mut maximum = f32::NEG_INFINITY;
        let mut index = lane;
        if head < PISA_GROUP_HEADS as u32 {
            while index < selected_tokens {
                maximum = maximum
                    .max(unsafe { scores.add((head * selected_tokens + index) as usize).read() });
                index += 32;
            }
            for offset in [16, 8, 4, 2, 1] {
                maximum = maximum.max(warp::shuffle_xor_f32(maximum, offset));
            }
            let mut subtotal = 0.0;
            index = lane;
            while index < selected_tokens {
                let address = (head * selected_tokens + index) as usize;
                let score = unsafe { scores.add(address).read() };
                let probability = if score.is_finite() {
                    (score - maximum).exp()
                } else {
                    0.0
                };
                unsafe { scores.add(address).write(probability) };
                subtotal += probability;
                index += 32;
            }
            let total = warp::reduce_sum_f32(subtotal).max(f32::MIN_POSITIVE);
            if lane == 0 {
                unsafe { totals.add(head as usize).write(total) };
            }
        }
        thread::sync_threads();

        let dim = tid % shape.head_width;
        let group = tid / shape.head_width;
        let groups = FBT_THREADS / shape.head_width;
        let mut sums = [0.0_f32; PISA_GROUP_HEADS];
        index = group;
        while index < selected_tokens {
            let slot = index / shape.block_tokens;
            let within = index % shape.block_tokens;
            let block = unsafe { selected_blocks.add(slot as usize).read() };
            let source_position = block.saturating_mul(shape.block_tokens) + within;
            if block != u32::MAX && source_position <= position {
                let source = sequence * shape.context + source_position;
                let value_index = u64::from(source) * u64::from(shape.kv_stride)
                    + u64::from(shape.key_offset + shape.head_width + dim);
                let value = half(kv[value_index as usize]);
                let mut head = 0;
                while head < PISA_GROUP_HEADS as u32 {
                    sums[head as usize] +=
                        unsafe { scores.add((head * selected_tokens + index) as usize).read() }
                            * value;
                    head += 1;
                }
            }
            index += groups;
        }
        let mut head = 0;
        while head < PISA_GROUP_HEADS as u32 {
            unsafe {
                values
                    .add((head * FBT_THREADS + tid) as usize)
                    .write(sums[head as usize])
            };
            head += 1;
        }
        thread::sync_threads();
        if tid < shape.head_width {
            let mut head = 0;
            while head < PISA_GROUP_HEADS as u32 {
                let mut sum = 0.0;
                let mut group = 0;
                while group < groups {
                    sum += unsafe {
                        values
                            .add((head * FBT_THREADS + group * shape.head_width + tid) as usize)
                            .read()
                    };
                    group += 1;
                }
                let destination = u64::from(row) * u64::from(shape.query_width())
                    + u64::from((head_base + head) * shape.head_width + tid);
                let total = unsafe { totals.add(head as usize).read() };
                unsafe { *output.get_unchecked_mut(destination as usize) = half_bits(sum / total) };
                head += 1;
            }
        }
    }

    /// Tiled routed gate/up projection with FP16 projection boundaries and a
    /// fused FP32 SiLU product. Grid X is expert columns; grid Y packs route
    /// tile and its 16-row tile as `route * row_tiles + row_tile`.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(FBT_THREADS)]
    #[launch_contract(domain = 2, block = (16, 16, 1), dynamic_shared = 0)]
    pub fn routed_gate(
        input: &[u16],
        weights: &[u16],
        routes: &[RoutedTile],
        mut activation: DisjointSlice<u16, thread::Runtime2DIndex>,
        shape: MoeShape,
    ) {
        static mut INPUT: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut GATE: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut UP: SharedArray<f32, 256> = SharedArray::UNINIT;
        let x = thread::threadIdx_x();
        let y = thread::threadIdx_y();
        let flat = y * TILE + x;
        let route_index = thread::blockIdx_y() / shape.row_tiles;
        let row_tile = thread::blockIdx_y() % shape.row_tiles;
        let route = routes[route_index as usize];
        if route.valid_rows == 0 {
            return;
        }
        let local_row = row_tile * TILE + y;
        let row = route.first_row + local_row;
        let column = thread::blockIdx_x() * TILE + x;
        let input_tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut INPUT) };
        let gate_tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut GATE) };
        let up_tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut UP) };
        let expert_stride = u64::from(shape.width) * u64::from(2 * shape.expert_width);
        let expert = u64::from(if route.reserved == 1 {
            0
        } else {
            route.expert + 1
        }) * expert_stride;
        let mut gate_value = 0.0;
        let mut up_value = 0.0;
        let mut start = 0;
        while start < shape.width {
            let inner_x = start + x;
            let inner_y = start + y;
            let a = if local_row < route.valid_rows && inner_x < shape.width {
                half(input[(u64::from(row) * u64::from(shape.width) + u64::from(inner_x)) as usize])
            } else {
                0.0
            };
            let (g, u) = if inner_y < shape.width && column < shape.expert_width {
                let base = expert
                    + u64::from(inner_y) * u64::from(2 * shape.expert_width)
                    + u64::from(column);
                (
                    half(weights[base as usize]),
                    half(weights[(base + u64::from(shape.expert_width)) as usize]),
                )
            } else {
                (0.0, 0.0)
            };
            unsafe {
                input_tile.add(flat as usize).write(a);
                gate_tile.add(flat as usize).write(g);
                up_tile.add(flat as usize).write(u);
            }
            thread::sync_threads();
            let mut k = 0;
            while k < TILE && start + k < shape.width {
                unsafe {
                    let a = input_tile.add((y * TILE + k) as usize).read();
                    gate_value += a * gate_tile.add((k * TILE + x) as usize).read();
                    up_value += a * up_tile.add((k * TILE + x) as usize).read();
                }
                k += 1;
            }
            thread::sync_threads();
            start += TILE;
        }
        if local_row < route.valid_rows && column < shape.expert_width {
            let gate_rounded = half_bits(gate_value);
            let up_rounded = half_bits(up_value);
            let value = silu(half(gate_rounded)) * half(up_rounded);
            let destination =
                u64::from(row) * u64::from(shape.activation_stride) + u64::from(column);
            unsafe { *activation.get_unchecked_mut(destination as usize) = half_bits(value) };
        }
    }

    /// Tiled routed down projection. Grid packing matches `routed_gate`.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(FBT_THREADS)]
    #[launch_contract(domain = 2, block = (16, 16, 1), dynamic_shared = 0)]
    pub fn routed_down(
        activation: &[u16],
        weights: &[u16],
        routes: &[RoutedTile],
        mut output: DisjointSlice<u16, thread::Runtime2DIndex>,
        shape: MoeShape,
    ) {
        static mut INPUT: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut WEIGHT: SharedArray<f32, 256> = SharedArray::UNINIT;
        let x = thread::threadIdx_x();
        let y = thread::threadIdx_y();
        let flat = y * TILE + x;
        let route_index = thread::blockIdx_y() / shape.row_tiles;
        let row_tile = thread::blockIdx_y() % shape.row_tiles;
        let route = routes[route_index as usize];
        if route.valid_rows == 0 {
            return;
        }
        let local_row = row_tile * TILE + y;
        let row = route.first_row + local_row;
        let column = thread::blockIdx_x() * TILE + x;
        let input_tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut INPUT) };
        let weight_tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WEIGHT) };
        let expert_stride = u64::from(shape.expert_width) * u64::from(shape.width);
        let expert = u64::from(if route.reserved == 1 {
            0
        } else {
            route.expert + 1
        }) * expert_stride;
        let mut value = 0.0;
        let mut start = 0;
        while start < shape.expert_width {
            let inner_x = start + x;
            let inner_y = start + y;
            let a = if local_row < route.valid_rows && inner_x < shape.expert_width {
                half(
                    activation[(u64::from(row) * u64::from(shape.activation_stride)
                        + u64::from(inner_x)) as usize],
                )
            } else {
                0.0
            };
            let b = if inner_y < shape.expert_width && column < shape.width {
                let offset =
                    expert + u64::from(inner_y) * u64::from(shape.width) + u64::from(column);
                half(weights[offset as usize])
            } else {
                0.0
            };
            unsafe {
                input_tile.add(flat as usize).write(a);
                weight_tile.add(flat as usize).write(b);
            }
            thread::sync_threads();
            let mut k = 0;
            while k < TILE && start + k < shape.expert_width {
                unsafe {
                    value += input_tile.add((y * TILE + k) as usize).read()
                        * weight_tile.add((k * TILE + x) as usize).read();
                }
                k += 1;
            }
            thread::sync_threads();
            start += TILE;
        }
        if local_row < route.valid_rows && column < shape.width {
            let destination = u64::from(row) * u64::from(shape.width) + u64::from(column);
            unsafe { *output.get_unchecked_mut(destination as usize) = half_bits(value) };
        }
    }

    /// Gather the three routed expert rows back into token order and add the
    /// shared expert branch. Each thread owns one output element, preserving
    /// route order and FP32 accumulation before the FP16 boundary.
    #[kernel(unchecked_indexing)]
    #[launch_bounds(FBT_THREADS)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn combine_routes(
        route_weights: &[u16],
        packed_rows: &[u32],
        routed_output: &[u16],
        shared_output: &[u16],
        mut output: DisjointSlice<u16>,
        route_shape: RouteShape,
        moe_shape: MoeShape,
    ) {
        let index = thread::blockIdx_x() * FBT_THREADS + thread::threadIdx_x();
        let elements = route_shape.rows * moe_shape.width;
        if index >= elements {
            return;
        }
        let token = index / moe_shape.width;
        let column = index % moe_shape.width;
        let mut value = half(shared_output[index as usize]);
        let mut route = 0;
        while route < route_shape.top_k {
            let assignment = token * route_shape.top_k + route;
            let packed = packed_rows[assignment as usize];
            let source = u64::from(packed) * u64::from(moe_shape.width) + u64::from(column);
            value +=
                half(route_weights[assignment as usize]) * half(routed_output[source as usize]);
            route += 1;
        }
        unsafe { *output.get_unchecked_mut(index as usize) = half_bits(value) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes() {
        assert_eq!(PisaShape::fbt(8192, 4096).unwrap().qkv_width(), 640);
        assert!(PisaShape::fbt(8191, 4096).is_err());
        assert!(FbtShape::new(8192, 512, VOCAB, 1.0e-5).is_ok());
        assert!(MoeShape::fbt().validate().is_ok());
    }

    #[test]
    fn matmul_shape() {
        let shape = MatmulShape {
            rows: 8192,
            columns: 640,
            inner: 512,
            input_stride: 512,
            weight_stride: 640,
            output_stride: 640,
            input_offset: 0,
            weight_offset: 0,
            output_offset: 0,
        };
        assert!(shape.validate().is_ok());
    }
}
