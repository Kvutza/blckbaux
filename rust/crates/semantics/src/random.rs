//! Integer random-stream identity. Wrapping arithmetic is part of the contract.

const SM64_OFFSET: u64 = 0xD2B74407B1CE6E93;

/// Prime constant for seed combination.
const SEED_PRIME: u64 = 1_000_003;

/// 2^53 as f64 constant.
const INV_2P53: f64 = 1.0 / 9007199254740992.0;

/// Minimum clip value to avoid log(0).
const CLIP_MIN: f64 = 1e-12;

/// Maximum clip value (exclusive upper bound).
const CLIP_MAX: f64 = 1.0 - 1e-12;

/// Convert u64 to f64 in [0, 1) using high 53 bits.
#[inline(always)]
pub fn u64_f53(x: u64) -> f64 {
    ((x >> 11) as f64) * INV_2P53
}

/// Box-Muller transform for standard normal variates.
///
/// Takes two uniform [0, 1) values and produces one standard normal.
#[inline(always)]
pub fn box_muller(u1: f64, u2: f64) -> f64 {
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[inline(always)]
pub fn normal_metric(seed_u64: u64, unique_idx: i64, metric: usize) -> f64 {
    let unique_u64 = unique_idx as u64;
    let base =
        (seed_u64.wrapping_mul(SEED_PRIME).wrapping_add(unique_u64)).wrapping_mul(SEED_PRIME);
    let metric_u64 = metric as u64;
    let combined1 = base.wrapping_add(metric_u64);
    let r1 = splitmix64(combined1);
    let combined2 = combined1 ^ SM64_OFFSET;
    let r2 = splitmix64(combined2);
    let mut u1 = u64_f53(r1);
    let u2 = u64_f53(r2);
    u1 = u1.clamp(CLIP_MIN, CLIP_MAX);
    box_muller(u1, u2)
}

#[inline(always)]
pub fn splitmix64(value: u64) -> u64 {
    let value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
