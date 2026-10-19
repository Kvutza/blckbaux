//! Activation jitter for DUST zeroth-order optimization.

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JitterParams {
    pub width: u32,
    pub rows: u32,
    pub seed: u32,
    pub sigma: f32,
}

/// Host CPU activation jitter emission for DUST.
/// Generates deterministic Rademacher activation noise.
pub fn emit_jitter(slice: &mut [f32], width: usize, rows: usize, seed: u32, sigma: f32) {
    if sigma <= 0.0 {
        return;
    }
    for row in 0..rows {
        for i in 0..width {
            let ix = row * width + i;
            if ix >= slice.len() {
                break;
            }
            let mut state = ((seed as u64) ^ (ix as u64 + 1)).wrapping_mul(0xbf58476d1ce4e5b9);
            state = (state ^ (state >> 30)).wrapping_mul(0x94d049bb133111eb);
            state ^= state >> 31;
            let noise = if (state & 1) != 0 { 1.0f32 } else { -1.0f32 };
            slice[ix] += sigma * noise;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let mut a = vec![0.0f32; 64];
        let mut b = vec![0.0f32; 64];
        emit_jitter(&mut a, 16, 4, 42, 0.01);
        emit_jitter(&mut b, 16, 4, 42, 0.01);
        assert_eq!(a, b);
        assert!(
            a.iter()
                .all(|&x| (x - 0.01).abs() < 1e-6 || (x + 0.01).abs() < 1e-6)
        );
    }

    #[test]
    fn inactive() {
        let mut a = vec![0.0f32; 64];
        emit_jitter(&mut a, 16, 4, 42, 0.0);
        assert!(a.iter().all(|&x| x == 0.0));
    }
}
