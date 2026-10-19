//! Shared streaming prefix selection for precision-weighted neighbours.
//! The distance coefficient is a fitted bias proxy, not a certified Lipschitz
//! constant. No covariance matrix or gradient estimate is constructed.

/// Resident observation capacity, shared by host state and device scratch.
pub const MAX_HISTORY: usize = 128;
pub mod query;

/// Append until full, then replace the oldest observation other than the
/// incumbent. Slots retain their order: Thompson draws follow identities,
/// not the current slot number. Parameter storage must use the same slot.
pub fn retention(identities: &[i64], incumbent: i64) -> Result<usize, &'static str> {
    if identities.len() > MAX_HISTORY {
        return Err("observation history exceeds its capacity");
    }
    if identities.len() < MAX_HISTORY {
        return Ok(identities.len());
    }
    identities
        .iter()
        .enumerate()
        .filter(|(_, identity)| **identity != incumbent)
        .min_by_key(|(_, identity)| **identity)
        .map(|(slot, _)| slot)
        .ok_or("observation history has no replaceable slot")
}

#[derive(Clone, Copy, Default)]
pub struct Prefix {
    weight: f32,
    noise: f32,
    best: f32,
    pub count: u32,
}

impl Prefix {
    #[inline(always)]
    pub fn push(&mut self, rank: u32, distance: f32, epistemic: f32, noise: f32) {
        let precision = 1.0 / (1.0e-9 + epistemic * distance + noise).max(1.0e-12);
        self.weight += precision;
        self.noise += precision * precision * noise;
        let variance_error = self.noise / (self.weight * self.weight).max(1.0e-24);
        let risk = epistemic * distance + variance_error.sqrt();
        if self.count == 0 || risk < self.best {
            self.best = risk;
            self.count = rank;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bias_noise() {
        let mut noisy = Prefix::default();
        for rank in 1..=8 {
            noisy.push(rank, 1.0, 0.0, 1.0);
        }
        assert_eq!(noisy.count, 8);
        let mut separated = Prefix::default();
        separated.push(1, 0.0, 1.0, 0.01);
        separated.push(2, 100.0, 1.0, 0.01);
        assert_eq!(separated.count, 1);
    }
}
