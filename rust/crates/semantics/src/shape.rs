//! Canonical fixed-patch H-Net dimensions and partial contraction work count.
pub const HNET_PATCH: u64 = 64;
pub const PISA_BYTES: u64 = 260;
pub const HNET_LAYERS: u64 = 5;
pub const HNET_WIDTH: u64 = 512;
pub const HNET_QUERY: u64 = 64;
pub const HNET_STREAMS: u64 = 4;
pub const HNET_HEADS: u64 = 8;
pub const EXPERT_WIDTH: u64 = 216;
pub const ROUTED_COUNT: u64 = 625;
pub const SHARED_COUNT: u64 = 1;
pub const EXPERT_COUNT: u64 = ROUTED_COUNT + SHARED_COUNT;
pub const ACTIVE_COUNT: u64 = 3;
pub const QKV_WIDTH: u64 = 640;
pub const GATE_WIDTH: u64 = 2 * EXPERT_WIDTH;
pub const MHC_INPUT: u64 = HNET_STREAMS * HNET_WIDTH;
pub const MHC_COEFS: u64 = HNET_STREAMS * HNET_STREAMS + 2 * HNET_STREAMS;
pub const HNET_PARAMS: u64 = HNET_LAYERS
    * (HNET_WIDTH * ROUTED_COUNT
        + HNET_WIDTH * QKV_WIDTH
        + HNET_WIDTH * HNET_WIDTH
        + EXPERT_COUNT * HNET_WIDTH * GATE_WIDTH
        + EXPERT_COUNT * EXPERT_WIDTH * HNET_WIDTH
        + 2 * HNET_WIDTH
        + 2 * MHC_INPUT * MHC_COEFS
        + 2 * MHC_COEFS
        + 8)
    + HNET_WIDTH * crate::hnet::VOCABULARY as u64
    + HNET_WIDTH
    + HNET_WIDTH
    + HNET_PATCH * HNET_WIDTH;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HnetShape {
    pub patch: u64,
    pub width: u64,
    pub query: u64,
    pub streams: u64,
    pub heads: u64,
    pub hidden: u64,
    pub experts: u64,
    pub active: u64,
    pub visits: u64,
    pub levels: u64,
    pub radix: u64,
}

impl HnetShape {
    pub const fn coeffs(self) -> u64 {
        let predictor =
            (self.streams * self.width) * (self.streams * self.streams + 2 * self.streams);
        let attention =
            self.width * ((self.heads + 2) * self.query) + (self.heads * self.query) * self.width;
        let experts = self.width * self.experts + (self.active + 1) * 3 * self.width * self.hidden;
        let encoder = self.patch * self.width;
        let decoder =
            self.width * self.query + self.patch * 2 * self.levels * self.radix * self.query;
        encoder + self.visits * (2 * predictor + attention + experts) + decoder
    }
}

pub const HNET_SHAPE: HnetShape = HnetShape {
    patch: 64,
    width: HNET_WIDTH,
    query: HNET_QUERY,
    streams: HNET_STREAMS,
    heads: HNET_HEADS,
    hidden: EXPERT_WIDTH,
    // `experts` is the routed population; the shared expert is separate.
    experts: ROUTED_COUNT,
    active: ACTIVE_COUNT,
    visits: 7,
    levels: 6,
    radix: 8,
};

/// Dense contraction coefficients in one complete fixed-patch forward pass.
/// Selector, reduction, normalization, memory, launch, and synchronization
/// work remain outside this deliberately partial count.
pub const HNET_COEFFS: u64 = HNET_SHAPE.coeffs();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_count() {
        assert_eq!(HNET_PARAMS, 1_177_756_440);
        assert_eq!(EXPERT_COUNT, 626);
        assert_eq!(QKV_WIDTH, (HNET_HEADS + 2) * HNET_QUERY);
        assert_eq!(MHC_INPUT, HNET_STREAMS * HNET_WIDTH);
        assert_eq!(MHC_COEFS, HNET_STREAMS * HNET_STREAMS + 2 * HNET_STREAMS);
    }
}
