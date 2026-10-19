//! One full-size initialized H-Net forward case, not a pretraining experiment.
use super::super::{PATCH, VOCABULARY, WIDTH};

pub const SEED: u64 = 17;
pub const ROWS: usize = 128;
pub const TOKENS: usize = 128;
pub const WORDS: usize = ROWS * WIDTH;

pub fn sequence() -> Vec<u32> {
    (0..=TOKENS)
        .map(|i| (crate::random::splitmix64(73 + i as u64) % VOCABULARY as u64) as u32)
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParameterFamily {
    pub name: &'static str,
    pub elements: u64,
}

/// Logical families in Metal's packed parameter order. CUDA's tied embedding
/// is a second physical view of embedding_readout, not another logical family.
pub const FAMILIES: [ParameterFamily; 17] = {
    use crate::shape as d;
    [
        ParameterFamily {
            name: "router",
            elements: d::HNET_LAYERS * d::HNET_WIDTH * d::ROUTED_COUNT,
        },
        ParameterFamily {
            name: "qkv",
            elements: d::HNET_LAYERS * d::HNET_WIDTH * d::QKV_WIDTH,
        },
        ParameterFamily {
            name: "attention_output",
            elements: d::HNET_LAYERS * d::HNET_WIDTH * d::HNET_WIDTH,
        },
        ParameterFamily {
            name: "expert_gate_up",
            elements: d::HNET_LAYERS * d::EXPERT_COUNT * d::HNET_WIDTH * d::GATE_WIDTH,
        },
        ParameterFamily {
            name: "expert_down",
            elements: d::HNET_LAYERS * d::EXPERT_COUNT * d::EXPERT_WIDTH * d::HNET_WIDTH,
        },
        ParameterFamily {
            name: "attention_norm",
            elements: d::HNET_LAYERS * d::HNET_WIDTH,
        },
        ParameterFamily {
            name: "ffn_norm",
            elements: d::HNET_LAYERS * d::HNET_WIDTH,
        },
        ParameterFamily {
            name: "embedding_readout",
            elements: d::HNET_WIDTH * VOCABULARY as u64,
        },
        ParameterFamily {
            name: "mhc_attention_predictor",
            elements: d::HNET_LAYERS * d::MHC_INPUT * d::MHC_COEFS,
        },
        ParameterFamily {
            name: "mhc_attention_bias",
            elements: d::HNET_LAYERS * d::MHC_COEFS,
        },
        ParameterFamily {
            name: "mhc_attention_control",
            elements: d::HNET_LAYERS * 4,
        },
        ParameterFamily {
            name: "mhc_moe_predictor",
            elements: d::HNET_LAYERS * d::MHC_INPUT * d::MHC_COEFS,
        },
        ParameterFamily {
            name: "mhc_moe_bias",
            elements: d::HNET_LAYERS * d::MHC_COEFS,
        },
        ParameterFamily {
            name: "mhc_moe_control",
            elements: d::HNET_LAYERS * 4,
        },
        ParameterFamily {
            name: "final_norm",
            elements: d::HNET_WIDTH,
        },
        ParameterFamily {
            name: "mask_embed",
            elements: d::HNET_WIDTH,
        },
        ParameterFamily {
            name: "index_query",
            elements: d::HNET_PATCH * d::HNET_WIDTH,
        },
    ]
};

pub const TARGETS: usize = TOKENS - (PATCH - 1);
