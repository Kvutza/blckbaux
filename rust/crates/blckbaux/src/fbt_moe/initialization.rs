use super::resources::{half_bits, mhc_bias};
use super::*;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};

#[derive(Debug, Clone, Copy)]
pub(super) struct HnetMhc4Initialization {
    policy: blckbaux_semantics::hnet::Initialization,
}

impl std::ops::Deref for HnetMhc4Initialization {
    type Target = blckbaux_semantics::hnet::Initialization;
    fn deref(&self) -> &Self::Target {
        &self.policy
    }
}

impl HnetMhc4Initialization {
    pub(super) fn new(architecture: ResidualArchitecture) -> Result<Self, String> {
        if architecture != ResidualArchitecture::HnetMhc4 {
            return Err(
                "fbt-pisa1-hnet-mhc4 initialization requires model fbt-pisa1-hnet-mhc4-v1".into(),
            );
        }

        Ok(Self {
            policy: blckbaux_semantics::hnet::Initialization::new(),
        })
    }
}

/// Draw independent Gaussian columns and normalize each column to an exact
/// FP32 norm before the single FP16 rounding step. Matrices use row-major
/// `[input, output]` storage, matching the Metal projection kernels.
fn spherical(
    runtime: &Runtime,
    matrices: usize,
    input: usize,
    output: usize,
    seed: u64,
    column_norm: f32,
) -> Buffer {
    let elements = matrices
        .checked_mul(input)
        .and_then(|v| v.checked_mul(output))
        .expect("fixed model tensor dimensions must fit usize");
    let buffer = runtime.buffer::<u16>(elements);
    let values =
        unsafe { std::slice::from_raw_parts_mut(buffer.contents().cast::<u16>(), elements) };
    let mut random = StdRng::seed_from_u64(seed);
    blckbaux_semantics::hnet::columns(
        values,
        input,
        output,
        column_norm,
        || StandardNormal.sample(&mut random),
        half_bits,
    )
    .expect("fixed H-Net Gaussian column initialization");
    buffer
}

fn mhc_controls(runtime: &Runtime, layers: usize, scale: f32, lambda: f32) -> Buffer {
    let mut values = Vec::with_capacity(layers * 4);
    for _ in 0..layers {
        values.extend([
            half_bits(scale),
            half_bits(scale),
            half_bits(scale),
            half_bits(lambda),
        ]);
    }
    runtime.buffer_with(&values)
}

struct HnetBuffers {
    router: Buffer,
    qkv: Buffer,
    output: Buffer,
    gate_up: Buffer,
    down: Buffer,
    attention_norm: Buffer,
    ffn_norm: Buffer,
    readout: Buffer,
    mask_embed: Buffer,
    index_query: Buffer,
    feedback_state: Buffer,
    feedback_gate: Buffer,
    mhc_attention_predictor: Buffer,
    mhc_attention_bias: Buffer,
    mhc_attention_control: Buffer,
    mhc_moe_predictor: Buffer,
    mhc_moe_bias: Buffer,
    mhc_moe_control: Buffer,
    final_norm: Buffer,
}

impl HnetBuffers {
    fn into_weights(self, architecture: ResidualArchitecture) -> CandidateWeights {
        CandidateWeights {
            architecture,
            initialization: crate::config::ModelInitialization::FbtPisa1HnetMhc4,
            router: self.router,
            qkv: self.qkv,
            output: self.output,
            gate_up: self.gate_up,
            down: self.down,
            attention_norm: self.attention_norm,
            ffn_norm: self.ffn_norm,
            readout: self.readout,
            mask_embed: self.mask_embed,
            index_query: self.index_query,
            feedback_state: self.feedback_state,
            feedback_gate: self.feedback_gate,
            mhc_attention_predictor: self.mhc_attention_predictor,
            mhc_attention_bias: self.mhc_attention_bias,
            mhc_attention_control: self.mhc_attention_control,
            mhc_moe_predictor: self.mhc_moe_predictor,
            mhc_moe_bias: self.mhc_moe_bias,
            mhc_moe_control: self.mhc_moe_control,
            final_norm: self.final_norm,
        }
    }
}

impl CandidateWeights {
    pub(super) fn block_scale(
        &self,
        policy: Option<HnetMhc4Initialization>,
        shape: crate::config::TrustRegionShape,
        family: &str,
        initial_rms: f64,
    ) -> Result<(f32, f32), String> {
        let multiplier = if shape == crate::config::TrustRegionShape::TensorFamilyStatic {
            trust_multiplier(family)
        } else {
            1.0
        };
        if let Some(policy) = policy {
            return Ok((policy.proposal_scale(family)? * multiplier, multiplier));
        }
        let floor = if matches!(family, "mhc_attention_control" | "mhc_moe_control") {
            0.02
        } else {
            1e-6
        };
        Ok((initial_rms.max(floor) as f32 * multiplier, multiplier))
    }

    pub(super) fn hnet_weights(
        runtime: &Runtime,
        seed: u64,
        architecture: ResidualArchitecture,
    ) -> Result<Self, String> {
        let policy = HnetMhc4Initialization::new(architecture)?;
        let layers = MODEL_LAYERS as usize;
        let experts = routing::MOE_EXPERTS as usize;
        let width = WIDTH as usize;
        let expert_width = routing::ROUTED_WIDTH as usize;
        let gate_up = routing::MOE_GATEUP as usize;
        let qkv_width = QKV_WIDTH as usize;
        let vocab = VOCAB as usize;
        let draw = |matrices, input, output, domain, norm| {
            spherical(
                runtime,
                matrices,
                input,
                output,
                crate::hash::splitmix64(seed ^ domain),
                norm,
            )
        };

        Ok(HnetBuffers {
            router: draw(
                layers,
                width,
                routing::ROUTED_EXPERTS as usize,
                0x6a09_e667,
                policy.router_column_norm,
            ),
            qkv: draw(layers, width, qkv_width, 0xbb67_ae85, 1.0),
            output: draw(layers, width, width, 0x3c6e_f372, policy.residual_gain),
            gate_up: draw(layers * experts, width, gate_up, 0xa54f_f53a, 1.0),
            down: draw(
                layers * experts,
                expert_width,
                width,
                0x510e_527f,
                policy.down_norm(),
            ),
            attention_norm: filled(runtime, layers * width, 0x3c00),
            ffn_norm: filled(runtime, layers * width, 0x3c00),
            // Tied embedding/readout token columns have unit norm, giving
            // unit logit variance after RMS normalization.
            readout: draw(1, width, vocab, 0x9b05_688c, 1.0),
            mask_embed: draw(1, width, 1, 0x4d41_534b, 1.0),
            // Each column has unit norm across the 64 within-block positions.
            // The same basis gates fine-token embeddings during compression
            // and supplies the position component to the local decoder.
            index_query: draw(1, 64, width, 0x494e_4458, 1.0),
            feedback_state: draw(1, width, width, 0x1f83_d9ab, 1.0),
            feedback_gate: draw(1, width, width, 0x5be0_cd19, 1.0),
            mhc_attention_predictor: draw(
                layers,
                MHC_INPUT as usize,
                MHC_COEFFICIENTS as usize,
                0x428a_2f98,
                1.0,
            ),
            mhc_attention_bias: mhc_bias(runtime, layers),
            mhc_attention_control: mhc_controls(
                runtime,
                layers,
                policy.predictor_control,
                policy.transport_lambda,
            ),
            mhc_moe_predictor: draw(
                layers,
                MHC_INPUT as usize,
                MHC_COEFFICIENTS as usize,
                0x7137_4491,
                1.0,
            ),
            mhc_moe_bias: mhc_bias(runtime, layers),
            mhc_moe_control: mhc_controls(
                runtime,
                layers,
                policy.predictor_control,
                policy.transport_lambda,
            ),
            final_norm: filled(runtime, width, 0x3c00),
        }
        .into_weights(architecture))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hnet_scales() {
        let policy = HnetMhc4Initialization::new(ResidualArchitecture::HnetMhc4).unwrap();
        assert_eq!(policy.residual_depth, 22.0);
        assert_eq!(policy.sites, 14);
        assert!((policy.residual_gain - 22.0f32.sqrt().recip()).abs() < 1.0e-7);
        assert!((policy.transport_lambda - 1.0 / 112.0).abs() < 1.0e-7);

        let retained = (1.0 - policy.transport_lambda).powi(policy.sites as i32);
        assert!((retained - 0.882).abs() < 0.001);
    }

    #[test]
    fn router_scale() {
        let policy = HnetMhc4Initialization::new(ResidualArchitecture::HnetMhc4).unwrap();
        let expected_max =
            policy.router_column_norm * (2.0 * (routing::ROUTED_EXPERTS as f32).ln()).sqrt();
        assert!((expected_max - 1.0).abs() < 1.0e-6);
    }
}
