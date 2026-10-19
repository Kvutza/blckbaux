use super::*;

impl Buffers {
    pub(super) fn moe_grad(&self) -> Result<&routing_grad::MoeGradBuffers, String> {
        self.fine_grained_grad
            .get_or_init(|| {
                Runtime::shared()
                    .map(|runtime| routing_grad::MoeGradBuffers::new(&runtime, self.rows))
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    pub(super) fn quantized_gateup(&self) -> Result<&BufferRef, String> {
        self.quantized_gateup
            .get_or_init(|| {
                Runtime::shared().map(|runtime| {
                    runtime.buffer::<i8>(MODEL_LAYERS as usize * routing::GATE_PARAMETERS)
                })
            })
            .as_ref()
            .map(|buffer| buffer.as_ref())
            .map_err(Clone::clone)
    }

    pub(super) fn logits(&self) -> Result<&BufferRef, String> {
        self.logits
            .get_or_init(|| {
                Runtime::shared().map(|runtime| runtime.buffer::<u16>((self.rows * VOCAB) as usize))
            })
            .as_ref()
            .map(|buffer| buffer.as_ref())
            .map_err(Clone::clone)
    }

    pub(super) fn proposal_partials(&self) -> Result<&BufferRef, String> {
        self.proposal_partials
            .get_or_init(|| {
                Runtime::shared().map(|runtime| {
                    runtime.buffer::<[f32; 4]>(self.rows as usize * (VOCAB / 64) as usize)
                })
            })
            .as_ref()
            .map(|buffer| buffer.as_ref())
            .map_err(Clone::clone)
    }

    pub(super) fn loss_partials(&self) -> Result<&BufferRef, String> {
        self.loss_partials
            .get_or_init(|| {
                Runtime::shared().map(|runtime| {
                    runtime.buffer::<[f32; 4]>(self.rows as usize * (VOCAB / 64) as usize)
                })
            })
            .as_ref()
            .map(|buffer| buffer.as_ref())
            .map_err(Clone::clone)
    }

    pub(super) fn new(runtime: &Runtime) -> Self {
        Self::sized(runtime, false, ROWS)
    }

    fn sized(runtime: &Runtime, hierarchical_readout: bool, rows: u32) -> Self {
        let half = |n: u32| n as usize;
        let partials = || {
            let cell = std::cell::OnceCell::new();
            if hierarchical_readout {
                let _ = cell.set(Ok(runtime.buffer::<[f32; 4]>(1)));
            }
            cell
        };
        let residual = residual_buffers(runtime, rows);
        Self {
            rows,
            fine_grained: routing::FineGrainedMoeBuffers::for_rows(runtime, rows),
            fine_grained_grad: std::cell::OnceCell::new(),
            quantized_gateup: std::cell::OnceCell::new(),
            input: patterned(runtime, half(rows * WIDTH), 0x243f_6a88, 0x2c00),
            router: patterned(runtime, half(WIDTH * EXPERTS), 0x85a3_08d3, 0x1800),
            gates: runtime.buffer::<u16>(half(rows)),
            grouped: runtime.buffer::<u16>(half(rows * WIDTH)),
            gate_up_base: patterned(
                runtime,
                half(EXPERTS * WIDTH * GATE_UP),
                0x1319_8a2e,
                0x2000,
            ),
            gate_up: canary(runtime, (rows * GATE_UP) as usize),
            gate_inner: factor(runtime, EXPERTS, 32, 32),
            gate_outer: factor(runtime, EXPERTS, 16, 54),
            activation: canary(runtime, (rows * EXPERT_WIDTH) as usize),
            down_base: patterned(
                runtime,
                half(EXPERTS * EXPERT_WIDTH * WIDTH),
                0x0370_7344,
                0x2000,
            ),
            down: canary(runtime, (rows * WIDTH) as usize),
            down_inner: factor(runtime, EXPERTS, 27, 16),
            down_outer: factor(runtime, EXPERTS, 32, 32),
            output: canary(runtime, (rows * WIDTH) as usize),
            mps_gate_up: canary(runtime, (rows * GATE_UP) as usize),
            mps_down: canary(runtime, (rows * WIDTH) as usize),
            materialized_gate_up: canary(runtime, (EXPERTS * WIDTH * GATE_UP) as usize),
            materialized_down: canary(runtime, (EXPERTS * EXPERT_WIDTH * WIDTH) as usize),
            qkv_weights: patterned(runtime, half(WIDTH * QKV_WIDTH), 0xa409_3822, 0x2000),
            qkv: canary(runtime, (rows * QKV_WIDTH) as usize),
            output_projection_weights: patterned(runtime, half(WIDTH * WIDTH), 0x299f_31d0, 0x2000),
            projected: canary(runtime, (rows * WIDTH) as usize),
            mps_qkv: canary(runtime, (rows * QKV_WIDTH) as usize),
            mps_projected: canary(runtime, (rows * WIDTH) as usize),
            feedback_state_weights: patterned(runtime, half(WIDTH * WIDTH), 0x082e_fa98, 0x2000),
            feedback_gate_weights: patterned(runtime, half(WIDTH * WIDTH), 0xec4e_6c89, 0x2000),
            feedback_state: residual.feedback_state,
            feedback_gate: residual.feedback_gate,
            feedback: residual.feedback,
            readout_weights: patterned(runtime, half(WIDTH * VOCAB), 0x4528_21e6, 0x1800),
            logits: std::cell::OnceCell::new(),
            loss_partials: partials(),
            proposal_partials: partials(),
            labels: synthetic_buffer(runtime, 1),
            losses: runtime.buffer_with(&vec![f32::NAN; rows as usize + 64]),
            score_mask: runtime.buffer_with(&vec![1u8; rows as usize]),
            sequence_scores: runtime.buffer_with(&[f32::NAN; BATCH as usize]),
            tokens: synthetic_buffer(runtime, 0),
            fine_tokens: runtime.buffer_with(&vec![u32::MAX; (CONTEXT * 64) as usize]),
            fine_losses: runtime.buffer::<f32>((CONTEXT * 64) as usize),
            hnet_readout: hierarchical_readout
                .then(|| runtime.buffer::<u16>((WIDTH * VOCAB) as usize)),
            unit_norm: residual.unit_norm,
            normalized: residual.normalized,
            attention_state: residual.attention_state,
            mhc_streams: residual.mhc_streams,
            mhc_coefficients: residual.mhc_coefficients,
            rope: rope_table(runtime, MAX_CONTEXT),
        }
    }

    pub(super) fn for_context(
        runtime: &Runtime,
        context: u32,
        architecture: ResidualArchitecture,
    ) -> Self {
        let mut buffers = Self::sized(
            runtime,
            architecture == ResidualArchitecture::HnetMhc4,
            context.max(ROWS),
        );
        buffers.tokens = runtime.buffer_with(&vec![0u32; context as usize]);
        buffers.labels = runtime.buffer_with(&vec![0u32; context as usize]);
        if context > MAX_CONTEXT {
            buffers.rope = rope_table(runtime, context);
        }
        buffers.fine_tokens = runtime.buffer_with(&vec![u32::MAX; context as usize * 64]);
        buffers.fine_losses = runtime.buffer::<f32>(context as usize * 64);
        buffers
    }
}

struct ResidualBuffers {
    feedback_state: Buffer,
    feedback_gate: Buffer,
    feedback: Buffer,
    unit_norm: Buffer,
    normalized: Buffer,
    attention_state: Buffer,
    mhc_streams: [Buffer; 2],
    mhc_coefficients: Buffer,
}

fn residual_buffers(runtime: &Runtime, rows: u32) -> ResidualBuffers {
    ResidualBuffers {
        feedback_state: runtime.buffer::<u16>((rows * WIDTH) as usize),
        feedback_gate: runtime.buffer::<u16>((rows * WIDTH) as usize),
        feedback: runtime.buffer::<u16>((rows * WIDTH) as usize),
        unit_norm: filled(runtime, WIDTH as usize, 0x3c00),
        normalized: runtime.buffer::<u16>((rows * WIDTH) as usize),
        attention_state: runtime.buffer::<u16>((rows * WIDTH) as usize),
        mhc_streams: [
            runtime.buffer::<u16>((rows * MHC_STREAMS * WIDTH) as usize),
            runtime.buffer::<u16>((rows * MHC_STREAMS * WIDTH) as usize),
        ],
        mhc_coefficients: runtime.buffer::<f32>((rows * MHC_COEFFICIENTS) as usize),
    }
}
