use super::*;

impl Scorer<'_> {
    fn hnet_input(&self, encoder: &ComputeCommandEncoderRef) {
        let shape = blckbaux_semantics::hnet::EncodeShape {
            rows: self.rows,
            width: WIDTH,
            vocabulary: VOCAB,
            patch: 64,
        };
        let offset = self.activation_offset();
        encoder.set_compute_pipeline_state(&self.pipelines.hnet_encode);
        encoder.set_buffer(
            0,
            Some(
                self.buffers
                    .hnet_readout
                    .as_ref()
                    .expect("H-Net scorer has transposed readout"),
            ),
            0,
        );
        encoder.set_buffer(
            1,
            Some(&self.buffers.fine_tokens),
            u64::from(self.row_start) * 64 * std::mem::size_of::<u32>() as u64,
        );
        encoder.set_buffer(2, Some(self.weights.buffer), self.weights.index_query);
        encoder.set_buffer(3, Some(self.weights.buffer), self.weights.mask_embed);
        encoder.set_buffer(4, Some(&self.buffers.input), offset);
        encoder.set_bytes(
            5,
            std::mem::size_of_val(&shape) as u64,
            std::ptr::from_ref(&shape).cast(),
        );
        encoder.dispatch_threads(
            thread_group(u64::from(self.rows) * u64::from(WIDTH)),
            thread_group(
                self.pipelines
                    .hnet_encode
                    .max_total_threads_per_threadgroup()
                    .min(256),
            ),
        );
    }

    pub(super) fn embed(&self) -> Result<(), String> {
        let encoder = self.active_encoder();
        let pipelines = self.pipelines;
        let buffers = self.buffers;
        let weights = self.weights;
        let activation_offset = self.activation_offset();
        if weights.architecture == ResidualArchitecture::HnetMhc4 {
            self.hnet_input(encoder);
        } else {
            encoder.set_compute_pipeline_state(if self.diffusion.is_some() {
                &pipelines.denoise_embed
            } else {
                &pipelines.embed
            });
            encoder.set_buffer(0, Some(weights.buffer), weights.readout);
            encoder.set_buffer(
                1,
                Some(&buffers.tokens),
                u64::from(self.row_start) * std::mem::size_of::<u32>() as u64,
            );
            encoder.set_buffer(2, Some(&buffers.input), activation_offset);
            encoder.set_bytes(
                3,
                std::mem::size_of_val(&VOCAB) as u64,
                std::ptr::from_ref(&VOCAB).cast(),
            );
        }
        if let Some(input) = self.diffusion {
            encoder.set_buffer(3, Some(weights.buffer), weights.mask_embed);
            encoder.set_buffer(4, Some(input.confidence), u64::from(self.row_start) * 4);
            encoder.dispatch_thread_groups(thread_group(u64::from(self.rows)), thread_group(128));
        } else if weights.architecture != ResidualArchitecture::HnetMhc4 {
            encoder.dispatch_threads(
                thread_group(u64::from(self.rows) * u64::from(WIDTH)),
                thread_group(pipelines.embed.max_total_threads_per_threadgroup().min(256)),
            );
        }
        encoder.memory_barrier_with_resources(&[&buffers.input]);

        if weights.architecture.is_multistream() {
            let shape = [self.rows, WIDTH, weights.architecture.kernel_code(), 0];
            encoder.set_compute_pipeline_state(&pipelines.mhc_replicate);
            encoder.set_buffer(0, Some(&buffers.input), activation_offset);
            encoder.set_buffer(
                1,
                Some(&buffers.mhc_streams[0]),
                half_bytes(u64::from(self.scratch_start() * MHC_STREAMS * WIDTH)),
            );
            encoder.set_bytes(
                2,
                std::mem::size_of_val(&shape) as u64,
                shape.as_ptr().cast(),
            );
            encoder.dispatch_threads(
                thread_group(u64::from(self.rows * WIDTH)),
                thread_group(
                    pipelines
                        .mhc_replicate
                        .max_total_threads_per_threadgroup()
                        .min(256),
                ),
            );
            encoder.memory_barrier_with_resources(&[&buffers.mhc_streams[0]]);
            return Ok(());
        }

        rms_offsets(
            &encoder,
            pipelines,
            (&buffers.input, activation_offset),
            (weights.buffer, weights.attention_norm),
            (&buffers.normalized, activation_offset),
            self.rows,
        );
        encoder.memory_barrier_with_resources(&[&buffers.normalized]);
        Ok(())
    }

    pub(super) fn capture_hidden(&self, output: &BufferRef) -> Result<(), String> {
        if self.weights.architecture != ResidualArchitecture::HnetMhc4 {
            return Ok(());
        }
        let encoder = self.active_encoder();
        let shape = [self.rows, self.row_start, WIDTH];
        encoder.memory_barrier_with_resources(&[&self.buffers.normalized]);
        encoder.set_compute_pipeline_state(&self.pipelines.hnet_capture);
        encoder.set_buffer(0, Some(&self.buffers.normalized), self.activation_offset());
        encoder.set_buffer(1, Some(output), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of_val(&shape) as u64,
            shape.as_ptr().cast(),
        );
        encoder.dispatch_threads(
            thread_group(u64::from(self.rows) * u64::from(WIDTH)),
            thread_group(
                self.pipelines
                    .hnet_capture
                    .max_total_threads_per_threadgroup()
                    .min(256),
            ),
        );
        Ok(())
    }

    pub(super) fn layer(
        &self,
        input: &BufferRef,
        layer: u32,
        pass: u32,
        execution: u32,
    ) -> Result<(), String> {
        if self.weights.architecture.is_multistream() {
            self.start_stage("mhc_attention_input")?;
            self.mhc_prepare(layer, execution, true)?;
            self.attention(layer, execution)?;
            self.start_stage("mhc_attention_update")?;
            self.mhc_update(layer, execution, true, &self.buffers.projected)?;
            self.start_stage("mhc_moe_input")?;
            self.mhc_prepare(layer, execution, false)?;
            self.feed_mhc(layer)?;
            self.start_stage("mhc_moe_update")?;
            self.mhc_update(layer, execution, false, &self.buffers.output)?;
            return Ok(());
        }
        self.attention(layer, execution)?;
        self.feed_forward(input, layer, pass)
    }
}
