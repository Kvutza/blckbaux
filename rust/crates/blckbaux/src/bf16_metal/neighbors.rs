//! Stratified coordinate quadrature for bounded model-state neighbour history.
//! Every proposal still materializes all coordinates. These sampled distances
//! are estimates of the weighted weight-space metric, not exact distances or
//! a proved sufficient representation of the objective.
use super::*;
use crate::apple_gpu::thread_group;

#[repr(C)]
#[derive(Clone, Copy)]
struct Point {
    element: u64,
    leaf: u32,
    mass: f32,
}

pub(super) struct Neighbors {
    points: Vec<Point>,
    indices: Vec<usize>,
    descriptors: Buffer,
    history: Buffer,
    pool: Buffer,
    components: Buffer,
    sample: ComputePipelineState,
    distance: ComputePipelineState,
}

impl Neighbors {
    pub(super) fn new(state: &SearchState) -> Result<Self, String> {
        let mut points = Vec::new();
        let mut indices = Vec::new();
        for (leaf, block) in state.blocks.iter().enumerate() {
            let count = (8192usize.saturating_mul(block.len) / state.dimensions)
                .max(8)
                .min(block.len);
            for stratum in 0..count {
                let start = stratum * block.len / count;
                let end = (stratum + 1) * block.len / count;
                let element = start
                    + crate::hash::splitmix64(
                        block.key ^ (stratum as u64).wrapping_mul(0x9e3779b97f4a7c15),
                    ) as usize
                        % (end - start);
                points.push(Point {
                    element: element as u64,
                    leaf: leaf as u32,
                    mass: (end - start) as f32,
                });
                indices.push(block.offset + element);
            }
        }
        let source = state.procedural_source(state.pool_layout);
        Ok(Self {
            descriptors: state.runtime.buffer_with(&points),
            history: state.runtime.buffer::<u16>(points.len() * MAX_HISTORY),
            pool: state.runtime.buffer::<u16>(points.len() * 4),
            components: state.runtime.buffer::<f32>(4 * MAX_HISTORY * FAMILIES),
            sample: state
                .runtime
                .pipeline(&source, "Neighbour geometry", "bf16_sample")?,
            distance: state.runtime.precise(
                &source,
                "Neighbour geometry",
                "bf16_sample_distances",
            )?,
            points,
            indices,
        })
    }

    pub(super) fn bytes(&self) -> u64 {
        self.descriptors.length()
            + self.history.length()
            + self.pool.length()
            + self.components.length()
    }

    pub(super) fn capture(&self, row: usize, model: &Buffer) {
        assert!(row < MAX_HISTORY);
        // Called only while the search is idle or after the proposal completed.
        unsafe {
            for (sample, &index) in self.indices.iter().enumerate() {
                self.history
                    .contents()
                    .cast::<u16>()
                    .add(row * self.points.len() + sample)
                    .write(model.contents().cast::<u16>().add(index).read());
            }
        }
    }

    pub(super) fn archive(&self, row: usize, candidate: usize) {
        assert!(row < MAX_HISTORY && candidate < 4);
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.pool
                    .contents()
                    .cast::<u16>()
                    .add(candidate * self.points.len()),
                self.history
                    .contents()
                    .cast::<u16>()
                    .add(row * self.points.len()),
                self.points.len(),
            );
        }
    }

    pub(super) fn encode(&self, state: &SearchState, command: &CommandBufferRef, root: u64) {
        let params = Params {
            stream_seed: root,
            tiles: self.points.len() as u32,
            history: state.history as u32,
            initialize: u32::from(state.family.is_some()),
            ..state.pool_params(root)
        };
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.sample);
        for (i, buffer) in [
            &state.base,
            &state.leaves_gpu,
            &self.descriptors,
            &self.pool,
        ]
        .iter()
        .enumerate()
        {
            encoder.set_buffer(i as u64, Some(buffer), 0);
        }
        encoder.set_bytes(
            4,
            size_of::<Params>() as u64,
            (&params as *const Params).cast(),
        );
        encoder.set_buffer(5, Some(&state.threshold_tables), 0);
        encoder.dispatch_thread_groups(
            thread_group(self.points.len().div_ceil(256) as u64),
            thread_group(256),
        );
        encoder.end_encoding();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.distance);
        for (i, buffer) in [
            &self.pool,
            &self.history,
            &self.descriptors,
            &state.leaves_gpu,
            &state.pool_distances,
            &state.family_base_weights,
            &state.family_groups,
            &self.components,
        ]
        .iter()
        .enumerate()
        {
            encoder.set_buffer(i as u64, Some(buffer), 0);
        }
        encoder.set_bytes(
            8,
            size_of::<Params>() as u64,
            (&params as *const Params).cast(),
        );
        encoder.dispatch_thread_groups(thread_group((4 * state.history) as u64), thread_group(256));
        encoder.end_encoding();
    }

    pub(super) fn components(&self, candidate: usize, history: usize) -> Vec<[f32; FAMILIES]> {
        let values = read::<f32>(&self.components, 4 * history * FAMILIES);
        values[candidate * history * FAMILIES..(candidate + 1) * history * FAMILIES]
            .chunks_exact(FAMILIES)
            .map(|row| row.try_into().unwrap())
            .collect()
    }
}

impl SearchState {
    pub fn neighbor_error(&self, round: &Proposals) -> Option<f64> {
        self.neighbors.as_ref()?;
        let sampled = round
            .history_distances
            .iter()
            .find(|(id, _)| *id == round.base_id)?
            .1;
        let exact: f64 = round
            .changes
            .iter()
            .zip(&self.blocks)
            .map(|(change, block)| change.1 * f64::from(block.weight))
            .sum();
        (exact > 0.0).then(|| (f64::from(sampled) - exact).abs() / exact)
    }

    pub fn neighbor_samples(&self) -> usize {
        self.neighbors
            .as_ref()
            .map_or(0, |neighbors| neighbors.points.len())
    }

    pub fn calibration_info(&self) -> (usize, f32, Option<f64>) {
        (
            self.calibration.count(),
            self.calibration.multiplier(),
            self.calibration.coverage(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DistanceScaling, HistoryGeometry, ResidentEnnConfig};
    use crate::procedural_pool::{ProceduralPoolConfig, ProposalMethod};

    fn search(arms: u32, slots: u32, mode: ProposalKernel, method: ProposalMethod) -> SearchState {
        // Include odd tensor starts, singleton leaves, tile boundaries, and tails.
        let lengths = [TILE_ELEMENTS + 3, 1, 10, 19];
        let mut base = Vec::new();
        let mut blocks = Vec::new();
        for (leaf, length) in lengths.into_iter().enumerate() {
            blocks.push(ParamBlock::new(71 + leaf as u64, base.len(), length, 0.25, 1.0).unwrap());
            base.extend((0..length).map(|index| {
                let magnitude = 0x3000 + ((index * 17 + leaf * 31) % 2048) as u16;
                magnitude | (((index + leaf) & 1) as u16) << 15
            }));
        }
        let mut search =
            SearchState::new_implicit(&base, blocks, 1, TRLengthConfig::new(0.1, 0.001, 0.4), mode)
                .unwrap();
        search
            .configure_enn(ResidentEnnConfig {
                pool: ProceduralPoolConfig {
                    arms,
                    candidates_per_arm: slots,
                }
                .resolve_resident()
                .unwrap(),
                proposal_method: method,
                ask: {
                    let mut config = RoundConfig::default();
                    config.surrogate.neighbors = 1;
                    config
                },
                num_candidates: 4,
                num_samples: 2,
                fit_params: false,
                adapt_neighbors: false,
                distance_scaling: DistanceScaling::Global,
                history_geometry: HistoryGeometry::Latent,
                local_scale_neighbors: 1,
            })
            .unwrap();
        search.observe_initial(0.0, 0.0).unwrap();
        search
    }

    #[test]
    fn sample_parity() {
        autoreleasepool(|| {
            for (arms, slots) in [(1, 1), (1, 4), (2, 2), (4, 1)] {
                for mode in [ProposalKernel::Gaussian, ProposalKernel::Rademacher] {
                    for method in [ProposalMethod::Additive, ProposalMethod::PairMix] {
                        let mut search = search(arms, slots, mode, method);
                        for iteration in 0..8 {
                            let candidate = iteration % search.pool_layout.count() as usize;
                            let root = crate::hash::splitmix64(iteration as u64 + 71);
                            let direct = search.test_candidate(root, candidate).unwrap();
                            let direct = read::<u16>(&direct, search.dimensions);
                            let command = search.runtime.queue.new_command_buffer();
                            search.encode_proposal(
                                command,
                                Params {
                                    history: 1,
                                    base_slot: 0,
                                    ..search.params(root, candidate)
                                },
                            );
                            finish(command).unwrap();
                            assert_eq!(read::<u16>(&search.proposal, search.dimensions), direct);
                            if method == ProposalMethod::PairMix {
                                assert_eq!(search.pool_params(root).stream_seed, root);
                            }
                            search
                                .begin_forced(root, RoundConfig::default(), candidate)
                                .unwrap();
                            let round = search.finish_ask().unwrap();
                            let actual = read::<u16>(&search.proposal, search.dimensions);
                            assert_eq!(actual, direct);
                            let neighbors = search.neighbors.as_ref().unwrap();
                            let count = neighbors.points.len();
                            let pool = read::<u16>(&neighbors.pool, 4 * count);
                            let observed = neighbors
                                .indices
                                .iter()
                                .map(|&index| actual[index])
                                .collect::<Vec<_>>();
                            for (sample, &expected) in observed.iter().enumerate() {
                                assert_eq!(
                                    pool[candidate * count + sample],
                                    expected,
                                    "sample/materializer mismatch: {arms}x{slots} {mode:?} {method:?} candidate={candidate} iteration={iteration} sample={sample}"
                                );
                            }
                            let row = search.history;
                            search.tell_noisy(&round, -1.0, 0.0).unwrap();
                            search.sync().unwrap();
                            let neighbors = search.neighbors.as_ref().unwrap();
                            let history = read::<u16>(&neighbors.history, (row + 1) * count);
                            assert_eq!(&history[row * count..], observed);
                        }
                    }
                }
            }
        });
    }
}
