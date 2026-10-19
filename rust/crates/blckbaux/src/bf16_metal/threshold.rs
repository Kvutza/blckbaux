use super::*;
use crate::threshold::{TABLE_WORDS, ThresholdModel, ThresholdTable};
use std::collections::BTreeMap;
use std::time::Instant;

#[derive(Clone)]
struct ProgramStep {
    table: ThresholdTable,
    norm: f32,
    base_id: i64,
}

struct ProgramMetric {
    norms: [f32; 4],
    // Family components use the original block metric; current learned weights
    // enter only when forming the scalar overlap. Table correlations retain the
    // existing latent-program approximation rather than realized FP16 geometry.
    components: Option<[[f32; FAMILIES]; 4]>,
    history: Vec<[f32; FAMILIES]>,
    weights: [f32; FAMILIES],
}

struct ProgramGeometry {
    addresses: Vec<(usize, [f64; FAMILIES])>,
    totals: [f64; FAMILIES],
}

impl ProgramGeometry {
    fn new(blocks: &[ParamBlock], groups: Option<&[usize]>) -> Result<Self, String> {
        if groups.is_some_and(|groups| groups.len() != blocks.len()) {
            return Err("Threshold geometry requires one family group per block".into());
        }
        let mut dense = vec![[0.0; FAMILIES]; 1 << 12];
        let mut totals = [0.0; FAMILIES];
        for (index, block) in blocks.iter().enumerate() {
            let group = groups.map_or(0, |groups| groups[index]);
            if group >= FAMILIES {
                return Err("Threshold geometry family group is out of range".into());
            }
            let energy =
                block.len as f64 * f64::from(block.weight) * f64::from(block.scale).powi(2);
            if !energy.is_finite() || energy <= 0.0 {
                return Err("Threshold geometry block energy must be positive and finite".into());
            }
            dense[usize::from(block.address.bits())][group] += energy;
            totals[group] += energy;
        }
        let addresses = dense
            .into_iter()
            .enumerate()
            .filter(|(_, energy)| energy.iter().any(|value| *value > 0.0))
            .collect();
        Ok(Self { addresses, totals })
    }

    fn correlations(&self, left: &ThresholdTable, right: &ThresholdTable) -> [f64; FAMILIES] {
        let mut overlaps = [0.0; FAMILIES];
        for &(address, energy) in &self.addresses {
            let correlation = left.prefix_correlation(right, address);
            for group in 0..FAMILIES {
                overlaps[group] += energy[group] * correlation;
            }
        }
        std::array::from_fn(|group| {
            if self.totals[group] > 0.0 {
                overlaps[group] / self.totals[group]
            } else {
                0.0
            }
        })
    }

    fn correlation(&self, left: &ThresholdTable, right: &ThresholdTable) -> f64 {
        let correlations = self.correlations(left, right);
        let total = self.totals.iter().sum::<f64>();
        correlations
            .iter()
            .zip(self.totals)
            .map(|(&correlation, energy)| correlation * energy)
            .sum::<f64>()
            / total
    }
}

pub(super) struct ThresholdSearch {
    basis_seed: Option<u64>,
    adaptive: bool,
    steps: usize,
    model: ThresholdModel,
    tables: Vec<ThresholdTable>,
    history: BTreeMap<i64, ProgramStep>,
    overlaps: Vec<[f32; FAMILIES]>,
    geometry: ProgramGeometry,
    prepare_ms: f32,
}

impl ThresholdSearch {
    pub(super) fn new(
        noise_floor: f64,
        adaptive: bool,
        blocks: &[ParamBlock],
    ) -> Result<Self, String> {
        Ok(Self {
            basis_seed: None,
            adaptive,
            steps: 0,
            model: ThresholdModel::new(noise_floor)?,
            tables: Vec::with_capacity(4),
            history: BTreeMap::new(),
            overlaps: Vec::new(),
            geometry: ProgramGeometry::new(blocks, None)?,
            prepare_ms: 0.0,
        })
    }

    pub(super) fn set_groups(
        &mut self,
        blocks: &[ParamBlock],
        groups: &[usize],
    ) -> Result<(), String> {
        self.geometry = ProgramGeometry::new(blocks, Some(groups))?;
        Ok(())
    }

    pub(super) fn compact(&mut self, incumbent: i64) {
        self.history.retain(|identity, _| *identity == incumbent);
        self.tables.clear();
        self.overlaps.clear();
    }

    pub(super) fn retain_identities(&mut self, identities: &[i64]) {
        self.history
            .retain(|identity, _| identities.contains(identity));
    }

    pub(super) fn basis(&self) -> u64 {
        self.basis_seed.unwrap_or(0)
    }

    pub(super) fn prepare_ms(&self) -> f32 {
        self.prepare_ms
    }

    #[cfg(test)]
    pub(super) fn correlations(
        &self,
        left: &ThresholdTable,
        right: &ThresholdTable,
    ) -> [f64; FAMILIES] {
        self.geometry.correlations(left, right)
    }

    #[cfg(test)]
    pub(super) fn correlation(&self, left: &ThresholdTable, right: &ThresholdTable) -> f64 {
        self.geometry.correlation(left, right)
    }

    fn prepare(
        &mut self,
        root: u64,
        buffer: &Buffer,
        identities: &[i64],
        incumbent: i64,
        metric: ProgramMetric,
    ) -> Result<(u64, Vec<f32>), String> {
        let start = Instant::now();
        let basis = *self
            .basis_seed
            .get_or_insert_with(|| crate::hash::splitmix64(root ^ 0x7074_665f_6261_7369));
        self.tables.clear();
        let draws = if self.adaptive {
            self.model.draw_batch(root, &[0, 1, 2, 3])?
        } else {
            (0..4)
                .map(|offset| ThresholdTable::basis(self.steps + offset))
                .collect()
        };
        self.tables.extend(draws);
        unsafe {
            for (candidate, table) in self.tables.iter().enumerate() {
                std::ptr::copy_nonoverlapping(
                    table.words().as_ptr(),
                    buffer.contents().cast::<u64>().add(candidate * TABLE_WORDS),
                    TABLE_WORDS,
                );
            }
        }
        let mut distances = Vec::with_capacity(5 * identities.len());
        self.overlaps.clear();
        for (candidate, table) in self.tables.iter().enumerate() {
            for (row, identity) in identities.iter().enumerate() {
                let step = self
                    .history
                    .get(identity)
                    .filter(|step| step.base_id == incumbent);
                let mut components = [0.0; FAMILIES];
                let correction = match (step, metric.components) {
                    (Some(step), Some(norms)) => {
                        let correlations = self.geometry.correlations(table, &step.table);
                        components = std::array::from_fn(|group| {
                            (-2.0
                                * (f64::from(norms[candidate][group])
                                    * f64::from(metric.history[row][group]))
                                .sqrt()
                                * correlations[group]) as f32
                        });
                        components
                            .iter()
                            .zip(metric.weights)
                            .map(|(&overlap, weight)| f64::from(overlap) * f64::from(weight))
                            .sum::<f64>() as f32
                    }
                    (Some(step), None) => overlap_correction(
                        self.geometry.correlation(table, &step.table),
                        metric.norms[candidate],
                        step,
                    ),
                    (None, _) => 0.0,
                };
                distances.push(correction);
                self.overlaps.push(components);
            }
        }
        // Incumbent-to-history distances are already in the pairwise matrix.
        distances.extend(std::iter::repeat_n(0.0, identities.len()));
        self.prepare_ms = start.elapsed().as_secs_f32() * 1000.0;
        Ok((basis, distances))
    }

    pub(super) fn table(&self, candidate: usize) -> Result<ThresholdTable, String> {
        self.tables
            .get(candidate)
            .cloned()
            .ok_or_else(|| "threshold candidate table is unavailable".into())
    }

    pub(super) fn overlaps(
        &self,
        candidate: usize,
        rows: usize,
    ) -> Result<&[[f32; FAMILIES]], String> {
        self.overlaps
            .get(candidate * rows..(candidate + 1) * rows)
            .ok_or_else(|| "threshold family overlap is unavailable".into())
    }

    pub(super) fn observe(
        &mut self,
        identity: i64,
        base_id: i64,
        table: &ThresholdTable,
        radius: f32,
        norm: f32,
        improvement: f64,
        variance: f64,
    ) -> Result<(), String> {
        if !norm.is_finite() || norm < 0.0 {
            return Err("threshold direction norm must be finite and nonnegative".into());
        }
        if self.adaptive {
            self.model
                .observe(table, f64::from(radius), improvement, variance)?;
        }
        self.steps = self.steps.saturating_add(1);
        self.history.insert(
            identity,
            ProgramStep {
                table: table.clone(),
                norm,
                base_id,
            },
        );
        Ok(())
    }
}

fn overlap_correction(correlation: f64, norm: f32, step: &ProgramStep) -> f32 {
    let cross = 2.0 * (f64::from(norm) * f64::from(step.norm)).sqrt() * correlation;
    -cross as f32
}

impl SearchState {
    pub(crate) fn program_ms(&self) -> Option<f32> {
        self.threshold.as_ref().map(ThresholdSearch::prepare_ms)
    }

    pub(super) fn prepare_threshold(&mut self, root: u64) -> Result<u64, String> {
        if self.threshold.is_none() {
            return Ok(0);
        }
        let identities = self.identities[..self.history].to_vec();
        let norms = std::array::from_fn(|candidate| self.latent_norm(candidate));
        let components = self
            .family
            .as_ref()
            .map(|_| std::array::from_fn(|candidate| self.latent_components(candidate)));
        let base = identities
            .iter()
            .position(|&identity| identity == self.base_id)
            .unwrap_or(0);
        let metric = ProgramMetric {
            norms,
            components,
            history: self.family.as_ref().map_or_else(Vec::new, |family| {
                (0..self.history)
                    .map(|row| family.components[base * MAX_HISTORY + row])
                    .collect()
            }),
            weights: self
                .family
                .as_ref()
                .map_or([1.0; FAMILIES], |family| family.weights),
        };
        let threshold = self.threshold.as_mut().unwrap();
        let (basis, distances) = threshold.prepare(
            root,
            &self.threshold_tables,
            &identities,
            self.base_id,
            metric,
        )?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                distances.as_ptr(),
                self.pool_distances.contents().cast::<f32>(),
                distances.len(),
            );
        }
        Ok(basis)
    }

    pub(super) fn threshold_table(
        &self,
        candidate: usize,
    ) -> Result<Option<ThresholdTable>, String> {
        self.threshold
            .as_ref()
            .map(|threshold| threshold.table(candidate))
            .transpose()
    }
}
