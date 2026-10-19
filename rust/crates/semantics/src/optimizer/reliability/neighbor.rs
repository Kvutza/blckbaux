//! Bounded neighbor-value estimation used by the current experiment.

use super::MAX_HISTORY;
use crate::SurrogateSpec;
use crate::core::ValuePrediction;
use rand::seq::index::sample;
use rand::{Rng, SeedableRng, rngs::StdRng};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct NeighborEstimatorParams {
    pub(super) neighbors: usize,
    pub(super) epistemic: f64,
    pub(super) aleatoric: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FitState {
    pub neighbors: usize,
    pub epistemic: f64,
    pub aleatoric: f64,
    pub scale: f64,
}

/// The bounded implicit-history neighbor estimator used by the Metal pretraining path.
/// Distances arrive from the CUDA implementation of Metal's stratified shadow
/// Gaussian geometry; fitting remains on the host on both backends.
pub struct NeighborValueEstimator {
    pub(super) distances: [[f32; MAX_HISTORY]; MAX_HISTORY],
    pub(super) outcomes: [f32; MAX_HISTORY],
    pub(super) variances: [f32; MAX_HISTORY],
    pub(super) len: usize,
    base: usize,
    identities: [i64; MAX_HISTORY],
    observation: i64,
    fit_candidates: usize,
    fit_samples: usize,
    fit_seed: u64,
    defaults: NeighborEstimatorParams,
    pub(super) params: NeighborEstimatorParams,
    pub(super) y_scale: f64,
    default_scale: f64,
    fit_params: bool,
    adapt_neighbors: bool,
    fitted: bool,
}

impl NeighborValueEstimator {
    pub fn new(initial_value: f32, initial_variance: f32, fit_seed: u64) -> Result<Self, String> {
        Self::configured(initial_value, initial_variance, fit_seed, false, false)
    }

    pub fn configured(
        initial_value: f32,
        initial_variance: f32,
        fit_seed: u64,
        fit_params: bool,
        adapt_neighbors: bool,
    ) -> Result<Self, String> {
        Self::with_spec(
            initial_value,
            initial_variance,
            fit_seed,
            SurrogateSpec {
                fit: fit_params,
                adaptive: adapt_neighbors,
                ..Default::default()
            },
        )
    }

    pub fn with_spec(
        initial_value: f32,
        initial_variance: f32,
        fit_seed: u64,
        spec: SurrogateSpec,
    ) -> Result<Self, String> {
        let spec = spec.validate()?;
        if !initial_value.is_finite() || !initial_variance.is_finite() || initial_variance < 0.0 {
            return Err("neighbor value estimator initial observation is invalid".into());
        }
        let defaults = NeighborEstimatorParams {
            neighbors: spec.neighbors,
            epistemic: f64::from(spec.epistemic),
            aleatoric: f64::from(spec.aleatoric),
        };
        let mut outcomes = [0.0; MAX_HISTORY];
        let mut variances = [0.0; MAX_HISTORY];
        outcomes[0] = initial_value;
        variances[0] = initial_variance;
        Ok(Self {
            distances: [[0.0; MAX_HISTORY]; MAX_HISTORY],
            outcomes,
            variances,
            len: 1,
            base: 0,
            identities: [1; MAX_HISTORY],
            observation: 1,
            fit_candidates: spec.fit_candidates,
            fit_samples: spec.fit_samples,
            fit_seed,
            defaults,
            params: defaults,
            y_scale: f64::from(spec.scale),
            default_scale: f64::from(spec.scale),
            fit_params: spec.fit,
            adapt_neighbors: spec.adaptive,
            fitted: false,
        })
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn base_value(&self) -> f32 {
        self.outcomes[self.base]
    }

    pub const fn base_variance(&self) -> f32 {
        self.variances[self.base]
    }

    /// Stable identities used by acquisition draws and bounded retention.
    pub fn identities(&self) -> &[i64] {
        &self.identities[..self.len]
    }

    /// Squared distances in the same slot order as `identities`.
    pub fn distance_row(&self, slot: usize) -> Option<&[f32]> {
        (slot < self.len).then(|| &self.distances[slot][..self.len])
    }

    /// The sampled GPU row and host observation must replace this same slot.
    pub fn next_slot(&self) -> Result<usize, String> {
        crate::neighbor::retention(self.identities(), self.identities[self.base])
            .map_err(str::to_string)
    }

    pub const fn fit_state(&self) -> FitState {
        FitState {
            neighbors: self.params.neighbors,
            epistemic: self.params.epistemic,
            aleatoric: self.params.aleatoric,
            scale: self.y_scale,
        }
    }

    pub fn compact(&mut self) {
        let value = self.base_value();
        let variance = self.base_variance();
        let identity = self.identities[self.base];
        self.distances = [[0.0; MAX_HISTORY]; MAX_HISTORY];
        self.outcomes = [0.0; MAX_HISTORY];
        self.variances = [0.0; MAX_HISTORY];
        self.outcomes[0] = value;
        self.variances[0] = variance;
        self.len = 1;
        self.base = 0;
        self.identities[0] = identity;
        self.fitted = false;
        self.y_scale = self.default_scale;
    }

    pub fn incumbent_prediction(&self) -> Result<ValuePrediction, String> {
        let prediction = self.query(
            &self.distances[self.base][..self.len],
            Some(self.base),
            0.0,
            None,
        )?;
        Ok(ValuePrediction {
            mean: prediction.mean,
            standard_error: prediction.standard_error,
        })
    }

    pub fn predict(&self, distances: &[f32]) -> Result<ValuePrediction, String> {
        let prediction = self.query(distances, None, 0.0, None)?;
        Ok(ValuePrediction {
            mean: prediction.mean,
            standard_error: prediction.standard_error,
        })
    }

    pub fn acquire(
        &self,
        distances: &[f32],
        beta: f32,
        draws: Option<&[f32]>,
    ) -> Result<crate::neighbor::query::Prediction, String> {
        self.query(distances, None, beta, draws)
    }

    fn query(
        &self,
        distances: &[f32],
        excluded: Option<usize>,
        beta: f32,
        draws: Option<&[f32]>,
    ) -> Result<crate::neighbor::query::Prediction, String> {
        if distances.len() != self.len {
            return Err("invalid query history length".into());
        }
        let params = self.prediction_params();
        crate::neighbor::query::Query {
            distances,
            outcomes: &self.outcomes[..self.len],
            variances: &self.variances[..self.len],
            draws,
            neighbors: params.neighbors,
            epistemic: params.epistemic as f32,
            aleatoric: params.aleatoric as f32,
            scale: self.y_scale as f32,
            beta,
            excluded,
            incumbent: self.base,
            adaptive: self.adapt_neighbors,
        }
        .predict()
        .map_err(str::to_string)
    }

    pub fn append(
        &mut self,
        row: &[f32],
        value: f32,
        variance: f32,
        accepted: bool,
    ) -> Result<(), String> {
        if row.len() != self.len
            || row
                .iter()
                .any(|distance| !distance.is_finite() || *distance < 0.0)
            || !value.is_finite()
            || !variance.is_finite()
            || variance < 0.0
        {
            return Err(
                "neighbor value estimator observation or sampled distance row is invalid".into(),
            );
        }
        let next = self.next_slot()?;
        let identity = self
            .observation
            .checked_add(1)
            .ok_or("observation identity overflow")?;
        for (previous, &distance) in row.iter().enumerate() {
            self.distances[next][previous] = distance;
            self.distances[previous][next] = distance;
        }
        self.outcomes[next] = value;
        self.variances[next] = variance;
        self.distances[next][next] = 0.0;
        self.identities[next] = identity;
        self.observation = identity;
        self.len = (self.len + 1).min(MAX_HISTORY);
        if accepted {
            self.base = next;
        }
        if self.fit_params && self.len >= self.defaults.neighbors {
            self.fit()?;
        }
        Ok(())
    }

    fn prediction_params(&self) -> NeighborEstimatorParams {
        if self.fitted {
            self.params
        } else {
            self.defaults
        }
    }

    fn fit(&mut self) -> Result<(), String> {
        let n = self.len;
        let sum = self.outcomes[..n]
            .iter()
            .map(|&value| f64::from(value))
            .sum::<f64>();
        let sumsq = self.outcomes[..n]
            .iter()
            .map(|&value| f64::from(value).powi(2))
            .sum::<f64>();
        let mean = sum / n as f64;
        let variance = (sumsq / n as f64 - mean * mean).max(0.0);
        self.y_scale = variance.sqrt();
        if !self.y_scale.is_finite() || self.y_scale <= 0.0 {
            self.y_scale = 1.0;
        }
        let mut rng = StdRng::seed_from_u64(self.fit_seed.wrapping_add(n as u64));
        let mut candidates = Vec::with_capacity(self.fit_candidates + 1);
        // ENNFitter draws every epistemic scale first, followed by every
        // aleatoric scale. Preserve that RNG order so CUDA and Metal select
        // the same likelihood candidate from the same fit seed.
        let epistemic = (0..self.fit_candidates)
            .map(|_| 10.0_f64.powf(rng.gen_range(-3.0..=3.0)))
            .collect::<Vec<_>>();
        let aleatoric = (0..self.fit_candidates)
            .map(|_| 10.0_f64.powf(rng.gen_range(-3.0..=3.0)))
            .collect::<Vec<_>>();
        for (&epistemic, &aleatoric) in epistemic.iter().zip(&aleatoric) {
            candidates.push(NeighborEstimatorParams {
                neighbors: self.params.neighbors,
                epistemic,
                aleatoric,
            });
        }
        candidates.push(self.params);
        let selected = sampled_rows(n, self.fit_samples, &mut rng);
        let mut best: Option<(NeighborEstimatorParams, f64)> = None;
        for candidate in candidates {
            let score = self.loo_score(candidate, &selected);
            if score.is_finite() && best.is_none_or(|(_, best_score)| score >= best_score) {
                best = Some((candidate, score));
            }
        }
        // Metal fits only the epistemic and aleatoric scales at the configured
        // k. Query-local prefix selection is controlled independently; k is
        // not another global likelihood parameter.
        self.params = best
            .ok_or("neighbor value estimator scale fitting produced no finite candidate")?
            .0;
        self.fitted = true;
        Ok(())
    }

    fn loo_score(&self, params: NeighborEstimatorParams, selected: &[usize]) -> f64 {
        let mut score = 0.0;
        for &held in selected {
            let (mean, standard_error) = posterior(
                &self.distances[held][..self.len],
                &self.outcomes[..self.len],
                &self.variances[..self.len],
                params,
                self.y_scale,
                Some(held),
                false,
            );
            if !mean.is_finite() || !standard_error.is_finite() || standard_error <= 0.0 {
                return f64::NEG_INFINITY;
            }
            let target = f64::from(self.outcomes[held]) / self.y_scale;
            let mean = mean / self.y_scale;
            let standard_error = standard_error / self.y_scale;
            let residual = (target - mean) / standard_error;
            score += -0.5 * (2.0 * std::f64::consts::PI).ln()
                - standard_error.ln()
                - 0.5 * residual * residual;
        }
        score
    }
}

fn sampled_rows(n: usize, count: usize, rng: &mut StdRng) -> Vec<usize> {
    if count >= n {
        (0..n).collect()
    } else {
        sample(rng, n, count).into_iter().collect()
    }
}

fn posterior(
    distances: &[f32],
    outcomes: &[f32],
    variances: &[f32],
    params: NeighborEstimatorParams,
    y_scale: f64,
    excluded: Option<usize>,
    adaptive_neighbors: bool,
) -> (f64, f64) {
    let mut nearest = (0..distances.len())
        .filter(|index| Some(*index) != excluded)
        .collect::<Vec<_>>();
    nearest.sort_by(|&left, &right| {
        distances[left]
            .total_cmp(&distances[right])
            .then(left.cmp(&right))
    });
    nearest.truncate(params.neighbors.min(nearest.len()));
    if nearest.is_empty() {
        let fallback = excluded.unwrap_or(0).min(outcomes.len() - 1);
        return (
            f64::from(outcomes[fallback]),
            params.aleatoric.max(1.0e-12).sqrt() * y_scale,
        );
    }
    let scale_squared = (y_scale * y_scale).max(1.0e-12);
    if adaptive_neighbors {
        // Match Metal's query-local prefix selection. Scale fitting itself uses
        // the configured maximum k, exactly as ENNFitter::ask_distances does.
        let mut precision_sum = 0.0;
        let mut noise_sum = 0.0;
        let mut best_risk = f64::INFINITY;
        let mut chosen = 1;
        for (position, &index) in nearest.iter().enumerate() {
            let noise = params.aleatoric + f64::from(variances[index]) / scale_squared;
            let precision = 1.0
                / (1.0e-9 + params.epistemic * f64::from(distances[index]) + noise).max(1.0e-12);
            precision_sum += precision;
            noise_sum += precision * precision * noise;
            let variance_error = noise_sum / (precision_sum * precision_sum).max(1.0e-24);
            let risk = params.epistemic * f64::from(distances[index]) + variance_error.sqrt();
            if risk < best_risk {
                best_risk = risk;
                chosen = position + 1;
            }
        }
        nearest.truncate(chosen);
    }
    let weights = nearest
        .iter()
        .map(|&index| {
            1.0 / (1.0e-9
                + params.epistemic * f64::from(distances[index])
                + params.aleatoric
                + f64::from(variances[index]) / scale_squared)
        })
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f64>().max(1.0e-12);
    let mean = nearest
        .iter()
        .zip(&weights)
        .map(|(&index, &weight)| weight * f64::from(outcomes[index]))
        .sum::<f64>()
        / total;
    let aleatoric = nearest
        .iter()
        .zip(&weights)
        .map(|(&index, &weight)| {
            weight / total * (params.aleatoric + f64::from(variances[index]) / scale_squared)
        })
        .sum::<f64>();
    let standard_error = (1.0 / total + aleatoric).max(1.0e-12).sqrt() * y_scale;
    (mean, standard_error)
}
