//! FP32 reference for the resident Metal posterior and acquisition operation.
//! Order of accumulation and first-index tie breaking are part of this rule.
use super::{MAX_HISTORY, Prefix};

pub struct Query<'a> {
    pub distances: &'a [f32],
    pub outcomes: &'a [f32],
    pub variances: &'a [f32],
    pub draws: Option<&'a [f32]>,
    pub neighbors: usize,
    pub epistemic: f32,
    pub aleatoric: f32,
    pub scale: f32,
    pub beta: f32,
    pub excluded: Option<usize>,
    pub incumbent: usize,
    pub adaptive: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prediction {
    pub mean: f32,
    /// Total predictive standard error, including the observation-noise term.
    pub standard_error: f32,
    /// Epistemic standard error used by UCB and the sampled function.
    pub epistemic_error: f32,
    pub score: f32,
    pub neighbors: usize,
}

impl Query<'_> {
    pub fn predict(&self) -> Result<Prediction, &'static str> {
        let len = self.distances.len();
        if len == 0
            || len > MAX_HISTORY
            || self.outcomes.len() != len
            || self.variances.len() != len
            || self.neighbors == 0
            || self.incumbent >= len
            || self.excluded.is_some_and(|i| i >= len)
            || self
                .draws
                .is_some_and(|d| d.len() != len || d.iter().any(|v| !v.is_finite()))
            || self
                .distances
                .iter()
                .chain(self.variances)
                .any(|v| !v.is_finite() || *v < 0.0)
            || self.outcomes.iter().any(|v| !v.is_finite())
            || [self.epistemic, self.aleatoric, self.scale, self.beta]
                .iter()
                .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err("invalid resident posterior query");
        }
        let mut nearest = [0; MAX_HISTORY];
        let mut count = 0;
        for i in 0..len {
            if Some(i) == self.excluded {
                continue;
            }
            let mut position = count;
            while position > 0 && self.distances[i] < self.distances[nearest[position - 1]] {
                nearest[position] = nearest[position - 1];
                position -= 1;
            }
            nearest[position] = i;
            count += 1;
        }
        count = count.min(self.neighbors);
        if count == 0 {
            return Ok(Prediction {
                mean: self.outcomes[self.incumbent],
                standard_error: self.aleatoric.max(1.0e-12).sqrt() * self.scale,
                epistemic_error: 0.0,
                score: self.outcomes[self.incumbent],
                neighbors: 0,
            });
        }
        let scale_squared = (self.scale * self.scale).max(1.0e-12);
        let noise = |i: usize| self.aleatoric + self.variances[i] / scale_squared;
        let precision = |i: usize| {
            1.0 / (1.0e-9
                + self.epistemic * self.distances[i]
                + self.aleatoric
                + self.variances[i] / scale_squared)
                .max(1.0e-12)
        };
        if self.adaptive {
            let mut prefix = Prefix::default();
            for (rank, &i) in nearest[..count].iter().enumerate() {
                prefix.push(rank as u32 + 1, self.distances[i], self.epistemic, noise(i));
            }
            count = prefix.count as usize;
        }
        let (mut sum, mut value, mut reference) = (0.0_f32, 0.0_f32, f32::MIN_POSITIVE);
        for &i in &nearest[..count] {
            let weight = precision(i);
            sum += weight;
            value += weight * self.outcomes[i];
            reference = reference.max(weight);
        }
        let mean = value / sum.max(1.0e-12);
        let mut aleatoric = 0.0;
        for &i in &nearest[..count] {
            aleatoric += precision(i) / sum.max(1.0e-12) * noise(i);
        }
        let variance = 1.0 / sum.max(1.0e-12);
        let error = (variance + aleatoric).sqrt() * self.scale;
        let epistemic = variance.sqrt() * self.scale;
        let score = if let Some(draws) = self.draws {
            let (mut noise, mut squared) = (0.0_f32, 0.0_f32);
            for &i in &nearest[..count] {
                let weight = precision(i) / reference;
                noise += weight * draws[i];
                squared += weight * weight;
            }
            mean + epistemic * (noise / squared.sqrt().max(1.0e-12))
        } else {
            mean + self.beta * epistemic
        };
        if [mean, error, epistemic, score]
            .iter()
            .any(|v| !v.is_finite())
        {
            return Err("resident posterior arithmetic overflow");
        }
        Ok(Prediction {
            mean,
            standard_error: error,
            epistemic_error: epistemic,
            score,
            neighbors: count,
        })
    }
}
