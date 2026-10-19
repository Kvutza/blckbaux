//! Resolved surrogate settings transported unchanged to accelerator hosts.
use deser::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize)]
#[deser(rename_all = "kebab-case", deny_unknown_fields)]
pub struct SurrogateSpec {
    pub neighbors: usize,
    /// Coefficient multiplying squared parameter distance in the variance proxy.
    pub epistemic: f32,
    /// Additive observation-noise variance proxy before output scaling.
    pub aleatoric: f32,
    /// Output standard-deviation scale, not a variance.
    pub scale: f32,
    pub fit_candidates: usize,
    pub fit_samples: usize,
    pub fit: bool,
    pub adaptive: bool,
}

impl Default for SurrogateSpec {
    fn default() -> Self {
        Self {
            neighbors: crate::INITIAL_OBSERVATIONS,
            epistemic: 0.05,
            aleatoric: 0.05,
            scale: 1.0,
            fit_candidates: 4,
            fit_samples: 4,
            fit: false,
            adaptive: false,
        }
    }
}

impl SurrogateSpec {
    pub fn validate(self) -> Result<Self, String> {
        if !(1..=crate::MAX_HISTORY).contains(&self.neighbors)
            || self.fit_candidates == 0
            || self.fit_samples == 0
            || [self.epistemic, self.aleatoric, self.scale]
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err("invalid resident surrogate settings".into());
        }
        Ok(self)
    }
}
