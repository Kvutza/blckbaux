//! Scientific roles in the TOML optimizer recipe, not executable algorithms.
//!
//! Proposal generation constructs trials; the surrogate estimates their values;
//! acquisition selects a trial; acceptance decides whether it replaces the
//! incumbent; radius control sets subsequent step sizes. Fitting parameters is
//! distinct from adapting a query's neighbor count. Definitions here describe
//! the configuration; `conversion` and runtime guards establish supported paths.
//! Deserialization success does not certify a theorem or learning performance.

use super::*;

/// A closed optimizer recipe. The selected algorithm determines which
/// scientific roles exist; unrelated sections are rejected by deserialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(
    tag = "algorithm",
    rename_all = "kebab-case",
    rename_all_fields = "kebab-case",
    deny_unknown_fields
)]
pub enum OptimizerSpec {
    #[deser(rename = "operator-bo")]
    OperatorBo {
        candidate_generation: CandidateGenerationSpec,
        surrogate: SurrogateSpec,
        acquisition: AcquisitionSpec,
        acceptance: AcceptanceSpec,
        radius_control: RadiusControlSpec,
    },
    #[deser(rename = "turbo-enn")]
    TurboEnn {
        candidate_generation: CandidateGenerationSpec,
        surrogate: SurrogateSpec,
        acquisition: AcquisitionSpec,
        trust_region: TrustRegionSpec,
    },
    #[deser(rename = "muon")]
    Muon,
    #[deser(rename = "adamw")]
    Adamw,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
/// Pool and transformation used to construct trials around an incumbent.
/// This does not set the number of forward evaluations or committed updates.
pub struct CandidateGenerationSpec {
    /// Transformation program, separate from the underlying noise distribution.
    pub generator: crate::procedural_pool::ProposalMethod,
    /// TOTAL proposals across arms, distinct from fitting trials or objective calls.
    pub candidates: u32,
    /// Seed namespaces sharing a center, not independent trust regions.
    pub arms: u32,
    /// Underlying sampler selection; program transformations can change the
    /// distribution of the resulting parameter displacement.
    pub perturbation: Option<ProposalKernelSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
/// Noise sampler requested inside a proposal program, not its complete law.
pub struct ProposalKernelSpec {
    pub distribution: crate::ProposalKernel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum SurrogateEstimator {
    #[deser(rename = "knn")]
    NeighborValue,
    #[deser(rename = "enn")]
    Enn,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
/// Estimator, geometry, uncertainty scales, and fitting policy for retained data.
/// These settings do not by themselves define how a trial direction is generated.
pub struct SurrogateSpec {
    pub estimator: SurrogateEstimator,
    #[deser(default = default_neighbors())]
    /// Fixed k, or the configured neighbor budget when a local rule is enabled.
    pub neighbors: i32,
    #[deser(default)]
    /// Query-local neighbor policy, independent of hyperparameter fitting.
    pub neighbor_rule: NeighborRule,
    #[deser(default = epistemic_default())]
    pub epistemic_scale: f64,
    #[deser(default = aleatoric_default())]
    pub aleatoric_scale: f64,
    #[deser(default = default_scale())]
    pub y_scale: f64,
    #[deser(default)]
    /// How historical parameter dissimilarities are represented/computed;
    /// `latent` is not a reduced-dimensional parameterization of the model.
    pub geometry: HistoryGeometry,
    #[deser(default)]
    pub scaling: DistanceScaling,
    pub local_neighbors: Option<usize>,
    /// Scale fitting rule. This is independent of query-local neighbor choice.
    #[deser(flatten)]
    pub fit: SurrogateFit,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum NeighborRule {
    /// Use the configured neighbor count, clamped to available history at ask.
    #[default]
    Fixed,
    /// Request the selection kernel's query-local risk-prefix rule.
    RiskPrefix,
}

/// Keep schema defaults aligned with the resident ask's initial neighbor budget.
pub(super) fn default_neighbors() -> i32 {
    crate::trials::RoundConfig::default().surrogate.neighbors as i32
}

/// Initial epistemic scale from the same ask defaults used by the runtime.
pub(super) fn epistemic_default() -> f64 {
    f64::from(
        crate::trials::RoundConfig::default()
            .surrogate
            .epistemic_scale,
    )
}

/// Initial aleatoric scale from the resident ask, before optional fitting.
pub(super) fn aleatoric_default() -> f64 {
    f64::from(
        crate::trials::RoundConfig::default()
            .surrogate
            .aleatoric_scale,
    )
}

/// Initial output scale from the resident ask, before optional fitting.
pub(super) fn default_scale() -> f64 {
    f64::from(crate::trials::RoundConfig::default().surrogate.output_scale)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(
    tag = "fit",
    rename_all = "kebab-case",
    rename_all_fields = "kebab-case",
    deny_unknown_fields
)]
pub enum SurrogateFit {
    /// Keep supplied hyperparameters; neighbor adaptation remains independent.
    Fixed,
    #[deser(rename = "leave-one-out")]
    LeaveOneOut {
        /// Hyperparameter trials in the fitter, not candidate parameter vectors.
        fit_candidates: usize,
        /// Fitter's leave-one-out scoring budget, not fresh model evaluations.
        fit_samples: usize,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(
    tag = "function",
    rename_all = "kebab-case",
    rename_all_fields = "kebab-case",
    deny_unknown_fields
)]
pub enum AcquisitionSpec {
    #[deser(rename = "upper-confidence-bound")]
    Ucb {
        #[deser(default = default_beta())]
        beta: f64,
    },
    #[deser(rename = "thompson")]
    Thompson,
    /// Uniform selection control over the generated candidate pool.
    Random,
    Pareto {
        scales: Vec<f32>,
    },
    AugmentedChebyshev {
        scales: Vec<f32>,
        preferences: Vec<f32>,
        alpha: f32,
        seed_domain: String,
    },
}

/// Default count of MORBO regions, distinct from procedural seed arms.
fn default_regions() -> usize {
    4
}

/// Reuse the shared UCB exploration default rather than a second local constant.
fn default_beta() -> f64 {
    let AcquisitionConfig::UCB { beta } = AcquisitionConfig::default() else {
        unreachable!()
    };
    beta
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum AcceptanceCriterion {
    #[deser(rename = "positive-improvement")]
    PositiveImprovement,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
pub struct AcceptanceSpec {
    pub criterion: AcceptanceCriterion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(default, deny_unknown_fields, rename_all = "kebab-case")]
/// Dimensionless radius bounds and optional tensor scaling geometry.
/// A radius multiplies block proposal scales; it is not a full-vector norm.
pub struct TrustRegionBounds {
    pub initial: f64,
    pub min: f64,
    pub max: f64,
    pub shape: Option<TrustRegionShape>,
}

impl Default for TrustRegionBounds {
    fn default() -> Self {
        let length = TRLengthConfig::default();
        Self {
            initial: length.length_init,
            min: length.length_min,
            max: length.length_max,
            shape: None,
        }
    }
}

impl TrustRegionBounds {
    /// Check finite ordered bounds `0 < min <= initial <= max` without mutation.
    /// Valid bounds do not establish that a useful step fits inside the region.
    pub fn validate(&self) -> Result<(), String> {
        if !self.initial.is_finite()
            || !self.min.is_finite()
            || !self.max.is_finite()
            || self.min <= 0.0
            || self.min > self.initial
            || self.initial > self.max
        {
            return Err(
                "trust region bounds must be finite and satisfy 0 < min <= initial <= max".into(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(
    tag = "policy",
    rename_all = "kebab-case",
    rename_all_fields = "kebab-case",
    deny_unknown_fields
)]
pub enum RadiusControlSpec {
    #[deser(rename = "reliability")]
    Reliability {
        #[deser(flatten)]
        bounds: TrustRegionBounds,
        #[deser(flatten)]
        policy: crate::ReliabilityPolicy,
    },
}

impl RadiusControlSpec {
    /// Borrow radius bounds only; evidence thresholds remain in the policy.
    pub fn bounds(&self) -> &TrustRegionBounds {
        match self {
            Self::Reliability { bounds, .. } => bounds,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(
    tag = "policy",
    rename_all = "kebab-case",
    rename_all_fields = "kebab-case",
    deny_unknown_fields
)]
pub enum TrustRegionSpec {
    #[deser(rename = "turbo")]
    Turbo {
        #[deser(flatten)]
        bounds: TrustRegionBounds,
    },
    #[deser(rename = "morbo")]
    Morbo {
        #[deser(flatten)]
        bounds: TrustRegionBounds,
        #[deser(default = default_regions())]
        regions: usize,
        rescalarize: Rescalarize,
        clip: bool,
    },
    #[deser(rename = "reliability")]
    Reliability {
        #[deser(flatten)]
        bounds: TrustRegionBounds,
        #[deser(flatten)]
        policy: crate::ReliabilityPolicy,
    },
}

impl TrustRegionSpec {
    /// Borrow the same radius-bound role across controller alternatives.
    /// This accessor neither initializes a controller nor equates their behavior.
    pub fn bounds(&self) -> &TrustRegionBounds {
        match self {
            Self::Turbo { bounds }
            | Self::Morbo { bounds, .. }
            | Self::Reliability { bounds, .. } => bounds,
        }
    }
}
