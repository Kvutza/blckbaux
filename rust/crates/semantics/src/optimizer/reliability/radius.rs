//! Calibrated acceptance and reliability-controlled proposal radius.

use crate::calibration::Calibration;

pub(super) const MIN_SCALE: f64 = 1.0e-12;
pub(super) const PRIOR: f64 = 1.0;
const SIGN_MASK: u64 = 0x8000_0000_0000_0000;
const EXP_MASK: u64 = 0x7ff0_0000_0000_0000;
const FRAC_MASK: u64 = 0x000f_ffff_ffff_ffff;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReliabilityAction {
    Hold,
    Expand,
    Shrink,
}

/// Accept a measured improvement only when it strictly clears its error margin.
///
/// This is the executable scalar commit boundary shared by the optimizer
/// backends.  The Mathlib-facing specification is
/// `BlckbauxResearch.Optimization.EnnCommit.accepts`.
pub fn accept_candidate(improvement: f64, margin: f64) -> bool {
    accept_bits(improvement.to_bits(), margin.to_bits())
}

fn nan_bits(bits: u64) -> bool {
    bits & EXP_MASK == EXP_MASK && bits & FRAC_MASK != 0
}

fn zero_bits(bits: u64) -> bool {
    bits & !SIGN_MASK == 0
}

fn order_bits(bits: u64) -> u64 {
    if bits & SIGN_MASK == 0 {
        bits | SIGN_MASK
    } else {
        !bits
    }
}

/// IEEE-754 binary64 strict comparison expressed only with integer operations.
///
/// NaNs are unordered and the two signed zeros compare equal, matching Rust's
/// native `f64` `>` relation.  Keeping the executable boundary in this form
/// lets Charon and Aeneas extract the actual comparison into Lean rather than
/// accepting a Boolean computed outside the checked model.
pub fn accept_bits(improvement: u64, margin: u64) -> bool {
    if nan_bits(improvement) || nan_bits(margin) {
        return false;
    }
    if zero_bits(improvement) && zero_bits(margin) {
        return false;
    }
    accept_decision(order_bits(improvement) > order_bits(margin))
}

/// Commit the candidate exactly when the measured gain clears its margin.
/// `accept_bits` performs the extracted binary64 comparison; this identity is
/// the final Boolean commit gate shared with the controller proof.
pub fn accept_decision(clears_margin: bool) -> bool {
    clears_margin
}

/// Select the radius adaptation action from the controller's Boolean gates.
///
/// This function is deliberately independent of floating-point arithmetic and
/// controller storage. Both production controllers call it after computing
/// their evidence predicates, and Aeneas translates this function to Lean.
pub fn decide_action(
    model_adequate: bool,
    model_inadequate: bool,
    sharp: bool,
    accepted: bool,
    positive_ratio: bool,
    boundary_step: bool,
    failure: bool,
) -> ReliabilityAction {
    if model_inadequate {
        ReliabilityAction::Shrink
    } else if model_adequate && sharp && accepted && positive_ratio && boundary_step {
        ReliabilityAction::Expand
    } else if failure && !accepted {
        ReliabilityAction::Shrink
    } else {
        ReliabilityAction::Hold
    }
}

impl ReliabilityAction {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Hold => "hold",
            Self::Expand => "expand",
            Self::Shrink => "shrink",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ReliabilityDecision {
    pub accepted: bool,
    pub improvement: f64,
    pub threshold: f64,
    pub predicted_improvement: f64,
    pub action: ReliabilityAction,
    pub next_radius: f32,
    pub standardized_error: f64,
    pub sharpness_ratio: f64,
    pub agreement_ratio: Option<f64>,
}

/// Calibration and scalar acceptance around the shared numerical controller.
/// The policy is identical to Metal's; only observation/calibration storage is
/// specific to this host adapter. Invalid evidence leaves both states intact.
pub struct RadiusController {
    pub(super) calibration: Calibration,
    control: super::control::RadiusControl,
}

impl RadiusController {
    pub fn new(initial: f32, minimum: f32, maximum: f32) -> Result<Self, String> {
        Self::configured(initial, minimum, maximum, Default::default())
    }

    pub fn configured(
        initial: f32,
        minimum: f32,
        maximum: f32,
        policy: super::control::ReliabilityPolicy,
    ) -> Result<Self, String> {
        Ok(Self {
            calibration: Calibration::default(),
            control: super::control::RadiusControl::new(
                policy,
                f64::from(initial),
                f64::from(minimum),
                f64::from(maximum),
            )?,
        })
    }

    pub fn radius(&self) -> f32 {
        self.control.length() as f32
    }

    /// Warm observations use the supplied margin without advancing evidence or
    /// calibration. The incumbent radius remains unchanged.
    pub fn observe_initial(
        &self,
        observed_improvement: f32,
        predicted_improvement: f32,
        threshold: f32,
    ) -> Result<ReliabilityDecision, String> {
        self.initial_evidence(
            f64::from(observed_improvement),
            f64::from(predicted_improvement),
            f64::from(threshold),
        )
    }

    pub fn initial_evidence(
        &self,
        observed_improvement: f64,
        predicted_improvement: f64,
        threshold: f64,
    ) -> Result<ReliabilityDecision, String> {
        super::control::validate_evidence(super::control::ReliabilityEvidence {
            observed_improvement,
            observation_variance: 0.0,
            predicted_improvement,
            prediction_variance: 0.0,
            accepted: false,
            nominal_radius: f64::from(self.radius()),
            response_radius: 0.0,
        })?;
        if !threshold.is_finite() {
            return Err("acceptance margin must be finite".into());
        }
        Ok(ReliabilityDecision {
            accepted: accept_candidate(observed_improvement, threshold),
            improvement: observed_improvement,
            threshold,
            predicted_improvement,
            action: ReliabilityAction::Hold,
            next_radius: self.radius(),
            standardized_error: 0.0,
            sharpness_ratio: 0.0,
            agreement_ratio: (predicted_improvement > 0.0)
                .then_some(observed_improvement / predicted_improvement),
        })
    }

    /// Validate one measured observation, apply the common policy, then update
    /// calibration when requested. The acceptance margin uses only past errors.
    pub fn observe(
        &mut self,
        observed_improvement: f32,
        observation_variance: f32,
        predicted_improvement: f32,
        prediction_variance: f32,
        response_radius: f32,
        calibrate: bool,
    ) -> Result<ReliabilityDecision, String> {
        self.observe_evidence(
            super::control::ReliabilityEvidence {
                observed_improvement: f64::from(observed_improvement),
                observation_variance: f64::from(observation_variance),
                predicted_improvement: f64::from(predicted_improvement),
                prediction_variance: f64::from(prediction_variance),
                accepted: false,
                nominal_radius: f64::from(self.radius()),
                response_radius: f64::from(response_radius),
            },
            calibrate,
        )
    }

    /// Preserve the FP64 evidence arithmetic used by the Metal adapter. The
    /// acceptance gate is computed here; the caller's accepted flag is ignored.
    pub fn observe_evidence(
        &mut self,
        mut evidence: super::control::ReliabilityEvidence,
        calibrate: bool,
    ) -> Result<ReliabilityDecision, String> {
        let improvement = evidence.observed_improvement;
        let threshold =
            f64::from(self.calibration.multiplier()) * evidence.observation_variance.sqrt();
        let accepted = accept_candidate(improvement, threshold);
        let mut calibration = self.calibration.clone();
        if calibrate {
            calibration.observe(
                improvement as f32,
                evidence.predicted_improvement as f32,
                (evidence.observation_variance + evidence.prediction_variance).sqrt() as f32,
            )?;
        }
        evidence.accepted = accepted;
        let result = self.control.update(evidence)?;
        self.calibration = calibration;
        Ok(ReliabilityDecision {
            accepted,
            improvement,
            threshold,
            predicted_improvement: evidence.predicted_improvement,
            action: result.action,
            next_radius: self.radius(),
            standardized_error: result.standardized_error.unwrap(),
            sharpness_ratio: result.sharpness_ratio.unwrap(),
            agreement_ratio: result.agreement_ratio,
        })
    }
}
