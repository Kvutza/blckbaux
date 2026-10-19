import BlckbauxResearch.Forward.Quality
import BlckbauxResearch.Optimization.Effectiveness

/-!
# Pretraining effectiveness profile

The first concrete quality axes separate held-out language loss, test failure,
and compilation failure. Runtime and memory are not quality axes; they remain
in the machine-resource vector. Additional quality axes can be introduced by
using another metric type without changing point, curve, or rate dominance.
-/

namespace BlckbauxResearch.Optimization.PretrainProfile

open Effectiveness ResearchProblem

inductive PretrainMetric where
  | heldoutNll
  | codeFailure
  | compileFailure
  deriving DecidableEq, Repr

/-- Thresholds defining zero gap on each initial pretraining quality axis. -/
structure QualityTarget where
  heldoutNll : ℝ
  codeFailure : ℝ
  compileFailure : ℝ

def PretrainMetric.loss
    (metric : PretrainMetric) (measurements : Forward.Quality.Measurements) : ℝ :=
  match metric with
  | .heldoutNll => measurements.heldoutLoss
  | .codeFailure => Forward.Quality.failure measurements.evidence
  | .compileFailure => if measurements.evidence.compiled then 0 else 1

def PretrainMetric.target
    (metric : PretrainMetric) (targets : QualityTarget) : ℝ :=
  match metric with
  | .heldoutNll => targets.heldoutNll
  | .codeFailure => targets.codeFailure
  | .compileFailure => targets.compileFailure

/-- Nonnegative excess above a declared quality threshold. -/
def qualityGap
    (targets : QualityTarget) (measurements : Forward.Quality.Measurements)
    (metric : PretrainMetric) : ℝ :=
  max 0 (metric.loss measurements - metric.target targets)

theorem qualityNonneg
    (targets : QualityTarget) (measurements : Forward.Quality.Measurements)
    (metric : PretrainMetric) :
    0 ≤ qualityGap targets measurements metric := by
  simp [qualityGap]

/-- Build a comparison profile from measured or proved budget-indexed bounds. -/
def profile
    (targets : QualityTarget)
    (measurements : ℕ → Forward.Quality.Measurements)
    (usage : ℕ → ResourceUsage)
    (tokenBound : ∀ tokens, usage tokens .tokensEvaluated ≤ tokens)
    (failure : ℕ → NNReal)
    (failureValid : ∀ tokens, failure tokens ≤ 1) :
    EffectProfile PretrainMetric where
  gap tokens metric := qualityGap targets (measurements tokens) metric
  gapNonneg tokens metric := qualityNonneg targets (measurements tokens) metric
  usage := usage
  tokenBound := tokenBound
  failure := failure
  failureValid := failureValid

#print axioms qualityNonneg

end BlckbauxResearch.Optimization.PretrainProfile
