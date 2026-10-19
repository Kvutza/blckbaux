import BlckbauxResearch.Optimization.Smooth
import BlckbauxResearch.Optimization.Moments
import BlckbauxResearch.Optimization.Selection

/-!
# Perturbation, execution, and selection in one bound

The pool has an incumbent and a mirrored full-parameter pair. The selected side
comes from an arbitrary feedback-based estimate, not a gradient oracle. Uniform
score accuracy and execution fidelity are explicit obligations. A measure can
be a history-conditioned law; no independence across rounds is assumed here.
This is a research mechanism, not an assertion that the deployed four-slot pool
implements mirrored search or satisfies these hypotheses.
-/
namespace BlckbauxResearch.Optimization.Search
noncomputable section
open scoped BigOperators
open MeasureTheory Set

abbrev Side := Fin 3

def candidate {Space : Type*} [AddGroup Space] (state delta : Space) (side : Side) : Space :=
  if side = 0 then state else if side = 1 then state + delta else state + (-delta)

variable {Space : Type*} [NormedAddCommGroup Space] [NormedSpace ℝ Space]

/-- Selection against executed candidates pays one comparison-side execution
penalty, plus the existing two-score-error and approximate-selection penalties. -/
theorem selected (objective : Space → ℝ) (state delta : Space)
    (execute : Side → Space) (estimate : Side → ℝ) (choice : Side)
    (error slack precision cap : ℝ)
    (accurate : UniformError (fun side => objective (execute side)) estimate error)
    (chosen : ApproxMin estimate choice slack)
    (faithful : ∀ side, objective (execute side) ≤
      objective (candidate state delta side) + precision)
    (mirrorBound : min (objective (state + delta)) (objective (state + (-delta))) -
      objective state ≤ -|fderiv ℝ objective state delta| + cap / 2) :
    objective (execute choice) - objective state ≤
      -|fderiv ℝ objective state delta| + cap / 2 + 2 * error + slack + precision := by
  have positive := Selection.regretBound
    (fun side => objective (execute side)) estimate choice 1 error slack accurate chosen
  have negative := Selection.regretBound
    (fun side => objective (execute side)) estimate choice 2 error slack accurate chosen
  have first := faithful 1
  have second := faithful 2
  simp only [candidate, show (1 : Side) ≠ 0 by decide, show (2 : Side) ≠ 0 by decide,
    show (2 : Side) ≠ 1 by decide, ↓reduceIte] at first second
  rcases le_total (objective (state + delta)) (objective (state + (-delta))) with left | right
  · rw [min_eq_left left] at mirrorBound
    linarith
  · rw [min_eq_right right] at mirrorBound
    linarith

/-- Nonlinear objective bounds are supplied by Mathlib calculus, not by an
assumed first-order surrogate. -/
theorem calculus (objective : Space → ℝ) (state delta : Space)
    (execute : Side → Space) (estimate : Side → ℝ) (choice : Side)
    (error slack precision cap : ℝ)
    (accurate : UniformError (fun side => objective (execute side)) estimate error)
    (chosen : ApproxMin estimate choice slack)
    (faithful : ∀ side, objective (execute side) ≤
      objective (candidate state delta side) + precision)
    (positive : ∀ time ∈ Icc (0 : ℝ) 1,
      ContDiffAt ℝ 2 objective (state + time • delta))
    (negative : ∀ time ∈ Icc (0 : ℝ) 1,
      ContDiffAt ℝ 2 objective (state + time • (-delta)))
    (positiveCap : ∀ time ∈ Icc (0 : ℝ) 1,
      iteratedFDeriv ℝ 2 objective (state + time • delta) (fun _ => delta) ≤ cap)
    (negativeCap : ∀ time ∈ Icc (0 : ℝ) 1,
      iteratedFDeriv ℝ 2 objective (state + time • (-delta)) (fun _ => -delta) ≤ cap) :
    objective (execute choice) - objective state ≤
      -|fderiv ℝ objective state delta| + cap / 2 + 2 * error + slack + precision :=
  selected objective state delta execute estimate choice error slack precision cap
    accurate chosen faithful
    (Smooth.mirrored objective state delta cap positive negative positiveCap negativeCap)

/-- A symbolic expected progress certificate keeps alignment, covariance,
proposal mean, selector error, and execution error separate. Integrability is
required explicitly to avoid Lean's default value for undefined integrals. -/
theorem expected {World : Type*} [MeasurableSpace World] {parameters : ℕ}
    (measure : Measure World) [IsProbabilityMeasure measure]
    (objective : Parameter.State parameters → ℝ) (state : Parameter.State parameters)
    (delta : World → Parameter.Proposal parameters) (next : World → Parameter.State parameters)
    (matrix : Matrix (Fin parameters) (Fin parameters) ℝ) (penalty : World → ℝ)
    (moments : ∀ index, MemLp (fun world => delta world index) 2 measure)
    (lossMoment : Integrable (fun world => objective (next world)) measure)
    (alignMoment : Integrable (fun world => |fderiv ℝ objective state (delta world)|) measure)
    (penaltyMoment : Integrable penalty measure)
    (step : ∀ world, objective (next world) - objective state ≤
      -|fderiv ℝ objective state (delta world)| +
        Moments.quadratic matrix (fun index => delta world index) / 2 + penalty world) :
    (∫ world, objective (next world) ∂measure) - objective state ≤
      -(∫ world, |fderiv ℝ objective state (delta world)| ∂measure) +
      (Matrix.trace (matrix * Moments.covariance measure (fun world index => delta world index)) +
        Moments.quadratic matrix (Moments.mean measure (fun world index => delta world index))) / 2 +
      ∫ world, penalty world ∂measure := by
  have curvature := Moments.integrable measure (fun world index => delta world index) matrix moments
  have negativeMoment : Integrable
      (fun world => -|fderiv ℝ objective state (delta world)|) measure := alignMoment.neg
  have mixedMoment : Integrable (fun world => -|fderiv ℝ objective state (delta world)| +
      Moments.quadratic matrix (fun index => delta world index) / 2) measure :=
    negativeMoment.add (curvature.div_const 2)
  have upper := mixedMoment.add penaltyMoment
  have bound := integral_mono (lossMoment.sub (integrable_const _)) upper step
  simp only [Pi.sub_apply, Pi.add_apply] at bound
  rw [integral_sub lossMoment (integrable_const _), integral_const,
    integral_add mixedMoment penaltyMoment,
    integral_add negativeMoment (curvature.div_const 2), integral_neg,
    integral_div, Moments.traceForm measure (fun world index => delta world index) matrix moments] at bound
  simpa using bound

/-- Dominating the curvature and error costs yields an actual expected decrease,
but that dominance is not inferred from a distribution's name. -/
theorem decrease (initial expectedLoss alignment curvature penalty gain : ℝ)
    (bound : expectedLoss - initial ≤ -alignment + curvature + penalty)
    (useful : gain ≤ alignment - curvature - penalty) :
    expectedLoss ≤ initial - gain := by linarith

/-- Operational research core: proposals, executed candidates, feedback scores,
and selection. The objective and its derivatives are not inputs to this core. -/
structure Round (parameters : ℕ) (World : Type*) where
  delta : World → Parameter.Proposal parameters
  execute : World → Side → Parameter.State parameters
  estimate : World → Side → ℝ
  choice : World → Side

def Round.next {parameters : ℕ} {World : Type*} (round : Round parameters World)
    (world : World) : Parameter.State parameters := round.execute world (round.choice world)

/-- The proof view names actual obligations, rather than asserting a desired
regret rate. This is not a source refinement or a calibration certificate. -/
structure Certificate {World : Type*} [MeasurableSpace World] {parameters : ℕ}
    (measure : Measure World) (objective : Parameter.State parameters → ℝ)
    (state : Parameter.State parameters) (round : Round parameters World)
    (matrix : Matrix (Fin parameters) (Fin parameters) ℝ) (error slack precision : ℝ) where
  accurate : ∀ world, UniformError (fun side => objective (round.execute world side))
    (round.estimate world) error
  chosen : ∀ world, ApproxMin (round.estimate world) (round.choice world) slack
  faithful : ∀ world side, objective (round.execute world side) ≤
    objective (candidate state (round.delta world) side) + precision
  positive : ∀ world time, time ∈ Icc (0 : ℝ) 1 →
    ContDiffAt ℝ 2 objective (state + time • round.delta world)
  negative : ∀ world time, time ∈ Icc (0 : ℝ) 1 →
    ContDiffAt ℝ 2 objective (state + time • (-round.delta world))
  positiveCap : ∀ world time, time ∈ Icc (0 : ℝ) 1 →
    iteratedFDeriv ℝ 2 objective (state + time • round.delta world) (fun _ => round.delta world) ≤
      Moments.quadratic matrix (fun index => round.delta world index)
  negativeCap : ∀ world time, time ∈ Icc (0 : ℝ) 1 →
    iteratedFDeriv ℝ 2 objective (state + time • (-round.delta world)) (fun _ => -round.delta world) ≤
      Moments.quadratic matrix (fun index => round.delta world index)
  moments : ∀ index, MemLp (fun world => round.delta world index) 2 measure
  lossMoment : Integrable (fun world => objective (round.next world)) measure
  alignMoment : Integrable (fun world => |fderiv ℝ objective state (round.delta world)|) measure

theorem Certificate.realized {World : Type*} [MeasurableSpace World] {parameters : ℕ}
    {measure : Measure World} {objective : Parameter.State parameters → ℝ}
    {state : Parameter.State parameters} {round : Round parameters World}
    {matrix : Matrix (Fin parameters) (Fin parameters) ℝ} {error slack precision : ℝ}
    (certificate : Certificate measure objective state round matrix error slack precision) (world : World) :
    objective (round.next world) - objective state ≤
      -|fderiv ℝ objective state (round.delta world)| +
        Moments.quadratic matrix (fun index => round.delta world index) / 2 +
          (2 * error + slack + precision) := by
  have bound := calculus objective state (round.delta world) (round.execute world)
    (round.estimate world) (round.choice world) error slack precision
    (Moments.quadratic matrix (fun index => round.delta world index))
    (certificate.accurate world) (certificate.chosen world) (certificate.faithful world)
    (certificate.positive world) (certificate.negative world)
    (certificate.positiveCap world) (certificate.negativeCap world)
  simpa only [Round.next, add_assoc] using bound

/-- The complete local composition: calculus, full covariance, executed states,
and feedback-based selection. No approximation inequality is an input here. -/
theorem Certificate.expected {World : Type*} [MeasurableSpace World] {parameters : ℕ}
    {measure : Measure World} [IsProbabilityMeasure measure]
    {objective : Parameter.State parameters → ℝ} {state : Parameter.State parameters}
    {round : Round parameters World} {matrix : Matrix (Fin parameters) (Fin parameters) ℝ}
    {error slack precision : ℝ}
    (certificate : Certificate measure objective state round matrix error slack precision) :
    (∫ world, objective (round.next world) ∂measure) - objective state ≤
      -(∫ world, |fderiv ℝ objective state (round.delta world)| ∂measure) +
      (Matrix.trace (matrix * Moments.covariance measure (fun world index => round.delta world index)) +
        Moments.quadratic matrix (Moments.mean measure (fun world index => round.delta world index))) / 2 +
      (2 * error + slack + precision) := by
  have bound := Search.expected measure objective state round.delta round.next matrix
    (fun _ => 2 * error + slack + precision) certificate.moments certificate.lossMoment
    certificate.alignMoment (integrable_const _) certificate.realized
  simpa using bound

#print axioms calculus
#print axioms expected
#print axioms Certificate.realized
#print axioms Certificate.expected
end
end BlckbauxResearch.Optimization.Search
