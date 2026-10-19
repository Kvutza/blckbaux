import BlckbauxResearch.Optimization.Step

/-!
# Candidate selection guarantees

This module isolates the selector used inside one optimizer transaction.  The
selection function may be a posterior mean, UCB score, Thompson sample, or any
other acquisition score.  Its name does not supply a guarantee: the required
property is a simultaneous error bound on the candidate pool actually searched.
-/

namespace BlckbauxResearch.Optimization

/-- Uniform error of a selection function on the represented candidate pool. -/
def UniformError
    {Candidate : Type*}
    (truth estimate : Candidate → ℝ)
    (epsilon : ℝ) : Prop :=
  ∀ candidate, |estimate candidate - truth candidate| ≤ epsilon

/-- The selected candidate minimizes the selection function up to `slack`. -/
def ApproxMin
    {Candidate : Type*}
    (estimate : Candidate → ℝ)
    (selected : Candidate)
    (slack : ℝ) : Prop :=
  ∀ candidate, estimate selected ≤ estimate candidate + slack

/-- The selected candidate maximizes a gain-oriented acquisition up to `slack`. -/
def ApproxMax
    {Candidate : Type*}
    (acquisition : Candidate → ℝ)
    (selected : Candidate)
    (slack : ℝ) : Prop :=
  ∀ candidate, acquisition candidate ≤ acquisition selected + slack

/-- Candidate-specific score error. Unlike `UniformError`, this retains the
local uncertainty assigned to every member of the represented pool. -/
def PointError
    {Candidate : Type*}
    (truth estimate error : Candidate → ℝ) : Prop :=
  ∀ candidate, |estimate candidate - truth candidate| ≤ error candidate

/-- Pointwise score errors charge the selected and comparison candidates
separately. This avoids replacing locally adaptive KNN envelopes by their
worst value over the complete pool. -/
theorem Selection.pointBound
    {Candidate : Type*}
    (truth estimate error : Candidate → ℝ)
    (selected comparison : Candidate)
    (slack : ℝ)
    (errorBound : PointError truth estimate error)
    (selectedWithin : ApproxMin estimate selected slack) :
    truth selected - truth comparison ≤
      error selected + error comparison + slack := by
  have selectedError := (abs_le.mp (errorBound selected)).1
  have comparisonError := (abs_le.mp (errorBound comparison)).2
  have selectionError := selectedWithin comparison
  linarith

/-- Simultaneous error `epsilon` and selection slack imply regret at most `2ε + ξ`. -/
theorem Selection.regretBound
    {Candidate : Type*}
    (truth estimate : Candidate → ℝ)
    (selected comparison : Candidate)
    (epsilon slack : ℝ)
    (errorBound : UniformError truth estimate epsilon)
    (selectedWithin : ApproxMin estimate selected slack) :
    truth selected - truth comparison ≤ 2 * epsilon + slack := by
  have selectedError := (abs_le.mp (errorBound selected)).1
  have comparisonError := (abs_le.mp (errorBound comparison)).2
  have selectionError := selectedWithin comparison
  linarith

/-- Gain error `epsilon` and maximization slack imply a gain gap at most `2ε + ξ`. -/
theorem Selection.gainBound
    {Candidate : Type*}
    (truth acquisition : Candidate → ℝ)
    (selected comparison : Candidate)
    (epsilon slack : ℝ)
    (errorBound : UniformError truth acquisition epsilon)
    (selectedWithin : ApproxMax acquisition selected slack) :
    truth comparison - truth selected ≤ 2 * epsilon + slack := by
  have selectedError := (abs_le.mp (errorBound selected)).2
  have comparisonError := (abs_le.mp (errorBound comparison)).1
  have selectionError := selectedWithin comparison
  linarith

/-- The selector loses at most `2ε + ξ` of the opportunity in its candidate pool. -/
theorem Progress.lowerBound
    {State : Type*}
    {Candidate : Type*}
    (objective : Objective State)
    (incumbent : State)
    (candidate : Candidate → State)
    (estimate : Candidate → ℝ)
    (selected comparison : Candidate)
    (epsilon slack : ℝ)
    (errorBound : UniformError
      (fun index => objective (candidate index)) estimate epsilon)
    (selectedWithin : ApproxMin estimate selected slack) :
    opportunity objective incumbent (candidate comparison) -
        (2 * epsilon + slack) ≤
      progress objective incumbent (candidate selected) := by
  have regretBound := Selection.regretBound
    (fun index => objective (candidate index))
    estimate selected comparison epsilon slack errorBound selectedWithin
  unfold opportunity progress at *
  linarith

/-- Candidate-specific KNN envelopes preserve a local opportunity bound. -/
theorem Progress.pointLower
    {State : Type*}
    {Candidate : Type*}
    (objective : Objective State)
    (incumbent : State)
    (candidate : Candidate → State)
    (estimate error : Candidate → ℝ)
    (selected comparison : Candidate)
    (slack : ℝ)
    (errorBound : PointError
      (fun index => objective (candidate index)) estimate error)
    (selectedWithin : ApproxMin estimate selected slack) :
    opportunity objective incumbent (candidate comparison) -
        (error selected + error comparison + slack) ≤
      progress objective incumbent (candidate selected) := by
  have regretBound := Selection.pointBound
    (fun index => objective (candidate index)) estimate error
    selected comparison slack errorBound selectedWithin
  unfold opportunity progress at *
  linarith

/-- A simultaneous derivative-free pool matches a reference update whenever
one represented candidate covers the reference's progress plus the local KNN
selection penalty. The reference may be one Muon epoch; no mirrored proposal,
gradient reconstruction, or covariance model is required by this theorem. -/
theorem Progress.referenceBound
    {State : Type*}
    {Candidate : Type*}
    (objective : Objective State)
    (incumbent reference : State)
    (candidate : Candidate → State)
    (estimate error : Candidate → ℝ)
    (selected comparison : Candidate)
    (slack : ℝ)
    (errorBound : PointError
      (fun index => objective (candidate index)) estimate error)
    (selectedWithin : ApproxMin estimate selected slack)
    (coverage : progress objective incumbent reference +
        error selected + error comparison + slack ≤
      opportunity objective incumbent (candidate comparison)) :
    progress objective incumbent reference ≤
      progress objective incumbent (candidate selected) := by
  have lower := Progress.pointLower objective incumbent candidate estimate error
    selected comparison slack errorBound selectedWithin
  linarith

/-- Opportunity larger than the error-and-selection penalty forces positive progress. -/
theorem Progress.positiveBound
    {State : Type*}
    {Candidate : Type*}
    (objective : Objective State)
    (incumbent : State)
    (candidate : Candidate → State)
    (estimate : Candidate → ℝ)
    (selected comparison : Candidate)
    (epsilon slack : ℝ)
    (errorBound : UniformError
      (fun index => objective (candidate index)) estimate epsilon)
    (selectedWithin : ApproxMin estimate selected slack)
    (enoughOpportunity :
      2 * epsilon + slack <
        opportunity objective incumbent (candidate comparison)) :
    0 < progress objective incumbent (candidate selected) := by
  have lowerBound := Progress.lowerBound
    objective incumbent candidate estimate selected comparison epsilon slack
    errorBound selectedWithin
  linarith

/-- A gain-oriented acquisition loses at most `2ε + ξ` of pool opportunity. -/
theorem Gain.lowerBound
    {State : Type*}
    {Candidate : Type*}
    (objective : Objective State)
    (incumbent : State)
    (candidate : Candidate → State)
    (acquisition : Candidate → ℝ)
    (selected comparison : Candidate)
    (epsilon slack : ℝ)
    (errorBound : UniformError
      (fun index => progress objective incumbent (candidate index))
      acquisition epsilon)
    (selectedWithin : ApproxMax acquisition selected slack) :
    opportunity objective incumbent (candidate comparison) -
        (2 * epsilon + slack) ≤
      progress objective incumbent (candidate selected) := by
  have gainBound := Selection.gainBound
    (fun index => progress objective incumbent (candidate index))
    acquisition selected comparison epsilon slack errorBound selectedWithin
  unfold opportunity progress at *
  linarith

/-- Pool opportunity above the gain-error penalty forces positive progress. -/
theorem Gain.positiveBound
    {State : Type*}
    {Candidate : Type*}
    (objective : Objective State)
    (incumbent : State)
    (candidate : Candidate → State)
    (acquisition : Candidate → ℝ)
    (selected comparison : Candidate)
    (epsilon slack : ℝ)
    (errorBound : UniformError
      (fun index => progress objective incumbent (candidate index))
      acquisition epsilon)
    (selectedWithin : ApproxMax acquisition selected slack)
    (enoughOpportunity :
      2 * epsilon + slack <
        opportunity objective incumbent (candidate comparison)) :
    0 < progress objective incumbent (candidate selected) := by
  have lowerBound := Gain.lowerBound
    objective incumbent candidate acquisition selected comparison epsilon slack
    errorBound selectedWithin
  linarith

/-- Mirrored candidate pair guarantees progress bounded from below by linear alignment
minus curvature and surrogate selection penalties. -/
theorem Progress.mirrorBound
    {State : Type*}
    {Candidate : Type*}
    (objective : Objective State)
    (incumbent : State)
    (candidate : Candidate → State)
    (estimate : Candidate → ℝ)
    (selected : Candidate)
    (posCand negCand : Candidate)
    (epsilon slack : ℝ)
    (linearTerm curvature energy : ℝ)
    (errorBound : UniformError
      (fun index => objective (candidate index)) estimate epsilon)
    (selectedWithin : ApproxMin estimate selected slack)
    (hOpp : linearTerm - curvature * energy ≤
      max (opportunity objective incumbent (candidate posCand))
          (opportunity objective incumbent (candidate negCand))) :
    linearTerm - curvature * energy - (2 * epsilon + slack) ≤
      progress objective incumbent (candidate selected) := by
  have hPos := Progress.lowerBound
    objective incumbent candidate estimate selected posCand epsilon slack
    errorBound selectedWithin
  have hNeg := Progress.lowerBound
    objective incumbent candidate estimate selected negCand epsilon slack
    errorBound selectedWithin
  rcases le_total
      (opportunity objective incumbent (candidate posCand))
      (opportunity objective incumbent (candidate negCand)) with hLe | hLe
  · rw [max_eq_right hLe] at hOpp
    linarith
  · rw [max_eq_left hLe] at hOpp
    linarith

/-- When linear alignment strictly exceeds curvature plus the surrogate selection penalty,
the selected candidate achieves strictly positive progress. -/
theorem Progress.mirrorPositive
    {State : Type*}
    {Candidate : Type*}
    (objective : Objective State)
    (incumbent : State)
    (candidate : Candidate → State)
    (estimate : Candidate → ℝ)
    (selected : Candidate)
    (posCand negCand : Candidate)
    (epsilon slack : ℝ)
    (linearTerm curvature energy : ℝ)
    (errorBound : UniformError
      (fun index => objective (candidate index)) estimate epsilon)
    (selectedWithin : ApproxMin estimate selected slack)
    (hOpp : linearTerm - curvature * energy ≤
      max (opportunity objective incumbent (candidate posCand))
          (opportunity objective incumbent (candidate negCand)))
    (enoughOpportunity :
      2 * epsilon + slack < linearTerm - curvature * energy) :
    0 < progress objective incumbent (candidate selected) := by
  have hBound := Progress.mirrorBound
    objective incumbent candidate estimate selected posCand negCand
    epsilon slack linearTerm curvature energy errorBound selectedWithin hOpp
  linarith

/-!
A selected candidate's realized objective is not enough to validate the
simultaneous pool error used above.  Even with the estimate function and the
selected truth value fixed, an unobserved candidate can either agree with the
estimate or violate any proposed nonnegative error radius.
-/

/-- Selected-only truth cannot identify a simultaneous candidate-pool error bound. -/
theorem Selected.unidentified
    {Candidate : Type*}
    [DecidableEq Candidate]
    (estimate : Candidate → ℝ)
    (selected other : Candidate)
    (epsilon : ℝ)
    (distinct : selected ≠ other)
    (nonnegative : 0 ≤ epsilon) :
    ∃ truthA truthB : Candidate → ℝ,
      truthA selected = truthB selected ∧
      UniformError truthA estimate epsilon ∧
      ¬ UniformError truthB estimate epsilon := by
  let truthA : Candidate → ℝ := estimate
  let truthB : Candidate → ℝ := fun candidate =>
    if candidate = other then estimate candidate + epsilon + 1
    else estimate candidate
  refine ⟨truthA, truthB, ?_, ?_, ?_⟩
  · simp [truthA, truthB, distinct]
  · intro candidate
    simp [truthA, nonnegative]
  · intro errorBound
    have atOther := errorBound other
    have nonpositive :
        estimate other - (estimate other + epsilon + 1) ≤ 0 := by
      linarith
    simp only [truthB, if_pos rfl] at atOther
    rw [abs_of_nonpos nonpositive] at atOther
    linarith

#print axioms Selection.pointBound
#print axioms Progress.pointLower
#print axioms Progress.referenceBound

end BlckbauxResearch.Optimization
