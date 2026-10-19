import BlckbauxResearch.Optimization.Selection
import BlckbauxResearch.Optimization.ResearchProblem
import BlckbauxResearch.Precision.Float16
import BlckbauxResearch.Precision.Discrete

/-!
# Native FP16 optimizer transaction

This module formalizes an optimizer transaction operating directly on the native
IEEE binary16 state space.  By eliminating continuous ℝ^D embeddings and
post-hoc quantization, every candidate state is an exact physical FP16 state.

There are no unquantized latent centers or post-selection rounding errors in
this model. A candidate may still equal the incumbent, and construction of a
source proposal that changes stored words remains a separate obligation.
-/

namespace BlckbauxResearch.Optimization.Native

open BlckbauxResearch.Precision.Float16
open BlckbauxResearch.Precision.Discrete
open BlckbauxResearch.Optimization.ResearchProblem

/-- An optimizer transaction operating directly on native FP16 parameter states. -/
structure Transaction (parameterCount candidateCount : ℕ) where
  incumbent : State parameterCount
  candidate : Fin candidateCount → State parameterCount
  objective : Objective (State parameterCount)
  acquisition : Fin candidateCount → ℝ
  selected : Fin candidateCount
  comparison : Fin candidateCount
  epsilon : ℝ
  slack : ℝ
  errorBound : UniformError
    (fun index => progress objective incumbent (candidate index))
    acquisition
    epsilon
  selectedWithin : ApproxMax acquisition selected slack
  machine : Machine
  usage : ResourceUsage

/-- The native FP16 state selected by the transaction. -/
def Transaction.selectedState
    {parameterCount candidateCount : ℕ}
    (transaction : Transaction parameterCount candidateCount) :
    State parameterCount :=
  transaction.candidate transaction.selected

/-- The native comparison state in the candidate pool. -/
def Transaction.comparisonState
    {parameterCount candidateCount : ℕ}
    (transaction : Transaction parameterCount candidateCount) :
    State parameterCount :=
  transaction.candidate transaction.comparison

/-- Strict improvement is the acceptance rule for the native transaction. -/
def Transaction.accepted
    {parameterCount candidateCount : ℕ}
    (transaction : Transaction parameterCount candidateCount) : Prop :=
  transaction.objective transaction.selectedState <
    transaction.objective transaction.incumbent

/-- The next incumbent produced by strict acceptance on native FP16 states. -/
noncomputable def Transaction.finalState
    {parameterCount candidateCount : ℕ}
    (transaction : Transaction parameterCount candidateCount) :
    State parameterCount := by
  classical
  exact if transaction.accepted then
      transaction.selectedState
    else
      transaction.incumbent

/-- Complete measured wall time is checked against the declared 200 ms target. -/
def Transaction.meetsTarget
    {parameterCount candidateCount : ℕ}
    (transaction : Transaction parameterCount candidateCount) : Prop :=
  transaction.usage .wallTimeMs ≤ currentTarget.targetMs

/-- The 1000 ms ceiling is distinct from the 200 ms operating target. -/
def Transaction.meetsCeiling
    {parameterCount candidateCount : ℕ}
    (transaction : Transaction parameterCount candidateCount) : Prop :=
  transaction.usage .wallTimeMs ≤ currentTarget.ceilingMs

/-- Every native transaction meeting the target stays within the ceiling. -/
theorem Transaction.targetCeiling
    {parameterCount candidateCount : ℕ}
    (transaction : Transaction parameterCount candidateCount)
    (meetsTarget : transaction.meetsTarget) :
    transaction.meetsCeiling := by
  unfold Transaction.meetsTarget Transaction.meetsCeiling at *
  exact meetsTarget.trans (by norm_num [currentTarget])

/-- Enough opportunity in the native candidate pool forces strict acceptance. -/
theorem Transaction.improves
    {parameterCount candidateCount : ℕ}
    (transaction : Transaction parameterCount candidateCount)
    (enoughOpportunity :
      2 * transaction.epsilon + transaction.slack <
        opportunity transaction.objective transaction.incumbent
          transaction.comparisonState) :
    transaction.accepted := by
  have positive := Gain.positiveBound
    transaction.objective
    transaction.incumbent
    transaction.candidate
    transaction.acquisition
    transaction.selected
    transaction.comparison
    transaction.epsilon
    transaction.slack
    transaction.errorBound
    transaction.selectedWithin
    enoughOpportunity
  unfold Transaction.accepted Transaction.selectedState
  unfold progress at positive
  linarith

/--
If the selected candidate does not improve, the comparison opportunity is bounded
by the estimator error and selection slack.
-/
theorem Transaction.stalledBound
    {parameterCount candidateCount : ℕ}
    (transaction : Transaction parameterCount candidateCount)
    (notBetter :
      transaction.objective transaction.incumbent ≤
        transaction.objective transaction.selectedState) :
    opportunity transaction.objective transaction.incumbent
        transaction.comparisonState ≤
      2 * transaction.epsilon + transaction.slack := by
  have lowerBound := Gain.lowerBound
    transaction.objective
    transaction.incumbent
    transaction.candidate
    transaction.acquisition
    transaction.selected
    transaction.comparison
    transaction.epsilon
    transaction.slack
    transaction.errorBound
    transaction.selectedWithin
  unfold Transaction.selectedState at notBetter
  unfold progress at lowerBound
  unfold Transaction.comparisonState
  linarith

/-- Under sufficient opportunity, the native state is committed without quantizer loss. -/
theorem Transaction.commits
    {parameterCount candidateCount : ℕ}
    (transaction : Transaction parameterCount candidateCount)
    (enoughOpportunity :
      2 * transaction.epsilon + transaction.slack <
        opportunity transaction.objective transaction.incumbent
          transaction.comparisonState) :
    transaction.finalState = transaction.selectedState := by
  unfold Transaction.finalState
  rw [if_pos (transaction.improves enoughOpportunity)]

/--
Exact bit preservation: The committed state is bit-for-bit identical to the selected
candidate. There is zero post-hoc quantization error or rounding discrepancy.
-/
theorem Transaction.exactCommit
    {parameterCount candidateCount : ℕ}
    (transaction : Transaction parameterCount candidateCount)
    (enoughOpportunity :
      2 * transaction.epsilon + transaction.slack <
        opportunity transaction.objective transaction.incumbent
          transaction.comparisonState) :
    ∀ parameter, (transaction.finalState parameter).1 =
      (transaction.selectedState parameter).1 := by
  intro parameter
  rw [transaction.commits enoughOpportunity]

end BlckbauxResearch.Optimization.Native
