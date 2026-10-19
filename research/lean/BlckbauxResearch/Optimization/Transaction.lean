import BlckbauxResearch.Optimization.Selection
import BlckbauxResearch.Optimization.ResearchProblem
import BlckbauxResearch.Precision.Commit

/-!
# One executable optimizer transaction

This transaction is parameterized by a precision-specific `CommitModel`. Each
real proposal has a counterfactual executed realization: the state that would
execute if that proposal were selected. Rust may score a procedural pool and
materialize only its winner, as the Metal path does. The selection guarantee
must nevertheless refer to those executable realizations rather than treating a
latent descriptor as the model state. BF16 and FP16 supply separate commit models.
-/

namespace BlckbauxResearch.Optimization

open BlckbauxResearch.Precision
open BlckbauxResearch.Optimization.ResearchProblem

/-- The data and proof obligations for one candidate-pool transaction. -/
structure Transaction
    (parameterCount candidateCount : ℕ) where
  model : CommitModel parameterCount
  incumbent : model.State
  proposal : Fin candidateCount → Parameter.Proposal parameterCount
  admissible : ∀ index, model.admissible incumbent (proposal index)
  objective : Objective model.State
  acquisition : Fin candidateCount → ℝ
  selected : Fin candidateCount
  comparison : Fin candidateCount
  epsilon : ℝ
  slack : ℝ
  errorBound : UniformError
    (fun index => progress objective incumbent
      (model.commit incumbent (proposal index) (admissible index)))
    acquisition
    epsilon
  selectedWithin : ApproxMax acquisition selected slack
  machine : Machine
  usage : ResourceUsage

/-- The backend state that would execute if this proposal were selected. -/
def Transaction.candidate
    {parameterCount candidateCount : ℕ}
    (transaction : Transaction parameterCount candidateCount)
    (index : Fin candidateCount) : transaction.model.State :=
  transaction.model.commit
    transaction.incumbent
    (transaction.proposal index)
    (transaction.admissible index)

def Transaction.selectedState
    {parameterCount candidateCount : ℕ}
    (transaction : Transaction parameterCount candidateCount) :
    transaction.model.State :=
  transaction.candidate transaction.selected

def Transaction.comparisonState
    {parameterCount candidateCount : ℕ}
    (transaction : Transaction parameterCount candidateCount) :
    transaction.model.State :=
  transaction.candidate transaction.comparison

/-- Strict improvement is the acceptance rule for this semantic transaction. -/
def Transaction.accepted
    {parameterCount candidateCount : ℕ}
    (transaction : Transaction parameterCount candidateCount) : Prop :=
  transaction.objective transaction.selectedState <
    transaction.objective transaction.incumbent

/-- The next incumbent produced by strict acceptance. -/
noncomputable def Transaction.finalState
    {parameterCount candidateCount : ℕ}
    (transaction : Transaction parameterCount candidateCount) :
    transaction.model.State := by
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

/-- Every transaction meeting the target also stays within the ceiling. -/
theorem Transaction.targetCeiling
    {parameterCount candidateCount : ℕ}
    (transaction : Transaction parameterCount candidateCount)
    (meetsTarget : transaction.meetsTarget) :
    transaction.meetsCeiling := by
  unfold Transaction.meetsTarget Transaction.meetsCeiling at *
  exact meetsTarget.trans (by norm_num [currentTarget])

/-- Enough executed-state opportunity forces the strict acceptance condition. -/
theorem Transaction.improves
    {parameterCount candidateCount : ℕ}
    (transaction : Transaction parameterCount candidateCount)
    (enoughOpportunity :
      2 * transaction.epsilon + transaction.slack <
        opportunity transaction.objective transaction.incumbent
          transaction.comparisonState) :
    transaction.accepted := by
  have errorBound : UniformError
      (fun index => progress transaction.objective transaction.incumbent
        (transaction.candidate index))
      transaction.acquisition
      transaction.epsilon := by
    simpa [Transaction.candidate] using transaction.errorBound
  have positive := Gain.positiveBound
    transaction.objective
    transaction.incumbent
    transaction.candidate
    transaction.acquisition
    transaction.selected
    transaction.comparison
    transaction.epsilon
    transaction.slack
    errorBound
    transaction.selectedWithin
    enoughOpportunity
  unfold Transaction.accepted Transaction.selectedState
  unfold progress at positive
  linarith

/--
If the selected executable state does not improve, then every chosen comparison
opportunity is no larger than the estimator-and-selection penalty.  This is the
checked diagnostic split between proposal opportunity and selection quality.
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
  have errorBound : UniformError
      (fun index ↦ progress transaction.objective transaction.incumbent
        (transaction.candidate index))
      transaction.acquisition
      transaction.epsilon := by
    simpa [Transaction.candidate] using transaction.errorBound
  have lowerBound := Gain.lowerBound
    transaction.objective
    transaction.incumbent
    transaction.candidate
    transaction.acquisition
    transaction.selected
    transaction.comparison
    transaction.epsilon
    transaction.slack
    errorBound
    transaction.selectedWithin
  unfold Transaction.selectedState at notBetter
  unfold progress at lowerBound
  unfold Transaction.comparisonState
  linarith

/-- Under the progress condition, the transaction commits the selected backend state. -/
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

end BlckbauxResearch.Optimization
