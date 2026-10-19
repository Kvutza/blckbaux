import BlckbauxResearch.Optimization.Regret
import Mathlib.Probability.Independence.Basic
import Mathlib.MeasureTheory.Measure.Real

/-!
# Candidate coverage and batch depth

This module separates three quantities that a large candidate pool can obscure:

* candidate coverage depends on the joint proposal law;
* selection error changes the objective threshold attained after coverage;
* execution latency depends on parallel width and sequential depth.

Independent candidates with a common miss probability `q` miss jointly with
probability `q ^ N`. Duplicating one candidate does not improve coverage. These
are not claims that source BLCKBAUX candidates are independent: that is an explicit
source-law obligation.
-/

namespace BlckbauxResearch.Optimization.Batch

open MeasureTheory ProbabilityTheory Set
open scoped ENNReal

variable {Ω ι : Type*}

/-- Every candidate misses the declared target. -/
def Miss (failure : ι → Set Ω) : Set Ω :=
  ⋂ index, failure index

/-- At least one candidate reaches the declared target. -/
def Hit (failure : ι → Set Ω) : Set Ω :=
  (Miss failure)ᶜ

/-- Repeating one failure event any nonzero number of times leaves it unchanged. -/
theorem duplicateMiss [Nonempty ι] (failure : Set Ω) :
    Miss (fun _ : ι => failure) = failure := by
  ext world
  simp [Miss]

variable [MeasurableSpace Ω] {measure : Measure Ω}

/-- Independent equal-probability failures multiply across a finite pool. -/
theorem missProb [Fintype ι] (failure : ι → Set Ω) (q : ℝ≥0∞)
    (independent : iIndepSet failure measure)
    (equalProb : ∀ index, measure (failure index) = q) :
    measure (Miss failure) = q ^ Fintype.card ι := by
  calc
    measure (Miss failure) =
        measure (⋂ index ∈ (Finset.univ : Finset ι), failure index) := by
      simp [Miss]
    _ = ∏ index ∈ (Finset.univ : Finset ι), measure (failure index) :=
      independent.meas_biInter Finset.univ
    _ = q ^ Fintype.card ι := by simp [equalProb]

/-- The probability of at least one hit is one minus the joint miss probability. -/
theorem hitProb [Fintype ι] [IsProbabilityMeasure measure]
    (failure : ι → Set Ω) (q : ℝ≥0∞)
    (measurable : ∀ index, MeasurableSet (failure index))
    (independent : iIndepSet failure measure)
    (equalProb : ∀ index, measure (failure index) = q) :
    measure (Hit failure) = 1 - q ^ Fintype.card ι := by
  rw [Hit, measure_compl]
  · rw [measure_univ, missProb failure q independent equalProb]
  · exact MeasurableSet.iInter measurable
  · exact measure_ne_top measure (Miss failure)

/-- Any certified joint miss bound transfers through an approximate selector.

This is the dependence-agnostic theorem. Independence, negative dependence,
geometric covering, or a source-specific argument may each be used to prove
`missBounded`; none is built into the selector result.
-/
theorem selectedBound [Fintype ι] [IsProbabilityMeasure measure]
    (loss : Ω → ι → ℝ) (selected : Ω → ℝ) (target penalty : ℝ)
    (failure : ι → Set Ω) (missBound : ℝ≥0∞)
    (failureEq : ∀ index, failure index = {world | target < loss world index})
    (measurable : ∀ index, MeasurableSet (failure index))
    (missBounded : measure (Miss failure) ≤ missBound)
    (selection : ∀ world index, selected world ≤ loss world index + penalty) :
    1 - missBound ≤ measure {world | selected world ≤ target + penalty} := by
  have hitMeasure : measure (Hit failure) = 1 - measure (Miss failure) := by
    rw [Hit, measure_compl, measure_univ]
    · exact MeasurableSet.iInter measurable
    · exact measure_ne_top measure (Miss failure)
  calc
    1 - missBound ≤ 1 - measure (Miss failure) :=
      tsub_le_tsub_left missBounded 1
    _ = measure (Hit failure) := hitMeasure.symm
    _ ≤ measure {world | selected world ≤ target + penalty} := by
      apply measure_mono
      intro world reached
      simp only [Hit, Miss, mem_compl_iff, mem_iInter, not_forall] at reached
      obtain ⟨index, reaches⟩ := reached
      rw [failureEq index] at reaches
      exact (selection world index).trans <| by
        simpa [add_comm] using add_le_add_right (le_of_not_gt reaches) penalty

/-- A selector penalty shifts the reached threshold but not pool coverage. -/
theorem selectedHit [Fintype ι] [IsProbabilityMeasure measure]
    (loss : Ω → ι → ℝ) (selected : Ω → ℝ) (target penalty : ℝ)
    (failure : ι → Set Ω) (q : ℝ≥0∞)
    (failureEq : ∀ index, failure index = {world | target < loss world index})
    (measurable : ∀ index, MeasurableSet (failure index))
    (independent : iIndepSet failure measure)
    (equalProb : ∀ index, measure (failure index) = q)
    (selection : ∀ world index, selected world ≤ loss world index + penalty) :
    1 - q ^ Fintype.card ι ≤ measure {world | selected world ≤ target + penalty} := by
  rw [← hitProb failure q measurable independent equalProb]
  apply measure_mono
  intro world reached
  simp only [Hit, Miss, mem_compl_iff, mem_iInter, not_forall] at reached
  obtain ⟨index, reaches⟩ := reached
  rw [failureEq index] at reaches
  exact (selection world index).trans <| by
    simpa [add_comm] using add_le_add_right (le_of_not_gt reaches) penalty

/-- Joint pool coverage composes with the concrete `2 * error + slack`
selection penalty to give a terminal simple-regret probability guarantee. -/
theorem regretBound [Fintype ι] [IsProbabilityMeasure measure]
    {State : Type*} (objective : State → ℝ) (best : State)
    (candidate : Ω → ι → State) (estimate : Ω → ι → ℝ) (choice : Ω → ι)
    (target error slack : ℝ) (failure : ι → Set Ω) (missBound : ℝ≥0∞)
    (failureEq : ∀ index, failure index =
      {world | target < Regret.simpleRegret objective best (candidate world index)})
    (measurable : ∀ index, MeasurableSet (failure index))
    (missBounded : measure (Miss failure) ≤ missBound)
    (accurate : ∀ world,
      UniformError (fun index => objective (candidate world index)) (estimate world) error)
    (chosen : ∀ world, ApproxMin (estimate world) (choice world) slack) :
    1 - missBound ≤ measure {world |
      Regret.simpleRegret objective best (candidate world (choice world)) ≤
        target + (2 * error + slack)} := by
  apply selectedBound
    (fun world index => Regret.simpleRegret objective best (candidate world index))
    (fun world => Regret.simpleRegret objective best (candidate world (choice world)))
    target (2 * error + slack) failure missBound failureEq measurable missBounded
  intro world index
  have bound := Selection.regretBound
    (fun candidateIndex => objective (candidate world candidateIndex))
    (estimate world) (choice world) index error slack (accurate world) (chosen world)
  unfold Regret.simpleRegret
  linarith

/-- To meet a benchmark regret, pool coverage must absorb the selector penalty.
The benchmark may later be instantiated by a comparable Muon run; this theorem
does not itself establish that experimental comparability. -/
theorem benchmarkBound [Fintype ι] [IsProbabilityMeasure measure]
    {State : Type*} (objective : State → ℝ) (best : State)
    (candidate : Ω → ι → State) (estimate : Ω → ι → ℝ) (choice : Ω → ι)
    (benchmark error slack : ℝ) (failure : ι → Set Ω) (missBound : ℝ≥0∞)
    (failureEq : ∀ index, failure index = {world |
      benchmark - (2 * error + slack) <
        Regret.simpleRegret objective best (candidate world index)})
    (measurable : ∀ index, MeasurableSet (failure index))
    (missBounded : measure (Miss failure) ≤ missBound)
    (accurate : ∀ world,
      UniformError (fun index => objective (candidate world index)) (estimate world) error)
    (chosen : ∀ world, ApproxMin (estimate world) (choice world) slack) :
    1 - missBound ≤ measure {world |
      Regret.simpleRegret objective best (candidate world (choice world)) ≤ benchmark} := by
  simpa only [sub_add_cancel] using
    regretBound objective best candidate estimate choice
      (benchmark - (2 * error + slack)) error slack failure missBound failureEq
      measurable missBounded accurate chosen

/-- Candidate count, parallel width, and sequential depth are distinct. -/
structure Plan where
  candidateCount : ℕ
  batchWidth : ℕ
  sequentialDepth : ℕ
  covered : candidateCount ≤ batchWidth * sequentialDepth

/-- Any deadline-feasible plan obeys a width-times-deadline capacity bound. -/
theorem Plan.capacity (plan : Plan) (evaluationMs deadlineMs : ℝ)
    (evaluationNonneg : 0 ≤ evaluationMs)
    (depthFits : (plan.sequentialDepth : ℝ) * evaluationMs ≤ deadlineMs) :
    (plan.candidateCount : ℝ) * evaluationMs ≤
      (plan.batchWidth : ℝ) * deadlineMs := by
  have coverage : (plan.candidateCount : ℝ) ≤
      (plan.batchWidth : ℝ) * plan.sequentialDepth := by
    exact_mod_cast plan.covered
  calc
    (plan.candidateCount : ℝ) * evaluationMs ≤
        ((plan.batchWidth : ℝ) * plan.sequentialDepth) * evaluationMs :=
      mul_le_mul_of_nonneg_right coverage evaluationNonneg
    _ = (plan.batchWidth : ℝ) *
        ((plan.sequentialDepth : ℝ) * evaluationMs) := by ring
    _ ≤ (plan.batchWidth : ℝ) * deadlineMs := by
      exact mul_le_mul_of_nonneg_left depthFits (Nat.cast_nonneg plan.batchWidth)

#print axioms missProb
#print axioms hitProb
#print axioms selectedBound
#print axioms selectedHit
#print axioms regretBound
#print axioms benchmarkBound
#print axioms duplicateMiss
#print axioms Plan.capacity

end BlckbauxResearch.Optimization.Batch
