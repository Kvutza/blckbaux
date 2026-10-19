import BlckbauxResearch.Objective.Aggregation
import Mathlib.Data.Real.Basic

/-!
# Group-relative objective scoring and candidate bounds

This module formalizes relative scoring across a candidate pool of size `K`.
Standardizing candidate evaluations against the pool mean eliminates uniform
prompt and difficulty shifts, while bounded scatter ensures numerical stability.
-/

namespace BlckbauxResearch.Objective.Relative

noncomputable section

open scoped BigOperators

/-- Group empirical mean across a pool of candidates. -/
def poolMean {count : ℕ} (scores : Fin count → ℝ) : ℝ :=
  (count : ℝ)⁻¹ * ∑ index, scores index

/-- Centered deviation of a candidate from the pool mean. -/
def poolCentered {count : ℕ} (scores : Fin count → ℝ) (index : Fin count) : ℝ :=
  scores index - poolMean scores

/-- Sum of squared centered deviations across the pool. -/
def poolScatter {count : ℕ} (scores : Fin count → ℝ) : ℝ :=
  ∑ index, (poolCentered scores index) ^ 2

/-- The sum of centered deviations across any finite candidate pool is zero. -/
theorem Group.sumZero {count : ℕ} (positive : 0 < count)
    (scores : Fin count → ℝ) :
    ∑ index, poolCentered scores index = 0 := by
  have countNonzero : (count : ℝ) ≠ 0 := by
    exact_mod_cast Nat.ne_of_gt positive
  unfold poolCentered poolMean
  rw [Finset.sum_sub_distrib]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have cancelMul : (count : ℝ) * ((count : ℝ)⁻¹ * ∑ index, scores index) =
      ∑ index, scores index := by
    rw [← mul_assoc, mul_inv_cancel₀ countNonzero, one_mul]
  rw [cancelMul, sub_self]

/-- Adding a uniform prompt or context shift leaves centered deviations invariant. -/
theorem Group.shift {count : ℕ} (positive : 0 < count)
    (scores : Fin count → ℝ) (offset : ℝ) (index : Fin count) :
    poolCentered (fun i => scores i + offset) index = poolCentered scores index := by
  have countNonzero : (count : ℝ) ≠ 0 := by
    exact_mod_cast Nat.ne_of_gt positive
  unfold poolCentered poolMean
  rw [Finset.sum_add_distrib]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have expandMean : (count : ℝ)⁻¹ * (∑ x, scores x + (count : ℝ) * offset) =
      (count : ℝ)⁻¹ * (∑ x, scores x) + offset := by
    rw [mul_add, ← mul_assoc, inv_mul_cancel₀ countNonzero, one_mul]
  rw [expandMean]
  ring

/-- Any single candidate squared deviation is bounded by total pool scatter. -/
theorem Group.bound {count : ℕ} (scores : Fin count → ℝ) (index : Fin count) :
    (poolCentered scores index) ^ 2 ≤ poolScatter scores := by
  unfold poolScatter
  have nonneg : ∀ i ∈ (Finset.univ : Finset (Fin count)), 0 ≤ (poolCentered scores i) ^ 2 := by
    intro i _
    exact sq_nonneg _
  exact Finset.single_le_sum nonneg (Finset.mem_univ index)

#print axioms Group.sumZero
#print axioms Group.shift
#print axioms Group.bound

end

end BlckbauxResearch.Objective.Relative
