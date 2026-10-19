import BlckbauxResearch.Objective.CorpusMemory
import BlckbauxResearch.Forward.Algebra

/-!
# The unseen-origin obstruction is compatible with log loss

Every strictly positive prescribed loss q is realized by finite logits for
any vocabulary with at least two tokens. Give the target logit 0 and all m
other logits log((exp(q)-1)/m); the partition sum is exp(q).

Adding 1 to the two bounded objectives in CorpusMemory makes their losses
strictly positive without changing comparisons or regret. The construction
therefore realizes the same indistinguishable two-state decision problem
using the existing Forward.logLoss definition. Repeating the logits across
any positive token count leaves the window mean unchanged.

This is a result about the objective FORM and arbitrary logit maps. It does
not construct weights of the deployed H-Net, a legal overlapping corpus with
these responses, BF16 execution, or a trajectory of the actual optimizer.
Those restrictions may rule out this witness; they cannot be omitted when
claiming that pretraining loss has favorable information structure.
-/

namespace BlckbauxResearch.Objective.CorpusWitness

open scoped BigOperators
open BlckbauxResearch.Forward BlckbauxResearch.Objective.NaturalWindow
open BlckbauxResearch.Objective.CorpusMemory

noncomputable section

/-- One target plus `alternatives` equal competing logits. -/
def logits (alternatives : ℕ) (value : ℝ) : Vector (alternatives + 1) :=
  Fin.cases 0 (fun _ => Real.log ((Real.exp value - 1) / (alternatives : ℝ)))

theorem lossExact (alternatives : ℕ) (positive : 0 < alternatives)
    (value : ℝ) (valuePositive : 0 < value) : logLoss (logits alternatives value) 0 = value := by
  have denominator : 0 < (alternatives : ℝ) := by exact_mod_cast positive
  have numerator : 0 < Real.exp value - 1 := sub_pos.mpr (Real.one_lt_exp_iff.mpr valuePositive)
  have argument := div_pos numerator denominator
  have partition : ∑ index, Real.exp (logits alternatives value index) = Real.exp value := by
    simp [logits, Fin.sum_univ_succ, Real.exp_log argument]
    field_simp
    ring
  rw [logLoss, partition]
  simp [logits]

/-- The token count alone cannot eliminate the example: the arithmetic mean
of repeated token losses is the same prescribed loss, including at 1M tokens.
No claim of token independence is made. -/
theorem windowExact (alternatives tokens : ℕ) (positive : 0 < alternatives)
    (tokenPositive : 0 < tokens) (value : ℝ) (valuePositive : 0 < value) :
    average (fun _ : Fin tokens => logLoss (logits alternatives value) 0) = value := by
  simp_rw [lossExact alternatives positive value valuePositive]
  simp [average, Nat.ne_of_gt tokenPositive]

def meanLoss {count : ℕ} (alternatives : ℕ) (seen : Finset (Fin count))
    (scale : ℝ) (favored state : Fin 2) : ℝ :=
  corpusMean (fun origin => logLoss (logits alternatives
    (1 + blindLoss seen scale favored state origin)) 0)

/-- Adding the same baseline 1 to every loss leaves corpus comparisons intact. -/
theorem meanExact {count : ℕ} (countPositive : 0 < count)
    (alternatives : ℕ) (positive : 0 < alternatives) (seen : Finset (Fin count))
    (scale : ℝ) (nonnegative : 0 ≤ scale) (favored state : Fin 2) :
    meanLoss alternatives seen scale favored state =
      1 + corpusMean (blindLoss seen scale favored state) := by
  have equality : ∀ origin, logLoss (logits alternatives
      (1 + blindLoss seen scale favored state origin)) 0 =
        1 + blindLoss seen scale favored state origin := by
    intro origin
    apply lossExact alternatives positive
    have lower := (blindBound seen scale nonnegative favored state origin).1
    linarith
  dsimp [meanLoss]
  simp_rw [equality]
  simp [corpusMean, Finset.sum_add_distrib, mul_add, Nat.ne_of_gt countPositive]

/-- Both finite-logit objectives give exactly the same observed response,
at every state on every observed origin. -/
theorem observedExact {count : ℕ} (alternatives : ℕ) (positive : 0 < alternatives)
    (seen : Finset (Fin count)) (scale : ℝ) (favored state : Fin 2)
    (origin : Fin count) (observed : origin ∈ seen) :
    logLoss (logits alternatives (1 + blindLoss seen scale favored state origin)) 0 = 1 := by
  simpa [blindLoss, observed] using lossExact alternatives positive 1 (by norm_num)

/-- Whichever of the two states is recommended, one compatible log-loss
objective has simple regret scale*unobserved_fraction, with a proved optimum.
This does not assert realizability by the deployed neural architecture. -/
theorem likelihoodRegret {count : ℕ} (countPositive : 0 < count)
    (alternatives : ℕ) (positive : 0 < alternatives) (seen : Finset (Fin count))
    (scale : ℝ) (nonnegative : 0 ≤ scale) (recommendation : Fin 2) :
    ∃ favored : Fin 2,
      (∀ state, meanLoss alternatives seen scale favored favored ≤
        meanLoss alternatives seen scale favored state) ∧
      meanLoss alternatives seen scale favored recommendation -
        meanLoss alternatives seen scale favored favored =
          (count : ℝ)⁻¹ * ((Finset.univ \ seen).card : ℝ) * scale := by
  obtain ⟨favored, optimal, gap⟩ := unseenRegret seen scale nonnegative recommendation
  refine ⟨favored, ?_, ?_⟩
  · intro state
    rw [meanExact countPositive alternatives positive seen scale nonnegative,
      meanExact countPositive alternatives positive seen scale nonnegative]
    linarith [optimal state]
  · rw [meanExact countPositive alternatives positive seen scale nonnegative,
      meanExact countPositive alternatives positive seen scale nonnegative]
    linarith

#print axioms lossExact
#print axioms windowExact
#print axioms meanExact
#print axioms observedExact
#print axioms likelihoodRegret

end
end BlckbauxResearch.Objective.CorpusWitness
