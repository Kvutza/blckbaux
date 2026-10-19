import BlckbauxResearch.Foundation
import BlckbauxResearch.Optimization.Selection
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Mathematical Evaluation Space for Bayesian Optimization

Grounded in:
* David Blackwell (1951, 1953): Comparison of experiments and sufficiency preorders.
* Dennis Lindley (1956): Information gain from experiments.
* Raul Astudillo & Peter Frazier (2019, 2021): Composite Bayesian Optimization `g(f(x))`.
* Javier González et al. (2017): Preferential / relative optimization under Common Random Numbers.
-/

namespace BlckbauxResearch.Optimization.EvaluationSpace

noncomputable section

open scoped BigOperators
open BlckbauxResearch.Optimization

/-- An experimental query space equipped with cost and information sufficiency.
Query `q_2 ⪯ q_1` asserts that experiment `q_1` is Blackwell-sufficient for `q_2`. -/
structure Experiment (Query : Type*) where
  cost : Query → ℝ
  costPos : ∀ q, 0 < cost q
  Suff : Query → Query → Prop

/-- A measurement channel executing experiments across parameter states and environments. -/
structure Channel (State World Query : Type*) (Obs : Query → Type*) where
  evaluate : State → World → (q : Query) → Obs q

/-- Coarsening reduction morphism: query `coarse` is obtained from `fine` via a
deterministic reduction map `reduce`, preserving evaluation across all environments. -/
def Coarsening {State World Query : Type*} {Obs : Query → Type*}
    (channel : Channel State World Query Obs)
    (fine coarse : Query)
    (reduce : Obs fine → Obs coarse) : Prop :=
  ∀ (state : State) (world : World),
    channel.evaluate state world coarse = reduce (channel.evaluate state world fine)

/-- General composite weighted evaluation in dimension m (Astudillo & Frazier 2019/2021). -/
def compMean {m : ℕ} (w : Fin m → ℝ) (v : Fin m → ℝ) : ℝ :=
  ∑ i : Fin m, w i * v i

/-- General composite aggregation is additive. -/
theorem compAdd {m : ℕ} (w : Fin m → ℝ) (u v : Fin m → ℝ) :
    compMean w (u + v) = compMean w u + compMean w v := by
  unfold compMean
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i _
  simp only [Pi.add_apply]
  ring

/-- General composite aggregation scales with scalar multipliers. -/
theorem compScale {m : ℕ} (w : Fin m → ℝ) (c : ℝ) (v : Fin m → ℝ) :
    compMean w (fun i => c * v i) = c * compMean w v := by
  unfold compMean
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  ring

/-- General composite error bound for non-negative weights (Astudillo & Frazier). -/
theorem compError {m : ℕ} (w : Fin m → ℝ)
    (hNonneg : ∀ i : Fin m, 0 ≤ w i)
    (truth estimate eps : Fin m → ℝ)
    (hEps : ∀ i : Fin m, |estimate i - truth i| ≤ eps i) :
    |compMean w estimate - compMean w truth| ≤ compMean w eps := by
  unfold compMean
  have hDiff : ∑ i, w i * estimate i - ∑ i, w i * truth i =
      ∑ i, w i * (estimate i - truth i) := by
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro i _
    ring
  rw [hDiff]
  calc
    |∑ i, w i * (estimate i - truth i)| ≤ ∑ i, |w i * (estimate i - truth i)| :=
      Finset.abs_sum_le_sum_abs (fun i => w i * (estimate i - truth i)) Finset.univ
    _ = ∑ i, w i * |estimate i - truth i| := by
      apply Finset.sum_congr rfl
      intro i _
      rw [abs_mul, abs_of_nonneg (hNonneg i)]
    _ ≤ ∑ i, w i * eps i := by
      apply Finset.sum_le_sum
      intro i _
      exact mul_le_mul_of_nonneg_left (hEps i) (hNonneg i)

/-- General composite confidence propagation: componentwise bounds propagate
without dimension distortion through positive linear aggregation. -/
theorem compConf {m : ℕ} (w : Fin m → ℝ)
    (hNonneg : ∀ i : Fin m, 0 ≤ w i)
    (truth estimate sigma : Fin m → ℝ)
    (beta : ℝ)
    (hConf : ∀ i : Fin m, |estimate i - truth i| ≤ beta * sigma i) :
    |compMean w estimate - compMean w truth| ≤ beta * compMean w sigma := by
  have hBound := compError w hNonneg truth estimate (fun i => beta * sigma i) hConf
  have hScale : compMean w (fun i => beta * sigma i) = beta * compMean w sigma := by
    unfold compMean
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    ring
  rwa [hScale] at hBound

/-- General composite UCB acquisition regret bound: optimizing composite UCB
over vector components preserves the scalar regret bound. -/
theorem compRegret {Candidate : Type*} {m : ℕ} (w : Fin m → ℝ)
    (hNonneg : ∀ i : Fin m, 0 ≤ w i)
    (truth mean sigma : Candidate → Fin m → ℝ)
    (selected comparison : Candidate)
    (beta slack : ℝ)
    (hConf : ∀ c, ∀ i : Fin m, |truth c i - mean c i| ≤ beta * sigma c i)
    (selectedWithin :
      ApproxMax
        (fun c => compMean w (mean c) + beta * compMean w (sigma c))
        selected slack) :
    compMean w (truth comparison) - compMean w (truth selected) ≤
      2 * beta * compMean w (sigma selected) + slack := by
  have hConfComp : ∀ c,
      |compMean w (truth c) - compMean w (mean c)| ≤
        beta * compMean w (sigma c) := by
    intro c
    rw [abs_sub_comm]
    have hSymm : ∀ i : Fin m, |mean c i - truth c i| ≤ beta * sigma c i := by
      intro i
      rw [abs_sub_comm]
      exact hConf c i
    exact compConf w hNonneg (truth c) (mean c) (sigma c) beta hSymm
  exact Selection.ucbBound
    (fun c => compMean w (truth c))
    (fun c => compMean w (mean c))
    (fun c => compMean w (sigma c))
    selected comparison beta slack hConfComp selectedWithin

/-- Paired observation structure recording mean difference and sample variance. -/
structure PairedObs where
  meanDiff : ℝ
  sampleVar : ℝ

/-- Paired consistency: the coupled difference evaluation matches the difference
of scalar evaluations evaluated in the same environment. -/
def PairedConsistent {State World : Type*}
    (scalarEval : State → World → ℝ)
    (pairedEval : State → State → World → PairedObs) : Prop :=
  ∀ cand base w, (pairedEval cand base w).meanDiff = scalarEval cand w - scalarEval base w

/-- Algebraic variance decomposition under environmental covariance. -/
theorem pairedVariance (varBase varCand covar : ℝ) :
    varBase + varCand - 2 * covar = varCand + (varBase - 2 * covar) := by
  ring

/-- Strict variance reduction under Common Random Numbers (CRN): when environmental
covariance exceeds half the baseline variance, paired evaluation has strictly less
variance than evaluating the candidate alone. -/
theorem pairedAdvantage
    (varBase varCand covar : ℝ)
    (hCovar : varBase / 2 < covar) :
    varBase + varCand - 2 * covar < varCand := by
  linarith

/-- Uniform channel weights for arbitrary dimension m (1 / m on m components). -/
def uniformWeights (m : ℕ) : Fin m → ℝ := fun _ => (1 / (m : ℝ))

/-- Uniform channel weights are non-negative. -/
theorem uniformNonneg (m : ℕ) : ∀ i : Fin m, 0 ≤ uniformWeights m i := by
  intro _
  unfold uniformWeights
  positivity

/-- Mean aggregation of m evaluation channels as a composite BO instance. -/
def uniformMean (m : ℕ) (v : Fin m → ℝ) : ℝ :=
  compMean (uniformWeights m) v

/-- Uniform channel mean formula matches scalar average. -/
theorem uniformFormula (m : ℕ) (v : Fin m → ℝ) :
    uniformMean m v = (1 / (m : ℝ)) * ∑ i : Fin m, v i := by
  unfold uniformMean compMean uniformWeights
  rw [← Finset.mul_sum]

/-- Uniform channel mean aggregation is additive. -/
theorem uniformAdd (m : ℕ) (u v : Fin m → ℝ) :
    uniformMean m (u + v) = uniformMean m u + uniformMean m v :=
  compAdd (uniformWeights m) u v

/-- Uniform channel mean scales with scalar multipliers. -/
theorem uniformScale (m : ℕ) (c : ℝ) (v : Fin m → ℝ) :
    uniformMean m (fun i => c * v i) = c * uniformMean m v :=
  compScale (uniformWeights m) c v

/-- Uniform channel error bound specialized from general composite BO. -/
theorem uniformError (m : ℕ)
    (truth estimate eps : Fin m → ℝ)
    (hEps : ∀ i : Fin m, |estimate i - truth i| ≤ eps i) :
    |uniformMean m estimate - uniformMean m truth| ≤ uniformMean m eps :=
  compError (uniformWeights m) (uniformNonneg m) truth estimate eps hEps

/-- Uniform channel confidence propagation specialized from general composite BO. -/
theorem uniformConf (m : ℕ)
    (truth estimate sigma : Fin m → ℝ)
    (beta : ℝ)
    (hConf : ∀ i : Fin m, |estimate i - truth i| ≤ beta * sigma i) :
    |uniformMean m estimate - uniformMean m truth| ≤ beta * uniformMean m sigma :=
  compConf (uniformWeights m) (uniformNonneg m) truth estimate sigma beta hConf

/-- Uniform channel composite UCB regret bound specialized from general composite BO. -/
theorem uniformRegret
    {Candidate : Type*} (m : ℕ)
    (truth mean sigma : Candidate → Fin m → ℝ)
    (selected comparison : Candidate)
    (beta slack : ℝ)
    (hConf : ∀ c, ∀ i : Fin m, |truth c i - mean c i| ≤ beta * sigma c i)
    (selectedWithin :
      ApproxMax
        (fun c => uniformMean m (mean c) + beta * uniformMean m (sigma c))
        selected slack) :
    uniformMean m (truth comparison) - uniformMean m (truth selected) ≤
      2 * beta * uniformMean m (sigma selected) + slack :=
  compRegret (uniformWeights m) (uniformNonneg m) truth mean sigma selected comparison beta slack hConf selectedWithin

end

end BlckbauxResearch.Optimization.EvaluationSpace
