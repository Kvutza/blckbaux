import BlckbauxResearch.Optimization.ResearchProblem
import Mathlib.Analysis.Asymptotics.Defs
import Mathlib.Analysis.Asymptotics.Lemmas

/-!
# Learning effectiveness

An optimizer is compared through attainable quality-resource profiles, not a
single run or a scalar efficiency score. A token budget indexes each profile.
At every budget the profile records an upper bound for every declared quality
gap, the complete machine-resource use, and a failure-probability bound.

Comparing these upper bounds orders certificates.  It does not order actual
optimizer risks: a smaller candidate upper bound than a reference upper bound
is insufficient for an outcome-superiority theorem.  Such a theorem needs
actual-risk semantics or a candidate upper bound below a reference lower bound.
Probability bounds also require a connection to a specified failure event.

Point dominance compares one token budget. Curve dominance compares every
budget in a declared scope. Rate dominance is strict asymptotic improvement of
every quality gap. A set of profiles represents alternative schedules,
hyperparameters, policies, or machines without choosing one in advance.
-/

namespace BlckbauxResearch.Optimization.Effectiveness

open Filter Asymptotics
open ResearchProblem

/-- A quality-resource guarantee indexed by total evaluated-token budget. -/
structure EffectProfile (Metric : Type*) where
  gap : ℕ → Metric → ℝ
  gapNonneg : ∀ tokens metric, 0 ≤ gap tokens metric
  usage : ℕ → ResourceUsage
  tokenBound : ∀ tokens, usage tokens .tokensEvaluated ≤ tokens
  failure : ℕ → NNReal
  failureValid : ∀ tokens, failure tokens ≤ 1

/-- The declared quality gap is below a target at this token budget. -/
def EffectProfile.reaches
    (profile : EffectProfile Metric) (metric : Metric)
    (target : ℝ) (tokens : ℕ) : Prop :=
  profile.gap tokens metric ≤ target

/-- Quality, resources, and reliability weakly dominate at one token budget. -/
def PointDominance
    (candidate reference : EffectProfile Metric) (tokens : ℕ) : Prop :=
  (∀ metric, candidate.gap tokens metric ≤ reference.gap tokens metric) ∧
    Resource.le (candidate.usage tokens) (reference.usage tokens) ∧
    candidate.failure tokens ≤ reference.failure tokens

/-- Point dominance holds at every token budget selected by `scope`. -/
def CurveDominance
    (scope : ℕ → Prop) (candidate reference : EffectProfile Metric) : Prop :=
  ∀ tokens, scope tokens → PointDominance candidate reference tokens

/-- Every quality gap has strictly smaller asymptotic order. -/
def RateDominance
    (candidate reference : EffectProfile Metric) : Prop :=
  ∀ metric,
    (fun tokens ↦ candidate.gap tokens metric) =o[atTop]
      (fun tokens ↦ reference.gap tokens metric)

/-- Alternative attainable profiles for one optimizer family. -/
abbrev ProfileSet (Metric : Type*) := Set (EffectProfile Metric)

/-- Every reference profile has a point-dominating candidate profile. -/
def PointFrontier
    (tokens : ℕ) (candidate reference : ProfileSet Metric) : Prop :=
  ∀ profile ∈ reference,
    ∃ witness ∈ candidate, PointDominance witness profile tokens

/-- Every reference profile has a curve-dominating candidate profile. -/
def CurveFrontier
    (scope : ℕ → Prop) (candidate reference : ProfileSet Metric) : Prop :=
  ∀ profile ∈ reference,
    ∃ witness ∈ candidate, CurveDominance scope witness profile

/-- Every reference profile has a strictly rate-dominating candidate profile. -/
def RateFrontier
    (candidate reference : ProfileSet Metric) : Prop :=
  ∀ profile ∈ reference,
    ∃ witness ∈ candidate, RateDominance witness profile

theorem CurveDominance.point
    {scope : ℕ → Prop} {candidate reference : EffectProfile Metric}
    (dominance : CurveDominance scope candidate reference)
    {tokens : ℕ} (inside : scope tokens) :
    PointDominance candidate reference tokens :=
  dominance tokens inside

theorem PointDominance.reaches
    {candidate reference : EffectProfile Metric} {tokens : ℕ}
    (dominance : PointDominance candidate reference tokens)
    {metric : Metric} {target : ℝ}
    (reached : reference.reaches metric target tokens) :
    candidate.reaches metric target tokens :=
  (dominance.1 metric).trans reached

theorem RateDominance.bigO
    {candidate reference : EffectProfile Metric}
    (dominance : RateDominance candidate reference) :
    ∀ metric,
      (fun tokens ↦ candidate.gap tokens metric) =O[atTop]
        (fun tokens ↦ reference.gap tokens metric) :=
  fun metric ↦ (dominance metric).isBigO

/-- Strict rate dominance over a vanishing reference forces the candidate gap
to vanish as well.  Thus an optimizer with a positive asymptotic error floor
cannot satisfy this comparison against a vanishing reference curve. -/
theorem RateDominance.vanishes
    {candidate reference : EffectProfile Metric}
    (dominance : RateDominance candidate reference) (metric : Metric)
    (referenceZero : Tendsto (fun tokens ↦ reference.gap tokens metric)
      atTop (nhds 0)) :
    Tendsto (fun tokens ↦ candidate.gap tokens metric) atTop (nhds 0) :=
  (dominance metric).trans_tendsto referenceZero

/-- A nonzero limit of the declared candidate curve rules out strict rate
dominance over a vanishing reference curve.  When the candidate curve is only
an upper bound, its limit is not a lower bound on actual optimizer regret. -/
theorem floorObstruct
    {candidate reference : EffectProfile Metric} (metric : Metric)
    {floor : ℝ} (floorNe : floor ≠ 0)
    (candidateFloor : Tendsto (fun tokens ↦ candidate.gap tokens metric)
      atTop (nhds floor))
    (referenceZero : Tendsto (fun tokens ↦ reference.gap tokens metric)
      atTop (nhds 0)) :
    ¬ RateDominance candidate reference := by
  intro dominance
  have candidateZero := dominance.vanishes metric referenceZero
  exact floorNe (tendsto_nhds_unique candidateFloor candidateZero)

theorem CurveFrontier.point
    {scope : ℕ → Prop} {candidate reference : ProfileSet Metric}
    (dominance : CurveFrontier scope candidate reference)
    {tokens : ℕ} (inside : scope tokens) :
    PointFrontier tokens candidate reference := by
  intro profile member
  obtain ⟨witness, witnessMember, curve⟩ := dominance profile member
  exact ⟨witness, witnessMember, curve.point inside⟩

#print axioms CurveDominance.point
#print axioms PointDominance.reaches
#print axioms RateDominance.bigO
#print axioms RateDominance.vanishes
#print axioms floorObstruct
#print axioms CurveFrontier.point

end BlckbauxResearch.Optimization.Effectiveness
