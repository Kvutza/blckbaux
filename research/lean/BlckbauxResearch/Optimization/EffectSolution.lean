import BlckbauxResearch.Optimization.Effectiveness
import BlckbauxResearch.Optimization.Rate
import BlckbauxResearch.Optimization.Target

/-!
# Effectiveness refinement

An effectiveness profile is a comparison contract.  It does not by itself
show that an executable optimizer attains the declared gaps or resources.
This module supplies that missing refinement boundary.

For every evaluated-token budget, a solution profile provides a machine task
and one policy carrying certificates for every admissible environment in that
task.  The declared quality gap must cover the task's expected terminal simple
regret bound, and the declared resource vector must cover the task's complete
machine limit.  Consequently, profile comparisons can be traced back to the
existing synthesis theorems rather than treated as unattached curves.
-/

namespace BlckbauxResearch.Optimization.EffectSolution

open Effectiveness Filter ResearchProblem Synthesis Target

variable {State Feedback Memory Noise : Type*} [Fintype Noise]

/-- A token-indexed profile refined by certified machine synthesis tasks. -/
structure SolutionProfile
    (State Feedback Memory Noise : Type*) [Fintype Noise] where
  profile : EffectProfile Unit
  task : ℕ → MachineTask State Feedback Memory Noise
  solution : ∀ tokens, MachineSolution (task tokens)
  gapCovers : ∀ tokens, (task tokens).bound ≤ profile.gap tokens ()
  usageCovers : ∀ tokens,
    Resource.le (task tokens).limit (profile.usage tokens)

/-- The profile gap is a proved upper bound on expected terminal simple regret. -/
theorem SolutionProfile.regretBound
    (family : SolutionProfile State Feedback Memory Noise)
    (tokens : ℕ) (environment : (family.task tokens).Environment) :
    expected (family.solution tokens).policy
        ((family.task tokens).oracle environment)
        ((family.task tokens).objective environment)
        ((family.task tokens).best environment) 0
        (family.task tokens).horizon
        (family.solution tokens).policy.initial ≤
      family.profile.gap tokens () :=
  ((family.solution tokens).regretBound environment).trans
    (family.gapCovers tokens)

/-- Every execution path is bounded by the profile's complete resource vector. -/
theorem SolutionProfile.resourceBound
    (family : SolutionProfile State Feedback Memory Noise)
    (tokens : ℕ) (environment : (family.task tokens).Environment)
    (path : ℕ → Noise) :
    Resource.le
      (spentUsage (family.solution tokens).policy
        ((family.task tokens).oracle environment)
        ((family.task tokens).usage environment) 0
        (family.task tokens).horizon
        (family.solution tokens).policy.initial path)
      (family.profile.usage tokens) := by
  intro resource
  exact ((family.solution tokens).resourceBound environment path resource).trans
    (family.usageCovers tokens resource)

/-- Every certified execution respects its indexing evaluated-token budget. -/
theorem SolutionProfile.tokenBound
    (family : SolutionProfile State Feedback Memory Noise)
    (tokens : ℕ) (environment : (family.task tokens).Environment)
    (path : ℕ → Noise) :
    spentUsage (family.solution tokens).policy
        ((family.task tokens).oracle environment)
        ((family.task tokens).usage environment) 0
        (family.task tokens).horizon
        (family.solution tokens).policy.initial path .tokensEvaluated ≤ tokens :=
  (family.resourceBound tokens environment path .tokensEvaluated).trans
    (family.profile.tokenBound tokens)

/-- The underlying machine certificate still controls every transaction. -/
theorem SolutionProfile.latencyBound
    (family : SolutionProfile State Feedback Memory Noise)
    (tokens time : ℕ) (environment : (family.task tokens).Environment)
    (before : time < (family.task tokens).horizon)
    (memory : Memory) (noise : Noise) :
    (family.task tokens).usage environment time memory
        ((family.solution tokens).policy.query time memory) noise .wallTimeMs ≤
      (family.task tokens).deadlineMs :=
  (family.solution tokens).latencyBound environment time before memory noise

/-- A certified profile using the geometric envelope has vanishing regret when
its token-indexed estimator penalty vanishes. -/
theorem SolutionProfile.gapVanishes
    (family : SolutionProfile State Feedback Memory Noise)
    (initial factor : ℝ) (penalty : ℕ → ℝ)
    (factorNonneg : 0 ≤ factor) (contractive : factor < 1)
    (penaltyZero : Tendsto penalty atTop (nhds 0))
    (gapExact : ∀ tokens,
      family.profile.gap tokens () =
        Rate.envelope initial factor (penalty tokens) tokens) :
    Tendsto (fun tokens ↦ family.profile.gap tokens ()) atTop (nhds 0) :=
  (Rate.movingFloor initial factor penalty factorNonneg contractive penaltyZero).congr'
    (Eventually.of_forall fun tokens ↦ (gapExact tokens).symm)

/-- Point comparison with a reference profile; the reference need not be implemented in Lean. -/
def PointClaim
    (candidate : SolutionProfile State Feedback Memory Noise)
    (reference : EffectProfile Unit)
    (tokens : ℕ) : Prop :=
  PointDominance candidate.profile reference tokens

/-- Curve comparison with a measured or proved reference profile. -/
def CurveClaim
    (scope : ℕ → Prop)
    (candidate : SolutionProfile State Feedback Memory Noise)
    (reference : EffectProfile Unit) : Prop :=
  CurveDominance scope candidate.profile reference

/-- Strict asymptotic comparison with a measured or proved reference profile. -/
def RateClaim
    (candidate : SolutionProfile State Feedback Memory Noise)
    (reference : EffectProfile Unit) : Prop :=
  RateDominance candidate.profile reference

theorem CurveClaim.point
    {scope : ℕ → Prop}
    {candidate : SolutionProfile State Feedback Memory Noise}
    {reference : EffectProfile Unit}
    (claim : CurveClaim scope candidate reference)
    {tokens : ℕ} (inside : scope tokens) :
    PointClaim candidate reference tokens :=
  claim tokens inside

#print axioms SolutionProfile.regretBound
#print axioms SolutionProfile.resourceBound
#print axioms SolutionProfile.tokenBound
#print axioms SolutionProfile.latencyBound
#print axioms SolutionProfile.gapVanishes
#print axioms CurveClaim.point

end BlckbauxResearch.Optimization.EffectSolution
