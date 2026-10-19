import BlckbauxResearch.Optimization.Selection
import BlckbauxResearch.Optimization.Step
import Mathlib.Topology.MetricSpace.Basic

/-!
# Trust-region length adaptation

This module formalizes the mathematical necessity and boundaries of trust-region
radius adaptation using Mathlib's metric space foundations.

In black-box optimization on high-dimensional parameter spaces, the local trust
region is a closed metric ball of radius `length`.  The model approximation error
scales with radius, while linear descent opportunity scales with radius.

We prove:
1. Under Lipschitz continuity, candidate opportunity inside the ball is bounded by `K * length`.
2. Under quadratic surrogate error `C * length^2`, guaranteed progress is bounded below.
3. If proposal progress fails (progress ≤ 0) while linear opportunity `γ * length` exists,
   the radius `length` must exceed the critical curvature threshold `(γ - slack / length) / (2 * C)`.
   This formally justifies trust-region contraction upon observed failure.
4. On a discrete state space with minimum separation `delta`, contracting `length < delta`
   collapses the closed ball to the singleton center `{incumbent}`, proving the lattice
   floor requires restart rather than continuous shrinking.
-/

namespace BlckbauxResearch.Optimization

variable {State : Type*}

/-- The closed metric trust region of radius `length` around an incumbent state. -/
def TrustRegion [MetricSpace State] (incumbent : State) (length : ℝ) : Set State :=
  Metric.closedBall incumbent length

/-- Under Lipschitz continuity with constant `K`, opportunity in the trust region is at most `K * length`. -/
theorem Opportunity.lipschitzBound
    [MetricSpace State]
    (objective : Objective State)
    (incumbent comparison : State)
    (length K : ℝ)
    (hK : 0 ≤ K)
    (hLip : ∀ x y, |objective x - objective y| ≤ K * dist x y)
    (hInBall : comparison ∈ TrustRegion incumbent length) :
    opportunity objective incumbent comparison ≤ K * length := by
  unfold opportunity TrustRegion at *
  have hDist : dist comparison incumbent ≤ length := Metric.mem_closedBall.mp hInBall
  have hDiff : objective incumbent - objective comparison ≤ |objective incumbent - objective comparison| :=
    le_abs_self _
  have hBound : |objective incumbent - objective comparison| ≤ K * dist incumbent comparison :=
    hLip incumbent comparison
  rw [dist_comm] at hBound
  have hKDist : K * dist comparison incumbent ≤ K * length :=
    mul_le_mul_of_nonneg_left hDist hK
  linarith

/-- Guaranteed progress under quadratic surrogate error `C * length^2`. -/
theorem Progress.curvatureBound
    (objective : Objective State)
    (incumbent : State)
    (comparison selected : State)
    (C length slack : ℝ)
    (hError : |objective selected - objective comparison| ≤ 2 * (C * length ^ 2) + slack) :
    opportunity objective incumbent comparison - (2 * (C * length ^ 2) + slack) ≤
      progress objective incumbent selected := by
  have hUpper := (abs_le.mp hError).2
  unfold opportunity progress at *
  linarith

/--
Contraction theorem: If progress fails (nonpositive) despite linear opportunity `γ * length`,
surrogate curvature forces `length` above the critical resolution threshold `(γ - slack / length) / (2 * C)`.
-/
theorem Adaptation.contractThreshold
    (opp progressVal C length γ slack : ℝ)
    (hPosLength : 0 < length)
    (hPosC : 0 < C)
    (hOpp : γ * length ≤ opp)
    (hFail : progressVal ≤ 0)
    (hBound : opp - (2 * (C * length ^ 2) + slack) ≤ progressVal) :
    (γ - slack / length) / (2 * C) ≤ length := by
  have hInequality : γ * length - (2 * (C * length ^ 2) + slack) ≤ 0 := by
    linarith
  have hRearrange : γ * length - slack ≤ 2 * C * length ^ 2 := by
    linarith
  have hDiv : (γ * length - slack) / length ≤ 2 * C * length := by
    rw [div_le_iff₀ hPosLength]
    nlinarith
  have hSplit : γ - slack / length ≤ 2 * C * length := by
    have hEq : (γ * length - slack) / length = γ - slack / length := by
      rw [sub_div, mul_div_cancel_right₀ γ (ne_of_gt hPosLength)]
    rwa [hEq] at hDiv
  have hFinal : (γ - slack / length) / (2 * C) ≤ length := by
    rw [div_le_iff₀ (by linarith)]
    linarith
  exact hFinal

/-- When surrogate slack is strictly below the optimal curvature-clearance margin `γ^2 / (8 * C)`,
evaluating at the optimal resolution `length = γ / (4 * C)` guarantees strictly positive progress. -/
theorem Adaptation.optimalProgress
    (opp progressVal C γ slack : ℝ)
    (hPosC : 0 < C)
    (hSlack : slack < γ ^ 2 / (8 * C))
    (hOpp : γ * (γ / (4 * C)) ≤ opp)
    (hBound : opp - (2 * (C * (γ / (4 * C)) ^ 2) + slack) ≤ progressVal) :
    0 < progressVal := by
  have hC : C ≠ 0 := ne_of_gt hPosC
  have hCurv : 2 * (C * (γ / (4 * C)) ^ 2) = γ ^ 2 / (8 * C) := by
    calc
      2 * (C * (γ / (4 * C)) ^ 2) = 2 * C * (γ ^ 2 / (16 * C ^ 2)) := by ring
      _ = (2 * C * γ ^ 2) / (16 * C ^ 2) := by ring
      _ = (2 * C * γ ^ 2) / (2 * C * (8 * C)) := by ring
      _ = γ ^ 2 / (8 * C) := by
        rw [mul_div_mul_left (γ ^ 2) (8 * C) (by linarith : 2 * C ≠ 0)]
  have hLinear : γ * (γ / (4 * C)) = 2 * (γ ^ 2 / (8 * C)) := by
    ring
  have hNet : 0 < γ ^ 2 / (8 * C) - slack := sub_pos.mpr hSlack
  have hLower : γ ^ 2 / (8 * C) - slack ≤ progressVal := by
    linarith
  exact lt_of_lt_of_le hNet hLower

/--
Lattice floor theorem: In a metric space where all distinct states are separated by at least
`delta`, any trust region of radius `length < delta` contains no distinct state from the incumbent.
-/
theorem Lattice.floorCollapse
    [MetricSpace State]
    (incumbent candidate : State)
    (delta length : ℝ)
    (hSeparation : ∀ x y : State, x ≠ y → delta ≤ dist x y)
    (hRadius : length < delta)
    (hInBall : candidate ∈ TrustRegion incumbent length) :
    candidate = incumbent := by
  unfold TrustRegion at hInBall
  have hDist : dist candidate incumbent ≤ length := Metric.mem_closedBall.mp hInBall
  by_contra hDiff
  have hMinDist : delta ≤ dist candidate incumbent := hSeparation candidate incumbent hDiff
  linarith

/--
Zero opportunity theorem: Below the lattice separation `delta`, the closed trust ball
contains only the incumbent, giving identically zero opportunity.
-/
theorem Lattice.zeroOpportunity
    [MetricSpace State]
    (objective : Objective State)
    (incumbent candidate : State)
    (delta length : ℝ)
    (hSeparation : ∀ x y : State, x ≠ y → delta ≤ dist x y)
    (hRadius : length < delta)
    (hInBall : candidate ∈ TrustRegion incumbent length) :
    opportunity objective incumbent candidate = 0 := by
  have hEq : candidate = incumbent :=
    Lattice.floorCollapse incumbent candidate delta length hSeparation hRadius hInBall
  unfold opportunity
  rw [hEq, sub_self]

/--
Zero progress theorem: Below the lattice separation `delta`, candidate evaluation
yields identically zero progress, proving trust-region shrinking must terminate in restart.
-/
theorem Lattice.zeroProgress
    [MetricSpace State]
    (objective : Objective State)
    (incumbent candidate : State)
    (delta length : ℝ)
    (hSeparation : ∀ x y : State, x ≠ y → delta ≤ dist x y)
    (hRadius : length < delta)
    (hInBall : candidate ∈ TrustRegion incumbent length) :
    progress objective incumbent candidate = 0 := by
  have hEq : candidate = incumbent :=
    Lattice.floorCollapse incumbent candidate delta length hSeparation hRadius hInBall
  unfold progress
  rw [hEq, sub_self]

#print axioms Adaptation.contractThreshold
#print axioms Adaptation.optimalProgress
#print axioms Lattice.floorCollapse
#print axioms Lattice.zeroProgress

end BlckbauxResearch.Optimization
