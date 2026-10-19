import BlckbauxResearch.Optimization.ObjectiveSearch
import BlckbauxResearch.Optimization.Regret

/-!
# Transporting a successful proposal

A derivative-free optimizer can reuse a previously evaluated proposal without
reconstructing a gradient.  The executable move is regenerated at the current
incumbent, while its earlier source state and realized gain remain evidence.

The result below quantifies the only geometry needed for that reuse.  If the
old and current incumbents are close, and applying the same move at both states
produces close trials, a Lipschitz objective preserves the historical gain up
to the two transport distances.  The subsequent theorems compose that result
with the existing incumbent-prediction, candidate-observation, acceptance, and
simple-regret semantics.

This is a proposal construction, not a claim that the deployed archive already
retains every required seed or that corpus NLL already satisfies the supplied
regularity and score-error bounds.

The open question is whether the retained observations and the source proposal
rule produce such a move repeatedly, with transport error small relative to
the current population regret.  `pastGain` records an earlier population gain;
it is evidence assumed by these theorems, not a consequence of retaining a
state.  A proof for pretraining must derive that evidence from admissible
observations and explain how the move remains useful as the incumbent changes.

Audit, 2026-10-18: this two-value bound cannot certify positive reuse of an
accepted move once the current objective is no greater than its recorded trial
objective. Write s = f(past), t = f(apply past move), x = f(incumbent), and
g <= s - t. The base bound implies s - x <= baseShift, hence
g - baseShift - trialShift <= x - t - trialShift <= 0 when x <= t and
trialShift >= 0. The latter nonnegativity follows from any valid trial bound.
This applies to every accepted record on a population-monotone trajectory;
it does not assert that noisy source acceptance gives such a trajectory.

The implication below remains valid. What fails is using its lower bound to
justify that exploitation regime. A different research question is how the
gain f(x) - f(apply x move) itself varies with x. Separate absolute-value
bounds discard cancellation between its two terms. Section 68 of the research
handoff contains the REPL-checked obstruction and a squared-distance example;
no new production rule or pretraining regularity is asserted here.
-/

namespace BlckbauxResearch.Optimization.TransferGuide

open scoped NNReal

variable {State Move : Type*}

/-- A retained move transfers through two local value bounds. Global
Lipschitz regularity is unnecessary when the two relevant pairs have direct
certificates from the forward program. -/
theorem progressLocal
    (objective : Objective State) (apply : State → Move → State)
    (past incumbent : State) (move : Move)
    (gain baseShift trialShift : ℝ)
    (pastGain : gain ≤ progress objective past (apply past move))
    (baseNear : |objective incumbent - objective past| ≤ baseShift)
    (trialNear : |objective (apply past move) -
      objective (apply incumbent move)| ≤ trialShift) :
    gain - baseShift - trialShift ≤
      progress objective incumbent (apply incumbent move) := by
  have baseLower := (abs_le.mp baseNear).1
  have trialLower := (abs_le.mp trialNear).1
  unfold progress at pastGain ⊢
  linarith

/-- Reapplying one recorded move transports its historical objective gain.
There is no derivative, linear response, or ambient-dimensional cover. -/
theorem progressBound [PseudoMetricSpace State]
    (objective : Objective State) (constant : ℝ≥0)
    (regular : LipschitzWith constant objective)
    (apply : State → Move → State) (past incumbent : State) (move : Move)
    (gain baseRadius trialRadius : ℝ)
    (pastGain : gain ≤ progress objective past (apply past move))
    (baseNear : dist incumbent past ≤ baseRadius)
    (trialNear : dist (apply past move) (apply incumbent move) ≤ trialRadius) :
    gain - constant * (baseRadius + trialRadius) ≤
      progress objective incumbent (apply incumbent move) := by
  have baseChange := regular.dist_le_mul incumbent past
  have trialChange := regular.dist_le_mul (apply past move) (apply incumbent move)
  rw [Real.dist_eq] at baseChange trialChange
  have baseCost := mul_le_mul_of_nonneg_left baseNear constant.coe_nonneg
  have trialCost := mul_le_mul_of_nonneg_left trialNear constant.coe_nonneg
  unfold progress at pastGain ⊢
  have baseLower := (abs_le.mp baseChange).1
  have trialLower := (abs_le.mp trialChange).1
  linarith

/-- The source-shaped modeled acceptance rule commits the transported move when
its retained gain clears the incumbent error, candidate error, and margin.
Rejection is not assumed away: the separation premise proves acceptance. -/
theorem commitBound [PseudoMetricSpace State]
    (objective : Objective State) (constant : ℝ≥0)
    (regular : LipschitzWith constant objective)
    (apply : State → Move → State) (past incumbent : State) (move : Move)
    (scores : ObjectiveSearch.Scores (Fin 1))
    (gain baseRadius trialRadius baseError candidateError margin : ℝ)
    (pastGain : gain ≤ progress objective past (apply past move))
    (baseNear : dist incumbent past ≤ baseRadius)
    (trialNear : dist (apply past move) (apply incumbent move) ≤ trialRadius)
    (baseAccurate : |scores.incumbent - objective incumbent| ≤ baseError)
    (candidateAccurate :
      |scores.candidate 0 - objective (apply incumbent move)| ≤ candidateError)
    (separated : baseError + candidateError + margin ≤
      gain - constant * (baseRadius + trialRadius)) :
    gain - constant * (baseRadius + trialRadius) ≤
      progress objective incumbent
        (ObjectiveSearch.commit incumbent (fun _ : Fin 1 => apply incumbent move)
          scores margin 0) := by
  let candidate : Fin 1 → State := fun _ => apply incumbent move
  have accurate : ObjectiveSearch.ScoreError objective incumbent candidate scores
      baseError candidateError := by
    exact ⟨baseAccurate, fun index => by simpa [candidate, Fin.eq_zero index] using candidateAccurate⟩
  have useful := progressBound objective constant regular apply past incumbent move
    gain baseRadius trialRadius pastGain baseNear trialNear
  exact ObjectiveSearch.goodCommit objective incumbent candidate scores
    baseError candidateError margin
    (gain - constant * (baseRadius + trialRadius)) 0 accurate useful separated

/-- A transported historical move gives a deterministic one-transaction
simple-regret contraction.  `gainShare` is the retained fraction of current
simple regret; `loss` contains the resolution left after transport. -/
theorem regretBound [PseudoMetricSpace State]
    (objective : Objective State) (constant : ℝ≥0)
    (regular : LipschitzWith constant objective)
    (apply : State → Move → State) (past incumbent best : State) (move : Move)
    (scores : ObjectiveSearch.Scores (Fin 1))
    (gain baseRadius trialRadius baseError candidateError margin gainShare loss : ℝ)
    (pastGain : gain ≤ progress objective past (apply past move))
    (baseNear : dist incumbent past ≤ baseRadius)
    (trialNear : dist (apply past move) (apply incumbent move) ≤ trialRadius)
    (baseAccurate : |scores.incumbent - objective incumbent| ≤ baseError)
    (candidateAccurate :
      |scores.candidate 0 - objective (apply incumbent move)| ≤ candidateError)
    (separated : baseError + candidateError + margin ≤
      gain - constant * (baseRadius + trialRadius))
    (retained : gainShare * Regret.simpleRegret objective best incumbent - loss ≤
      gain - constant * (baseRadius + trialRadius)) :
    Regret.simpleRegret objective best
        (ObjectiveSearch.commit incumbent (fun _ : Fin 1 => apply incumbent move)
          scores margin 0) ≤
      (1 - gainShare) * Regret.simpleRegret objective best incumbent + loss := by
  have progressLower := commitBound objective constant regular apply past incumbent move
    scores gain baseRadius trialRadius baseError candidateError margin pastGain
    baseNear trialNear baseAccurate candidateAccurate separated
  simp only [Regret.simpleRegret, progress] at progressLower retained ⊢
  linarith

#print axioms progressBound
#print axioms progressLocal
#print axioms commitBound
#print axioms regretBound

end BlckbauxResearch.Optimization.TransferGuide
