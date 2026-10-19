import BlckbauxResearch.Objective.Adaptive
import BlckbauxResearch.Objective.MarginalWindow
import BlckbauxResearch.Optimization.TransferGuide
import BlckbauxResearch.Optimization.EnnCommit

/-!
# Population regret from a natural-window score

The deployed transaction evaluates one proposed parameter state on a natural
token window and compares that observed loss with an ENN score for the current
incumbent.  This file states the corresponding population-objective result.

A previously useful move is transported to the current incumbent.  If its
transported population gain clears the ENN incumbent error, the adaptive-window
error allowance, and the acceptance margin, then failure of the desired
simple-regret contraction is contained in the event that the candidate window
overestimates its population loss.  Mathlib's conditional sub-Gaussian
Azuma--Hoeffding theorem bounds exactly that event.

The theorem does not assert that the current contiguous corpus sampler satisfies
the conditional sub-Gaussian premises.  `AdaptiveBatch` is the precise source
obligation that must be discharged for that sampler.
-/

namespace BlckbauxResearch.Optimization.PretrainRegret

/-!
For `sourceTail`, put `θ = incumbent`, `θ⋆ = best`,
`θ′ = apply incumbent move`, and `b = gain - baseShift - trialShift`.
The `retained` hypothesis is `gainShare * R_f(θ; θ⋆) - loss ≤ b`;
`progressLocal` gives `b ≤ f(θ) - f(θ′)`.  Hence
`R_f(θ′; θ⋆) ≤ (1 - gainShare) * R_f(θ; θ⋆) + loss` is algebraic.
The missing result is a source-policy derivation of useful `b` and its
persistence, alongside the stated ENN and window-observation conditions.
-/

noncomputable section

open scoped BigOperators NNReal
open MeasureTheory ProbabilityTheory
open BlckbauxResearch.Objective.Adaptive
open BlckbauxResearch.Objective.NaturalWindow

variable {World State Move : Type*}

/-- Scores used by the current single-proposal transaction: the incumbent score
comes from ENN and the candidate score is a natural-window mean loss. -/
def windowScores
    [MeasurableSpace World]
    [StandardBorelSpace World]
    {measure : Measure World}
    [IsZeroOrProbabilityMeasure measure]
    (batch : AdaptiveBatch World measure)
    (sampleCount : ℕ)
    (baseScore : ℝ)
    (world : World) : ObjectiveSearch.Scores (Fin 1) where
  incumbent := baseScore
  candidate := fun _ => meanLoss batch sampleCount world

/-- Scores for one complete natural window drawn from the finite set of aligned
corpus origins. -/
def marginalScores
    {count : ℕ}
    (loss : Fin count → ℝ)
    (baseScore : ℝ)
    (origin : Fin count) : ObjectiveSearch.Scores (Fin 1) where
  incumbent := baseScore
  candidate := fun _ => loss origin

/-- A uniformly sampled complete natural window yields a population simple-
regret contraction for a deterministically useful transported proposal. The
tail scale is the range of whole-window losses, not the token count inside the
window. -/
theorem marginalTail
    (count : ℕ)
    [NeZero count]
    [PseudoMetricSpace State]
    (objective : Objective State)
    (constant : ℝ≥0)
    (regular : LipschitzWith constant objective)
    (apply : State → Move → State)
    (past incumbent best : State)
    (move : Move)
    (candidateLoss : Fin count → ℝ)
    (baseScore lower upper : ℝ)
    (gain baseRadius trialRadius baseError margin : ℝ)
    (gainShare loss epsilon : ℝ)
    (bounded : ∀ origin, candidateLoss origin ∈ Set.Icc lower upper)
    (epsilonNonnegative : 0 ≤ epsilon)
    (marginalEq : corpusMean candidateLoss = objective (apply incumbent move))
    (pastGain : gain ≤ progress objective past (apply past move))
    (baseNear : dist incumbent past ≤ baseRadius)
    (trialNear : dist (apply past move) (apply incumbent move) ≤ trialRadius)
    (baseAccurate : |baseScore - objective incumbent| ≤ baseError)
    (separated : baseError + epsilon + margin ≤
      gain - constant * (baseRadius + trialRadius))
    (retained : gainShare * Regret.simpleRegret objective best incumbent - loss ≤
      gain - constant * (baseRadius + trialRadius)) :
    (Objective.MarginalWindow.originMeasure count).real {origin |
        (1 - gainShare) * Regret.simpleRegret objective best incumbent + loss <
          Regret.simpleRegret objective best
            (ObjectiveSearch.commit incumbent
              (fun _ : Fin 1 => apply incumbent move)
              (marginalScores candidateLoss baseScore origin) margin 0)} ≤
      Real.exp (-epsilon ^ 2 /
        (2 * ((‖upper - lower‖₊ / 2) ^ 2))) := by
  let candidate : Fin 1 → State := fun _ => apply incumbent move
  have useful := TransferGuide.progressBound objective constant regular apply
    past incumbent move gain baseRadius trialRadius pastGain baseNear trialNear
  have eventBound := Objective.MarginalWindow.upperTail count candidateLoss
    lower upper epsilon bounded epsilonNonnegative
  calc
    (Objective.MarginalWindow.originMeasure count).real {origin |
        (1 - gainShare) * Regret.simpleRegret objective best incumbent + loss <
          Regret.simpleRegret objective best
            (ObjectiveSearch.commit incumbent candidate
              (marginalScores candidateLoss baseScore origin) margin 0)} ≤
      (Objective.MarginalWindow.originMeasure count).real {origin |
        epsilon ≤ candidateLoss origin - corpusMean candidateLoss} := by
          refine measureReal_mono ?_ (by finiteness)
          intro origin failed
          simp only [Set.mem_setOf_eq] at failed ⊢
          by_contra scoreNotHigh
          have scoreUpper :
              candidateLoss origin - corpusMean candidateLoss < epsilon :=
            lt_of_not_ge scoreNotHigh
          have baseLower := (abs_le.mp baseAccurate).1
          have accepted : ObjectiveSearch.Accept
              (marginalScores candidateLoss baseScore origin) margin (0 : Fin 1) := by
            unfold ObjectiveSearch.Accept marginalScores
            change candidateLoss origin + margin ≤ baseScore
            unfold progress at useful
            rw [marginalEq] at scoreUpper
            linarith
          have contracted :
              Regret.simpleRegret objective best (candidate 0) ≤
                (1 - gainShare) *
                    Regret.simpleRegret objective best incumbent + loss := by
            unfold progress at useful
            simp only [Regret.simpleRegret] at retained ⊢
            dsimp [candidate]
            linarith
          rw [ObjectiveSearch.commit, if_pos accepted] at failed
          exact (not_lt_of_ge contracted) failed
    _ ≤ Real.exp (-epsilon ^ 2 /
        (2 * ((‖upper - lower‖₊ / 2) ^ 2))) := eventBound

/-- A natural-window candidate score yields a high-probability population
simple-regret contraction.  The only probabilistic failure charged here is
candidate-loss overestimation, because the proposed move already has a
deterministic transported population-gain certificate. -/
theorem regretTail
    [MeasurableSpace World]
    [StandardBorelSpace World]
    (measure : Measure World)
    [IsZeroOrProbabilityMeasure measure]
    (batch : AdaptiveBatch World measure)
    (sampleCount : ℕ)
    (countPositive : 0 < sampleCount)
    [PseudoMetricSpace State]
    (objective : Objective State)
    (constant : ℝ≥0)
    (regular : LipschitzWith constant objective)
    (apply : State → Move → State)
    (past incumbent best : State)
    (move : Move)
    (baseScore : ℝ)
    (gain baseRadius trialRadius baseError margin : ℝ)
    (gainShare loss bias epsilon : ℝ)
    (epsilonNonnegative : 0 ≤ epsilon)
    (pastGain : gain ≤ progress objective past (apply past move))
    (baseNear : dist incumbent past ≤ baseRadius)
    (trialNear : dist (apply past move) (apply incumbent move) ≤ trialRadius)
    (baseAccurate : |baseScore - objective incumbent| ≤ baseError)
    (centerBound : ∀ world,
      meanCenter batch sampleCount world - objective (apply incumbent move) ≤ bias)
    (separated : baseError + (epsilon + bias) + margin ≤
      gain - constant * (baseRadius + trialRadius))
    (retained : gainShare * Regret.simpleRegret objective best incumbent - loss ≤
      gain - constant * (baseRadius + trialRadius)) :
    measure.real {world |
        (1 - gainShare) * Regret.simpleRegret objective best incumbent + loss <
          Regret.simpleRegret objective best
            (ObjectiveSearch.commit incumbent
              (fun _ : Fin 1 => apply incumbent move)
              (windowScores batch sampleCount baseScore world) margin 0)} ≤
      Real.exp (-((sampleCount : ℝ) * epsilon) ^ 2 /
        (2 * ∑ index ∈ Finset.range sampleCount, batch.scale index)) := by
  let candidate : Fin 1 → State := fun _ => apply incumbent move
  let target := objective (apply incumbent move)
  have useful := TransferGuide.progressBound objective constant regular apply
    past incumbent move gain baseRadius trialRadius pastGain baseNear trialNear
  have eventBound := fixedTail measure batch sampleCount countPositive target bias
    centerBound epsilonNonnegative
  calc
    measure.real {world |
        (1 - gainShare) * Regret.simpleRegret objective best incumbent + loss <
          Regret.simpleRegret objective best
            (ObjectiveSearch.commit incumbent candidate
              (windowScores batch sampleCount baseScore world) margin 0)} ≤
      measure.real {world |
        epsilon + bias ≤ meanLoss batch sampleCount world - target} := by
          refine measureReal_mono ?_ (by finiteness)
          intro world failed
          simp only [Set.mem_setOf_eq] at failed ⊢
          by_contra scoreNotHigh
          have scoreUpper :
              meanLoss batch sampleCount world - target < epsilon + bias :=
            lt_of_not_ge scoreNotHigh
          have baseLower := (abs_le.mp baseAccurate).1
          have accepted : ObjectiveSearch.Accept
              (windowScores batch sampleCount baseScore world) margin (0 : Fin 1) := by
            unfold ObjectiveSearch.Accept windowScores
            change meanLoss batch sampleCount world + margin ≤ baseScore
            unfold progress at useful
            dsimp [target] at scoreUpper
            linarith
          have contracted :
              Regret.simpleRegret objective best (candidate 0) ≤
                (1 - gainShare) *
                    Regret.simpleRegret objective best incumbent + loss := by
            unfold progress at useful
            simp only [Regret.simpleRegret] at retained ⊢
            dsimp [candidate]
            linarith
          rw [ObjectiveSearch.commit, if_pos accepted] at failed
          exact (not_lt_of_ge contracted) failed
    _ ≤ Real.exp (-((sampleCount : ℝ) * epsilon) ^ 2 /
        (2 * ∑ index ∈ Finset.range sampleCount, batch.scale index)) := eventBound

/-- End-to-end marginal-window regret contraction for the actual source ENN
decision (`EnnCommit.scalarCommit` with `Neighbor.mean` and its denominator
floor), discharging the Hoeffding window-loss range directly from
`LossGeometry.trustRange`.

Research problem: for which population objectives and retained histories does
the source proposal law supply `pastGain` and `retained` with a positive regret
share and controlled loss?  The proposal may use previous evaluations and ENN
state, but not the unknown optimum, population loss at an unevaluated state, or
a gradient.  This theorem checks the consequence of those premises for one
commit; it does not prove that the deployed generator produces such a move.
The hypotheses `marginalEq`, `observed`, `near`, and `separated` likewise need
independent justification for the actual data and estimator. -/
theorem sourceTail
    {capacity count tokens width : ℕ}
    [NeZero count]
    (tokenPositive : 0 < tokens)
    [PseudoMetricSpace State]
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters) (scale : ℝ)
    (distances : Forward.Vector capacity) (nearest : List (Fin capacity))
    (objective : Objective State)
    (points : Fin capacity → State)
    (apply : State → Move → State)
    (past incumbent best : State) (move : Move)
    (noise geometry : Forward.Vector capacity)
    (logits : Fin count → State → Forward.Rows tokens width)
    (targets : Fin count → Fin tokens → Fin width)
    (logitLip : ℝ≥0)
    (logitRegular : ∀ origin, LipschitzWith logitLip (logits origin))
    (logitCap trustRadius : ℝ)
    (logitBounded : ∀ origin token index,
      |logits origin incumbent token index| ≤ logitCap)
    (trustNear : dist (apply incumbent move) incumbent ≤ trustRadius)
    (gain baseShift trialShift margin : ℝ)
    (gainShare loss epsilon : ℝ)
    (epsilonNonnegative : 0 ≤ epsilon)
    (weights : ∀ position : Fin nearest.length,
      0 ≤ Neighbor.weight data params scale distances nearest[position.val])
    (observed : ∀ position : Fin nearest.length,
      |data.values nearest[position.val] -
        -objective (points nearest[position.val])| ≤ noise nearest[position.val])
    (near : ∀ position : Fin nearest.length,
      |-objective (points nearest[position.val]) - -objective incumbent| ≤
        geometry nearest[position.val])
    (marginalEq :
      corpusMean (fun origin =>
        Forward.average (fun token =>
          Forward.logLoss (logits origin (apply incumbent move) token)
            (targets origin token))) =
        objective (apply incumbent move))
    (pastGain : gain ≤ progress objective past (apply past move))
    (baseNear : |objective incumbent - objective past| ≤ baseShift)
    (trialNear : |objective (apply past move) -
      objective (apply incumbent move)| ≤ trialShift)
    (separated :
      EnnCommit.errorRadius data params scale distances nearest
        (-objective incumbent) noise geometry + epsilon + margin <
        gain - baseShift - trialShift)
    (retained : gainShare * Regret.simpleRegret objective best incumbent - loss ≤
      gain - baseShift - trialShift) :
    (Objective.MarginalWindow.originMeasure count).real {origin |
        (1 - gainShare) * Regret.simpleRegret objective best incumbent + loss <
          Regret.simpleRegret objective best
            (EnnCommit.scalarCommit incumbent (apply incumbent move)
              (Neighbor.mean data params scale distances nearest)
              (-Forward.average (fun token =>
                Forward.logLoss (logits origin (apply incumbent move) token)
                  (targets origin token)))
              margin)} ≤
      Real.exp (-epsilon ^ 2 /
        (2 * ((‖(Real.log width + 2 * (logitCap + logitLip * trustRadius)) - 0‖₊ / 2) ^ 2))) := by
  let windowLoss : Fin count → ℝ := fun origin =>
    Forward.average (fun token =>
      Forward.logLoss (logits origin (apply incumbent move) token)
        (targets origin token))
  let upper : ℝ := Real.log width + 2 * (logitCap + logitLip * trustRadius)
  have bounded : ∀ origin, windowLoss origin ∈ Set.Icc 0 upper := by
    intro origin
    exact Forward.LossGeometry.trustRange tokenPositive (logits origin)
      (targets origin) logitLip (logitRegular origin) incumbent
      (apply incumbent move) logitCap trustRadius (logitBounded origin) trustNear
  have eventBound := Objective.MarginalWindow.upperTail count windowLoss
    0 upper epsilon bounded epsilonNonnegative
  have baseAccurate := EnnCommit.baseBound data params scale distances nearest
    (fun state => -objective state) points incumbent noise geometry weights
    observed near
  have useful := TransferGuide.progressLocal objective apply past incumbent move
    gain baseShift trialShift pastGain baseNear trialNear
  calc
    (Objective.MarginalWindow.originMeasure count).real {origin |
        (1 - gainShare) * Regret.simpleRegret objective best incumbent + loss <
          Regret.simpleRegret objective best
            (EnnCommit.scalarCommit incumbent (apply incumbent move)
              (Neighbor.mean data params scale distances nearest)
              (-windowLoss origin) margin)} ≤
      (Objective.MarginalWindow.originMeasure count).real {origin |
        epsilon ≤ windowLoss origin - corpusMean windowLoss} := by
          refine measureReal_mono ?_ (by finiteness)
          intro origin failed
          simp only [Set.mem_setOf_eq] at failed ⊢
          by_contra scoreNotHigh
          have scoreUpper :
              windowLoss origin - corpusMean windowLoss < epsilon :=
            lt_of_not_ge scoreNotHigh
          have baseUpper := (abs_le.mp baseAccurate).2
          have accepted : EnnCommit.accepts
              (-windowLoss origin - Neighbor.mean data params scale distances nearest)
              margin := by
            unfold EnnCommit.accepts
            unfold progress at useful
            rw [marginalEq] at scoreUpper
            linarith
          have contracted :
              Regret.simpleRegret objective best (apply incumbent move) ≤
                (1 - gainShare) *
                    Regret.simpleRegret objective best incumbent + loss := by
            unfold progress at useful
            simp only [Regret.simpleRegret] at retained ⊢
            linarith
          rw [EnnCommit.scalarCommit, if_pos accepted] at failed
          exact (not_lt_of_ge contracted) failed
    _ ≤ Real.exp (-epsilon ^ 2 /
        (2 * ((‖upper - 0‖₊ / 2) ^ 2))) := eventBound

#print axioms regretTail
#print axioms marginalTail
#print axioms sourceTail

end

end BlckbauxResearch.Optimization.PretrainRegret
