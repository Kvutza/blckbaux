import BlckbauxResearch.Foundation
import Mathlib.Data.Real.Basic

/-!
# Active pretraining workload

The one-million-token value is the size of one objective-evaluation window, not
the duration of pretraining. A pretraining horizon is expressed by the total
number of evaluated tokens. Optimizer transactions merely partition that token
stream, so this module fixes no transaction count.
-/

/-!
Let `T(n) = ∑_{t<n} tokensAt(t)` be evaluated-token time.  Under the current
fixed natural-window protocol, `T(n) = n * windowTokens`.  This is accounting,
not a claim that `n` transactions suffice for a useful pretrained checkpoint;
the latter depends on a separately defined loss path and threshold.
-/

namespace BlckbauxResearch.Optimization.PretrainWorkload

open scoped BigOperators

/-- Tokens in one active objective-evaluation window. -/
def windowTokens : ℕ := currentTarget.contextTokens

/-- Complete transaction target latency in milliseconds. -/
def targetMs : ℕ := currentTarget.targetMs

/-- Model parameter count for `fbt-pisa1-hnet-mhc4-v1`. -/
def paramCount : ℕ := currentTarget.parameterCount

/-- Complete candidate states evaluated per transaction. -/
def candidateCount : ℕ := currentTarget.candidates

/-- Nearest neighbors queried by the ENN surrogate. -/
def neighborK : ℕ := currentTarget.neighbors

theorem activeContract :
    windowTokens = currentTarget.contextTokens ∧
    targetMs = currentTarget.targetMs ∧
    paramCount = currentTarget.parameterCount ∧
    candidateCount = 1 ∧ neighborK = 10 := by
  decide

/-- Cumulative evaluated tokens after an arbitrary number of transactions. -/
def tokenClock (tokensAt : ℕ → ℕ) (steps : ℕ) : ℕ :=
  ∑ step ∈ Finset.range steps, tokensAt step

theorem clockSucc (tokensAt : ℕ → ℕ) (steps : ℕ) :
    tokenClock tokensAt (steps + 1) = tokenClock tokensAt steps + tokensAt steps := by
  simp [tokenClock, Finset.sum_range_succ]

/-- With the active natural-window protocol, `steps` transactions mean exactly
`steps * windowTokens` evaluated candidate tokens. -/
theorem clockFixed (steps : ℕ) :
    tokenClock (fun _ => windowTokens) steps = steps * windowTokens := by
  simp [tokenClock]

/-- A declared base-model quality criterion stated as a loss threshold. -/
structure LossGoal where
  lossAfter : ℕ → ℝ
  threshold : ℝ

/-- The quality criterion has been reached after processing this many tokens. -/
def LossGoal.reached (goal : LossGoal) (tokens : ℕ) : Prop :=
  goal.lossAfter tokens ≤ goal.threshold

/-- A token budget is sufficient when the criterion is reached within it. -/
def LossGoal.sufficient (goal : LossGoal) (budget : ℕ) : Prop :=
  ∃ tokens ≤ budget, goal.reached tokens

/-- A token requirement is the least sufficient budget for the declared criterion. -/
def LossGoal.required (goal : LossGoal) (budget : ℕ) : Prop :=
  goal.sufficient budget ∧ ∀ smaller < budget, ¬goal.sufficient smaller

theorem LossGoal.sufficientMono
    {goal : LossGoal} {smaller larger : ℕ}
    (enough : goal.sufficient smaller) (order : smaller ≤ larger) :
    goal.sufficient larger := by
  obtain ⟨tokens, within, reached⟩ := enough
  exact ⟨tokens, within.trans order, reached⟩

theorem LossGoal.requiredUnique
    {goal : LossGoal} {left right : ℕ}
    (leftReq : goal.required left) (rightReq : goal.required right) :
    left = right := by
  rcases leftReq with ⟨leftEnough, leftLeast⟩
  rcases rightReq with ⟨rightEnough, rightLeast⟩
  apply Nat.le_antisymm
  · by_contra notOrder
    exact leftLeast right (Nat.lt_of_not_ge notOrder) rightEnough
  · by_contra notOrder
    exact rightLeast left (Nat.lt_of_not_ge notOrder) leftEnough

#print axioms clockSucc
#print axioms clockFixed
#print axioms LossGoal.requiredUnique
#print axioms activeContract

end BlckbauxResearch.Optimization.PretrainWorkload
