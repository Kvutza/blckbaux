import BlckbauxResearch.Optimization.Regret
import BlckbauxResearch.Optimization.ResearchProblem

/-!
# Experiment traces and alternative regret views

This module supplies a common deterministic trace semantics without selecting a
single notion of regret. A `World` may contain an entire task/randomness path, so
fixing it here does not assert that the experiment itself is non-random.

Terminal gap, cumulative gap, target hitting, matched-budget comparison, and
pairwise excess are separate views. Theorems record when they coincide. A
counterexample records that reaching a target does not imply ending at that
target unless a stability condition is supplied.
-/

/-!
For a fixed world and machine, write `θₜ` for the trace state and `Lₜ` for
its score.  Terminal gap uses `L_T`, cumulative gap sums scores before `T`,
and target hitting asserts existence of some `t ≤ T`.  These are distinct
predicates; no glyph should identify them without an additional theorem.
Resource usage is a vector indexed by kind, not an implicit scalar FLOP cost.
-/

namespace BlckbauxResearch.Semantics.Experiment

open scoped BigOperators
open BlckbauxResearch.Optimization
open BlckbauxResearch.Optimization.ResearchProblem

/-- Oriented score difference: positive means the candidate is worse. -/
def scoreGap (sense : ObjectiveSense) (candidate reference : ℝ) : ℝ :=
  match sense with
  | .minimize => candidate - reference
  | .maximize => reference - candidate

/-- A best state is defined relative to the declared objective direction. -/
def Best {State : Type*} (sense : ObjectiveSense)
    (objective : State → ℝ) (best : State) : Prop :=
  ∀ state, sense.noWorse (objective best) (objective state)

theorem gapIff (sense : ObjectiveSense) (candidate reference : ℝ) :
    sense.noWorse candidate reference ↔ scoreGap sense candidate reference ≤ 0 := by
  cases sense <;> simp [ObjectiveSense.noWorse, scoreGap]

theorem orderTrans (sense : ObjectiveSense) (first second third : ℝ)
    (firstBetter : sense.noWorse first second)
    (secondBetter : sense.noWorse second third) :
    sense.noWorse first third := by
  cases sense <;> simp only [ObjectiveSense.noWorse] at * <;> linarith

theorem gap_nonneg {State : Type*} (sense : ObjectiveSense)
    (objective : State → ℝ) (best state : State)
    (optimal : Best sense objective best) :
    0 ≤ scoreGap sense (objective state) (objective best) := by
  have comparison := optimal state
  cases sense <;>
    simp only [ObjectiveSense.noWorse, scoreGap] at comparison ⊢ <;> linarith

/-- One fully specified execution path under a fixed objective world and machine. -/
structure Trace {State : Type*}
    (construction : ObjectiveConstruction State)
    (world : construction.World)
    (machine : Machine) where
  initial : State
  state : ℕ → State
  starts : state 0 = initial
  usage : ℕ → ResourceUsage
  usageMono : ∀ {first second}, first ≤ second → Resource.le (usage first) (usage second)

namespace Trace

variable {State : Type*}
  {construction : ObjectiveConstruction State}
  {world : construction.World}
  {machine : Machine}

def score (trace : Trace construction world machine) (time : ℕ) : ℝ :=
  construction.score (construction.evaluate (trace.state time) world)

def objective (state : State) : ℝ :=
  construction.score (construction.evaluate state world)

/-- Terminal gap to a state comparator at one declared horizon. -/
def terminal (trace : Trace construction world machine)
    (comparator : State) (horizon : ℕ) : ℝ :=
  scoreGap construction.sense (trace.score horizon)
    (objective (construction := construction) (world := world) comparator)

/-- Sum of pre-terminal gaps at times `0, ..., horizon - 1`. -/
def cumulative (trace : Trace construction world machine)
    (comparator : State) (horizon : ℕ) : ℝ :=
  ∑ time ∈ Finset.range horizon,
    scoreGap construction.sense (trace.score time)
      (objective (construction := construction) (world := world) comparator)

/-- Pairwise terminal excess; positive means `candidate` ends worse. -/
def excess (candidate reference : Trace construction world machine)
    (horizon : ℕ) : ℝ :=
  scoreGap construction.sense (candidate.score horizon) (reference.score horizon)

/-- The trace reaches the scalar target at or before the horizon. -/
def Hits (trace : Trace construction world machine)
    (target : ℝ) (horizon : ℕ) : Prop :=
  ∃ time, time ≤ horizon ∧ construction.sense.noWorse (trace.score time) target

/-- Later scores never become worse than earlier scores. -/
def Stable (trace : Trace construction world machine) : Prop :=
  ∀ {first second}, first ≤ second →
    construction.sense.noWorse (trace.score second) (trace.score first)

/-- Resource feasibility against a componentwise budget at one horizon. -/
def Within (trace : Trace construction world machine)
    (budget : ResourceUsage) (horizon : ℕ) : Prop :=
  Resource.le (trace.usage horizon) budget

/-- A comparison fixes the start, horizon, machine, and componentwise budget. -/
def Matched (candidate reference : Trace construction world machine)
    (budget : ResourceUsage) (horizon : ℕ) : Prop :=
  candidate.initial = reference.initial ∧
    candidate.Within budget horizon ∧ reference.Within budget horizon

/-- Common-envelope terminal dominance, without selecting a regret-to-optimum view. -/
def Beats (candidate reference : Trace construction world machine)
    (budget : ResourceUsage) (horizon : ℕ) : Prop :=
  Matched candidate reference budget horizon ∧
    construction.sense.noWorse (candidate.score horizon) (reference.score horizon)

/-- Terminal and componentwise resource dominance on comparable traces. -/
def Fair (candidate reference : Trace construction world machine)
    (horizon : ℕ) : Prop :=
  candidate.initial = reference.initial ∧
    construction.sense.noWorse (candidate.score horizon) (reference.score horizon) ∧
    Resource.le (candidate.usage horizon) (reference.usage horizon)

/-- Evidence needed to turn one trace horizon into an accounted run result. -/
structure Account (trace : Trace construction world machine) (horizon : ℕ) where
  outerQueries : ℕ
  queryAccounted : trace.usage horizon .outerObjectiveQueries = outerQueries
  evaluationCovered : Resource.le
    (construction.evaluationUsage (trace.state horizon) world)
    (trace.usage horizon)
  search : SearchCount
  searchCovered : search.evaluated ≤ outerQueries

/-- The terminal snapshot of a trace, with every accounting obligation retained. -/
def result (trace : Trace construction world machine) (horizon : ℕ)
    (account : Account trace horizon) : RunResult construction world machine where
  initialState := trace.initial
  finalState := trace.state horizon
  finalScore := trace.score horizon
  scoreExact := rfl
  outerObjectiveQueries := account.outerQueries
  usage := trace.usage horizon
  queryAccounted := account.queryAccounted
  evaluationCovered := account.evaluationCovered
  search := account.search
  searchCovered := account.searchCovered

/-- Trace fairness is exactly sufficient for fairness of accounted snapshots. -/
theorem fairResult
    (candidate reference : Trace construction world machine) (horizon : ℕ)
    (candidateAccount : Account candidate horizon)
    (referenceAccount : Account reference horizon)
    (fair : candidate.Fair reference horizon) :
    Run.fair (candidate.result horizon candidateAccount)
      (reference.result horizon referenceAccount) :=
  fair

/-- One accounted query and terminal dominance imply the one-query run contract. -/
theorem queryResult
    (candidate reference : Trace construction world machine) (horizon : ℕ)
    (candidateAccount : Account candidate horizon)
    (referenceAccount : Account reference horizon)
    (sameStart : candidate.initial = reference.initial)
    (oneQuery : candidateAccount.outerQueries = 1)
    (noWorse : construction.sense.noWorse
      (candidate.score horizon) (reference.score horizon)) :
    Run.oneQuery (candidate.result horizon candidateAccount)
      (reference.result horizon referenceAccount) :=
  ⟨sameStart, oneQuery, noWorse⟩

theorem terminal_min
    (trace : Trace construction world machine) (best : State) (horizon : ℕ)
    (sense : construction.sense = .minimize) :
    trace.terminal best horizon =
      Regret.simpleRegret
        (objective (construction := construction) (world := world))
        best (trace.state horizon) := by
  simp [terminal, score, objective, scoreGap, sense, Regret.simpleRegret]

/-- A common comparator cancels for either objective direction. -/
theorem excess_eq
    (candidate reference : Trace construction world machine)
    (comparator : State) (horizon : ℕ) :
    candidate.terminal comparator horizon - reference.terminal comparator horizon =
      candidate.excess reference horizon := by
  cases sense : construction.sense <;>
    simp [terminal, excess, scoreGap, sense]

theorem terminal_nonneg
    (trace : Trace construction world machine) (best : State) (horizon : ℕ)
    (optimal : Best construction.sense
      (objective (construction := construction) (world := world)) best) :
    0 ≤ trace.terminal best horizon :=
  gap_nonneg construction.sense
    (objective (construction := construction) (world := world))
    best (trace.state horizon) optimal

theorem cumulative_nonneg
    (trace : Trace construction world machine) (best : State) (horizon : ℕ)
    (optimal : Best construction.sense
      (objective (construction := construction) (world := world)) best) :
    0 ≤ trace.cumulative best horizon := by
  simp only [cumulative]
  apply Finset.sum_nonneg
  intro time _
  exact gap_nonneg construction.sense
    (objective (construction := construction) (world := world))
    best (trace.state time) optimal

/-- Reaching a target transfers to the terminal state only under stability. -/
theorem hit_terminal
    (trace : Trace construction world machine) (target : ℝ) (horizon : ℕ)
    (stable : trace.Stable) (hit : trace.Hits target horizon) :
    construction.sense.noWorse (trace.score horizon) target := by
  obtain ⟨time, before, reached⟩ := hit
  exact orderTrans construction.sense _ _ _ (stable before) reached

theorem usage_before
    (trace : Trace construction world machine) (budget : ResourceUsage)
    {first second : ℕ} (before : first ≤ second) (within : trace.Within budget second) :
    trace.Within budget first := by
  intro resource
  exact (trace.usageMono before resource).trans (within resource)

end Trace

/-- Target hitting alone does not imply a terminal guarantee. -/
theorem hitCounterexample :
    ∃ scores : ℕ → ℝ,
      (∃ time, time ≤ 1 ∧ ObjectiveSense.minimize.noWorse (scores time) 0) ∧
      ¬ ObjectiveSense.minimize.noWorse (scores 1) 0 := by
  refine ⟨fun time => if time = 0 then -1 else 1, ?_, ?_⟩
  · exact ⟨0, by omega, by simp [ObjectiveSense.noWorse]⟩
  · simp [ObjectiveSense.noWorse]

#print axioms gapIff
#print axioms gap_nonneg
#print axioms Trace.excess_eq
#print axioms Trace.hit_terminal
#print axioms Trace.usage_before
#print axioms Trace.fairResult
#print axioms Trace.queryResult
#print axioms hitCounterexample

end BlckbauxResearch.Semantics.Experiment
