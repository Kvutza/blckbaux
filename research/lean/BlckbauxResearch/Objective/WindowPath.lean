import BlckbauxResearch.Objective.NaturalWindow
import Mathlib.Topology.MetricSpace.Lipschitz

/-!
# Window-dependent gains along a parameter path

Let l(x,o) be loss at parameter state x on corpus origin o, and let F(x)
be its uniform mean over origins. An exact same-window improvement is
g(x,y,o) = l(x,o)-l(y,o). Summing such improvements along accepted states
need not give F(x₀)-F(xₜ): each edge may use a different origin.

This module isolates the error without assigning a probability law to the
deterministic source schedule. It also characterizes when all two-edge cycles
cancel, and constructs a bounded-loss cycle with arbitrarily large accumulated
score but zero population improvement. The counterexample concerns the stated
function class, not a claim that this cycle occurred in a neural-network run.

The inspected Metal long-context runner stores a baseline plus same-window
improvement. The path below abstracts accepted transitions only. Rejected
proposals still need their own row error bound; floating-point arithmetic,
patch aggregation, and the controller's actual transitions are not refined here.
-/

namespace BlckbauxResearch.Objective.WindowPath

open scoped BigOperators NNReal
open BlckbauxResearch.Objective.NaturalWindow

noncomputable section

variable {State Origin : Type*}

/-- Positive values mean lower loss at the proposed state, on this origin. -/
def gain (loss : State → Origin → ℝ) (left right : State) (origin : Origin) : ℝ :=
  loss left origin - loss right origin

/-- Zero-anchored score accumulated along accepted transitions. -/
def accumulated (loss : State → Origin → ℝ) (path : ℕ → State)
    (origin : ℕ → Origin) (steps : ℕ) : ℝ :=
  ∑ step ∈ Finset.range steps, gain loss (path step) (path (step + 1)) (origin step)

theorem meanGain {count : ℕ} (loss : State → Fin count → ℝ) (left right : State) :
    corpusMean (gain loss left right) = corpusMean (loss left) - corpusMean (loss right) := by
  simp only [corpusMean, gain, Finset.sum_sub_distrib, mul_sub]

/-- The error is exactly the sum of edge biases against the SAME population
objective. Exact forward scores do not make these biases vanish. -/
theorem errorSplit {count : ℕ} (loss : State → Fin count → ℝ)
    (path : ℕ → State) (origin : ℕ → Fin count) (steps : ℕ) :
    accumulated loss path origin steps -
        (corpusMean (loss (path 0)) - corpusMean (loss (path steps))) =
      ∑ step ∈ Finset.range steps,
        (gain loss (path step) (path (step + 1)) (origin step) -
          corpusMean (gain loss (path step) (path (step + 1)))) := by
  induction steps with
  | zero => simp [accumulated]
  | succ steps induction =>
      simp only [accumulated, Finset.sum_range_succ] at induction ⊢
      rw [meanGain]
      linarith

/-- A directly usable deterministic radius: bound the variation across origins
of each improvement, rather than the much larger variation of raw losses.
Neither independence nor a fresh random origin is required. -/
theorem errorBound {count : ℕ} (positive : 0 < count)
    (loss : State → Fin count → ℝ) (path : ℕ → State)
    (origin : ℕ → Fin count) (steps : ℕ) (radius : ℕ → ℝ)
    (oscillation : ∀ step < steps, ∀ other,
      |gain loss (path step) (path (step + 1)) other -
        gain loss (path step) (path (step + 1)) (origin step)| ≤ radius step) :
    |accumulated loss path origin steps -
      (corpusMean (loss (path 0)) - corpusMean (loss (path steps)))| ≤
        ∑ step ∈ Finset.range steps, radius step := by
  rw [errorSplit]
  refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
  apply Finset.sum_le_sum
  intro step member
  exact biasBound positive (gain loss (path step) (path (step + 1)))
    (origin step) (radius step) (oscillation step (Finset.mem_range.mp member))

/-- A proposed row inherits the incumbent's accumulated error and adds its own
window error. Here target is initial population loss minus current population
loss, matching a zero initial score and positive improvement rewards. -/
theorem rowBound (stored target measured truth baseError edgeError : ℝ)
    (base : |stored - target| ≤ baseError)
    (edge : |measured - truth| ≤ edgeError) :
    |(stored + measured) - (target + truth)| ≤ baseError + edgeError := by
  calc
    _ = |(stored - target) + (measured - truth)| := by congr 1; ring
    _ ≤ |stored - target| + |measured - truth| := abs_add_le _ _
    _ ≤ baseError + edgeError := add_le_add base edge

/-- No-error accumulation for arbitrary states and arbitrary origins has a
strong exact characterization: loss is a parameter-only term plus an
origin-only term. Two-edge cycle cancellation already suffices. The supplied
reference state and origin avoid hidden nonemptiness assumptions. -/
theorem separableIff (loss : State → Origin → ℝ) (base : State) (reference : Origin) :
    (∀ left right first second,
      gain loss left right first + gain loss right left second = 0) ↔
      ∃ objective : State → ℝ, ∃ difficulty : Origin → ℝ,
        ∀ state origin, loss state origin = objective state + difficulty origin := by
  constructor
  · intro cancels
    refine ⟨fun state => loss state reference,
      fun origin => loss base origin - loss base reference, ?_⟩
    intro state origin
    have identity := cancels state base origin reference
    dsimp [gain] at identity
    linarith
  · rintro ⟨objective, difficulty, separates⟩ left right first second
    simp only [gain, separates]
    ring

/-- A shared corpus-difficulty term cancels exactly. This is a sufficient
structural condition, not an asserted property of the pretraining objective. -/
theorem separableExact {count : ℕ} (positive : 0 < count)
    (loss : State → Fin count → ℝ) (objective : State → ℝ)
    (difficulty : Fin count → ℝ)
    (separates : ∀ state origin, loss state origin = objective state + difficulty origin)
    (path : ℕ → State) (origin : ℕ → Fin count) (steps : ℕ) :
    accumulated loss path origin steps =
      corpusMean (loss (path 0)) - corpusMean (loss (path steps)) := by
  have bound := errorBound positive loss path origin steps (fun _ => 0) (by
    intro step before other
    simp only [gain, separates]
    have identity :
        (objective (path step) + difficulty other -
          (objective (path (step + 1)) + difficulty other)) -
        (objective (path step) + difficulty (origin step) -
          (objective (path (step + 1)) + difficulty (origin step))) = 0 := by ring
    rw [identity, abs_zero])
  simp only [Finset.sum_const_zero] at bound
  exact sub_eq_zero.mp (abs_eq_zero.mp (le_antisymm bound (abs_nonneg _)))

/-- A metric controls this error through the variation of the window-centered
loss, not necessarily through the full loss. The assumption is a mixed
state/origin difference bound; deriving a SMALL constant remains necessary. -/
theorem movementBound [PseudoMetricSpace State] {count : ℕ} (positive : 0 < count)
    (loss : State → Fin count → ℝ) (constant : ℝ≥0)
    (mixed : ∀ left right first second,
      |gain loss left right first - gain loss left right second| ≤
        constant * dist left right)
    (path : ℕ → State) (origin : ℕ → Fin count) (steps : ℕ) :
    |accumulated loss path origin steps -
      (corpusMean (loss (path 0)) - corpusMean (loss (path steps)))| ≤
        constant * ∑ step ∈ Finset.range steps, dist (path step) (path (step + 1)) := by
  simpa only [Finset.mul_sum] using
    errorBound positive loss path origin steps
      (fun step => constant * dist (path step) (path (step + 1)))
      (fun step _ other => mixed _ _ other (origin step))

/-- An available but potentially loose fallback: if every origin's loss is
L-Lipschitz in the same metric, the mixed difference is at most 2*L*d.
This uses Mathlib's metric Lipschitz bound, with no derivatives. -/
theorem mixedBound [PseudoMetricSpace State] (loss : State → Origin → ℝ)
    (constant : ℝ≥0) (regular : ∀ origin, LipschitzWith constant (fun state => loss state origin))
    (left right : State) (first second : Origin) :
    |gain loss left right first - gain loss left right second| ≤
      (2 * constant : ℝ≥0) * dist left right := by
  have firstBound := (regular first).dist_le_mul left right
  have secondBound := (regular second).dist_le_mul left right
  rw [Real.dist_eq] at firstBound secondBound
  calc
    _ ≤ |gain loss left right first| + |gain loss left right second| := abs_sub _ _
    _ ≤ constant * dist left right + constant * dist left right := add_le_add firstBound secondBound
    _ = _ := by simp only [NNReal.coe_mul, NNReal.coe_ofNat]; ring

/-- The two states have opposite window preferences but equal corpus means. -/
def cycleLoss (state origin : Fin 2) : ℝ := if state = origin then 1 else 0

theorem cycleBound (state origin : Fin 2) : cycleLoss state origin ∈ Set.Icc (0 : ℝ) 1 := by
  simp only [cycleLoss]
  split_ifs <;> constructor <;> norm_num

theorem cycleMean (state : Fin 2) : corpusMean (cycleLoss state) = 1 / 2 := by
  fin_cases state <;> norm_num [corpusMean, cycleLoss, Fin.sum_univ_two]

/-- Alternate two states, and use the identically indexed origin at each edge. -/
def cycleState (step : ℕ) : Fin 2 := ⟨step % 2, Nat.mod_lt _ (by decide)⟩

theorem cycleReturns (cycles : ℕ) : cycleState (2 * cycles) = cycleState 0 := by
  apply Fin.ext
  simp [cycleState]

theorem cycleGain (step : ℕ) :
    gain cycleLoss (cycleState step) (cycleState (step + 1)) (cycleState step) = 1 := by
  have different : cycleState (step + 1) ≠ cycleState step := by
    intro equality
    have values := congrArg Fin.val equality
    dsimp [cycleState] at values
    omega
  simp [gain, cycleLoss, different]

/-- The actual accumulated-path definition grows by one on EVERY transition,
although the population improvement is always zero. The path returns to its
initial state after every two transitions. All losses lie in [0,1]; neither
noisy evaluations, an unbounded loss, nor arithmetic error causes the drift. -/
theorem cycleGrowth (steps : ℕ) :
    accumulated cycleLoss cycleState cycleState steps = (steps : ℝ) ∧
      corpusMean (cycleLoss (cycleState 0)) -
        corpusMean (cycleLoss (cycleState steps)) = 0 := by
  constructor
  · simp [accumulated, cycleGain]
  · rw [cycleMean, cycleMean, sub_self]

#print axioms meanGain
#print axioms errorSplit
#print axioms errorBound
#print axioms rowBound
#print axioms separableIff
#print axioms separableExact
#print axioms movementBound
#print axioms mixedBound
#print axioms cycleBound
#print axioms cycleMean
#print axioms cycleReturns
#print axioms cycleGain
#print axioms cycleGrowth

end
end BlckbauxResearch.Objective.WindowPath
