import BlckbauxResearch.Foundation

/-!
# One exact optimizer step

This module separates three quantities without assuming a proposal law,
surrogate, acquisition rule, smooth objective, or noise model.

* `opportunity` is the improvement offered by a comparison candidate.
* `selectionRegret` is how much worse the selected candidate is than that
  comparison candidate.
* `progress` is the incumbent-to-selected objective decrease.

The decomposition theorem is an algebraic identity.  Nonnegativity and positive
progress require the hypotheses stated by the subsequent theorems.
-/

/-!
For an objective `f`, incumbent `θ`, comparison state `c`, and selected state
`θ′`, the exact identity is
`f(θ) - f(θ′) = (f(θ) - f(c)) - (f(θ′) - f(c))`.
Thus `progress = opportunity - selectionRegret` introduces no statistical
guarantee: a useful comparison and controlled selection error are separate
premises.
-/

namespace BlckbauxResearch.Optimization

variable {State : Type*}

abbrev Objective (State : Type*) := State → ℝ

def opportunity
    (objective : Objective State)
    (incumbent comparison : State) : ℝ :=
  objective incumbent - objective comparison

def selectionRegret
    (objective : Objective State)
    (comparison selected : State) : ℝ :=
  objective selected - objective comparison

def progress
    (objective : Objective State)
    (incumbent selected : State) : ℝ :=
  objective incumbent - objective selected

theorem progress_decomposition
    (objective : Objective State)
    (incumbent comparison selected : State) :
    progress objective incumbent selected =
      opportunity objective incumbent comparison -
        selectionRegret objective comparison selected := by
  unfold progress opportunity selectionRegret
  ring

theorem opportunity_nonnegative
    (objective : Objective State)
    (incumbent comparison : State)
    (comparisonNoWorse : objective comparison ≤ objective incumbent) :
    0 ≤ opportunity objective incumbent comparison := by
  exact sub_nonneg.mpr comparisonNoWorse

theorem Selection.nonneg
    (objective : Objective State)
    (comparison selected : State)
    (comparisonNoWorse : objective comparison ≤ objective selected) :
    0 ≤ selectionRegret objective comparison selected := by
  exact sub_nonneg.mpr comparisonNoWorse

theorem Progress.positive
    (objective : Objective State)
    (incumbent comparison selected : State)
    (dominates :
      selectionRegret objective comparison selected <
        opportunity objective incumbent comparison) :
    0 < progress objective incumbent selected := by
  rw [progress_decomposition]
  exact sub_pos.mpr dominates

end BlckbauxResearch.Optimization
