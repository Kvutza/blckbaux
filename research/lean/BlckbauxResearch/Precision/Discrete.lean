import BlckbauxResearch.Precision.Float16

/-!
# Native discrete parameter proposals

Rather than drawing proposals in continuous Euclidean space ℝ^D and passing
them through a lossy quantizer at commit time, a native discrete proposal
operates directly on the finite IEEE binary16 lattice.

This directly avoids three failure modes of continuous quantization:
* update vanishing: infinitesimal continuous updates are rounded to zero;
* latent drift: the continuous controller center diverges from VRAM bit patterns;
* quantization error: discrepancy between proposed and executed candidates.
-/

namespace BlckbauxResearch.Precision.Discrete

open BlckbauxResearch.Precision.Float16

/-- A discrete coordinate step on the IEEE binary16 lattice. -/
structure Step where
  delta : Int
  deriving DecidableEq, Repr

/-- Zero step corresponds to maintaining the parameter unchanged. -/
instance : Zero Step where
  zero := ⟨0⟩

/-- A step is active if its displacement is nonzero. -/
def Step.IsActive (step : Step) : Prop :=
  step.delta ≠ 0

instance (step : Step) : Decidable step.IsActive :=
  inferInstanceAs (Decidable (step.delta ≠ 0))

/-- A native discrete proposal assigns a discrete step to every learned parameter. -/
abbrev Proposal (parameterCount : ℕ) :=
  ParameterIndex parameterCount → Step

/-- A dense discrete proposal has an active displacement at every parameter coordinate. -/
def IsDense {parameterCount : ℕ} (proposal : Proposal parameterCount) : Prop :=
  ∀ parameter, (proposal parameter).IsActive

/-- A transition rule specifies how a discrete step updates a finite word. -/
structure Transition where
  apply : FiniteWord → Step → Word
  applyZero : ∀ word, apply word 0 = word.1
  applyActive : ∀ (word : FiniteWord) (step : Step),
    step.IsActive → apply word step ≠ word.1

/-- A transition is finite-preserving on a given state and proposal. -/
def Admissible
    {parameterCount : ℕ}
    (transition : Transition)
    (state : State parameterCount)
    (proposal : Proposal parameterCount) : Prop :=
  ∀ parameter, (transition.apply (state parameter) (proposal parameter)).IsFinite

/-- Apply a discrete proposal directly to an executed FP16 state. -/
def commit
    {parameterCount : ℕ}
    (transition : Transition)
    (state : State parameterCount)
    (proposal : Proposal parameterCount)
    (admissible : Admissible transition state proposal) :
    State parameterCount :=
  fun parameter =>
    ⟨transition.apply (state parameter) (proposal parameter), admissible parameter⟩

/-- The zero discrete proposal is always admissible. -/
theorem zeroValid
    {parameterCount : ℕ}
    (transition : Transition)
    (state : State parameterCount) :
    Admissible transition state 0 := by
  intro parameter
  change (transition.apply (state parameter) 0).IsFinite
  rw [transition.applyZero]
  exact (state parameter).property

/-- Applying the zero proposal preserves the state exactly. -/
theorem commitZero
    {parameterCount : ℕ}
    (transition : Transition)
    (state : State parameterCount) :
    commit transition state 0 (zeroValid transition state) = state := by
  funext parameter
  apply Subtype.ext
  simp [commit, transition.applyZero]

/-- A coordinate is physically modified by the discrete proposal. -/
def ChangedAt
    {parameterCount : ℕ}
    (transition : Transition)
    (state : State parameterCount)
    (proposal : Proposal parameterCount)
    (admissible : Admissible transition state proposal)
    (parameter : ParameterIndex parameterCount) : Prop :=
  commit transition state proposal admissible parameter ≠ state parameter

/--
Every active coordinate in an admissible proposal is guaranteed to produce
an executed state change. Unlike continuous quantization, active discrete steps
cannot vanish.
-/
theorem activeEffect
    {parameterCount : ℕ}
    (transition : Transition)
    (state : State parameterCount)
    (proposal : Proposal parameterCount)
    (admissible : Admissible transition state proposal)
    (parameter : ParameterIndex parameterCount)
    (active : (proposal parameter).IsActive) :
    ChangedAt transition state proposal admissible parameter := by
  unfold ChangedAt commit
  intro equalWord
  have wordEqual := congrArg Subtype.val equalWord
  have neq := transition.applyActive (state parameter) (proposal parameter) active
  exact neq wordEqual

/--
A dense discrete proposal is guaranteed to change EVERY stored parameter in the model.
This establishes 100% update density without quantization extinction.
-/
theorem denseEffect
    {parameterCount : ℕ}
    (transition : Transition)
    (state : State parameterCount)
    (proposal : Proposal parameterCount)
    (admissible : Admissible transition state proposal)
    (dense : IsDense proposal) :
    ∀ parameter, ChangedAt transition state proposal admissible parameter := by
  intro parameter
  exact activeEffect transition state proposal admissible parameter (dense parameter)

/--
A dense proposal forces the entire committed state to differ from the initial state,
provided parameterCount is positive.
-/
theorem denseDiff
    {parameterCount : ℕ}
    (transition : Transition)
    (state : State parameterCount)
    (proposal : Proposal parameterCount)
    (admissible : Admissible transition state proposal)
    (dense : IsDense proposal)
    (positiveCount : 0 < parameterCount) :
    commit transition state proposal admissible ≠ state := by
  intro equalStates
  let firstParam : ParameterIndex parameterCount := ⟨0, positiveCount⟩
  have changed := denseEffect transition state proposal admissible dense firstParam
  exact changed (congrFun equalStates firstParam)

/-- Native discrete FP16 proposals guarantee stored state changes. -/
def NativeClaim
    {parameterCount : ℕ}
    (transition : Transition)
    (state : State parameterCount)
    (proposal : Proposal parameterCount)
    (admissible : Admissible transition state proposal)
    (dense : IsDense proposal) : ResearchClaim where
  statement := "A dense native FP16 proposal changes every stored model parameter without quantization loss."
  proposition := ∀ parameter, ChangedAt transition state proposal admissible parameter
  status := .proved (denseEffect transition state proposal admissible dense)

end BlckbauxResearch.Precision.Discrete
