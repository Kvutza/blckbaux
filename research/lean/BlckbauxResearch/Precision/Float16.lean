import BlckbauxResearch.Precision.Commit

/-!
# FP16 parameter-state semantics

The full-parameter Metal pretraining path uses IEEE binary16 storage through
`SearchState::new_implicit` and the `FP16_INDEPENDENT` Metal branch. CUDA also
has FP16 proposal paths. The legacy `bf16_*` filenames do not determine the
encoding that a configured run executes.

`Quantizer` is an abstract backend obligation. This file does not claim that
a Metal or CUDA conversion implements its decoding, rounding, or round-trip
laws; each backend requires a refinement proof.
-/

/-!
The stored state is `z : Fin p → FiniteWord`, not an element of the ideal real
space `Θₚ`.  Given a backend quantizer `q`, committing a real proposal `u`
acts coordinatewise as `q(decode(zᵢ) + uᵢ)` when every result is finite.
The abstract round-trip premise does not prove that a Metal or CUDA kernel
implements this map.
-/

namespace BlckbauxResearch.Precision.Float16

/-- One raw IEEE binary16 word, including finite values, infinities, and NaNs. -/
structure Word where
  bits : Fin 65_536
  deriving DecidableEq, Repr

/-- Bits 10 through 14 are the five-bit binary16 exponent field. -/
def Word.exponentField (word : Word) : ℕ :=
  (word.bits.val / 1024) % 32

/-- A binary16 word is finite exactly when its exponent field is not all ones. -/
def Word.IsFinite (word : Word) : Prop :=
  word.exponentField ≠ 31

instance (word : Word) : Decidable word.IsFinite :=
  inferInstanceAs (Decidable (word.exponentField ≠ 31))

/-- One finite binary16 storage word. -/
abbrev FiniteWord := { word : Word // word.IsFinite }

/-- FP16 storage has one finite word per learned parameter. -/
abbrev State (parameterCount : ℕ) :=
  ParameterIndex parameterCount → FiniteWord

/--
The semantics required from a concrete FP16 conversion.

The round-trip law makes exact preservation of a stored finite word an explicit
backend obligation.
-/
structure Quantizer where
  decode : FiniteWord → ℝ
  roundEven : ℝ → Word
  roundTrip : ∀ value, roundEven (decode value) = value.1

/-- The raw FP16 word produced for one proposed parameter update. -/
def roundedWord
    {parameterCount : ℕ}
    (quantizer : Quantizer)
    (state : State parameterCount)
    (proposal : Parameter.Proposal parameterCount)
    (parameter : ParameterIndex parameterCount) : Word :=
  quantizer.roundEven (quantizer.decode (state parameter) + proposal parameter)

/-- Every committed word must remain finite. -/
def Admissible
    {parameterCount : ℕ}
    (quantizer : Quantizer)
    (state : State parameterCount)
    (proposal : Parameter.Proposal parameterCount) : Prop :=
  ∀ parameter, (roundedWord quantizer state proposal parameter).IsFinite

/-- Round a real proposal into an FP16 model state. -/
def commit
    {parameterCount : ℕ}
    (quantizer : Quantizer)
    (state : State parameterCount)
    (proposal : Parameter.Proposal parameterCount)
    (admissible : Admissible quantizer state proposal) :
    State parameterCount :=
  fun parameter =>
    ⟨roundedWord quantizer state proposal parameter, admissible parameter⟩

/-- The FP16 quantizer exposed through the precision-neutral commit boundary. -/
def Quantizer.commitModel
    {parameterCount : ℕ}
    (quantizer : Quantizer) : CommitModel parameterCount where
  State := State parameterCount
  admissible := Admissible quantizer
  commit := commit quantizer

/-- The real proposal survives storage at this parameter. -/
def SurvivesAt
    {parameterCount : ℕ}
    (quantizer : Quantizer)
    (state : State parameterCount)
    (proposal : Parameter.Proposal parameterCount)
    (admissible : Admissible quantizer state proposal)
    (parameter : ParameterIndex parameterCount) : Prop :=
  commit quantizer state proposal admissible parameter ≠ state parameter

/-- The proposal changes at least one stored FP16 parameter. -/
def StoredEffect
    {parameterCount : ℕ}
    (quantizer : Quantizer)
    (state : State parameterCount)
    (proposal : Parameter.Proposal parameterCount)
    (admissible : Admissible quantizer state proposal) : Prop :=
  ∃ parameter, SurvivesAt quantizer state proposal admissible parameter

theorem Commit.zeroValid
    {parameterCount : ℕ}
    (quantizer : Quantizer)
    (state : State parameterCount) :
    Admissible quantizer state 0 := by
  intro parameter
  unfold roundedWord
  change (quantizer.roundEven (quantizer.decode (state parameter) + 0)).IsFinite
  rw [add_zero]
  rw [quantizer.roundTrip]
  exact (state parameter).property

theorem commit_zero
    {parameterCount : ℕ}
    (quantizer : Quantizer)
    (state : State parameterCount) :
    commit quantizer state 0 (Commit.zeroValid quantizer state) = state := by
  funext parameter
  apply Subtype.ext
  simp [commit, roundedWord, quantizer.roundTrip]

theorem StoredEffect.commitNe
    {parameterCount : ℕ}
    (quantizer : Quantizer)
    (state : State parameterCount)
    (proposal : Parameter.Proposal parameterCount)
    (admissible : Admissible quantizer state proposal) :
    StoredEffect quantizer state proposal admissible ↔
      commit quantizer state proposal admissible ≠ state := by
  simp only [StoredEffect, SurvivesAt]
  constructor
  · rintro ⟨parameter, changed⟩ equalStates
    exact changed (congrFun equalStates parameter)
  · intro statesDiffer
    by_contra noChangedParameter
    push Not at noChangedParameter
    apply statesDiffer
    funext parameter
    exact noChangedParameter parameter

/-- Realized parameter displacement after FP16 rounding and decoding. -/
def realizedStep
    {parameterCount : ℕ}
    (quantizer : Quantizer)
    (state : State parameterCount)
    (proposal : Parameter.Proposal parameterCount)
    (admissible : Admissible quantizer state proposal)
    (parameter : ParameterIndex parameterCount) : ℝ :=
  quantizer.decode (commit quantizer state proposal admissible parameter) -
    quantizer.decode (state parameter)

/-- A parameter that does not survive FP16 rounding has zero realized
decoded displacement. -/
theorem deadzoneStep
    {parameterCount : ℕ}
    (quantizer : Quantizer)
    (state : State parameterCount)
    (proposal : Parameter.Proposal parameterCount)
    (admissible : Admissible quantizer state proposal)
    (parameter : ParameterIndex parameterCount)
    (unchanged : ¬ SurvivesAt quantizer state proposal admissible parameter) :
    realizedStep quantizer state proposal admissible parameter = 0 := by
  unfold SurvivesAt at unchanged
  push Not at unchanged
  unfold realizedStep
  rw [unchanged, sub_self]

/-- When no parameter survives FP16 rounding, the entire committed state equals the incumbent. -/
theorem deadzoneAll
    {parameterCount : ℕ}
    (quantizer : Quantizer)
    (state : State parameterCount)
    (proposal : Parameter.Proposal parameterCount)
    (admissible : Admissible quantizer state proposal)
    (allUnchanged : ∀ parameter, ¬ SurvivesAt quantizer state proposal admissible parameter) :
    commit quantizer state proposal admissible = state := by
  funext parameter
  have unchanged := allUnchanged parameter
  unfold SurvivesAt at unchanged
  push Not at unchanged
  exact unchanged

/-- Full real dimension alone does not imply an executed FP16 state change. -/
def Quantization.claim
    {parameterCount : ℕ}
    (quantizer : Quantizer)
    (state : State parameterCount)
    (proposal : Parameter.Proposal parameterCount)
    (admissible : Admissible quantizer state proposal) : ResearchClaim where
  statement := "The proposed real update changes at least one stored FP16 parameter."
  proposition := StoredEffect quantizer state proposal admissible
  status := .conjecture

end BlckbauxResearch.Precision.Float16
