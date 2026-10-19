import BlckbauxResearch.Precision.Commit

/-!
# BF16 parameter-state semantics

This module is a BF16 alternative to the real-vector model. It keeps those
state spaces separate and states the finite-BF16 invariant explicitly. The
current full-parameter Metal pretraining path instead stores FP16 words.

`BFloat16Quantizer` is a contract, not an implementation claim. A matching BF16
backend must separately establish that its conversion implements the contract
it supplies here. The current Metal and recorded CUDA FP16 paths use a different
format and are not instances of this contract.
-/

namespace BlckbauxResearch.Precision

/-- One raw BF16 word, including finite values, infinities, and NaNs. -/
structure BFloat16Word where
  bits : Fin 65_536
  deriving DecidableEq, Repr

/-- Bits 7 through 14 are the eight-bit BF16 exponent field. -/
def BFloat16Word.exponentField (word : BFloat16Word) : ℕ :=
  (word.bits.val / 128) % 256

/-- A BF16 word is finite exactly when its exponent field is not all ones. -/
def BFloat16Word.IsFinite (word : BFloat16Word) : Prop :=
  word.exponentField ≠ 255

instance (word : BFloat16Word) : Decidable word.IsFinite :=
  inferInstanceAs (Decidable (word.exponentField ≠ 255))

/-- A stored optimizer parameter satisfying the finite-value invariant. -/
abbrev FiniteBFloat16 := { word : BFloat16Word // word.IsFinite }

/-- Executed parameter storage has one finite BF16 word per learned parameter. -/
abbrev StoredState (parameterCount : ℕ) :=
  ParameterIndex parameterCount → FiniteBFloat16

/--
The semantics required from the concrete BF16 conversion used by a backend.

The round-trip law is the first backend parity obligation: committing a zero
real update must preserve every finite stored word exactly.
-/
structure BFloat16Quantizer where
  decodeFinite : FiniteBFloat16 → ℝ
  roundToNearestEven : ℝ → BFloat16Word
  roundTrip : ∀ value, roundToNearestEven (decodeFinite value) = value.1

/-- The raw word produced for one proposed parameter update. -/
def roundedWord
    {parameterCount : ℕ}
    (quantizer : BFloat16Quantizer)
    (state : StoredState parameterCount)
    (proposal : Parameter.Proposal parameterCount)
    (parameter : ParameterIndex parameterCount) : BFloat16Word :=
  quantizer.roundToNearestEven
    (quantizer.decodeFinite (state parameter) + proposal parameter)

/-- A proposal is admissible for model storage when every rounded word is finite. -/
def CommitAdmissible
    {parameterCount : ℕ}
    (quantizer : BFloat16Quantizer)
    (state : StoredState parameterCount)
    (proposal : Parameter.Proposal parameterCount) : Prop :=
  ∀ parameter, (roundedWord quantizer state proposal parameter).IsFinite

/-- The real update is rounded only when it is committed to model storage. -/
def commitProposal
    {parameterCount : ℕ}
    (quantizer : BFloat16Quantizer)
    (state : StoredState parameterCount)
    (proposal : Parameter.Proposal parameterCount)
    (admissible : CommitAdmissible quantizer state proposal) :
    StoredState parameterCount :=
  fun parameter =>
    ⟨roundedWord quantizer state proposal parameter, admissible parameter⟩

/-- The BF16 quantizer exposed through the precision-neutral commit boundary. -/
def BFloat16Quantizer.commitModel
    {parameterCount : ℕ}
    (quantizer : BFloat16Quantizer) : CommitModel parameterCount where
  State := StoredState parameterCount
  admissible := CommitAdmissible quantizer
  commit := commitProposal quantizer

/-- A proposal survives BF16 storage at an index when its stored word changes. -/
def SurvivesAt
    {parameterCount : ℕ}
    (quantizer : BFloat16Quantizer)
    (state : StoredState parameterCount)
    (proposal : Parameter.Proposal parameterCount)
    (admissible : CommitAdmissible quantizer state proposal)
    (parameter : ParameterIndex parameterCount) : Prop :=
  commitProposal quantizer state proposal admissible parameter ≠ state parameter

/-- A proposal has an executed effect when it changes at least one BF16 word. -/
def StoredEffect
    {parameterCount : ℕ}
    (quantizer : BFloat16Quantizer)
    (state : StoredState parameterCount)
    (proposal : Parameter.Proposal parameterCount)
    (admissible : CommitAdmissible quantizer state proposal) : Prop :=
  ∃ parameter, SurvivesAt quantizer state proposal admissible parameter

theorem Commit.zeroValid
    {parameterCount : ℕ}
    (quantizer : BFloat16Quantizer)
    (state : StoredState parameterCount) :
    CommitAdmissible quantizer state 0 := by
  intro parameter
  unfold roundedWord
  change
    (quantizer.roundToNearestEven
      (quantizer.decodeFinite (state parameter) + 0)).IsFinite
  rw [add_zero]
  rw [quantizer.roundTrip]
  exact (state parameter).property

theorem commit_zero
    {parameterCount : ℕ}
    (quantizer : BFloat16Quantizer)
    (state : StoredState parameterCount) :
    commitProposal quantizer state 0 (Commit.zeroValid quantizer state) = state := by
  funext parameter
  apply Subtype.ext
  simp [commitProposal, roundedWord, quantizer.roundTrip]

theorem StoredEffect.commitNe
    {parameterCount : ℕ}
    (quantizer : BFloat16Quantizer)
    (state : StoredState parameterCount)
    (proposal : Parameter.Proposal parameterCount)
    (admissible : CommitAdmissible quantizer state proposal) :
    StoredEffect quantizer state proposal admissible ↔
      commitProposal quantizer state proposal admissible ≠ state := by
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

/-- Realized parameter displacement after BF16 rounding and decoding. -/
def realizedStep
    {parameterCount : ℕ}
    (quantizer : BFloat16Quantizer)
    (state : StoredState parameterCount)
    (proposal : Parameter.Proposal parameterCount)
    (admissible : CommitAdmissible quantizer state proposal)
    (parameter : ParameterIndex parameterCount) : ℝ :=
  quantizer.decodeFinite (commitProposal quantizer state proposal admissible parameter) -
    quantizer.decodeFinite (state parameter)

/-- A parameter that does not survive BF16 rounding has zero realized
decoded displacement. -/
theorem deadzoneStep
    {parameterCount : ℕ}
    (quantizer : BFloat16Quantizer)
    (state : StoredState parameterCount)
    (proposal : Parameter.Proposal parameterCount)
    (admissible : CommitAdmissible quantizer state proposal)
    (parameter : ParameterIndex parameterCount)
    (unchanged : ¬ SurvivesAt quantizer state proposal admissible parameter) :
    realizedStep quantizer state proposal admissible parameter = 0 := by
  unfold SurvivesAt at unchanged
  push Not at unchanged
  unfold realizedStep
  rw [unchanged, sub_self]

/-- When no parameter survives BF16 rounding, the entire committed state equals the incumbent. -/
theorem deadzoneAll
    {parameterCount : ℕ}
    (quantizer : BFloat16Quantizer)
    (state : StoredState parameterCount)
    (proposal : Parameter.Proposal parameterCount)
    (admissible : CommitAdmissible quantizer state proposal)
    (allUnchanged : ∀ parameter, ¬ SurvivesAt quantizer state proposal admissible parameter) :
    commitProposal quantizer state proposal admissible = state := by
  funext parameter
  have unchanged := allUnchanged parameter
  unfold SurvivesAt at unchanged
  push Not at unchanged
  exact unchanged

/--
Full real dimension alone does not imply an executed BF16 change.  That bridge
is precisely the `StoredEffect` obligation for the chosen proposal rule.
-/
def Quantization.claim
    {parameterCount : ℕ}
    (quantizer : BFloat16Quantizer)
    (state : StoredState parameterCount)
    (proposal : Parameter.Proposal parameterCount)
    (admissible : CommitAdmissible quantizer state proposal) : ResearchClaim where
  statement := "The proposed real update changes at least one stored BF16 parameter."
  proposition := StoredEffect quantizer state proposal admissible
  status := .conjecture

end BlckbauxResearch.Precision
