/-!
This module records authors, evidence kinds, and dialogue moves.  Its objects
are provenance-bearing protocol data, not optimizer quantities.  Replacing
their names with mathematical glyphs would hide distinctions that the protocol
exists to preserve.
-/

namespace BlckbauxResearch

abbrev MoveId := Nat

inductive Author where
  | first
  | second
  deriving DecidableEq, Repr

inductive EvidenceKind where
  | assumption
  | derivation
  | leanProof
  | codeFact
  | measurement
  | externalSource
  | conjecture
  deriving DecidableEq, Repr

structure Evidence where
  kind : EvidenceKind
  locator : String
  deriving DecidableEq, Repr

inductive Verdict where
  | accepted
  | rejected
  | parked
  deriving DecidableEq, Repr

inductive MoveKind where
  | question
  | proposal
  | paraphrase
  | confirm
  | correct
  | support
  | challenge
  | counterexample
  | prediction
  | test
  | observation
  | revision
  | resolve (verdict : Verdict)
  deriving DecidableEq, Repr

structure Move where
  id : MoveId
  author : Author
  kind : MoveKind
  target : Option MoveId
  statement : String
  evidence : Option Evidence := none
  settlingCondition : Option String := none
  deriving DecidableEq, Repr

structure DraftMove where
  author : Author
  kind : MoveKind
  target : Option MoveId
  statement : String
  evidence : Option Evidence := none
  settlingCondition : Option String := none
  deriving DecidableEq, Repr

inductive ProtocolError where
  | emptyStatement
  | missingTarget
  | unknownTarget (target : MoveId)
  | targetDoesNotPrecede (target current : MoveId)
  | invalidParent (child parent : MoveKind)
  | sameAuthorParaphrase
  | invalidConfirmationAuthor
  | unconfirmedClaim (target : MoveId)
  | missingEvidence
  | observationIsNotMeasurement
  | missingSettlingCondition
  deriving DecidableEq, Repr

structure Dialogue where
  private mk ::
  moves : List Move := []
  deriving DecidableEq, Repr

def Dialogue.empty : Dialogue := ⟨[]⟩

def Dialogue.nextId (dialogue : Dialogue) : MoveId := dialogue.moves.length

def Dialogue.find? (dialogue : Dialogue) (id : MoveId) : Option Move :=
  dialogue.moves.find? fun move => move.id == id

def MoveKind.requiresTarget : MoveKind → Bool
  | .question => false
  | _ => true

def MoveKind.requiresEvidence : MoveKind → Bool
  | .support | .challenge | .counterexample | .observation => true
  | _ => false

def MoveKind.needsConfirmation : MoveKind → Bool
  | .support | .challenge | .counterexample | .prediction | .revision | .resolve _ => true
  | _ => false

def allowedParent : MoveKind → MoveKind → Bool
  | .question, _ => true
  | .proposal, .question => true
  | .paraphrase, .question | .paraphrase, .proposal | .paraphrase, .revision => true
  | .confirm, .paraphrase | .correct, .paraphrase => true
  | .support, .proposal | .support, .revision | .support, .prediction => true
  | .challenge, .proposal | .challenge, .revision | .challenge, .prediction => true
  | .counterexample, .proposal | .counterexample, .revision | .counterexample, .prediction => true
  | .prediction, .proposal | .prediction, .revision => true
  | .test, .prediction => true
  | .observation, .test => true
  | .revision, .proposal | .revision, .revision => true
  | .resolve _, .question | .resolve _, .proposal | .resolve _, .revision => true
  | _, _ => false

def Dialogue.meaningConfirmed (dialogue : Dialogue) (claim : MoveId) : Bool :=
  dialogue.moves.any fun paraphrase =>
    paraphrase.kind == .paraphrase && paraphrase.target == some claim &&
      dialogue.moves.any fun confirmation =>
        confirmation.kind == .confirm && confirmation.target == some paraphrase.id

private def requireSome (value : Option α) (error : ProtocolError) :
    Except ProtocolError α :=
  match value with
  | some item => .ok item
  | none => .error error

private def validateTarget
    (dialogue : Dialogue) (id : MoveId) (kind : MoveKind) (target : Option MoveId) :
    Except ProtocolError (Option Move) := do
  match target with
  | some targetId =>
    if ¬targetId < id then
      throw (.targetDoesNotPrecede targetId id)
    let parent ← requireSome (dialogue.find? targetId) (.unknownTarget targetId)
    if ¬allowedParent kind parent.kind then
      throw (.invalidParent kind parent.kind)
    pure (some parent)
  | none =>
    if kind.requiresTarget then
      throw .missingTarget
    pure none

private def validateHandshake
    (dialogue : Dialogue) (draft : DraftMove) (parent : Option Move) :
    Except ProtocolError Unit := do
  match draft.kind, parent with
  | .paraphrase, some original =>
      if draft.author == original.author then
        throw .sameAuthorParaphrase
  | .confirm, some paraphrase =>
      let originalId ← requireSome paraphrase.target .missingTarget
      let original ← requireSome (dialogue.find? originalId) (.unknownTarget originalId)
      if draft.author != original.author then
        throw .invalidConfirmationAuthor
  | kind, some target =>
      if kind.needsConfirmation && ¬dialogue.meaningConfirmed target.id then
        throw (.unconfirmedClaim target.id)
  | _, _ => pure ()

private def validateEvidence (draft : DraftMove) : Except ProtocolError Unit := do
  if draft.kind.requiresEvidence && draft.evidence.isNone then
    throw .missingEvidence
  if draft.kind == .observation then
    match draft.evidence with
    | some evidence =>
        if evidence.kind != .measurement then
          throw .observationIsNotMeasurement
    | none => throw .missingEvidence
  if draft.kind == .prediction && draft.settlingCondition.isNone then
    throw .missingSettlingCondition

def Dialogue.append (dialogue : Dialogue) (draft : DraftMove) :
    Except ProtocolError Dialogue := do
  if draft.statement.isEmpty then
    throw .emptyStatement
  let id := dialogue.nextId
  let parent ← validateTarget dialogue id draft.kind draft.target
  validateHandshake dialogue draft parent
  validateEvidence draft
  let move : Move := {
    id
    author := draft.author
    kind := draft.kind
    target := draft.target
    statement := draft.statement
    evidence := draft.evidence
    settlingCondition := draft.settlingCondition
  }
  pure ⟨dialogue.moves ++ [move]⟩

def Move.targetPrecedes (move : Move) : Bool :=
  match move.target with
  | none => true
  | some target => decide (target < move.id)

def Dialogue.targetsPrecede (dialogue : Dialogue) : Bool :=
  dialogue.moves.all Move.targetPrecedes

theorem Move.targetLt
    (move : Move) (target : MoveId)
    (checked : move.targetPrecedes = true)
    (isTarget : move.target = some target) : target < move.id := by
  unfold Move.targetPrecedes at checked
  rw [isTarget] at checked
  exact of_decide_eq_true checked

end BlckbauxResearch
