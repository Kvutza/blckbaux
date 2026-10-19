import BlckbauxResearch.Optimization.TransferGuide
import Mathlib.Data.List.MinMax

/-!
# Selecting a retained move

The optimizer evaluates one proposal in a transaction.  A retained record
stores a complete move and a lower bound on the population gain measured at
its source state.  Transport to the current state subtracts separate source
and trial errors.  The exact-real specification takes the record with the
greatest remaining lower bound; finite-precision refinement remains separate.

The results below do not assume that a useful record exists.  They prove the
precise conditional statement needed by the regret argument: if one retained
record clears a required gain, the selected record also clears it and its
proposal realizes that gain at the current state.

Audit, 2026-10-18: a record with valid source gain and transport bounds has
lowerGain <= objective incumbent - objective (apply record.source record.move)
- trialShift record. Therefore, if the incumbent is no worse than that recorded
trial, lowerGain is nonpositive. For a history of accepted moves followed by
population-monotone updates, every record has this obstruction. Increasing the
archive or finding its exact argmax cannot make this particular bound positive.
At a strictly positive eligibility threshold none of those records can qualify;
a zero threshold may admit a zero bound but gives no strict improvement.

This is a limitation of the bound, not a claim that replayed moves cannot help.
For f(x) = (x - 3)^2, the move 0 -> 1 gains 5 and repeating it 1 -> 2 gains 3,
yet the exact separate shifts give lowerGain = 5 - 5 - 3 = -3. The general
real-valued inequality was checked through ./blckbaux research repl and is recorded
in handoff section 68. Existing declarations are unchanged by this audit.
-/

namespace BlckbauxResearch.Optimization.RetainedGuide

open scoped BigOperators

variable {State Move : Type*}

/-- A complete move with its source state and a lower bound on source gain. -/
structure Record (State Move : Type*) where
  source : State
  move : Move
  gain : ℝ

/-- Observable score gap after subtracting separate source and trial errors. -/
def scoreGain (sourceScore trialScore sourceError trialError : ℝ) : ℝ :=
  sourceScore - trialScore - sourceError - trialError

/-- Score accuracy turns the observable score gap into a population-gain lower
bound.  No derivative or unknown optimum occurs in the statement. -/
theorem scoreBound
    (objective : Objective State)
    (source trial : State)
    (sourceScore trialScore sourceError trialError : ℝ)
    (sourceAccurate : |sourceScore - objective source| ≤ sourceError)
    (trialAccurate : |trialScore - objective trial| ≤ trialError) :
    scoreGain sourceScore trialScore sourceError trialError ≤
      progress objective source trial := by
  have sourceLower := (abs_le.mp sourceAccurate).2
  have trialUpper := (abs_le.mp trialAccurate).1
  unfold scoreGain progress
  linarith

/-- Pointwise score bounds establish the source-gain premise for every retained
record when its stored gain is the corresponding corrected score gap. -/
theorem historyBound
    (history : List (Record State Move))
    (objective : Objective State)
    (apply : State → Move → State)
    (sourceScore trialScore sourceError trialError : Record State Move → ℝ)
    (gainEq : ∀ record ∈ history, record.gain =
      scoreGain (sourceScore record) (trialScore record)
        (sourceError record) (trialError record))
    (sourceAccurate : ∀ record ∈ history,
      |sourceScore record - objective record.source| ≤ sourceError record)
    (trialAccurate : ∀ record ∈ history,
      |trialScore record - objective (apply record.source record.move)| ≤
        trialError record) :
    ∀ record ∈ history, record.gain ≤
      progress objective record.source (apply record.source record.move) := by
  intro record member
  rw [gainEq record member]
  exact scoreBound objective record.source (apply record.source record.move)
    (sourceScore record) (trialScore record) (sourceError record)
    (trialError record) (sourceAccurate record member)
    (trialAccurate record member)

/-- Lower gain remaining after transporting a record to the current state. -/
def lowerGain (record : Record State Move)
    (baseShift trialShift : Record State Move → ℝ) : ℝ :=
  record.gain - baseShift record - trialShift record

/-- Choose the earliest retained record attaining the greatest transported
gain lower bound.  Empty history has no selected move. -/
noncomputable def choose (history : List (Record State Move))
    (baseShift trialShift : Record State Move → ℝ) : Option (Record State Move) :=
  List.argmax (fun record => lowerGain record baseShift trialShift) history

/-- Keep the best retained move only when its transported lower bound clears
the supplied eligibility threshold.  The threshold is where the caller accounts
for fresh score error and the acceptance margin. -/
noncomputable def eligible (history : List (Record State Move))
    (baseShift trialShift : Record State Move → ℝ)
    (threshold : ℝ) : Option (Record State Move) :=
  match choose history baseShift trialShift with
  | some record =>
      if threshold ≤ lowerGain record baseShift trialShift then some record else none
  | none => none

/-- Emit one complete move.  A certified retained move is deterministic;
otherwise the round emits the move indexed by its single exploration seed. -/
noncomputable def propose (history : List (Record State Move))
    (baseShift trialShift : Record State Move → ℝ)
    (threshold : ℝ) (explore : Seed → Move) (seed : Seed) : Move :=
  match eligible history baseShift trialShift threshold with
  | some record => record.move
  | none => explore seed

/-- An eligible record was selected by the retained-history argmax. -/
theorem eligibleMem
    (history : List (Record State Move))
    (baseShift trialShift : Record State Move → ℝ)
    (threshold : ℝ) (selected : Record State Move)
    (selectedEq : eligible history baseShift trialShift threshold = some selected) :
    selected ∈ choose history baseShift trialShift := by
  unfold eligible at selectedEq
  generalize choiceEq : choose history baseShift trialShift = choice at selectedEq
  cases choice with
  | none => simp at selectedEq
  | some record =>
      by_cases clears : threshold ≤ lowerGain record baseShift trialShift
      · simp [clears] at selectedEq
        subst selected
        simp
      · simp [clears] at selectedEq

/-- Eligibility exposes the exact transported-gain threshold checked by the
generator; it does not inspect regret or an unknown optimum. -/
theorem eligibleClears
    (history : List (Record State Move))
    (baseShift trialShift : Record State Move → ℝ)
    (threshold : ℝ) (selected : Record State Move)
    (selectedEq : eligible history baseShift trialShift threshold = some selected) :
    threshold ≤ lowerGain selected baseShift trialShift := by
  unfold eligible at selectedEq
  generalize choiceEq : choose history baseShift trialShift = choice at selectedEq
  cases choice with
  | none => simp at selectedEq
  | some record =>
      by_cases clears : threshold ≤ lowerGain record baseShift trialShift
      · simp [clears] at selectedEq
        subst selected
        exact clears
      · simp [clears] at selectedEq

/-- In the retained branch the exploration seed cannot change the proposal. -/
theorem proposeEligible
    (history : List (Record State Move))
    (baseShift trialShift : Record State Move → ℝ)
    (threshold : ℝ) (explore : Seed → Move)
    (selected : Record State Move)
    (selectedEq : eligible history baseShift trialShift threshold = some selected)
    (seed : Seed) :
    propose history baseShift trialShift threshold explore seed = selected.move := by
  simp [propose, selectedEq]

/-- In the fallback branch exactly one seed-indexed exploration move is emitted. -/
theorem proposeExplore
    (history : List (Record State Move))
    (baseShift trialShift : Record State Move → ℝ)
    (threshold : ℝ) (explore : Seed → Move)
    (guideNone : eligible history baseShift trialShift threshold = none)
    (seed : Seed) :
    propose history baseShift trialShift threshold explore seed = explore seed := by
  simp [propose, guideNone]

/-- The selected record belongs to retained history. -/
theorem chooseMem
    (history : List (Record State Move))
    (baseShift trialShift : Record State Move → ℝ)
    (selected : Record State Move)
    (selectedMem : selected ∈ choose history baseShift trialShift) :
    selected ∈ history := by
  exact List.argmax_mem (by simpa [choose] using selectedMem)

/-- Selection preserves every lower bound already present in history. -/
theorem chooseMax
    (history : List (Record State Move))
    (baseShift trialShift : Record State Move → ℝ)
    (candidate selected : Record State Move)
    (candidateMem : candidate ∈ history)
    (selectedMem : selected ∈ choose history baseShift trialShift) :
    lowerGain candidate baseShift trialShift ≤
      lowerGain selected baseShift trialShift := by
  exact List.le_of_mem_argmax candidateMem (by simpa [choose] using selectedMem)

/-- A selected record's transported lower bound is a true gain lower bound at
the current state when the three constituent record bounds are valid. -/
theorem progressBound
    (history : List (Record State Move))
    (baseShift trialShift : Record State Move → ℝ)
    (selected : Record State Move)
    (objective : Objective State)
    (apply : State → Move → State)
    (incumbent : State)
    (sourceGain : ∀ record ∈ history,
      record.gain ≤ progress objective record.source
        (apply record.source record.move))
    (baseBound : ∀ record ∈ history,
      |objective incumbent - objective record.source| ≤ baseShift record)
    (trialBound : ∀ record ∈ history,
      |objective (apply record.source record.move) -
        objective (apply incumbent record.move)| ≤ trialShift record)
    (selectedMem : selected ∈ choose history baseShift trialShift) :
    lowerGain selected baseShift trialShift ≤
      progress objective incumbent (apply incumbent selected.move) := by
  have retained := chooseMem history baseShift trialShift selected selectedMem
  have bound := TransferGuide.progressLocal objective apply selected.source
    incumbent selected.move selected.gain (baseShift selected)
    (trialShift selected) (sourceGain selected retained)
    (baseBound selected retained) (trialBound selected retained)
  simpa [lowerGain] using bound

/-- If any retained record clears a required gain, the selected proposal
clears the same gain at the current state. -/
theorem selectedBound
    (history : List (Record State Move))
    (baseShift trialShift : Record State Move → ℝ)
    (witness selected : Record State Move)
    (objective : Objective State)
    (apply : State → Move → State)
    (incumbent : State)
    (required : ℝ)
    (sourceGain : ∀ record ∈ history,
      record.gain ≤ progress objective record.source
        (apply record.source record.move))
    (baseBound : ∀ record ∈ history,
      |objective incumbent - objective record.source| ≤ baseShift record)
    (trialBound : ∀ record ∈ history,
      |objective (apply record.source record.move) -
        objective (apply incumbent record.move)| ≤ trialShift record)
    (witnessMem : witness ∈ history)
    (selectedMem : selected ∈ choose history baseShift trialShift)
    (useful : required ≤ lowerGain witness baseShift trialShift) :
    required ≤ progress objective incumbent (apply incumbent selected.move) := by
  exact useful.trans ((chooseMax history baseShift trialShift witness selected
    witnessMem selectedMem).trans (progressBound history baseShift trialShift
      selected objective apply incumbent sourceGain baseBound trialBound selectedMem))

/-- The best certified retained move is the single proposal evaluated by the
transaction.  If its transported lower bound also clears the fresh score
errors and acceptance margin, the transaction commits the move and realizes
the required population gain. -/
theorem commitBound
    (history : List (Record State Move))
    (baseShift trialShift : Record State Move → ℝ)
    (witness selected : Record State Move)
    (objective : Objective State)
    (apply : State → Move → State)
    (incumbent : State)
    (scores : ObjectiveSearch.Scores (Fin 1))
    (required baseError candidateError margin : ℝ)
    (sourceGain : ∀ record ∈ history,
      record.gain ≤ progress objective record.source
        (apply record.source record.move))
    (baseBound : ∀ record ∈ history,
      |objective incumbent - objective record.source| ≤ baseShift record)
    (trialBound : ∀ record ∈ history,
      |objective (apply record.source record.move) -
        objective (apply incumbent record.move)| ≤ trialShift record)
    (witnessMem : witness ∈ history)
    (selectedMem : selected ∈ choose history baseShift trialShift)
    (useful : required ≤ lowerGain witness baseShift trialShift)
    (baseAccurate : |scores.incumbent - objective incumbent| ≤ baseError)
    (candidateAccurate :
      |scores.candidate 0 - objective (apply incumbent selected.move)| ≤
        candidateError)
    (separated : baseError + candidateError + margin ≤ required) :
    required ≤ progress objective incumbent
      (ObjectiveSearch.commit incumbent
        (fun _ : Fin 1 => apply incumbent selected.move) scores margin 0) := by
  let candidate : Fin 1 → State := fun _ => apply incumbent selected.move
  have accurate : ObjectiveSearch.ScoreError objective incumbent candidate scores
      baseError candidateError := by
    exact ⟨baseAccurate, fun index => by
      simpa [candidate, Fin.eq_zero index] using candidateAccurate⟩
  have good := selectedBound history baseShift trialShift witness selected
    objective apply incumbent required sourceGain baseBound trialBound witnessMem
    selectedMem useful
  exact ObjectiveSearch.goodCommit objective incumbent candidate scores
    baseError candidateError margin required 0 accurate good separated

/-- A selected retained move gives one deterministic simple-regret recurrence.
The theorem names the precise missing condition: history must contain a move
whose transported, error-corrected gain covers the requested share of current
regret. -/
theorem regretBound
    (history : List (Record State Move))
    (baseShift trialShift : Record State Move → ℝ)
    (witness selected : Record State Move)
    (objective : Objective State)
    (apply : State → Move → State)
    (incumbent best : State)
    (scores : ObjectiveSearch.Scores (Fin 1))
    (required baseError candidateError margin gainShare loss : ℝ)
    (sourceGain : ∀ record ∈ history,
      record.gain ≤ progress objective record.source
        (apply record.source record.move))
    (baseBound : ∀ record ∈ history,
      |objective incumbent - objective record.source| ≤ baseShift record)
    (trialBound : ∀ record ∈ history,
      |objective (apply record.source record.move) -
        objective (apply incumbent record.move)| ≤ trialShift record)
    (witnessMem : witness ∈ history)
    (selectedMem : selected ∈ choose history baseShift trialShift)
    (useful : required ≤ lowerGain witness baseShift trialShift)
    (baseAccurate : |scores.incumbent - objective incumbent| ≤ baseError)
    (candidateAccurate :
      |scores.candidate 0 - objective (apply incumbent selected.move)| ≤
        candidateError)
    (separated : baseError + candidateError + margin ≤ required)
    (retained : gainShare * Regret.simpleRegret objective best incumbent - loss ≤
      required) :
    Regret.simpleRegret objective best
        (ObjectiveSearch.commit incumbent
          (fun _ : Fin 1 => apply incumbent selected.move) scores margin 0) ≤
      (1 - gainShare) * Regret.simpleRegret objective best incumbent + loss := by
  have progressLower := commitBound history baseShift trialShift witness selected
    objective apply incumbent scores required baseError candidateError margin
    sourceGain baseBound trialBound witnessMem selectedMem useful baseAccurate
    candidateAccurate separated
  simp only [Regret.simpleRegret, progress] at progressLower retained ⊢
  linarith

/-- When the retained gate succeeds, every seed denotes the same single
transaction.  The conditional mean therefore pays no exploration-mass factor. -/
theorem eligibleRegret [Fintype Seed]
    (law : ObjectiveSearch.ProposalLaw Seed)
    (history : List (Record State Move))
    (baseShift trialShift : Record State Move → ℝ)
    (selected : Record State Move)
    (objective : Objective State)
    (apply : State → Move → State)
    (incumbent best : State)
    (scores : ObjectiveSearch.Scores (Fin 1))
    (threshold baseError candidateError margin gainShare loss : ℝ)
    (explore : Seed → Move)
    (sourceGain : ∀ record ∈ history,
      record.gain ≤ progress objective record.source
        (apply record.source record.move))
    (baseBound : ∀ record ∈ history,
      |objective incumbent - objective record.source| ≤ baseShift record)
    (trialBound : ∀ record ∈ history,
      |objective (apply record.source record.move) -
        objective (apply incumbent record.move)| ≤ trialShift record)
    (selectedEq : eligible history baseShift trialShift threshold = some selected)
    (baseAccurate : |scores.incumbent - objective incumbent| ≤ baseError)
    (candidateAccurate :
      |scores.candidate 0 - objective (apply incumbent selected.move)| ≤
        candidateError)
    (separated : baseError + candidateError + margin ≤ threshold)
    (retained : gainShare * Regret.simpleRegret objective best incumbent - loss ≤
      threshold) :
    (∑ seed, law.weight seed * Regret.simpleRegret objective best
      (ObjectiveSearch.commit incumbent
        (fun _ : Fin 1 => apply incumbent
          (propose history baseShift trialShift threshold explore seed))
        scores margin 0)) ≤
      (1 - gainShare) * Regret.simpleRegret objective best incumbent + loss := by
  have selectedMem := eligibleMem history baseShift trialShift threshold selected selectedEq
  have retainedMem := chooseMem history baseShift trialShift selected selectedMem
  have clears := eligibleClears history baseShift trialShift threshold selected selectedEq
  have oneStep := regretBound history baseShift trialShift selected selected objective apply
    incumbent best scores threshold baseError candidateError margin gainShare loss
    sourceGain baseBound trialBound retainedMem selectedMem clears baseAccurate
    candidateAccurate separated retained
  calc
    (∑ seed, law.weight seed * Regret.simpleRegret objective best
        (ObjectiveSearch.commit incumbent
          (fun _ : Fin 1 => apply incumbent
            (propose history baseShift trialShift threshold explore seed))
          scores margin 0)) =
        (∑ seed, law.weight seed) * Regret.simpleRegret objective best
          (ObjectiveSearch.commit incumbent
            (fun _ : Fin 1 => apply incumbent selected.move) scores margin 0) := by
      rw [Finset.sum_mul]
      apply Finset.sum_congr rfl
      intro seed _
      rw [proposeEligible history baseShift trialShift threshold explore selected
        selectedEq seed]
    _ = Regret.simpleRegret objective best
          (ObjectiveSearch.commit incumbent
            (fun _ : Fin 1 => apply incumbent selected.move) scores margin 0) := by
      rw [law.total, one_mul]
    _ ≤ (1 - gainShare) * Regret.simpleRegret objective best incumbent + loss := oneStep

/-- When no retained move clears the gate, the emitted proposal follows the
exploration law.  One safe exploration atom with useful gain gives the exact
one-round expected-regret factor; no pool of objective evaluations is used. -/
theorem exploreRegret [Fintype Seed]
    (law : ObjectiveSearch.ProposalLaw Seed)
    (history : List (Record State Move))
    (baseShift trialShift : Record State Move → ℝ)
    (threshold : ℝ) (explore : Seed → Move)
    (guideNone : eligible history baseShift trialShift threshold = none)
    (objective : Objective State) (incumbent best : State)
    (commit : Move → State) (guide : Seed)
    (gain gainShare loss : ℝ)
    (safe : ∀ seed, 0 ≤ progress objective incumbent (commit (explore seed)))
    (guideGain : gain ≤ progress objective incumbent (commit (explore guide)))
    (gainBound : gainShare * Regret.simpleRegret objective best incumbent - loss ≤ gain) :
    (∑ seed, law.weight seed * Regret.simpleRegret objective best
      (commit (propose history baseShift trialShift threshold explore seed))) ≤
        (1 - law.weight guide * gainShare) *
          Regret.simpleRegret objective best incumbent + law.weight guide * loss := by
  classical
  simp_rw [proposeExplore history baseShift trialShift threshold explore guideNone]
  have gainLower : law.weight guide * gain ≤
      ∑ seed, law.weight seed * progress objective incumbent (commit (explore seed)) := by
    calc
      law.weight guide * gain ≤
          law.weight guide * progress objective incumbent (commit (explore guide)) :=
        mul_le_mul_of_nonneg_left guideGain (law.nonnegative guide)
      _ ≤ ∑ seed, law.weight seed *
          progress objective incumbent (commit (explore seed)) :=
        Finset.single_le_sum
          (fun seed _ => mul_nonneg (law.nonnegative seed) (safe seed))
          (Finset.mem_univ guide)
  have split :
      (∑ seed, law.weight seed * Regret.simpleRegret objective best
          (commit (explore seed))) +
        (∑ seed, law.weight seed * progress objective incumbent
          (commit (explore seed))) =
        Regret.simpleRegret objective best incumbent := by
    rw [← Finset.sum_add_distrib]
    calc
      _ = ∑ seed, law.weight seed *
          Regret.simpleRegret objective best incumbent := by
        apply Finset.sum_congr rfl
        intro seed _
        unfold Regret.simpleRegret progress
        ring
      _ = Regret.simpleRegret objective best incumbent := by
        rw [← Finset.sum_mul, law.total, one_mul]
  calc
    (∑ seed, law.weight seed * Regret.simpleRegret objective best
        (commit (explore seed))) =
        Regret.simpleRegret objective best incumbent -
          (∑ seed, law.weight seed * progress objective incumbent
            (commit (explore seed))) := by
      linarith
    _ ≤ Regret.simpleRegret objective best incumbent - law.weight guide * gain :=
      sub_le_sub_left gainLower _
    _ ≤ Regret.simpleRegret objective best incumbent - law.weight guide *
        (gainShare * Regret.simpleRegret objective best incumbent - loss) :=
      sub_le_sub_left
        (mul_le_mul_of_nonneg_left gainBound (law.nonnegative guide)) _
    _ = (1 - law.weight guide * gainShare) *
          Regret.simpleRegret objective best incumbent + law.weight guide * loss := by
      ring

end BlckbauxResearch.Optimization.RetainedGuide
