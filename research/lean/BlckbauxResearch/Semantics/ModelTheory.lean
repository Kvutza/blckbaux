import Mathlib.ModelTheory.Satisfiability
import BlckbauxResearch.Optimization.ResearchProblem

/-!
# A first-order boundary for optimizer claims

This is mathematical model theory, not a neural-network model, surrogate model,
or model architecture.  The domain elements are complete optimizer trials.  A
structure interprets named relations between trials, while `beatTheory` fixes
what every structure must establish before it may label a trial as beating its
Muon reference.

The numerical and probabilistic proofs remain in their native Mathlib theories.
They feed facts into an interpretation of this language.  First-order semantics
then lets us ask whether a claimed consequence holds in every admissible
interpretation or whether a countermodel still exists.
-/

namespace BlckbauxResearch.Semantics

open FirstOrder
open FirstOrder.Language
open FirstOrder.Language.Structure
open BlckbauxResearch.Optimization.ResearchProblem

/-- Relation symbols for complete optimizer trials. -/
inductive TrialRel : ℕ → Type
  | muonBeat : TrialRel 2
  | outcomeBeat : TrialRel 2
  | computeBeat : TrialRel 2
  | targetFit : TrialRel 1
  | ceilingFit : TrialRel 1
  | scopeFit : TrialRel 1
  deriving DecidableEq

/-- The relational first-order language used for cross-implementation claims. -/
def trialLang : FirstOrder.Language :=
  ⟨fun _ ↦ Empty, TrialRel⟩
  deriving IsRelational

abbrev muonBeat : trialLang.Relations 2 := .muonBeat
abbrev outcomeBeat : trialLang.Relations 2 := .outcomeBeat
abbrev computeBeat : trialLang.Relations 2 := .computeBeat
abbrev targetFit : trialLang.Relations 1 := .targetFit
abbrev ceilingFit : trialLang.Relations 1 := .ceilingFit
abbrev scopeFit : trialLang.Relations 1 := .scopeFit

/-- Universal implication between two binary trial relations. -/
def pairImplies
    (left right : trialLang.Relations 2) : trialLang.Sentence :=
  ∀' ∀' (left.boundedFormula₂ (&0) &1 ⟹ right.boundedFormula₂ (&0) &1)

/-- A binary claim about a candidate forces a unary property of that candidate. -/
def leftImplies
    (pair : trialLang.Relations 2)
    (property : trialLang.Relations 1) : trialLang.Sentence :=
  ∀' ∀' (pair.boundedFormula₂ (&0) &1 ⟹ property.boundedFormula₁ (&0))

/-- Universal implication between two unary trial properties. -/
def itemImplies
    (left right : trialLang.Relations 1) : trialLang.Sentence :=
  ∀' (left.boundedFormula₁ (&0) ⟹ right.boundedFormula₁ (&0))

/-- Semantic form of one binary-relation implication. -/
def PairRule
    {Trial : Type*}
    [trialLang.Structure Trial]
    (left right : trialLang.Relations 2) : Prop :=
  ∀ candidate reference : Trial,
    RelMap left ![candidate, reference] →
      RelMap right ![candidate, reference]

/-- Semantic form of a binary claim forcing a property of its left argument. -/
def LeftRule
    {Trial : Type*}
    [trialLang.Structure Trial]
    (pair : trialLang.Relations 2)
    (property : trialLang.Relations 1) : Prop :=
  ∀ candidate reference : Trial,
    RelMap pair ![candidate, reference] → RelMap property ![candidate]

/-- Semantic form of one unary-property implication. -/
def ItemRule
    {Trial : Type*}
    [trialLang.Structure Trial]
    (left right : trialLang.Relations 1) : Prop :=
  ∀ candidate : Trial, RelMap left ![candidate] → RelMap right ![candidate]

@[simp]
theorem pairRealize
    {Trial : Type*}
    [trialLang.Structure Trial]
    (left right : trialLang.Relations 2) :
    Trial ⊨ pairImplies left right ↔ PairRule (Trial := Trial) left right := by
  rw [pairImplies, FirstOrder.Language.Sentence.Realize,
    FirstOrder.Language.Formula.Realize]
  exact forall₂_congr fun _ _ ↦
    imp_congr BoundedFormula.realize_rel₂ BoundedFormula.realize_rel₂

@[simp]
theorem leftRealize
    {Trial : Type*}
    [trialLang.Structure Trial]
    (pair : trialLang.Relations 2)
    (property : trialLang.Relations 1) :
    Trial ⊨ leftImplies pair property ↔
      LeftRule (Trial := Trial) pair property := by
  rw [leftImplies, FirstOrder.Language.Sentence.Realize,
    FirstOrder.Language.Formula.Realize]
  exact forall₂_congr fun _ _ ↦
    imp_congr BoundedFormula.realize_rel₂ BoundedFormula.realize_rel₁

@[simp]
theorem itemRealize
    {Trial : Type*}
    [trialLang.Structure Trial]
    (left right : trialLang.Relations 1) :
    Trial ⊨ itemImplies left right ↔ ItemRule (Trial := Trial) left right := by
  rw [itemImplies, FirstOrder.Language.Sentence.Realize,
    FirstOrder.Language.Formula.Realize]
  exact forall_congr' fun _ ↦
    imp_congr BoundedFormula.realize_rel₁ BoundedFormula.realize_rel₁

/--
The first-order contract for the phrase "beats Muon" in this research program.
It requires outcome and compute dominance, the 200 ms target, and the explicitly
declared proposal-scope predicate.  Meeting the target also entails the weaker
1000 ms ceiling.
-/
def beatTheory : trialLang.Theory :=
  { pairImplies muonBeat outcomeBeat,
    pairImplies muonBeat computeBeat,
    leftImplies muonBeat targetFit,
    leftImplies muonBeat scopeFit,
    itemImplies targetFit ceilingFit }

/-- Satisfaction of `beatTheory` exposes exactly its five semantic rules. -/
theorem beatModel
    {Trial : Type*}
    [trialLang.Structure Trial] :
    Trial ⊨ beatTheory ↔
      PairRule (Trial := Trial) muonBeat outcomeBeat ∧
      PairRule (Trial := Trial) muonBeat computeBeat ∧
      LeftRule (Trial := Trial) muonBeat targetFit ∧
      LeftRule (Trial := Trial) muonBeat scopeFit ∧
      ItemRule (Trial := Trial) targetFit ceilingFit := by
  simp [beatTheory, FirstOrder.Language.Theory.model_iff]

/-- A concrete interpretation of the optimizer-trial relation symbols. -/
structure TrialWorld (Trial : Type*) where
  muonBeat : Trial → Trial → Prop
  outcomeBeat : Trial → Trial → Prop
  computeBeat : Trial → Trial → Prop
  targetFit : Trial → Prop
  ceilingFit : Trial → Prop
  scopeFit : Trial → Prop

/-- Interpret the first-order language in a concrete family of trials. -/
@[instance_reducible]
def TrialWorld.interpret
    {Trial : Type*}
    (world : TrialWorld Trial) : trialLang.Structure Trial where
  RelMap
    | .muonBeat => fun trials ↦ world.muonBeat (trials 0) (trials 1)
    | .outcomeBeat => fun trials ↦ world.outcomeBeat (trials 0) (trials 1)
    | .computeBeat => fun trials ↦ world.computeBeat (trials 0) (trials 1)
    | .targetFit => fun trials ↦ world.targetFit (trials 0)
    | .ceilingFit => fun trials ↦ world.ceilingFit (trials 0)
    | .scopeFit => fun trials ↦ world.scopeFit (trials 0)

/--
Interpret an accounted optimizer run as a first-order trial.  The caller must
supply the exact evidence predicate for the declared proposal scope; the word
"full" is deliberately not assigned to an unspecified reachability property.
-/
def runWorld
    {State : Type*}
    {construction : ObjectiveConstruction State}
    {world : construction.World}
    {machine : Machine}
    (scopeHolds : RunResult construction world machine → Prop) :
    TrialWorld (RunResult construction world machine) where
  muonBeat candidate reference :=
    Run.sameStart candidate reference ∧
      Run.noWorse candidate reference ∧
      Resource.le candidate.usage reference.usage ∧
      candidate.usage .wallTimeMs ≤ currentTarget.targetMs ∧
      scopeHolds candidate
  outcomeBeat := Run.noWorse
  computeBeat candidate reference :=
    Resource.le candidate.usage reference.usage
  targetFit candidate :=
    candidate.usage .wallTimeMs ≤ currentTarget.targetMs
  ceilingFit candidate :=
    candidate.usage .wallTimeMs ≤ currentTarget.ceilingMs
  scopeFit := scopeHolds

/-- The concrete run interpretation satisfies the first-order beat contract. -/
theorem runModel
    {State : Type*}
    {construction : ObjectiveConstruction State}
    {world : construction.World}
    {machine : Machine}
    (scopeHolds : RunResult construction world machine → Prop) :
    @FirstOrder.Language.Theory.Model
      trialLang
      (RunResult construction world machine)
      (runWorld scopeHolds).interpret
      beatTheory := by
  letI := (runWorld scopeHolds).interpret
  apply beatModel.mpr
  change
    (∀ candidate reference,
      (Run.sameStart candidate reference ∧
        Run.noWorse candidate reference ∧
        Resource.le candidate.usage reference.usage ∧
        candidate.usage .wallTimeMs ≤ currentTarget.targetMs ∧
        scopeHolds candidate) →
      Run.noWorse candidate reference) ∧
    (∀ candidate reference,
      (Run.sameStart candidate reference ∧
        Run.noWorse candidate reference ∧
        Resource.le candidate.usage reference.usage ∧
        candidate.usage .wallTimeMs ≤ currentTarget.targetMs ∧
        scopeHolds candidate) →
      Resource.le candidate.usage reference.usage) ∧
    (∀ candidate reference,
      (Run.sameStart candidate reference ∧
        Run.noWorse candidate reference ∧
        Resource.le candidate.usage reference.usage ∧
        candidate.usage .wallTimeMs ≤ currentTarget.targetMs ∧
        scopeHolds candidate) →
      candidate.usage .wallTimeMs ≤ currentTarget.targetMs) ∧
    (∀ candidate reference,
      (Run.sameStart candidate reference ∧
        Run.noWorse candidate reference ∧
        Resource.le candidate.usage reference.usage ∧
        candidate.usage .wallTimeMs ≤ currentTarget.targetMs ∧
        scopeHolds candidate) →
      scopeHolds candidate) ∧
    (∀ candidate : RunResult construction world machine,
      candidate.usage .wallTimeMs ≤ currentTarget.targetMs →
      candidate.usage .wallTimeMs ≤ currentTarget.ceilingMs)
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro _ _ claim
    exact claim.2.1
  · intro _ _ claim
    exact claim.2.2.1
  · intro _ _ claim
    exact claim.2.2.2.1
  · intro _ _ claim
    exact claim.2.2.2.2
  · intro _ withinTarget
    exact withinTarget.trans (by norm_num [currentTarget])

end BlckbauxResearch.Semantics
