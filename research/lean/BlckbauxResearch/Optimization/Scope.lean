import BlckbauxResearch.Optimization.Synthesis

/-!
# Bounded theorem synthesis

`Scope` is a finite, executable search whose successful checks imply a stated
mathematical goal. An empty ordinary scope means only that this checker found
nothing. `ExactScope` additionally proves completeness, so an empty exact scope
proves that no candidate in the declared finite grammar satisfies the goal.
This keeps verified discovery, bounded refutation, and unresolved search apart.
-/

namespace BlckbauxResearch.Optimization.Synthesis

variable {Candidate : Type*}

/-- A finite executable checker with a proof that every accepted candidate
satisfies the research goal. -/
structure Scope (Goal : Candidate → Prop) where
  candidates : Finset Candidate
  check : Candidate → Bool
  sound : ∀ candidate, candidate ∈ candidates → check candidate = true → Goal candidate

/-- Candidates accepted by the executable checker. -/
def Scope.pool {Goal : Candidate → Prop} (scope : Scope Goal) : Finset Candidate :=
  scope.candidates.filter fun candidate ↦ scope.check candidate

/-- Membership in the executable pool is exactly successful checking. -/
theorem Scope.memPool {Goal : Candidate → Prop} (scope : Scope Goal) (candidate : Candidate) :
    candidate ∈ scope.pool ↔ candidate ∈ scope.candidates ∧ scope.check candidate = true := by
  simp [Scope.pool]

/-- Any candidate found by a sound checker carries the requested theorem. -/
theorem Scope.found {Goal : Candidate → Prop} (scope : Scope Goal)
    (nonempty : scope.pool.Nonempty) : ∃ candidate, Goal candidate := by
  obtain ⟨candidate, member⟩ := nonempty
  have accepted := (scope.memPool candidate).mp member
  exact ⟨candidate, scope.sound candidate accepted.1 accepted.2⟩

/-- An exact scope also proves that the executable checker recognizes every
candidate satisfying the goal. -/
structure ExactScope (Goal : Candidate → Prop) extends Scope Goal where
  complete : ∀ candidate, candidate ∈ candidates → Goal candidate → check candidate = true

/-- Exhausting an exact finite scope is a bounded nonexistence theorem. -/
theorem ExactScope.exhausted {Goal : Candidate → Prop} (scope : ExactScope Goal)
    (empty : scope.toScope.pool = ∅) :
    ∀ candidate, candidate ∈ scope.candidates → ¬Goal candidate := by
  intro candidate declared goal
  have checked : scope.check candidate = true := scope.complete candidate declared goal
  have member : candidate ∈ scope.toScope.pool :=
    (scope.toScope.memPool candidate).mpr ⟨declared, checked⟩
  simp [empty] at member

/-- Exhausting a merely sound checker reports absence of a checked witness; it
is deliberately weaker than `ExactScope.exhausted`. -/
theorem Scope.empty {Goal : Candidate → Prop} (scope : Scope Goal)
    (empty : scope.pool = ∅) :
    ∀ candidate, candidate ∈ scope.candidates → scope.check candidate = false := by
  intro candidate declared
  apply Bool.eq_false_iff.mpr
  intro checked
  have member : candidate ∈ scope.pool := (scope.memPool candidate).mpr ⟨declared, checked⟩
  simp [empty] at member

end BlckbauxResearch.Optimization.Synthesis
