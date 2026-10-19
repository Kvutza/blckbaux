import BlckbauxResearch.Foundation

/-!
# Precision-neutral commit boundary

An optimizer transaction needs to know which real proposals are admissible and
which executed state each admissible proposal produces. It does not need to
identify BF16 with FP16. Backend-specific quantizers provide separate instances
of this small contract.
-/

/-!
Read a commit model as a partial action `Q(z, u)`: storage state `z`, real
proposal `u`, an admissibility proof, and an executed storage state.  A real
perturbation `u` must not be identified with the actual storage difference
without a precision-specific survival theorem.
-/

namespace BlckbauxResearch.Precision

/-- A backend's admissibility and commit semantics for real proposals. -/
structure CommitModel (parameterCount : ℕ) where
  State : Type*
  admissible : State → Parameter.Proposal parameterCount → Prop
  commit : ∀ (state) (proposal), admissible state proposal → State

end BlckbauxResearch.Precision
