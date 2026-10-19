import BlckbauxResearch.Generated.Optimizer.Funs
import BlckbauxResearch.Optimization.EnnCommit
import BlckbauxResearch.Optimization.Reliability

/-!
# Executable optimizer semantics

The generated definition in this module is extracted from the production Rust
implementation by Charon and Aeneas.  The handwritten definition remains the
mathematical interface used by the rest of the formalization.  `actionMatch`
checks all Boolean controller states and proves that the two definitions agree.
-/

namespace BlckbauxResearch.Semantics.Optimizer

private def signMask : BitVec 64 := BitVec.ofNat 64 0x8000000000000000
private def expMask : BitVec 64 := BitVec.ofNat 64 0x7ff0000000000000
private def fracMask : BitVec 64 := BitVec.ofNat 64 0x000fffffffffffff

private def nanBits (bits : BitVec 64) : Bool :=
  decide ((bits &&& expMask) = expMask ∧ (bits &&& fracMask) ≠ 0)

private def zeroBits (bits : BitVec 64) : Bool :=
  decide ((bits &&& ~~~signMask) = 0)

private def orderBits (bits : BitVec 64) : BitVec 64 :=
  if bits &&& signMask = 0 then bits ||| signMask else ~~~bits

/-- IEEE-754 binary64 strict order, including unordered NaNs and equal signed
zeros, stated independently of the generated Rust translation. -/
def acceptBits (improvement margin : BitVec 64) : Bool :=
  if nanBits improvement || nanBits margin then false
  else if zeroBits improvement && zeroBits margin then false
  else decide (orderBits margin < orderBits improvement)

private theorem signMatch :
    BlckbauxResearch.Generated.Optimizer.reliability.radius.SIGN_MASK.bv = signMask := by
  native_decide

private theorem expMatch :
    BlckbauxResearch.Generated.Optimizer.reliability.radius.EXP_MASK.bv = expMask := by
  native_decide

private theorem fracMatch :
    BlckbauxResearch.Generated.Optimizer.reliability.radius.FRAC_MASK.bv = fracMask := by
  native_decide

private theorem nanMatch (bits : Aeneas.Std.U64) :
    BlckbauxResearch.Generated.Optimizer.reliability.radius.nan_bits bits =
      Aeneas.Std.Result.ok (nanBits bits.bv) := by
  simp only [BlckbauxResearch.Generated.Optimizer.reliability.radius.nan_bits,
    Aeneas.Std.lift, Aeneas.Std.bind_ok]
  rw [show BlckbauxResearch.Generated.Optimizer.reliability.radius.EXP_MASK =
      (⟨expMask⟩ : Aeneas.Std.U64) by exact Aeneas.Std.U64.bv_eq_imp_eq _ _ expMatch]
  rw [show BlckbauxResearch.Generated.Optimizer.reliability.radius.FRAC_MASK =
      (⟨fracMask⟩ : Aeneas.Std.U64) by exact Aeneas.Std.U64.bv_eq_imp_eq _ _ fracMatch]
  simp [Aeneas.Std.UScalar.eq_equiv_bv_eq, nanBits]
  split <;> simp_all [bne, BEq.beq]

private theorem zeroMatch (bits : Aeneas.Std.U64) :
    BlckbauxResearch.Generated.Optimizer.reliability.radius.zero_bits bits =
      Aeneas.Std.Result.ok (zeroBits bits.bv) := by
  simp only [BlckbauxResearch.Generated.Optimizer.reliability.radius.zero_bits,
    Aeneas.Std.lift, Aeneas.Std.bind_ok]
  rw [show BlckbauxResearch.Generated.Optimizer.reliability.radius.SIGN_MASK =
      (⟨signMask⟩ : Aeneas.Std.U64) by exact Aeneas.Std.U64.bv_eq_imp_eq _ _ signMatch]
  simp [Aeneas.Std.UScalar.eq_equiv_bv_eq, zeroBits]

private theorem orderMatch (bits : Aeneas.Std.U64) :
    BlckbauxResearch.Generated.Optimizer.reliability.radius.order_bits bits =
      Aeneas.Std.Result.ok (⟨orderBits bits.bv⟩ : Aeneas.Std.U64) := by
  simp only [BlckbauxResearch.Generated.Optimizer.reliability.radius.order_bits,
    Aeneas.Std.lift, Aeneas.Std.bind_ok]
  rw [show BlckbauxResearch.Generated.Optimizer.reliability.radius.SIGN_MASK =
      (⟨signMask⟩ : Aeneas.Std.U64) by exact Aeneas.Std.U64.bv_eq_imp_eq _ _ signMatch]
  simp [Aeneas.Std.UScalar.eq_equiv_bv_eq, orderBits]
  split <;> simp_all [Aeneas.Std.UScalar.eq_equiv_bv_eq]

private theorem bitLt (left right : BitVec 64) :
    ((↑(⟨left⟩ : Aeneas.Std.U64) : Nat) < ↑(⟨right⟩ : Aeneas.Std.U64)) =
      (left < right) := by
  rfl

/-- The integer-only comparison extracted from production Rust implements the
independent binary64 bit-level specification for every pair of words. -/
theorem bitsMatch (improvement margin : Aeneas.Std.U64) :
    BlckbauxResearch.Generated.Optimizer.reliability.radius.accept_bits improvement margin =
      Aeneas.Std.Result.ok (acceptBits improvement.bv margin.bv) := by
  simp [BlckbauxResearch.Generated.Optimizer.reliability.radius.accept_bits,
    BlckbauxResearch.Generated.Optimizer.reliability.radius.accept_decision,
    nanMatch, zeroMatch, orderMatch, bitLt, acceptBits]
  cases hni : nanBits improvement.bv <;>
    cases hnm : nanBits margin.bv <;>
    cases hzi : zeroBits improvement.bv <;>
    cases hzm : zeroBits margin.bv <;>
    cases hlt : decide (orderBits margin.bv < orderBits improvement.bv) <;>
    simp_all

/-- The extracted production commit boundary agrees with the Mathlib-facing
acceptance law once the runtime comparison has supplied its Boolean result. -/
theorem acceptMatch (improvement margin : ℝ) :
    BlckbauxResearch.Generated.Optimizer.reliability.radius.accept_decision
      (decide (BlckbauxResearch.Optimization.EnnCommit.accepts improvement margin)) =
        Aeneas.Std.Result.ok
          (decide (BlckbauxResearch.Optimization.EnnCommit.accepts improvement margin)) := by
  rfl

abbrev CodeAction :=
  BlckbauxResearch.Generated.Optimizer.reliability.radius.ReliabilityAction

/-- Translate the handwritten action vocabulary to the Rust model's vocabulary. -/
def codeAction : BlckbauxResearch.Optimization.ReliabilityAction → CodeAction
  | .hold => .Hold
  | .expand => .Expand
  | .shrink => .Shrink

/-- The Aeneas model of the production Rust decision law equals its handwritten
Mathlib-facing specification for every controller state. -/
theorem actionMatch
    (modelAdequate modelInadequate sharp accepted positiveRatio boundaryStep failure : Bool) :
    BlckbauxResearch.Generated.Optimizer.reliability.radius.decide_action
      modelAdequate modelInadequate sharp accepted positiveRatio
      boundaryStep failure =
        Aeneas.Std.Result.ok (codeAction (BlckbauxResearch.Optimization.decideAction
          modelAdequate modelInadequate sharp accepted positiveRatio boundaryStep failure)) := by
  cases modelAdequate <;> cases modelInadequate <;> cases sharp <;> cases accepted <;>
    cases positiveRatio <;> cases boundaryStep <;> cases failure <;> rfl

end BlckbauxResearch.Semantics.Optimizer
