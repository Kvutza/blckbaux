import BlckbauxResearch.Foundation
import BlckbauxResearch.Precision.Discrete
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Basis
import Mathlib.LinearAlgebra.Matrix.Trace
import Mathlib.Algebra.Lie.Classical
import Mathlib.LinearAlgebra.UnitaryGroup

set_option linter.unusedSectionVars false

/-!
# Lie algebra and orthogonal operator views

This module formalizes commutator and orthogonal-multiplication identities.
It does not establish unrestricted parameter exploration, a rank guarantee,
objective improvement, or a production materializer.

Key theorems:
1. `commutatorSkew`: The commutator of two symmetric matrices is skew-symmetric.
2. `memSo`: The commutator of symmetric matrices belongs to `LieAlgebra.Orthogonal.so`.
3. `skewTrace`: The trace product of any skew-symmetric matrix and any symmetric
   matrix is identically zero. This does not eliminate the quadratic curvature
   of the actual induced weight perturbation.
4. `frobeniusInvariant`: Right multiplication by an orthogonal matrix strictly
   preserves the Frobenius energy of the weight matrix.
-/

namespace BlckbauxResearch.Optimization.LieOperator

open Matrix

variable {n m : Type*} [Fintype n] [DecidableEq n] [Fintype m] [DecidableEq m]

/-- The commutator of two square matrices. -/
def commutator (A B : Matrix n n ℝ) : Matrix n n ℝ :=
  A * B - B * A

/-- Transpose of a commutator reverses the bracket. -/
theorem transposeCommutator
    (A B : Matrix n n ℝ) :
    (commutator A B)ᵀ = commutator Bᵀ Aᵀ := by
  unfold commutator
  simp only [transpose_sub, transpose_mul]

/-- The commutator of two symmetric matrices is skew-symmetric. -/
theorem commutatorSkew
    (A B : Matrix n n ℝ)
    (hA : Aᵀ = A)
    (hB : Bᵀ = B) :
    (commutator A B)ᵀ = -commutator A B := by
  unfold commutator
  simp only [transpose_sub, transpose_mul, hA, hB]
  exact (neg_sub (A * B) (B * A)).symm

/-- The commutator of two symmetric matrices belongs to the Lie algebra `so(n)`. -/
theorem memSo
    (A B : Matrix n n ℝ)
    (hA : Aᵀ = A)
    (hB : Bᵀ = B) :
    commutator A B ∈ LieAlgebra.Orthogonal.so n ℝ := by
  rw [LieAlgebra.Orthogonal.mem_so]
  exact commutatorSkew A B hA hB

/-- Elementary pairwise rotation generator between coordinates `i` and `j`. -/
def pairGen (i j : n) : Matrix n n ℝ :=
  Matrix.single i j (1 : ℝ) - Matrix.single j i (1 : ℝ)

/-- Pairwise rotation generator is skew-symmetric. -/
theorem pairSkew (i j : n) : (pairGen i j)ᵀ = -pairGen i j := by
  unfold pairGen
  simp only [transpose_sub, transpose_single]
  exact (neg_sub (Matrix.single i j (1 : ℝ)) (Matrix.single j i (1 : ℝ))).symm

/-- Pairwise rotation generator belongs to the Lie algebra `so(n)`. -/
theorem pairSo (i j : n) : pairGen i j ∈ LieAlgebra.Orthogonal.so n ℝ := by
  rw [LieAlgebra.Orthogonal.mem_so]
  exact pairSkew i j

/-- The commutator of two adjacent pairwise rotation generators yields the
composed pairwise rotation generator. -/
theorem pairComm
    (i j k : n)
    (hij : i ≠ j) (hjk : j ≠ k) (hik : i ≠ k) :
    commutator (pairGen i j) (pairGen j k) = pairGen i k := by
  unfold commutator pairGen
  simp only [sub_mul, mul_sub, Matrix.single_mul_single_same, mul_one]
  rw [Matrix.single_mul_single_of_ne 1 i j k hjk 1]
  rw [Matrix.single_mul_single_of_ne 1 j i j hij 1]
  rw [Matrix.single_mul_single_of_ne 1 j i k hik 1]
  rw [Matrix.single_mul_single_of_ne 1 j k i hik.symm 1]
  rw [Matrix.single_mul_single_of_ne 1 j k j hjk.symm 1]
  rw [Matrix.single_mul_single_of_ne 1 k j i hij.symm 1]
  simp only [sub_zero, zero_sub, sub_neg_eq_add]
  abel

/-- Every skew-symmetric matrix decomposes as a weighted sum of elementary
pairwise rotation generators. -/
theorem pairSum (A : Matrix n n ℝ) (hA : Aᵀ = -A) :
    A = (1 / 2 : ℝ) • ∑ i : n, ∑ j : n, A i j • pairGen i j := by
  have hTrans : ∀ i j, A j i = -A i j := by
    intro i j
    have h1 : Aᵀ i j = (-A) i j := by rw [hA]
    rw [transpose_apply, Matrix.neg_apply] at h1
    exact h1
  ext a b
  simp only [pairGen, Matrix.smul_apply, Matrix.sum_apply, Matrix.sub_apply,
    smul_sub, smul_eq_mul, Matrix.single_apply]
  have hTerm : ∀ i j, A i j * (if i = a ∧ j = b then (1 : ℝ) else 0) =
      if i = a ∧ j = b then A a b else 0 := by
    intro i j
    split_ifs with h
    · rcases h with ⟨rfl, rfl⟩
      ring
    · ring
  have hTerm2 : ∀ i j, A i j * (if j = a ∧ i = b then (1 : ℝ) else 0) =
      if j = a ∧ i = b then A b a else 0 := by
    intro i j
    split_ifs with h
    · rcases h with ⟨rfl, rfl⟩
      ring
    · ring
  simp_rw [hTerm, hTerm2]
  simp only [Finset.sum_sub_distrib]
  rw [Fintype.sum_eq_single a]
  · rw [Fintype.sum_eq_single b]
    · rw [Fintype.sum_eq_single b]
      · rw [Fintype.sum_eq_single a]
        · have hba := hTrans a b
          simp only [and_self, ↓reduceIte]
          rw [hba]
          ring
        · intro j hj
          split_ifs with h
          · exact False.elim (hj h.1)
          · rfl
      · intro i hi
        have : ∀ j : n, (if j = a ∧ i = b then A b a else (0 : ℝ)) = 0 := by
          intro j
          split_ifs with h
          · exact False.elim (hi h.2)
          · rfl
        simp [this]
    · intro j hj
      split_ifs with h
      · exact False.elim (hj h.2)
      · rfl
  · intro i hi
    have : ∀ j : n, (if i = a ∧ j = b then A a b else (0 : ℝ)) = 0 := by
      intro j
      split_ifs with h
      · exact False.elim (hi h.1)
      · rfl
    simp [this]

/-- The Grammian of any weight matrix is symmetric. -/
theorem grammianSymm (W : Matrix m n ℝ) :
    (Wᵀ * W)ᵀ = Wᵀ * W := by
  simp only [transpose_mul, transpose_transpose]

/-- The Lie generator formed by the weight Grammian and activation covariance. -/
def lieGenerator (W : Matrix m n ℝ) (cov : Matrix n n ℝ) : Matrix n n ℝ :=
  commutator (Wᵀ * W) cov

/-- The forward activation covariance commutator is in `so(n)`. -/
theorem genSo
    (W : Matrix m n ℝ)
    (cov : Matrix n n ℝ)
    (hCov : covᵀ = cov) :
    lieGenerator W cov ∈ LieAlgebra.Orthogonal.so n ℝ := by
  unfold lieGenerator
  exact memSo (Wᵀ * W) cov (grammianSymm W) hCov

/-- The trace of the product of a skew-symmetric and a symmetric matrix is zero. -/
theorem skewTrace
    (Ω M : Matrix n n ℝ)
    (hΩ : Ωᵀ = -Ω)
    (hM : Mᵀ = M) :
    trace (Ω * M) = 0 := by
  have h1 : trace (Ω * M) = trace (Ω * M)ᵀ := (trace_transpose (Ω * M)).symm
  have h2 : (Ω * M)ᵀ = Mᵀ * Ωᵀ := transpose_mul Ω M
  rw [h2, hM, hΩ, mul_neg, trace_neg] at h1
  have h3 : trace (M * Ω) = trace (Ω * M) := trace_mul_comm M Ω
  rw [h3] at h1
  linarith

/-- Frobenius inner product of matrices. -/
def frobeniusSq (W : Matrix m n ℝ) : ℝ :=
  trace (W * Wᵀ)

/-- Orthogonal right-multiplication preserves Frobenius norm. -/
theorem frobeniusInvariant
    (W : Matrix m n ℝ)
    (Q : Matrix n n ℝ)
    (hQ : Q ∈ Matrix.orthogonalGroup n ℝ) :
    frobeniusSq (W * Q) = frobeniusSq W := by
  unfold frobeniusSq
  have hOrth : Q * Qᵀ = 1 := by
    rw [Matrix.mem_orthogonalGroup_iff] at hQ
    exact hQ
  have h1 : (W * Q) * (W * Q)ᵀ = W * (Q * Qᵀ) * Wᵀ := by
    rw [transpose_mul]
    simp only [Matrix.mul_assoc]
  rw [h1, hOrth, Matrix.mul_one]

/-- The skew-symmetric perturbation trace with weight Grammian is identically zero. -/
theorem grammianTrace
    (W : Matrix m n ℝ)
    (Ω : Matrix n n ℝ)
    (hΩ : Ωᵀ = -Ω) :
    trace (Ω * (Wᵀ * W)) = 0 :=
  skewTrace Ω (Wᵀ * W) hΩ (grammianSymm W)

/-- The first-order Frobenius variation under skew-symmetric perturbation is identically zero. -/
theorem frobeniusLinear
    (W : Matrix m n ℝ)
    (Ω : Matrix n n ℝ)
    (hΩ : Ωᵀ = -Ω) :
    trace ((W * Ω) * Wᵀ) = 0 := by
  have h1 : trace ((W * Ω) * Wᵀ) = trace (W * (Ω * Wᵀ)) := by
    rw [Matrix.mul_assoc]
  have h2 : trace (W * (Ω * Wᵀ)) = trace ((Ω * Wᵀ) * W) := trace_mul_comm W (Ω * Wᵀ)
  have h3 : (Ω * Wᵀ) * W = Ω * (Wᵀ * W) := Matrix.mul_assoc Ω Wᵀ W
  rw [h1, h2, h3]
  exact grammianTrace W Ω hΩ

variable {k : Type*} [Fintype k] [DecidableEq k]

/-- Bilinear attention kernel is invariant under GL(k) gauge transformation. -/
theorem gaugeInvar
    (Wq Wk : Matrix m k ℝ)
    (A Ainv : Matrix k k ℝ)
    (hA : A * Ainv = 1) :
    (Wq * A) * (Wk * Ainvᵀ)ᵀ = Wq * Wkᵀ := by
  have h1 : (Wk * Ainvᵀ)ᵀ = Ainv * Wkᵀ := by
    rw [transpose_mul, transpose_transpose]
  rw [h1, Matrix.mul_assoc Wq A, ← Matrix.mul_assoc A, hA, Matrix.one_mul]


/-- Optimal descent drop on a 1D quadratic surrogate along the geodesic. -/
theorem geodesicDrop (q0 g H : ℝ) (hH : 0 < H) :
    let α := -g / H
    q0 - (q0 + g * α + (1 / 2) * H * α^2) = g^2 / (2 * H) := by
  intro α
  dsimp [α]
  have hHne : H ≠ 0 := ne_of_gt hH
  field_simp
  ring

/-- Normalization of a scaled vector is invariant under positive scaling. -/
theorem radialInvar (c : ℝ) (hc : 0 < c) (v : ℝ) (hv : 0 < v) :
    (c * v) / Real.sqrt ((c * v)^2) = v / Real.sqrt (v^2) := by
  have hcv : 0 < c * v := mul_pos hc hv
  rw [Real.sqrt_sq (le_of_lt hcv), Real.sqrt_sq (le_of_lt hv)]
  rw [mul_div_mul_left _ _ (ne_of_gt hc)]

#print axioms commutatorSkew
#print axioms memSo
#print axioms pairSkew
#print axioms pairSo
#print axioms pairComm
#print axioms pairSum
#print axioms skewTrace
#print axioms grammianTrace
#print axioms frobeniusLinear
#print axioms frobeniusInvariant
#print axioms gaugeInvar
#print axioms geodesicDrop
#print axioms radialInvar

end BlckbauxResearch.Optimization.LieOperator
