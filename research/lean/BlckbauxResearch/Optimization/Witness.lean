import BlckbauxResearch.Optimization.Synthesis

/-!
# A two-query synthesis witness

This finite example exercises the policy/certificate interface. The policy
queries both Boolean states and recommends the state with the smaller observed
value. For arbitrary objectives and bounded observation errors `noise`, its
terminal simple regret is at most `2 * noise`. Each query costs at most one
declared work unit. The certificate is derived from the update and observation
error, rather than assuming a regret guarantee.

The example establishes neither BLCKBAUX scalability nor a shrinking regret rate.
The final counterexample proves that the noise penalty can be attained by this
recommendation rule; exact observations are needed for its zero-regret claim.
-/

namespace BlckbauxResearch.Optimization.Witness

open scoped BigOperators
open Regret Synthesis

structure Memory where
  first : ℚ
  second : ℚ

def choose (memory : Memory) : Bool :=
  if memory.first ≤ memory.second then false else true

def policy : Policy Bool ℚ Memory where
  initial := ⟨0, 0⟩
  query := fun time _ => if time = 0 then false else true
  update := fun time memory _ value =>
    if time = 0 then ⟨value, memory.second⟩ else ⟨memory.first, value⟩
  recommend := choose

theorem selectBound (objective : Bool → ℝ) (best : Bool)
    (memory : Memory) (noise : ℝ)
    (noiseNonneg : 0 ≤ noise)
    (secondError : |(memory.second : ℝ) - objective true| ≤ noise) :
    simpleRegret objective best (choose memory) ≤
      |(memory.first : ℝ) - objective false| + noise := by
  have firstError := abs_le.mp (le_refl |(memory.first : ℝ) - objective false|)
  have secondBounds := abs_le.mp secondError
  have firstNonneg := abs_nonneg ((memory.first : ℝ) - objective false)
  cases best with
  | false =>
      by_cases ordered : memory.first ≤ memory.second
      · simp only [simpleRegret, choose, if_pos ordered, sub_self]
        linarith
      · have orderedReal : (memory.second : ℝ) < (memory.first : ℝ) := by
          exact_mod_cast (lt_of_not_ge ordered)
        simp only [simpleRegret, choose, if_neg ordered]
        linarith
  | true =>
      by_cases ordered : memory.first ≤ memory.second
      · have orderedReal : (memory.first : ℝ) ≤ (memory.second : ℝ) := by
          exact_mod_cast ordered
        simp only [simpleRegret, choose, if_pos ordered]
        linarith
      · simp only [simpleRegret, choose, if_neg ordered, sub_self]
        linarith

noncomputable def potential (objective : Bool → ℝ) (best : Bool) (noise : ℝ)
    (time : ℕ) (memory : Memory) : ℝ :=
  if time = 0 then 2 * noise
  else if time = 1 then |(memory.first : ℝ) - objective false| + noise
  else simpleRegret objective best (choose memory)

noncomputable def certificate {Noise : Type*} [Fintype Noise]
    (objective : Bool → ℝ) (best : Bool)
    (oracle : Oracle Bool ℚ Memory Noise) (noise : ℝ) (noiseNonneg : 0 ≤ noise)
    (errorBound : ∀ time memory state branch,
      |(oracle.value time memory state branch : ℝ) - objective state| ≤ noise)
    (chargeBound : ∀ time memory state branch, oracle.charge time memory state branch ≤ 1) :
    Certificate policy oracle objective best 2 (2 * noise) 2 where
  potential := potential objective best noise
  initialBound := by simp [potential]
  terminalBound := by intro memory; simp [potential, policy]
  stepBound := by
    intro time before memory
    have cases : time = 0 ∨ time = 1 := by omega
    rcases cases with rfl | rfl
    · calc
        _ ≤ ∑ branch, oracle.weight 0 memory (policy.query 0 memory) branch * (2 * noise) := by
          apply Finset.sum_le_sum
          intro branch _
          apply mul_le_mul_of_nonneg_left _ (oracle.nonnegative _ _ _ _)
          change |(oracle.value 0 memory false branch : ℝ) - objective false| + noise ≤ 2 * noise
          have error := errorBound 0 memory false branch
          linarith
        _ = _ := by rw [← Finset.sum_mul, oracle.mass]; simp [potential]
    · calc
        _ ≤ ∑ branch, oracle.weight 1 memory (policy.query 1 memory) branch *
            (|(memory.first : ℝ) - objective false| + noise) := by
          apply Finset.sum_le_sum
          intro branch _
          apply mul_le_mul_of_nonneg_left _ (oracle.nonnegative _ _ _ _)
          simp only [potential, advance, policy, show (1 : ℕ) ≠ 0 by decide,
            show (2 : ℕ) ≠ 0 by decide, show (2 : ℕ) ≠ 1 by decide, ite_false]
          exact selectBound objective best _ noise noiseNonneg
            (errorBound 1 memory true branch)
        _ = _ := by rw [← Finset.sum_mul, oracle.mass]; simp [potential]
  credit := fun time _ => 2 - time
  workInitial := by simp
  workStep := by
    intro time before memory branch
    have charged := chargeBound time memory (policy.query time memory) branch
    omega

/-- The bound holds for every feedback path law satisfying the oracle contracts. -/
theorem regretBound {Noise : Type*} [Fintype Noise]
    (objective : Bool → ℝ) (best : Bool)
    (oracle : Oracle Bool ℚ Memory Noise) (noise : ℝ) (noiseNonneg : 0 ≤ noise)
    (errorBound : ∀ time memory state branch,
      |(oracle.value time memory state branch : ℝ) - objective state| ≤ noise)
    (chargeBound : ∀ time memory state branch, oracle.charge time memory state branch ≤ 1) :
    expected policy oracle objective best 0 2 policy.initial ≤ 2 * noise :=
  (certificate objective best oracle noise noiseNonneg errorBound chargeBound).regretBound

/-- All admissible objectives and query-dependent finite noise laws for the witness. -/
structure Environment (noise : ℝ) where
  objective : Bool → ℝ
  best : Bool
  optimal : Optimal objective best
  oracle : Oracle Bool ℚ Memory Bool
  errorBound : ∀ time memory state branch,
    |(oracle.value time memory state branch : ℝ) - objective state| ≤ noise
  chargeBound : ∀ time memory state branch, oracle.charge time memory state branch ≤ 1

/-- An explicit admissible environment prevents an empty synthesis task. -/
def zeroWorld (noise : ℝ) (noiseNonneg : 0 ≤ noise) : Environment noise where
  objective := fun _ => 0
  best := false
  optimal := by intro state; rfl
  oracle := {
    value := fun _ _ _ _ => 0
    weight := fun _ _ _ branch => if branch then 0 else 1
    nonnegative := by intro time memory state branch; cases branch <;> norm_num
    mass := by intro time memory state; simp
    charge := fun _ _ _ _ => 1 }
  errorBound := by intro time memory state branch; simpa using noiseNonneg
  chargeBound := by intro time memory state branch; rfl

def task (noise : ℝ) (noiseNonneg : 0 ≤ noise) : Task Bool ℚ Memory Bool where
  Environment := Environment noise
  examples := ⟨zeroWorld noise noiseNonneg⟩
  oracle := Environment.oracle
  objective := Environment.objective
  best := Environment.best
  optimal := Environment.optimal
  horizon := 2
  bound := 2 * noise
  budget := 2

/-- One computable policy and its certificates solve the entire fixed objective class. -/
noncomputable def solution (noise : ℝ) (noiseNonneg : 0 ≤ noise) :
    Solution (task noise noiseNonneg) where
  policy := policy
  certificate := fun environment => certificate environment.objective environment.best
    environment.oracle noise noiseNonneg environment.errorBound environment.chargeBound

/-- Exact feedback gives zero simple regret for every objective in this finite task. -/
theorem exactBound (environment : (task 0 (by norm_num)).Environment) :
    expected policy environment.oracle environment.objective environment.best
      0 2 policy.initial ≤ 0 := by
  simpa [solution, task] using (solution 0 (by norm_num)).regretBound environment

/-- The noisy bound also holds on every realized path, not only in expectation. -/
theorem pathBound (noise : ℝ) (noiseNonneg : 0 ≤ noise)
    (environment : Environment noise) (path : ℕ → Bool) :
    simpleRegret environment.objective environment.best
      (policy.recommend (run policy environment.oracle 0 2 policy.initial path)) ≤
        2 * noise := by
  let first := advance policy environment.oracle 0 policy.initial (path 0)
  let second := advance policy environment.oracle 1 first (path 1)
  have firstError : |(first.first : ℝ) - environment.objective false| ≤ noise := by
    simpa [first, advance, policy] using
      environment.errorBound 0 policy.initial false (path 0)
  have secondError : |(second.second : ℝ) - environment.objective true| ≤ noise := by
    simpa [second, advance, policy] using
      environment.errorBound 1 first true (path 1)
  have selected := selectBound environment.objective environment.best second noise
    noiseNonneg secondError
  have retained : second.first = first.first := by simp [second, advance, policy]
  rw [retained] at selected
  have bounded : simpleRegret environment.objective environment.best (choose second) ≤
      2 * noise := by linarith
  simpa [run, policy, first, second] using bounded

/-- Execute the exact-feedback witness with rational scores and no objective access. -/
def simulate (first second : ℚ) : Bool :=
  let memory := policy.update 0 policy.initial false first
  policy.recommend (policy.update 1 memory true second)

#eval simulate 2 1
#eval simulate 1 2

/-- A constant recommendation cannot satisfy that same zero-regret target. -/
theorem constantFails :
    let objective : Bool → ℝ := fun state => if state then 0 else 1
    Optimal objective true ∧ ¬ simpleRegret objective true false ≤ 0 := by
  dsimp
  constructor
  · intro state; cases state <;> norm_num
  · norm_num [simpleRegret]

/-- A noisy tie can attain the full penalty despite a globally optimal comparator. -/
theorem noiseSharp :
    let objective : Bool → ℝ := fun state => if state then 0 else 2
    let memory : Memory := ⟨1, 1⟩
    Optimal objective true ∧
      |(memory.first : ℝ) - objective false| ≤ 1 ∧
      |(memory.second : ℝ) - objective true| ≤ 1 ∧
      simpleRegret objective true (choose memory) = 2 := by
  dsimp
  constructor
  · intro state; cases state <;> norm_num
  · norm_num [simpleRegret, choose]

#print axioms Certificate.regretBound
#print axioms Certificate.workBound
#print axioms Solution.regretBound
#print axioms exactBound
#print axioms pathBound
#print axioms constantFails
#print axioms noiseSharp

end BlckbauxResearch.Optimization.Witness
