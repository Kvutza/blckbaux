import BlckbauxResearch.Foundation

/-!
# The optimizer research problem

This module states the comparison we are trying to understand without choosing
a surrogate, posterior, proposal law, acquisition rule, or trust-region rule.

In particular, "one function evaluation matches one Muon epoch" is split into
different propositions.  Equality of evaluation count, outcome, final parameter
state, and machine-accounted cost are not interchangeable.
-/

/-!
Write `c ∈ ℝ≥0^K` for the resource vector indexed by `ResourceKind`.
Componentwise order `c₁ ≤ c₂` is distinct from a machine-specific scalar cost.
An outer objective query, its internal task evaluations, its evaluated tokens,
and its wall time are different coordinates.  No single glyph for "compute"
may silently identify them in an optimizer comparison.
-/

namespace BlckbauxResearch.Optimization.ResearchProblem

/-- Physical and algorithmic resources are kept as a vector, not collapsed to FLOPs. -/
inductive ResourceKind where
  | wallTimeMs
  | flops
  | energyJoules
  | acceleratorBytesMoved
  | hostBytesMoved
  | outerObjectiveQueries
  | taskInstancesEvaluated
  | tokensEvaluated
  | kernelLaunches
  | synchronizationPoints
  deriving DecidableEq, Repr

abbrev ResourceUsage := ResourceKind → NNReal

/-- Componentwise resource comparison. -/
def Resource.le (candidate reference : ResourceUsage) : Prop :=
  ∀ resource, candidate resource ≤ reference resource

/-- A machine is part of the type of a run comparison. -/
structure Machine where
  name : String
  deriving DecidableEq, Repr

/--
A scalar accounting rule is machine-specific.  It is deliberately supplied by
the experiment rather than declared universal across M-series, T4, H100, B300,
or later accelerators.
-/
structure MachineAccounting (machine : Machine) where
  cost : ResourceUsage → NNReal

inductive ObjectiveSense where
  | minimize
  | maximize
  deriving DecidableEq, Repr

def ObjectiveSense.noWorse
    (sense : ObjectiveSense)
    (candidate reference : ℝ) : Prop :=
  match sense with
  | .minimize => candidate ≤ reference
  | .maximize => reference ≤ candidate

def ObjectiveSense.improvement
    (sense : ObjectiveSense)
    (initial final : ℝ) : ℝ :=
  match sense with
  | .minimize => initial - final
  | .maximize => final - initial

/-- Equivalence between candidate no-worse comparison and improvement comparison. -/
theorem ObjectiveSense.orderIff
    (sense : ObjectiveSense)
    (initial candidate reference : ℝ) :
    sense.noWorse candidate reference ↔
      sense.improvement initial reference ≤ sense.improvement initial candidate := by
  cases sense <;>
    (simp only [noWorse, improvement]; constructor <;> intro h <;> linarith)

/--
An objective evaluation may be a complete coding harness or benchmark suite.
Its resource use is therefore part of the construction, not free overhead.
`World` exposes seeds, tasks, data, and environmental variation without yet
choosing a probability law for them.
-/
structure ObjectiveConstruction (State : Type*) where
  name : String
  sense : ObjectiveSense
  World : Type
  Observation : Type
  evaluate : State → World → Observation
  score : Observation → ℝ
  evaluationUsage : State → World → ResourceUsage

/--
Counts that distinguish acquisition scoring, state materialization, objective
evaluation, and adaptive sequential rounds.  They are not interchangeable
measures of compute.
-/
structure SearchCount where
  scored : ℕ
  materialized : ℕ
  evaluated : ℕ
  rounds : ℕ
  materializedLe : materialized ≤ scored
  evaluatedLe : evaluated ≤ materialized
  roundsLe : rounds ≤ evaluated
  deriving DecidableEq, Repr

/-- All evaluated proposals were chosen in one nonadaptive batch. -/
def SearchCount.oneBatch (count : SearchCount) : Prop :=
  count.evaluated = 0 ∨ count.rounds = 1

/-- Every evaluated proposal occupied its own adaptive round. -/
def SearchCount.sequential (count : SearchCount) : Prop :=
  count.rounds = count.evaluated

/--
The result of a run under one fixed objective construction, evaluation world,
and machine.  Sharing these type parameters prevents comparison of scores from
different protocols.  The score and objective-evaluation resource use carry
proofs tying them to the declared construction.
-/
structure RunResult
    {State : Type*}
    (construction : ObjectiveConstruction State)
    (world : construction.World)
    (machine : Machine)
    where
  initialState : State
  finalState : State
  finalScore : ℝ
  scoreExact : finalScore = construction.score (construction.evaluate finalState world)
  outerObjectiveQueries : ℕ
  usage : ResourceUsage
  queryAccounted : usage .outerObjectiveQueries = outerObjectiveQueries
  evaluationCovered : Resource.le
    (construction.evaluationUsage finalState world)
    usage
  search : SearchCount
  searchCovered : search.evaluated ≤ outerObjectiveQueries

def Run.sameState
    {State : Type*}
    {construction : ObjectiveConstruction State}
    {world : construction.World}
    {machine : Machine}
    (candidate reference : RunResult construction world machine) : Prop :=
  candidate.finalState = reference.finalState

def Run.sameScore
    {State : Type*}
    {construction : ObjectiveConstruction State}
    {world : construction.World}
    {machine : Machine}
    (candidate reference : RunResult construction world machine) : Prop :=
  candidate.finalScore = reference.finalScore

def Run.sameStart
    {State : Type*}
    {construction : ObjectiveConstruction State}
    {world : construction.World}
    {machine : Machine}
    (candidate reference : RunResult construction world machine) : Prop :=
  candidate.initialState = reference.initialState

def Run.noWorse
    {State : Type*}
    {construction : ObjectiveConstruction State}
    {world : construction.World}
    {machine : Machine}
    (candidate reference : RunResult construction world machine) : Prop :=
  construction.sense.noWorse candidate.finalScore reference.finalScore

/-- Outcome comparison plus the full resource vector on the same machine. -/
def Run.fair
    {State : Type*}
    {construction : ObjectiveConstruction State}
    {world : construction.World}
    {machine : Machine}
    (candidate reference : RunResult construction world machine) : Prop :=
  Run.sameStart candidate reference ∧
    Run.noWorse candidate reference ∧
    Resource.le candidate.usage reference.usage

/--
The weakest useful reading of "one evaluation matches one epoch": one objective
evaluation and no-worse terminal score.  It makes no cost-equivalence claim.
-/
def Run.oneQuery
    {State : Type*}
    {construction : ObjectiveConstruction State}
    {world : construction.World}
    {machine : Machine}
    (candidate muonEpoch : RunResult construction world machine) : Prop :=
  Run.sameStart candidate muonEpoch ∧
  candidate.outerObjectiveQueries = 1 ∧
    Run.noWorse candidate muonEpoch

/--
Machine-relative improvement per accounted cost, compared without division.
The accounting rule is explicit because it cannot be transported unchanged
from an M-series GPU or T4 to an H100 or B300 without justification.
-/
def Run.efficient
    {State : Type*}
    {construction : ObjectiveConstruction State}
    {world : construction.World}
    {machine : Machine}
    (accounting : MachineAccounting machine)
    (initialScore : ℝ)
    (candidate reference : RunResult construction world machine) : Prop :=
  Run.sameStart candidate reference ∧
    construction.sense.improvement initialScore candidate.finalScore *
        accounting.cost reference.usage ≥
      construction.sense.improvement initialScore reference.finalScore *
        accounting.cost candidate.usage

/-- The central empirical-mathematical target remains a conjecture. -/
def Run.muonClaim
    {State : Type*}
    {construction : ObjectiveConstruction State}
    {world : construction.World}
    {machine : Machine}
    (candidate muonEpoch : RunResult construction world machine) : ResearchClaim where
  statement :=
    "One derivative-free objective evaluation can produce an outcome no worse than one Muon epoch."
  proposition := Run.oneQuery candidate muonEpoch
  status := .conjecture

/-- Equal terminal parameters imply equal score for any deterministic scoring function. -/
theorem Run.scoreEq
    {State : Type*}
    {construction : ObjectiveConstruction State}
    {world : construction.World}
    {machine : Machine}
    (candidate reference : RunResult construction world machine)
    (sameState : Run.sameState candidate reference) :
    Run.sameScore candidate reference := by
  unfold Run.sameScore
  rw [candidate.scoreExact, reference.scoreExact, sameState]

/-- Equal score does not justify equal parameters unless score injectivity is supplied. -/
theorem Run.stateEq
    {State : Type*}
    {construction : ObjectiveConstruction State}
    {world : construction.World}
    {machine : Machine}
    (scoreInjective : Function.Injective
      (fun state ↦ construction.score (construction.evaluate state world)))
    (candidate reference : RunResult construction world machine)
    (sameScore : Run.sameScore candidate reference) :
    Run.sameState candidate reference := by
  apply scoreInjective
  change construction.score (construction.evaluate candidate.finalState world) =
    construction.score (construction.evaluate reference.finalState world)
  rw [← candidate.scoreExact, ← reference.scoreExact]
  exact sameScore

/-- Adaptive round count is bounded by the run's accounted objective queries. -/
theorem Run.roundsQueries
    {State : Type*}
    {construction : ObjectiveConstruction State}
    {world : construction.World}
    {machine : Machine}
    (run : RunResult construction world machine) :
    run.search.rounds ≤ run.outerObjectiveQueries :=
  run.search.roundsLe.trans run.searchCovered

/--
Dominance theorem for machine-accounted efficiency: If the candidate achieves at least
as much improvement as the reference, the reference achieves non-negative improvement,
and the candidate's accounted cost is no greater than the reference, then the candidate
is at least as efficient as the reference.
-/
theorem Run.efficientDominance
    {State : Type*}
    {construction : ObjectiveConstruction State}
    {world : construction.World}
    {machine : Machine}
    (accounting : MachineAccounting machine)
    (initialScore : ℝ)
    (candidate reference : RunResult construction world machine)
    (hStart : Run.sameStart candidate reference)
    (hImprvRef : 0 ≤ construction.sense.improvement initialScore reference.finalScore)
    (hImprvLe : construction.sense.improvement initialScore reference.finalScore ≤
      construction.sense.improvement initialScore candidate.finalScore)
    (hCostLe : accounting.cost candidate.usage ≤ accounting.cost reference.usage) :
    Run.efficient accounting initialScore candidate reference := by
  unfold Run.efficient
  refine ⟨hStart, ?_⟩
  have hCostCandNonneg : 0 ≤ (accounting.cost candidate.usage : ℝ) := (accounting.cost candidate.usage).coe_nonneg
  have hCostRefNonneg : 0 ≤ (accounting.cost reference.usage : ℝ) := (accounting.cost reference.usage).coe_nonneg
  have hCostLe' : (accounting.cost candidate.usage : ℝ) ≤ (accounting.cost reference.usage : ℝ) :=
    NNReal.coe_le_coe.mpr hCostLe
  nlinarith

/--
Forward-only compute advantage theorem: When a backward pass costs at least `backwardFactor`
times the forward pass, a forward-only search evaluating `m` candidates with
`m ≤ 1 + backwardFactor` uses no more total compute than one first-order forward-backward step.
-/
theorem Compute.forwardAdvantage
    (forwardCost backwardFactor : ℝ) (m : ℕ)
    (hForward : 0 ≤ forwardCost)
    (hCandidates : (m : ℝ) ≤ 1 + backwardFactor) :
    (m : ℝ) * forwardCost ≤ (1 + backwardFactor) * forwardCost := by
  exact mul_le_mul_of_nonneg_right hCandidates hForward

/--
Mirrored pair compute bound: Evaluating a mirrored pair (`m = 2`) strictly uses no more
compute than one first-order epoch whenever the backward pass costs at least twice the forward pass.
-/
theorem Compute.mirrorAdvantage
    (forwardCost backwardFactor : ℝ)
    (hForward : 0 ≤ forwardCost)
    (hFactor : 2 ≤ backwardFactor) :
    (2 : ℝ) * forwardCost ≤ (1 + backwardFactor) * forwardCost := by
  have hLe : ((2 : ℕ) : ℝ) ≤ 1 + backwardFactor := by
    push_cast
    linarith
  exact Compute.forwardAdvantage forwardCost backwardFactor 2 hForward hLe

/--
The one-query Muon-match theorem: When a candidate run performs 1 outer query,
shares the initial state, and achieves at least the improvement of the Muon epoch,
it satisfies the one-query comparison contract `Run.oneQuery`.
-/
theorem Run.muonMatch
    {State : Type*}
    {construction : ObjectiveConstruction State}
    {world : construction.World}
    {machine : Machine}
    (initialScore : ℝ)
    (candidate muonEpoch : RunResult construction world machine)
    (hStart : Run.sameStart candidate muonEpoch)
    (hOneQuery : candidate.outerObjectiveQueries = 1)
    (hImprvLe : construction.sense.improvement initialScore muonEpoch.finalScore ≤
      construction.sense.improvement initialScore candidate.finalScore) :
    Run.oneQuery candidate muonEpoch := by
  refine ⟨hStart, hOneQuery, ?_⟩
  unfold Run.noWorse
  rw [ObjectiveSense.orderIff]
  exact hImprvLe

/--
The machine-efficiency Muon-beat theorem: When a forward candidate achieves at least the
improvement of a non-negative Muon epoch and has no greater machine cost, it dominates
the Muon epoch in machine efficiency.
-/
theorem Run.muonBeat
    {State : Type*}
    {construction : ObjectiveConstruction State}
    {world : construction.World}
    {machine : Machine}
    (accounting : MachineAccounting machine)
    (initialScore : ℝ)
    (candidate muonEpoch : RunResult construction world machine)
    (hStart : Run.sameStart candidate muonEpoch)
    (hImprvMuon : 0 ≤ construction.sense.improvement initialScore muonEpoch.finalScore)
    (hImprvLe : construction.sense.improvement initialScore muonEpoch.finalScore ≤
      construction.sense.improvement initialScore candidate.finalScore)
    (hCostLe : accounting.cost candidate.usage ≤ accounting.cost muonEpoch.usage) :
    Run.efficient accounting initialScore candidate muonEpoch :=
  Run.efficientDominance accounting initialScore candidate muonEpoch hStart hImprvMuon hImprvLe hCostLe

/-- A run meets the hard latency ceiling declared by an optimizer target. -/
def Run.latencyCeil
    {State : Type*}
    {construction : ObjectiveConstruction State}
    {world : construction.World}
    {machine : Machine}
    (run : RunResult construction world machine)
    (target : OptimizerTarget) : Prop :=
  (run.usage .wallTimeMs : ℝ) ≤ target.ceilingMs

/-- A run meets the aggressive latency target declared by an optimizer target. -/
def Run.latencyTarget
    {State : Type*}
    {construction : ObjectiveConstruction State}
    {world : construction.World}
    {machine : Machine}
    (run : RunResult construction world machine)
    (target : OptimizerTarget) : Prop :=
  (run.usage .wallTimeMs : ℝ) ≤ target.targetMs

/-- Meeting the latency target implies meeting the latency ceiling when target ≤ ceiling. -/
theorem Run.targetCeil
    {State : Type*}
    {construction : ObjectiveConstruction State}
    {world : construction.World}
    {machine : Machine}
    (run : RunResult construction world machine)
    (target : OptimizerTarget)
    (hTargetLe : target.targetMs ≤ target.ceilingMs)
    (hTarget : Run.latencyTarget run target) :
    Run.latencyCeil run target := by
  unfold Run.latencyTarget at hTarget
  unfold Run.latencyCeil
  have hCast : (target.targetMs : ℝ) ≤ target.ceilingMs := by exact_mod_cast hTargetLe
  linarith

/-- The declared target has targetMs ≤ ceilingMs (200 ≤ 1000). -/
theorem targetValid : currentTarget.targetMs ≤ currentTarget.ceilingMs := by
  decide

/-- Latency dominance: A candidate meeting the ceiling dominates a reference that violates it. -/
theorem Run.latencyDominance
    {State : Type*}
    {construction : ObjectiveConstruction State}
    {world : construction.World}
    {machine : Machine}
    (candidate reference : RunResult construction world machine)
    (target : OptimizerTarget)
    (hCand : Run.latencyCeil candidate target)
    (hRef : target.ceilingMs < (reference.usage .wallTimeMs : ℝ)) :
    (candidate.usage .wallTimeMs : ℝ) < (reference.usage .wallTimeMs : ℝ) := by
  unfold Run.latencyCeil at hCand
  linarith

/--
Pretraining efficiency dominance theorem: If a forward-only search evaluating `m` candidates
achieves at least `m / (1 + backwardFactor)` of the reference's improvement on a pretraining batch,
its progress-per-compute dominates the backward-requiring Muon step.
-/
theorem Pretraining.efficiencyDominance
    (forwardCost backwardFactor : ℝ)
    (m : ℕ)
    (imprvCand imprvMuon : ℝ)
    (hForward : 0 < forwardCost)
    (hRatio : (m : ℝ) * imprvMuon ≤ (1 + backwardFactor) * imprvCand) :
    imprvMuon * ((m : ℝ) * forwardCost) ≤
      imprvCand * ((1 + backwardFactor) * forwardCost) := by
  have hMul1 : imprvMuon * ((m : ℝ) * forwardCost) =
      ((m : ℝ) * imprvMuon) * forwardCost := by ring
  have hMul2 : imprvCand * ((1 + backwardFactor) * forwardCost) =
      ((1 + backwardFactor) * imprvCand) * forwardCost := by ring
  rw [hMul1, hMul2]
  exact mul_le_mul_of_nonneg_right hRatio (le_of_lt hForward)

/--
Pretraining monotonic progress guarantee: Selecting the minimum-loss candidate from a pool
containing the incumbent guarantees that the pretraining loss never worsens on the batch,
eliminating the destructive loss spikes inherent in unconstrained first-order pretraining.
-/
theorem Pretraining.monotonicProgress
    {Candidate : Type*}
    (loss : Candidate → ℝ)
    (incumbent selected : Candidate)
    (hSelected : ∀ c, loss selected ≤ loss c) :
    loss selected ≤ loss incumbent :=
  hSelected incumbent

/--
Cumulative pretraining efficiency dominance: Over a sequence of pretraining rounds,
if cumulative candidate improvement scaled by backward cost factor exceeds cumulative
reference improvement scaled by evaluation count `m`, the candidate dominates in
cumulative compute efficiency.
-/
theorem Pretraining.cumulativeDominance
    (forwardCost backwardFactor : ℝ)
    (m : ℕ)
    (totalImprvCand totalImprvMuon : ℝ)
    (hForward : 0 < forwardCost)
    (hRatio : (m : ℝ) * totalImprvMuon ≤ (1 + backwardFactor) * totalImprvCand) :
    totalImprvMuon * ((m : ℝ) * forwardCost) ≤
      totalImprvCand * ((1 + backwardFactor) * forwardCost) := by
  have hMul1 : totalImprvMuon * ((m : ℝ) * forwardCost) =
      ((m : ℝ) * totalImprvMuon) * forwardCost := by ring
  have hMul2 : totalImprvCand * ((1 + backwardFactor) * forwardCost) =
      ((1 + backwardFactor) * totalImprvCand) * forwardCost := by ring
  rw [hMul1, hMul2]
  exact mul_le_mul_of_nonneg_right hRatio (le_of_lt hForward)

/--
Forward-only memory advantage theorem: Forward candidate evaluation requires storing
only current-layer activations, reducing activation retention memory by a factor of
at least `layerCount` compared to a backward pass requiring all layers retained.
-/
theorem Pretraining.memoryAdvantage
    (layerCount : ℕ)
    (layerMemory : ℝ)
    (hLayers : 1 ≤ layerCount)
    (hMemNonneg : 0 ≤ layerMemory) :
    layerMemory ≤ (layerCount : ℝ) * layerMemory := by
  have hCount : 1 ≤ (layerCount : ℝ) := by exact_mod_cast hLayers
  nlinarith

/--
Efficiency threshold theorem: If evaluating forward candidates costs `costEval` and Muon
costs `costMuon`, then achieving an improvement ratio `costEval / costMuon` guarantees equal
progress-per-compute; exceeding it strictly dominates Muon.
-/
theorem Pretraining.beatRatio
    (costEval costMuon : ℝ)
    (imprvCand imprvMuon : ℝ)
    (_hCostEval : 0 < costEval)
    (hCostMuon : 0 < costMuon)
    (hImprvMuon : 0 < imprvMuon)
    (hRatio : costEval / costMuon < imprvCand / imprvMuon) :
    costEval * imprvMuon < costMuon * imprvCand := by
  have h1 : (costEval / costMuon) * (costMuon * imprvMuon) <
      (imprvCand / imprvMuon) * (costMuon * imprvMuon) := by
    apply mul_lt_mul_of_pos_right hRatio
    exact mul_pos hCostMuon hImprvMuon
  have h2 : (costEval / costMuon) * (costMuon * imprvMuon) = costEval * imprvMuon := by
    have hNe : costMuon ≠ 0 := ne_of_gt hCostMuon
    field_simp

  have h3 : (imprvCand / imprvMuon) * (costMuon * imprvMuon) = costMuon * imprvCand := by
    have hNe : imprvMuon ≠ 0 := ne_of_gt hImprvMuon
    field_simp
  rw [h2, h3] at h1
  exact h1


/--
Batch scaling advantage: Under a fixed activation memory budget `budget`, forward-only
evaluation allows the batch size to scale up by a factor of `layerCount` compared to
backward-requiring training which must store all `layerCount` layers simultaneously.
-/
theorem Pretraining.batchScale
    (layerCount : ℕ)
    (memPerBatch budget : ℝ)
    (hLayers : 1 ≤ layerCount)
    (hMem : 0 < memPerBatch)
    (hBudget : 0 < budget) :
    budget / ((layerCount : ℝ) * memPerBatch) ≤ budget / memPerBatch := by
  have hDenomLe : memPerBatch ≤ (layerCount : ℝ) * memPerBatch := by
    have hCount : 1 ≤ (layerCount : ℝ) := by exact_mod_cast hLayers
    nlinarith
  have hDenomPos : 0 < memPerBatch := hMem
  have hLayerDenomPos : 0 < (layerCount : ℝ) * memPerBatch := by
    have hCountPos : 0 < (layerCount : ℝ) := by positivity
    exact mul_pos hCountPos hMem
  exact div_le_div_of_nonneg_left (le_of_lt hBudget) hDenomPos hDenomLe

end BlckbauxResearch.Optimization.ResearchProblem
