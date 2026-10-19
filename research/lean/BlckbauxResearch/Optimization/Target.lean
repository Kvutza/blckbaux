import BlckbauxResearch.Optimization.Synthesis
import BlckbauxResearch.Optimization.ResearchProblem

/-!
# Machine-aware optimizer synthesis

The synthesis core proves regret and an abstract natural-number work bound.
This module adds the machine boundary without identifying abstract work with
milliseconds.  Every oracle branch receives a complete resource vector.
A machine certificate bounds cumulative resources along every feedback path
and bounds the wall time of every optimizer transaction separately.

The same policy must carry all three certificates for every environment:
terminal expected regret, cumulative resource feasibility, and per-round
latency.  Backend measurements are evidence for these fields; they do not
change the optimizer semantics.
-/

namespace BlckbauxResearch.Optimization.Target

open ResearchProblem Synthesis

variable {State Feedback Memory Noise : Type*} [Fintype Noise]

/-- Machine resources charged by one possible oracle branch. -/
abbrev StepUsage (State Memory Noise : Type*) :=
  ℕ → Memory → State → Noise → ResourceUsage

/-- Cumulative machine resources along one complete feedback path. -/
def spentUsage
    (policy : Policy State Feedback Memory)
    (oracle : Oracle State Feedback Memory Noise)
    (usage : StepUsage State Memory Noise) :
    ℕ → ℕ → Memory → (ℕ → Noise) → ResourceUsage
  | _, 0, _, _ => fun _ => 0
  | time, steps + 1, memory, path => fun resource =>
      usage time memory (policy.query time memory) (path time) resource +
        spentUsage policy oracle usage (time + 1) steps
          (advance policy oracle time memory (path time)) path resource

/-- A fixed token charge per transaction makes the token horizon exact. -/
theorem spentTokens
    (policy : Policy State Feedback Memory)
    (oracle : Oracle State Feedback Memory Noise)
    (usage : StepUsage State Memory Noise)
    (tokens : ℕ)
    (exact : ∀ time memory noise,
      usage time memory (policy.query time memory) noise .tokensEvaluated = tokens)
    (steps time : ℕ) (memory : Memory) (path : ℕ → Noise) :
    spentUsage policy oracle usage time steps memory path .tokensEvaluated =
      steps * tokens := by
  induction steps generalizing time memory with
  | zero => simp [spentUsage]
  | succ steps induction =>
      simp only [spentUsage, exact]
      rw [induction]
      push_cast
      ring

/-- A fixed synthesis task with an explicit machine resource account. -/
structure MachineTask (State Feedback Memory Noise : Type*) [Fintype Noise]
    extends Synthesis.Task State Feedback Memory Noise where
  machine : Machine
  parameterCount : ℕ
  contextTokens : ℕ
  candidates : ℕ
  neighbors : ℕ
  objectiveMode : ObjectiveMode
  samplingMode : SamplingMode
  usage : Environment → StepUsage State Memory Noise
  limit : ResourceUsage
  deadlineMs : NNReal

/-- The fixed BLCKBAUX operating envelope, independent of any chosen run horizon. -/
structure Envelope (task : MachineTask State Feedback Memory Noise) : Prop where
  parameters : task.parameterCount = currentTarget.parameterCount
  context : task.contextTokens = currentTarget.contextTokens
  candidates : task.candidates = currentTarget.candidates
  neighbors : task.neighbors = currentTarget.neighbors
  objective : task.objectiveMode = currentTarget.objective
  sampling : task.samplingMode = currentTarget.sampling
  latency : task.deadlineMs = currentTarget.targetMs

/-- Meeting the operating target also meets the weaker latency ceiling. -/
theorem Envelope.ceiling
    {task : MachineTask State Feedback Memory Noise}
    (envelope : Envelope task) :
    task.deadlineMs ≤ currentTarget.ceilingMs := by
  rw [envelope.latency]
  exact_mod_cast ResearchProblem.targetValid

/-- Regret, resource, and latency evidence for one environment. -/
structure MachineCert
    (policy : Policy State Feedback Memory)
    (task : MachineTask State Feedback Memory Noise)
    (environment : task.Environment) where
  algorithm : Synthesis.Certificate policy (task.oracle environment)
    (task.objective environment) (task.best environment)
    task.horizon task.bound task.budget
  credit : ℕ → Memory → ResourceUsage
  initial : Resource.le (credit 0 policy.initial) task.limit
  step : ∀ time, time < task.horizon → ∀ memory noise resource,
    task.usage environment time memory (policy.query time memory) noise resource +
        credit (time + 1)
          (advance policy (task.oracle environment) time memory noise) resource ≤
      credit time memory resource
  deadline : ∀ time, time < task.horizon → ∀ memory noise,
    task.usage environment time memory (policy.query time memory) noise
      .wallTimeMs ≤ task.deadlineMs

theorem MachineCert.controls
    {policy : Policy State Feedback Memory}
    {task : MachineTask State Feedback Memory Noise}
    {environment : task.Environment}
    (certificate : MachineCert policy task environment)
    (steps time : ℕ) (memory : Memory) (path : ℕ → Noise)
    (ends : time + steps = task.horizon) :
    Resource.le
      (spentUsage policy (task.oracle environment) (task.usage environment)
        time steps memory path)
      (certificate.credit time memory) := by
  intro resource
  induction steps generalizing time memory with
  | zero =>
      simp [spentUsage]
  | succ steps induction =>
      have before : time < task.horizon := by omega
      have nextEnds : time + 1 + steps = task.horizon := by omega
      change
        task.usage environment time memory (policy.query time memory)
              (path time) resource +
            spentUsage policy (task.oracle environment) (task.usage environment)
              (time + 1) steps
              (advance policy (task.oracle environment) time memory (path time))
              path resource ≤ certificate.credit time memory resource
      calc
        _ ≤ task.usage environment time memory (policy.query time memory)
              (path time) resource +
            certificate.credit (time + 1)
              (advance policy (task.oracle environment) time memory (path time))
              resource := by
          gcongr
          exact induction (time + 1)
            (advance policy (task.oracle environment) time memory (path time))
            nextEnds
        _ ≤ certificate.credit time memory resource :=
          certificate.step time before memory (path time) resource

/-- Every feedback path respects the declared componentwise machine budget. -/
theorem MachineCert.resourceBound
    {policy : Policy State Feedback Memory}
    {task : MachineTask State Feedback Memory Noise}
    {environment : task.Environment}
    (certificate : MachineCert policy task environment)
    (path : ℕ → Noise) :
    Resource.le
      (spentUsage policy (task.oracle environment) (task.usage environment)
        0 task.horizon policy.initial path)
      task.limit := by
  intro resource
  exact (certificate.controls task.horizon 0 policy.initial path (by omega) resource).trans
    (certificate.initial resource)

/-- One policy and certificates solve every environment in the machine task. -/
structure MachineSolution (task : MachineTask State Feedback Memory Noise) where
  policy : Policy State Feedback Memory
  certificate : ∀ environment, MachineCert policy task environment

theorem MachineSolution.regretBound
    {task : MachineTask State Feedback Memory Noise}
    (solution : MachineSolution task) (environment : task.Environment) :
    expected solution.policy (task.oracle environment) (task.objective environment)
      (task.best environment) 0 task.horizon solution.policy.initial ≤ task.bound :=
  (solution.certificate environment).algorithm.regretBound

theorem MachineSolution.resourceBound
    {task : MachineTask State Feedback Memory Noise}
    (solution : MachineSolution task) (environment : task.Environment)
    (path : ℕ → Noise) :
    Resource.le
      (spentUsage solution.policy (task.oracle environment) (task.usage environment)
        0 task.horizon solution.policy.initial path)
      task.limit :=
  (solution.certificate environment).resourceBound path

/-- Evaluated tokens are bounded over the entire chosen horizon. -/
theorem MachineSolution.tokenBound
    {task : MachineTask State Feedback Memory Noise}
    (solution : MachineSolution task) (environment : task.Environment)
    (path : ℕ → Noise) :
    spentUsage solution.policy (task.oracle environment) (task.usage environment)
        0 task.horizon solution.policy.initial path .tokensEvaluated ≤
      task.limit .tokensEvaluated :=
  solution.resourceBound environment path .tokensEvaluated

/-- The transaction horizon induces, but does not define, the token budget. -/
theorem MachineSolution.windowBound
    {task : MachineTask State Feedback Memory Noise}
    (solution : MachineSolution task) (environment : task.Environment)
    (path : ℕ → Noise) (tokens : ℕ)
    (exact : ∀ time memory noise,
      task.usage environment time memory (solution.policy.query time memory) noise
        .tokensEvaluated = tokens) :
    task.horizon * tokens ≤ task.limit .tokensEvaluated := by
  rw [← spentTokens solution.policy (task.oracle environment)
    (task.usage environment) tokens exact task.horizon 0 solution.policy.initial path]
  exact solution.tokenBound environment path

theorem MachineSolution.latencyBound
    {task : MachineTask State Feedback Memory Noise}
    (solution : MachineSolution task) (environment : task.Environment)
    (time : ℕ) (before : time < task.horizon) (memory : Memory) (noise : Noise) :
    task.usage environment time memory (solution.policy.query time memory) noise
      .wallTimeMs ≤ task.deadlineMs :=
  (solution.certificate environment).deadline time before memory noise

#print axioms MachineCert.resourceBound
#print axioms Envelope.ceiling
#print axioms MachineSolution.regretBound
#print axioms MachineSolution.resourceBound
#print axioms MachineSolution.tokenBound
#print axioms MachineSolution.windowBound
#print axioms MachineSolution.latencyBound

end BlckbauxResearch.Optimization.Target
