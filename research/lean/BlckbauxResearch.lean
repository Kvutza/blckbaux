import BlckbauxResearch.Foundation
import BlckbauxResearch.Representation.Blocks
import BlckbauxResearch.Representation.RecordedCuda
import BlckbauxResearch.Semantics.ModelTheory
import BlckbauxResearch.Semantics.Experiment
import BlckbauxResearch.Precision.Commit
import BlckbauxResearch.Precision.BFloat16
import BlckbauxResearch.Precision.Float16
import BlckbauxResearch.Precision.Discrete
import BlckbauxResearch.Optimization.Step
import BlckbauxResearch.Optimization.ObjectiveSearch
import BlckbauxResearch.Optimization.Reachability
import BlckbauxResearch.Optimization.Guide
import BlckbauxResearch.Optimization.Interpolation
import BlckbauxResearch.Optimization.PairRotation
import BlckbauxResearch.Optimization.Perturbation
import BlckbauxResearch.Optimization.Smooth
import BlckbauxResearch.Optimization.Moments
import BlckbauxResearch.Optimization.Field
import BlckbauxResearch.Optimization.Search
import BlckbauxResearch.Optimization.Rate
import BlckbauxResearch.Optimization.Selection
import BlckbauxResearch.Optimization.Transaction
import BlckbauxResearch.Optimization.Native
import BlckbauxResearch.Optimization.ResearchProblem
import BlckbauxResearch.Optimization.Effectiveness
import BlckbauxResearch.Optimization.EffectSolution
import BlckbauxResearch.Optimization.Information
import BlckbauxResearch.Optimization.Regret
import BlckbauxResearch.Optimization.Synthesis
import BlckbauxResearch.Optimization.Process
import BlckbauxResearch.Optimization.Target
import BlckbauxResearch.Optimization.Scope
import BlckbauxResearch.Optimization.Witness
import BlckbauxResearch.Optimization.Adaptation
import BlckbauxResearch.Optimization.Surrogate
import BlckbauxResearch.Optimization.Neighbor
import BlckbauxResearch.Optimization.Fitting
import BlckbauxResearch.Optimization.Accuracy
import BlckbauxResearch.Optimization.AdaptiveKnn
import BlckbauxResearch.Optimization.Bayes
import BlckbauxResearch.Optimization.Statistic
import BlckbauxResearch.Optimization.Retention
import BlckbauxResearch.Optimization.EnvelopeMemory
import BlckbauxResearch.Optimization.LocalMemory
import BlckbauxResearch.Optimization.KnnLearning
import BlckbauxResearch.Optimization.Calibration
import BlckbauxResearch.Optimization.KnnDecision
import BlckbauxResearch.Optimization.Quadrature
import BlckbauxResearch.Optimization.SampledEnn
import BlckbauxResearch.Optimization.RecentMemory
import BlckbauxResearch.Optimization.TransferGuide
import BlckbauxResearch.Optimization.RetainedGuide
import BlckbauxResearch.Optimization.PretrainRegret
import BlckbauxResearch.Optimization.PretrainHorizon
import BlckbauxResearch.Forward.LossGeometry
import BlckbauxResearch.Optimization.Batch
import BlckbauxResearch.Optimization.Cost
import BlckbauxResearch.Optimization.ContextualEnn
import BlckbauxResearch.Optimization.Trajectory
import BlckbauxResearch.Optimization.Reliability
import BlckbauxResearch.Optimization.Operator
import BlckbauxResearch.Optimization.LieOperator
import BlckbauxResearch.Optimization.Metric
import BlckbauxResearch.Optimization.PretrainWorkload
import BlckbauxResearch.Optimization.PretrainProfile
import BlckbauxResearch.Optimization.PatchFeedback
import BlckbauxResearch.Optimization.EvaluationSpace
import BlckbauxResearch.Semantics.FbtProgram
import BlckbauxResearch.Semantics.Optimizer
import BlckbauxResearch.Forward.Encoder
import BlckbauxResearch.Forward.Transport
import BlckbauxResearch.Forward.Projective
import BlckbauxResearch.Forward.Residual
import BlckbauxResearch.Forward.SeedSensitivity
import BlckbauxResearch.Forward.SinkhornRatio
import BlckbauxResearch.Forward.MixingSensitivity
import BlckbauxResearch.Forward.NormSensitivity
import BlckbauxResearch.Optimization.EnnCommit
import BlckbauxResearch.Optimization.SeedSupport
import BlckbauxResearch.Optimization.ActiveEnergy
import BlckbauxResearch.Forward.Attention
import BlckbauxResearch.Forward.Experts
import BlckbauxResearch.Forward.Mixing
import BlckbauxResearch.Forward.Sensitivity
import BlckbauxResearch.Forward.Regularity
import BlckbauxResearch.Forward.Selection
import BlckbauxResearch.Forward.SelectionCell
import BlckbauxResearch.Forward.LocalSensitivity
import BlckbauxResearch.Forward.Clipping
import BlckbauxResearch.Forward.Decoder
import BlckbauxResearch.Forward.Program
import BlckbauxResearch.Forward.Parameters
import BlckbauxResearch.Forward.Quality
import BlckbauxResearch.Forward.Work
import BlckbauxResearch.Objective.MultiObjective
import BlckbauxResearch.Objective.MarginalWindow
import BlckbauxResearch.Objective.NaturalWindow
import BlckbauxResearch.Objective.WindowPath
import BlckbauxResearch.Objective.CorpusMemory
import BlckbauxResearch.Objective.CorpusWitness
import BlckbauxResearch.Objective.Construction
import BlckbauxResearch.Objective.Aggregation
import BlckbauxResearch.Objective.Response
import BlckbauxResearch.Objective.Concentration
import BlckbauxResearch.Objective.Adaptive
import BlckbauxResearch.Objective.Relative
import BlckbauxResearch.Protocol
import BlckbauxResearch.Agenda
import BlckbauxResearch.Export.Latex

/-!
# Mathematical reading convention

This comment applies to the imported BLCKBAUX formalization; it declares no Lean
notation.  Let `p` be the learned-parameter count, `Θₚ` the ideal real
parameter space, `θ ∈ Θₚ` a state, and `u ∈ Θₚ` a proposed update.  A descriptor
`d` belongs to its own space and is not an element of `Θₚ`.  Let `z` denote a
stored finite-precision state; the relation between `z` and `θ` requires the
particular precision contract, not an implicit identification.

For a population objective `f`, write `G_f(θ, θ′) = f(θ) - f(θ′)` for progress and
`R_f(θ; θ⋆) = f(θ) - f(θ⋆)` for simple regret relative to a proof-side
comparator.  A sampled loss, an ENN estimate, and `f` are different functions.
The symbol `θ⋆` is not information available to the optimizer.  Proposal laws,
commit rules, and cost measures must be identified in their own modules; a
shared glyph must not conceal different constructions or assumptions.

These symbols are a reading aid, not replacements for searchable Lean names.
Every mathematical abbreviation should expose, rather than erase, the
hypotheses on which a checked result depends.
-/
