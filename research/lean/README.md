# BLCKBAUX optimizer research formalization

`./blckbaux research list` lists the formalized construction topics.
`./blckbaux research show TOPIC` reads one exact construction/Mathlib/hypothesis/
finding/opportunity/source-obligation record from
`../../docs/mathlib-constructions.txt`. These are fast text operations, not a
build, theorem check, or assertion that every listed obligation is resolved.
Examples: `parameters`, `proposal`, `surrogate`, `transport`, `precision`.

`./blckbaux research repl` starts the version-matched Lean community REPL. Import
`BlckbauxResearch` once, then reuse the returned environment and proof-state
identifiers across JSON requests. This avoids repeated imports and repeated
elaboration while an agent branches over tactics. Commands are separated by a
blank line. This is the latency-sensitive path; the one-shot commands below are
convenient cold-start fallbacks.

```text
./blckbaux research repl
{"cmd":"import BlckbauxResearch"}

{"cmd":"example : ∀ x : ℝ, 0 ≤ x ^ 2 := by sorry","env":0}

{"tactic":"intro x <;> positivity","proofState":0}
```

REPL completion is search feedback, not the acceptance boundary. A suggested
script may contain a placeholder or rely on the provisional `sorry` used to
create its goal. Copy the candidate into an ordinary theorem, remove every
placeholder, and run `./blckbaux research check`; only the replayed kernel-checked
declaration counts as a result.

`Semantics/Experiment.lean` is the common trace layer for regret research. It
derives terminal gap, cumulative gap, target hitting, pairwise excess, and
matched-budget comparison as separate views of the same fixed objective world.
It proves the exact relationships that hold without choosing one view as the
research target, and gives a counterexample showing that target hitting does not
imply a terminal guarantee without a stability hypothesis.
`Trace.Account` is the backend refinement boundary: Rust, Metal, and CUDA must
supply objective-query accounting, final-evaluation coverage, and distinct
search counts. An accounted horizon becomes the existing typed `RunResult`;
trace fairness and the one-query property then transfer to the corresponding
comparison contracts without trusting an unrelated reported score.

`./blckbaux research show DECLARATION` asks the pinned Lean/Mathlib environment
for a fully qualified declaration's elaborated type, axioms, home module, and
minimal import. `--body` additionally prints the complete proof or definition.
Inspection normally loads the BLCKBAUX import closure; `--mathlib` loads every
Mathlib module when the requested declaration is outside that closure.
`./blckbaux research prove 'PROPOSITION'` asks Lean's indexed `exact?` search for
a proof term. The proposition must bind every variable and hypothesis explicitly;
`--mathlib` expands the environment beyond the BLCKBAUX import closure. Pass `-` as
the proposition to read a single-line proposition from standard input.
These commands send an ephemeral module through Lean's standard input; they do
not create a source file or treat a search result as evidence that its hypotheses
hold. Examples:

```sh
./blckbaux research show BlckbauxResearch.Optimization.Rate.expected
./blckbaux research show ConvexOn.map_centerMass_le --mathlib
./blckbaux research prove '∀ x : ℝ, 0 ≤ x ^ 2'
```

`Forward/Parameters.lean` now enumerates every independent scalar field and
proves two-way conversions between the actual structured real weights and the
full Euclidean analytical parameter state. Cardinality follows symbolically
from the field types. Decoder aliases and their logical perturbations are
explicit. Generic Fin numbering is not a backend buffer-order proof.
`Forward/Regularity.lean` proves smoothness of actual exact-real primitives,
including joint weight/input paths through RMS and projection. Each tensor
coordinate is a smooth linear projection of the full parameter vector.
`Optimization/Accuracy.lean` uses Mathlib convex distance/Jensen to bound the
source clamped neighbor mean, retaining a floor-induced anchor-bias term.
`Optimization/Field.lean` describes the source-shaped frozen-basis affine random
field and derives mean/full covariance. Diagonal positive covariance requires
explicit balanced/unit/uncorrelated-sign and nonzero-innovation hypotheses;
these are not assumed consequences of the source hash or Gaussian kernel name.
The response-posterior proposal family was rejected rather than patched. The
source audit established that `round_condition` changes the corpus window,
condition seed, and evaluation seed every round, even when the incumbent is
unchanged. Its stored rows therefore did not denote repeated observations of one
stationary response. No common-law or transport theorem was present, while a
reset at every condition would leave no learned response before selection. The
response model, posterior proposal variant, response-radius posterior, Metal and
CUDA kernels, schema, tests, and `Optimization/ResponseField.lean` were deleted as one
mechanism. Paired patch differences remain measurements for acceptance and
reliability; they no longer train a cross-round response posterior.
Persistent patch-index posteriors were also removed because a patch index is
not a stable statistical identity across corpus slices or rollout counts.
Patches now reduce to one round mean/scatter statistic.
`PatchFeedback.meanPerm` and `PatchFeedback.energyPerm` prove that the retained
raw statistics are invariant under patch reindexing. These algebraic results do
not supply the still-missing common evaluation law.

The perturbation research track is recorded in
`../../docs/perturbation-research.txt`. `Optimization/Smooth.lean` derives
segment-local nonlinear bounds from Mathlib's multidimensional Taylor integral
formula. `Moments.lean` proves the full covariance/mean trace identity and its
library Gaussian specialization. `Search.lean` composes these with feedback
selection and executed-candidate error through an explicit round certificate.
`Rate.lean` derives finite-horizon simple-regret bounds from local progress
conditions, including the additive-error floor and a conditional zero-error
convergence result. These are general mathematical results, not established
rates for the deployed BLCKBAUX policy. `Optimization/Reachability.lean` now connects
finite proposal laws, candidate-specific KNN score envelopes, accepted updates,
and metric accessibility to that recurrence through per-round certificates.
It also derives the contraction factor from separate proposal-mass and
regret-fraction lower bounds rather than assuming a desired rate. Mathlib
covering numbers now impose both a local-regret certificate and a necessary
candidate-count/deadline constraint at a declared objective resolution.
An alternative local theorem needs no ambient-space cover: when a proposed
guide state's predicted score gap clears the incumbent and guide error bounds,
the guide is truly improving, and its own proposal-law weight is a valid local
mass floor. The resulting contraction factor is
`1 - guide_weight * retained_regret_share`. Establishing that retained share
for a history-conditioned KNN guide is now the principal estimator-alignment
obligation; it is not assumed to follow from the name KNN. A sharper sequential
theorem bypasses metric balls entirely: one accepted guide atom with certified
gain suffices. For a deterministic one-proposal transaction its weight is one,
so the factor becomes `1 - retained_regret_share`. The source weighted-neighbor
mean instantiates both forms from its explicit normalization and local-error
premises.
`Optimization/Information.lean` proves that no fixed proposal can retain any
positive fraction of regret uniformly over unrestricted objectives when an
unproposed state may contain the optimum. `Optimization/Guide.lean` therefore
states one explicit structured route rather than hiding that obstruction:
two-sided comparison between objective regret and distance in a task metric,
together with contraction of that distance by a history-selected proposal,
implies simple-regret contraction. The metric is not assumed to be ambient
Euclidean distance; choosing and proving a useful metric for pretraining remains
open. The module now constructs a target-directed perturbation explicitly as
`step * (anchor - incumbent)` and keeps the updated state as a separate object.
For the squared-distance test class, `squareProgress` gives the exact
derivative-free identity for one dense signed direction: progress is twice the
step times its inner-product alignment with `best - incumbent`, minus the
squared-step energy of the direction. Thus one evaluated candidate improves
whenever alignment clears that energy penalty; “one candidate” does not mean
that every parameter coordinate has a positive sign.
If the anchor has relative distance error `anchor_error`, a step in `[0,1]`
has certified distance factor `1 - step + step * anchor_error`; two-sided objective growth
turns this into regret factor
`upper_growth * distance_ratio / lower_growth`. Scalar KNN scores enter without
pretending to estimate an optimum vector: candidate coverage, pointwise score
error, and selection slack bound the distance of the selected parameter state,
which then becomes the anchor. Candidate-specific score accuracy and the
acceptance margin transport the metric theorem to the executable commit rule.
`Optimization/Information.lean` now also bounds finite screening: for any joint
candidate/acquisition law, selected expected nonnegative useful value is at
most the sum of the slot-wise expectations. Equal marginal upper bounds give
the candidate-count factor, without independence or a surrogate assumption.
This applies to improvement indicators and positive-part true gains. The work
corollary is a necessary condition in declared work units, not a deadline proof.
It does not assign small marginal gain to the pretraining objective or restrict
all adaptive generators to an unchanged base law. A bounded rejection prototype
was rejected as a standalone accessibility construction and removed from both
Rust and Lean; its general limitation, not an unused algorithm, is retained.

`Optimization/Interpolation.lean` studies a different, explicit construction:
interpolate two retained full parameter states using their observed loss gap
and a proved Mathlib `UniformConvexOn` modulus. If a nonnegative upper gap
`delta` is smaller than a certified modulus lower bound `B`, the coefficient
`(B-delta)/(2*B)` guarantees ideal gain `(B-delta)^2/(4*B)`. Radius clipping,
observation errors, conservative sampled-distance bounds, and execution error
are explicit. The quadratic family satisfies the imported property and attains
the exact gain in every inner-product dimension; no derivative or optimum is
an input to the coefficient. This does not establish the required modulus for
pretraining, reconstruct missing full history states, supply repeated useful
pairs, or give an end-to-end convergence or latency guarantee. It is not a
production algorithm: no Rust, backend, or TOML change accompanies this study.
Inspect with `./blckbaux research show interpolation`.

`Optimization/PairRotation.lean` now describes the actual additive parameter-pair
operation, separately from the matrix commutator view in `LieOperator.lean`.
Its pure rotation has positive quadratic energy drift; even interpolating or
extrapolating along that same chord cannot lower squared parameter norm. That
is not a statement about neural loss or the noisy accept/reject optimizer.
More constructively, the real operation has an explicit inverse for every
recorded coefficient choice. It forms a Mathlib `Equiv`; `Equiv.piCongrRight`
assembles all complete parameter pairs, and fixed-trace compositions inherit
inverses. The exact inverse contracts forward execution-error energy by
`1/(1+(scale*rotation)^2)`. It therefore supplies a concrete approximate-history
reconstruction route without assuming objective curvature or an accurate
optimum anchor. Backend reverse arithmetic, accumulated rounding, trace
retention, unpaired tails, and replay cost still need implementation accounts.
No exact quantized reversibility or seed-only weight compression is claimed.
Inspect with `./blckbaux research show inverse` or
`./blckbaux research show 'pair rotation'`.

`Forward/Sensitivity.lean` retains learned
coefficient and branch changes rather than treating fixed transport stability
as full-block contraction. It also proves `updateEntry` for entrywise exact mHC
update movement and `scheduleBound` for multi-layer forward perturbation
propagation across arbitrary layer schedules via `List.foldl`.
`Forward/SinkhornRatio.lean` bridges `SeedSensitivity` / `MixingSensitivity`
log-seed bounds and `Residual.energy`: floored row and column normalizations
each square a common multiplicative entrywise ratio without dividing by the
`1e-20` denominator floor (`rowRatio`, `columnRatio`, `stepRatio`,
`sinkhornRatio`), so finite Sinkhorn transport differences are controlled by
the multiplicative ratio and the incumbent matrix's own row/column residual
(`energyDiff`, `sourceDiff`). `Forward/NormSensitivity.lean` adds the tight
incumbent-conditioned RMS-state movement bound `incumbentBound`, proving
`gridEnergy (normState second - normState first) <= 6 * stateScale(first)^2 * gridEnergy (second - first)`
by canceling the candidate activation energy against the candidate RMS
denominator in `scaleRadius` (`scaleTerm`), avoiding the worst-case `200000`
(`2 / eps`) factor. `Optimization/EnnCommit.lean` connects `Sensitivity.scheduleBound`
to `logitBase` in `scheduleBase`, `Optimization/PretrainRegret.lean` connects
`EnnCommit.scalarCommit` (with `Neighbor.mean` and its denominator floor) and
`LossGeometry.trustRange` to `MarginalWindow.upperTail` in `sourceTail`, and
`Optimization/PretrainHorizon.lean` connects the multi-transaction regret
horizon to `PretrainWorkload.tokenClock` and `PretrainWorkload.LossGoal` in
`goalBound`. Source geometry, routing, calibration, precision, memory, and
machine-cost obligations remain explicit.

This Lean project uses Mathlib to separate proved optimizer facts from explicit
assumptions, measurements, and conjectures. `BlckbauxResearch/Foundation.lean`
defines the full model-parameter proposal space and the fixed research targets.
Compressed descriptors are separate types and cannot silently replace the full
parameter proposal.

The exploratory architecture uses three names. A `Core` is an operational
definition of the object under study. A `View` is a mathematical construction
used to expose selected structure; different views need not agree, reconstruct
one another, or be chosen as the single correct account. A `Finding` is an
ordinary checked definition, theorem, or counterexample produced through a
view. Findings are deliberately not wrapped in a generic proof record. Any
claim that a finding applies to executable Rust, Metal, or CUDA receives its
own hypotheses and proof at the point of use.

The implementation-grounded mathematical blueprint is
`../../docs/optimizer-formalization.txt`. It maps observation construction,
fitted neighbor uncertainty, sampled proposal geometry, regret, and resources
to explicit proof dependencies and open executable-refinement obligations.
`Objective/Response.lean` uses Mathlib's measures, covariance, square-integrable
random variables, and Gaussian distribution. It checks weighted covariance,
paired/baseline variance, shared-noise cancellation, fixed-world anchor
cancellation, and the residual when evaluation conditions change. Its concrete
counterexample has zero patch scatter but unit population mean variance for
any positive patch count. None of these results asserts an BLCKBAUX regret rate,
calibrated deployed uncertainty, floating-point parity, or a timing guarantee.

The broad forward-program track is `BlckbauxResearch/Forward/Program.lean`, with
component modules for algebra, encoding, transport, mixing, attention, fresh
selection, experts, and decoding. It composes explicit real-arithmetic operations
across the model; it is not yet a complete verified GPU implementation.
`../../docs/forward-formalization.txt` records source provenance, checked
properties, and missing paths. In particular, the generative forward view models
autonomous student token rollouts without teacher forcing: the generator evaluates
real sampled token trajectories directly. Quality and work modules connect conditional

objective approximations to regret and expose projection counts, separately from
full machine costs. Birkhoff decomposition, exact transport energy bounds, and
approximate row/column mass bounds are checked without pretending finite-step
Sinkhorn already produces an exactly doubly stochastic matrix.

`Optimization/Neighbor.lean` defines the current bounded neighbor estimator's
real-arithmetic ordering, adaptive prefix, prediction, supplied LOOCV parameter
selection, append, and compaction. Its library view is Mathlib's
`Finset.centerMass`: convex-set membership and Jensen bounds are imported from
general convex analysis. Ordinary normalization requires the denominator floor
to be inactive; the floor also has a checked zero-anchor barycenter view.
`Optimization/Fitting.lean` proves that the source-shaped fitting fold
maximizes its sampled LOO score over the supplied finite parameter trials.
For the average criterion, the target-score deficit is at most trial-coverage
error plus twice the criterion error. Those errors are not proved to vanish.
The existing Appendix C proofs are in the separate local project
`/Users/mehulbafna/research/yubo/yubo-pixi-env/formalization/enn-lean/`, with
`ENN/Paper/AppendixC.lean` as the theorem index. They already supply conditional
MSE, random-design bounds, consistency/rate transfers, parameter stability,
argmax convergence, and measurable exact/approximate selectors. They are not
currently a dependency of this Lake project. The finite source correspondence
must not be confused with establishing their statistical hypotheses.

`Optimization/Calibration.lean` connects the actual inverse-precision weighted
mean to total precision W. If the objective is L-Lipschitz, selected observation
errors are bounded by epsilon, and supplied dissimilarities dominate squared
task-metric distance, its absolute error is at most
`sqrt(k * (L^2 / epistemic + epsilon^2 / aleatoric)) * sqrt(1 / W)`.
The fitted scales must be positive and the mean's denominator floor inactive.
Mathlib Jensen and Cauchy-Schwarz derive this pathwise bound; it does not require
fitted weights to be independent of the responses. It does not certify the
configured acquisition multiplier or the sampled source geometry.

The generalization `Calibration.sourceRemainder` preserves those actual weights
when squared metric distance is bounded by `sampled_distance + remainder`.
Writing `C = L^2/epistemic + epsilon^2/aleatoric`, total precision `W`, and the
source-weighted average remainder `Rbar`, its certificate is
`absolute_error <= sqrt(C * (k/W + epistemic * Rbar))`.
This extra term is an upper-bound penalty, not a proved actual-error floor.
`Optimization/SampledEnn.lean` composes it with stratified quadrature and exposes
a certified cached selection row. Its weighted Euclidean view includes every
parameter and has a checked inverse for positive tensor weights; it does not
replace the physical objective by a latent objective.

The same module proves retrieval distortion for the actual source sort: if
squared-distance errors are at most R and k eligible points lie within true
radius r, selected neighbors lie within `sqrt(r^2 + 2*R)`. The source weighted
mean then has error at most `noise + L*sqrt(r^2 + 2*R)` under its existing
observation, positivity, regularity, and normalization conditions. Neither the
sort nor the prediction uses substituted corrected distances.

`Quadrature.roundedError` also propagates an explicit executed-versus-ideal
displacement cap into the quadrature residual. It does not establish that cap
for the backend BF16 converter. Tight residuals, actual proposal/sample
agreement, self-tuned distance scaling, corpus-error laws, and complete forward
regularity in this geometry remain source/mathematical obligations.

`Optimization/Retention.lean` gives a matching kind of information obstruction:
exact observations on a retained set cannot uniformly predict an L-Lipschitz
objective with error smaller than L times distance to that set. Mathlib's
Lipschitz extension theorem also constructs two regular objective worlds whose
archives become identical under the actual `Neighbor.compact` map, although
their better next updates differ. This concerns the archive, not every item of
optimizer state or an established pretraining lower bound.

`Optimization/LocalMemory.lean` constructs an alternative statistic: the latest
k observation identifiers per cell of a supplied finite metric cover. The
bounded-state update equals the complete-history specification and stores at
most k times the cell count, independent of training horizon. `KnnLearning.lean`
connects those retained counts to the source distance/index sort, explicitly
including squared dissimilarities. It proves a conditional finite-horizon gap
bound `2 * noise + 2 * L * cover_radius + selection_slack` once query count exceeds
k times the cell count. This uses optimistic selection, local recall, and a
terminal recommendation no worse than the observed minimum up to noise; it
does not assume a desired contraction rate. Cover construction, cell lookup,
state representation, acquisition search cost, and source adoption remain open.
The present source compaction is not silently assigned this statistic's recall.

`Optimization/RecentMemory.lean` analyzes the current implicit archive's
oldest-non-incumbent eviction instead of proposing another retention mechanism.
Its ID-set trace has capacity at most H, protects a retained incumbent, and
retains the latest H-1 observation IDs at every horizon. Thus k < H guarantees
retention of the latest k observations once those observations exist, without
a supplied metric cover or a hypothesis that the archive already has local recall.
The protected ID may follow any adaptive accepted/rejected sequence.

Mathlib's polygon inequality then accounts for intervening incumbent movement.
If realized trial-to-incumbent distances in the relevant window are at most R,
and each commit either stays or accepts that trial, the latest k trials lie
within (k+2)*R of the next trial. With coherent slots and squared-distance error
at most G, the existing source sort and mean give absolute prediction error at
most `noise + L*sqrt(((k+2)*R)^2 + 2*G)` under the stated observation, Lipschitz,
positivity, and normalization-floor conditions. R is a full-state metric
distance, not the TOML radius coefficient. This proves temporal recall and a
local geometric consequence; useful proposal gain, corpus-noise bounds, tight
sampled-distance error, changing-metric transport, backend slot refinement, and
terminal regret remain open. No Rust program or experiment changes.
The trace covers ordinary archive insertions, not explicit compaction or resets.

`Optimization/KnnDecision.lean` constructs the finite optimistic selector rather
than assuming an acquisition minimizer. It scans cached `(state, mean, error)`
rows, returns one state, refines Mathlib's stable `List.argmin`, and counts
exactly N-1 scalar comparisons for N cached candidates. The history minimum can
be maintained incrementally. Source mean and inverse-precision certificates
instantiate the cached rows. A new horizon theorem derives optimistic selection
from that scan, with an explicit candidate-coverage penalty; `stopBound` gives
a terminal certificate from an upper bound and the smallest optimistic score.
This is a selector construction, not a complete deployed policy: candidate
generation, score computation, coverage, retained-data provenance, observation
accuracy, and machine cost remain obligations. N is symbolic, not a new four-
candidate configuration. For N=1 the selector cannot improve the generator.

`Forward/LossGeometry.lean` derives actual token-objective regularity without
derivatives. Mathlib exp/log order laws prove coefficient-one control by
target-relative logit discrepancies and coefficient-two control by uncentered
logit discrepancies, independently of vocabulary size. Aligned token averaging
preserves the corresponding average bound. `logitLipschitz` is the Mathlib
sup-metric Lipschitz object; `parameterBound` composes it with a supplied
parameter-to-logit bound. It does not prove that bound for the whole model or
equate different corpus windows or autonomously sampled trajectories.

`Optimization/Bayes.lean` uses Mathlib's existing posterior-kernel construction
by disintegration and inherits its joint-law identity. It does not assert that
the fitted ENN predictor has that posterior law. `Optimization/Cost.lean` uses
Mathlib `IsTheta`/`IsBigO` along explicit filters for logical array payload and
candidate writes. Sorting and full latency are not hidden in a scan count.
These modules do not complete the entire source optimizer state transition.

The joint algorithm-and-proof experiment starts in
`BlckbauxResearch/Optimization/Regret.lean`, `Synthesis.lean`, and `Witness.lean`.
`Regret.simpleRegret` measures the terminal recommendation's true objective
gap to a comparator. Global optimality is an explicit proof about the declared
state type; the objective and its optimum are unavailable through the policy
interface. The existing selector bound composes with a coverage bound to give
simple regret at most `coverage + 2 * error + slack`.

The operational `Core` is a policy with initialization, queries, feedback
updates, and a final recommendation. Feedback has its own type, allowing a
computable implementation while real-valued objectives remain in the analysis.
The finite conditional oracle law can depend on time, retained memory, and the
queried state. This covers dependent observations within that model; proving
that retained memory determines the actual conditional law is a separate
obligation. The first query is deterministic; fresh policy randomness needs an
explicit input or retained generator state. No posterior or calibration property
is inferred from the policy interface.

`Optimization/Process.lean` now realizes that existing finite policy/oracle as
a Mathlib Markov transition kernel, with measurability hypotheses explicit.
`Kernel.traj` supplies the complete memory trajectory. `prefixLaw` and `marginal`
prove that every finite marginal equals the existing terminal recursion;
`pathIntegral` identifies its real terminal-gap expectation with
`Synthesis.expected`. `pathBound` transports the existing policy certificate to
that trajectory measure. The same policy and oracle determine every horizon.
This is a correspondence of the current mathematical models; identifying the
Rust execution and its actual conditional observation law remains open.

`Optimization/Statistic.lean` uses Mathlib decision theory on a fixed experiment.
`decisionRisk` identifies a decision after compression with the same composed
rule on the original data. `riskSplit` separates excess Bayes risk into decision
error and information lost by compression, using nonnegative extended-real
subtraction; finite risks are needed to read it as ordinary real differences.
`decisionBound` composes bounds on those two terms without subtraction. These
comparisons hold the observation experiment fixed. A different adaptive policy
changes that experiment; library Bayes risk over all terminal decisions does not
automatically optimize feasible training policies or respect machine budgets.

The certificate `View` consists of a terminal-regret potential and a work
credit. An initial bound, a one-step conditional expected potential inequality,
and terminal domination imply the expected simple-regret bound. A separate
one-step credit inequality bounds the sum of declared work charges along every
feedback path. A `Task` fixes a nonempty objective/oracle class, horizon, regret
target, and work budget. A `Solution` supplies one policy and certificates for
every environment in that class. Bounds and assumptions belong to the task,
so the policy's proof cannot silently choose an easier objective class.
Candidate policies and their runtime dependencies must compile as ordinary Lean
definitions; a noncomputable minimizer is not an executable search result.
Potentials and mathematical certificates may use noncomputable analysis.
Kernel checking must be accompanied by an axiom audit: unfinished proofs and
new axioms cannot discharge a candidate's obligations. Revising an objective
class or target is a separate research proposal, not a successful search result
for the original task. Failure to find a proof is not a proof of impossibility.

The initial checked `Finding` uses two Boolean states, rational observations,
and two evaluations. Its policy compares the recorded observations; its
certificate proves simple regret at most twice a uniform observation-error
bound. The result holds on every realized path as well as in expectation, and
the certificate also proves at most two query work units. Exact observations
give zero regret for every objective in this finite class. Checked
counterexamples reject a constant recommendation and show that a noisy tie can
attain the full error penalty. Both score orderings are evaluated by the actual
computable policy during the build, and the new result dependencies are printed
for an axiom audit.

This is the first synthesis experiment, not a BO discovery engine or an BLCKBAUX
pretraining guarantee. Rational memory is not a bound on physical storage;
query work units are not FLOPs or milliseconds. A shrinking regret rate,
finite-precision BLCKBAUX instantiation, noise/geometry assumptions, memory bound,
and implementation/machine cost refinement remain research obligations.

`BlckbauxResearch/Optimization/Perturbation.lean` is an abstract perturbation model.
Its core maps a source value to an update for every model parameter. Its energy
view observes the sum of squared parameter updates. Lean proves that a signed
field with fixed per-parameter scale `s` has energy `d * s^2` for parameter
count `d`, and that scale `r / sqrt d` gives exact energy `r^2` when `d` is
positive. The resulting square-root scaling is a mathematical candidate, not a
claim of objective improvement, convergence, or current source behavior. The
module's linear-functional lemmas are generic algebra and do not reinstate the
deleted response posterior or prescribe mirrored source proposals.

Earlier documentation described `Optimization/Latency.lean` and
`Evidence/Latency.lean`, but neither module is present in this checkout. Those
descriptions must not be counted as currently kernel-checked latency results.
The historical T4/A100 runs and their workload-specific timings are recorded in
`../../docs/optimizer-research-handoff.txt`. They are not universal hardware
bounds, and a stage timing is not a complete optimizer transaction. Current
source changes, including fixed-task incumbent caching, require separate
execution accounting. The blueprint keeps symbolic work, executable refinement,
machine envelopes, and the 200 ms/1000 ms targets as distinct obligations.

`BlckbauxResearch/Representation/Blocks.lean` separates a parameter block layout,
blockwise scaling, a fixed block-coefficient proposal family, and a seeded full
parameter field. A layout only assigns parameters to execution or scaling
blocks. A fixed family with one coefficient per block is proved to stay inside
the span of its block fields, even if each materialized proposal is a dense
parameter vector. A seeded field instead returns a complete parameter proposal;
its seed size is not renamed as parameter dimension, and activity of every
coordinate is not renamed as joint reachability.

`BlckbauxResearch/Representation/RecordedCuda.lean` encodes the source-audited
fixed-field representation from the completed T4 run: 70 buffer coefficients
against persistent dense fields. Conditional on the Rust/CUDA map refining that
`BlockBasis`, Lean proves its real proposal span has rank at most 70 and strictly
less than the 1,177,756,440-dimensional full proposal space. This does not claim
that FP16 rounding preserves the affine span or complete the Rust refinement.
The same module records the audited execution counts separately: 40,000 scored
coefficient proposals, 10,000 materialized and candidate-evaluated proposals,
10,000 adaptive rounds, and 20,001 complete objective evaluations after counting
the initial baseline and each round's paired incumbent evaluation. The measured
mean complete round time was 1202.43794 ms, which exceeds the 1000 ms ceiling and
six times the 200 ms operating target. These are properties of the recorded run,
not universal CUDA performance claims.

The obsolete `Optimization/CurrentPool.lean` model was deleted. It encoded a
legacy one-arm pool as two direction streams crossed with two radii. Historical
operator-BO examples use four arms with one slot each, but the active 1M-token
pretraining contract evaluates one complete candidate per round with `k = 10`.
Any new reachability theorem must be parameterized by the resolved proposal
configuration and proved against the source seed and perturbation map.

`BlckbauxResearch/Precision/BFloat16.lean` defines an alternative finite BF16
execution state and an abstract rounding contract. It is not the current
full-parameter Metal pretraining target. That source path uses FP16 through
`SearchState::new_implicit` and `FP16_INDEPENDENT`; the recorded CUDA runner
also materializes FP16 proposals. `BlckbauxResearch/Precision/Float16.lean` gives
FP16 its own word, finite-state, rounding, commit, and stored-effect contract.
`BlckbauxResearch/Precision/Discrete.lean` eliminates continuous ℝ^D embeddings
and post-hoc quantization by formalizing native discrete proposals directly
on the IEEE binary16 lattice. Under its explicit `Transition.applyActive`
contract, every active coordinate step changes a stored word. The current
Metal generator does not implement that discrete transition merely because
this theorem exists. Neither abstract quantizer is yet a claim of backend
parity. Full real dimension therefore does not silently imply that an update
survives either format's quantization.

`BlckbauxResearch/Optimization/Step.lean` begins the optimizer theory with an exact
one-step decomposition. It introduces no distributional or smoothness
assumptions: every later guarantee must state those hypotheses explicitly.

`BlckbauxResearch/Optimization/Target.lean` upgrades policy synthesis from one
abstract work counter to a complete machine resource vector. A machine solution
must certify expected terminal regret for every environment, cumulative resource
feasibility along every feedback path, and the wall-time bound of every round.
Its separate active envelope fixes the present research target at 1,177,756,440
parameters, 1,048,576 context tokens, 1,000 rounds, and 200 ms; the generic
synthesis definitions remain hardware-independent.

`BlckbauxResearch/Optimization/ObjectiveSearch.lean` gives the derivative-free
proposal bridge in metric and objective language. A nearby candidate retains a
reachable gain up to a declared value modulus. Simultaneous pool-score error
then transfers that gain to the selected member with the exact penalty
`2 * error + slack`. For stochastic generation, the probability law is over a
round seed that deterministically generates the whole dependent pool; it is not
over a uniformly chosen slot. `seedBound` turns the probability that any slot
hits the useful neighborhood into expected selected progress, provided rejected
or missed rounds cannot increase the objective. The separate uniform-slot result
is only a baseline and is not the deployed acquisition law.

`BlckbauxResearch/Optimization/Selection.lean` proves the selector contract used by
one optimizer transaction. The executable transaction uses the implementation's
gain orientation: positive values mean predicted objective decrease and the
selector maximizes acquisition. Simultaneous acquisition-to-realized-gain error
`epsilon` and approximate-maximization slack `xi` bound candidate-pool selection
regret by `2 * epsilon + xi`; opportunity larger than that penalty forces improvement.
The score may be a Thompson draw, UCB score, or another selection function—the
theorem does not silently identify it with a posterior mean. A second theorem
proves that observing only the selected candidate's realized objective cannot
identify this simultaneous error bound: an unobserved pool member can agree
with the estimator or violate any proposed nonnegative bound while preserving
the selected observation.

`BlckbauxResearch/Optimization/Batch.lean` separates raw pool size from effective
coverage and execution depth. Its general result accepts any certified upper
bound on the probability that every candidate misses a target; approximate
selection then reaches the penalty-shifted target with at least one minus that
bound. A second theorem specializes the penalty to the existing selector law:
terminal simple regret is at most `target + 2 * error + slack` with the same
probability. The benchmark corollary makes the comparison margin explicit: to
meet regret `R`, the pool must cover `R - (2 * error + slack)`. Under the
additional Mathlib `iIndepSet` hypothesis and common miss probability `q`, this
specializes to joint miss probability `q^N` and hit
probability `1-q^N`. Independence is not claimed for BLCKBAUX. The same module proves
that duplicating one candidate any nonzero number of times leaves the miss event
unchanged, and that a deadline-feasible schedule must satisfy
`candidate_count * evaluation_time <= batch_width * deadline`. Thus widening a
logical pool helps only through its joint coverage law, while hardware batching
controls how much of that pool fits within 200 ms or 1000 ms. The current four
independent seed streams still do not license a `q^4` claim: seed distinction is
not a proof that the objective-relative miss events are independent.

`BlckbauxResearch/Optimization/Transaction.lean` packages one precision-neutral
semantic transaction: full real parameter proposals, a backend commit model,
executed candidates, objective and acquisition values, candidate selection,
strict acceptance, and machine resource use. BF16 and FP16 provide separate
commit-model adapters. Its progress theorem is about committed backend states,
so an ideal proposal cannot receive credit for an update erased by quantization.
The pool states are counterfactual realizations; Metal may score its procedural
pool and materialize only the selected row.
If the selected executable state fails to improve while the simultaneous error
bound holds, a checked converse diagnostic bounds the chosen comparison's pool
opportunity by `2 * epsilon + xi`. This identifies the quantities that must be
measured to distinguish weak proposals from inaccurate or slack selection.
The recorded run and the current runner expose only the selected candidate's
realized objective, so they do not yet supply that diagnostic's pool-wide error
hypothesis. Certifying it requires either counterfactual pool evaluations on an
explicit audit schedule or a theorem deriving the bound from stated estimator
and objective assumptions.

`BlckbauxResearch/Optimization/Native.lean` formalizes an optimizer transaction
operating directly on native FP16 states, executing candidates and committing
accepted updates bit-for-bit without unquantized latent centers, rounding
discrepancies, or quantizer loss.

`BlckbauxResearch/Optimization/SeedSupport.lean` records the deterministic
fixed-seed boundary: after a completed ask, useful-move mass is zero or one,
and no finite pool guarantees improvement over every unrestricted objective.
`ActiveEnergy.lean` uses Mathlib covariance and Markov's inequality to show
that an idealized centered isotropic law assigns an active parameter set its
cardinality share of expected total perturbation energy. Its `scaledMean`
theorem gives the source-shaped pre-rounding second moment with per-parameter
scale and metric weight, rather than assuming the m/D share persists after
block scaling. `activeShare` proves that zeroing block scales outside an active
set (`routedScale`) concentrates 100% of expected pre-rounding weighted energy
onto the active set, while `deadzoneZero` and `roundingGap` connect
`Float16.deadzoneStep` to post-rounding weighted energy vanishing and quadratic
rounding perturbation. `EnnCommit.windowError` and `windowCommit` prove that
centering archive observations against a window baseline whose shift tracks the
incumbent shift within `delta` cancels common window difficulty drift from the
unpaired ENN incumbent estimate and scalar commit. Neither result asserts that
the source generator has a randomized isotropic law or proves pretraining regret.

`Forward/Experts.lean` proves that, for one fixed input, arbitrary replacement
of an unselected routed expert leaves that branch output unchanged when the
router and shared expert are fixed. The long Metal run already counts changed
FP16 words and rejects a selected non-DUST row that changes none. Neither fact
identifies which changed words influence the complete 1M-token objective;
that requires a forward-dependency argument across all token positions and
feedback passes.

The current Metal correspondence is direct but not yet proved: candidate
generation and `begin_ask` supply proposals; the Metal selection kernel supplies
the acquisition values and selected index; `encode_row` materializes the winner;
the external harness supplies its objective observation; `tell_absolute` and
`commit_weights` perform acceptance and incumbent update; ask/tell profiles feed
resource accounting. Binary64 acceptance now has a checked source
correspondence: Aeneas extracts the production Rust comparison, and
`Semantics/Optimizer.lean` proves it equal to an independent bit-level
specification for all input words, including NaNs and signed zero. Remaining
refinement proofs for the current long run include FP16 conversion parity,
counterfactual proposal identity, acquisition-score direction and uniform error,
and complete transaction timing. CUDA requires a separate
refinement proof against its own FP16 path. The BF16 alternative remains a
distinct, inactive semantic contract for this run.

The complete transaction has a `200 ms` operating target and a distinct
`1000 ms` ceiling. Both refer to end-to-end measured wall time, including
proposal construction, acquisition, selection, synchronization, and state
commit; neither may be discharged with a kernel-only timing.

`BlckbauxResearch/Optimization/ResearchProblem.lean` defines the broader search
target before choosing an algorithm. It distinguishes state equivalence,
objective equivalence, outcome dominance, resource dominance, and
machine-relative efficiency. A coding harness or benchmark evaluation carries
its full measured cost; its evaluation count is not treated as compute cost.
Comparable run results now share the same objective construction, evaluation
world, initial state, parameter count, and machine by type or proof. Their final
score is proved to be the declared construction's score, and the construction's
evaluation cost must be included in total usage. `SearchCount` separately records
acquisition-scored proposals, materialized parameter states, objective-evaluated
proposals, and adaptive sequential rounds, exposing batching rather than hiding
it inside a candidate count.

`BlckbauxResearch/Optimization/Information.lean` proves the first boundary result:
for an unrestricted objective, one observation cannot order two unqueried
parameter states. Positive one-query results must therefore state the objective
or representation structure that supplies the missing information.

`BlckbauxResearch/Semantics/ModelTheory.lean` starts the actual first-order
model-theory layer from Mathlib. Its domain elements are complete optimizer
trials, not neural-network models. A language names outcome dominance, compute
dominance, the 200 ms target, the 1000 ms ceiling, an explicit proposal-scope
predicate, and the claimed Muon comparison. Its theory forbids calling a trial a Muon beat
unless those obligations hold. Concrete Rust, Metal, CUDA, simulator, and
mathematical constructions can become separate structures for this same
language. A purported theorem is universal only when it holds in every model of
the stated theory; an admissible countermodel identifies a missing assumption.

`BlckbauxResearch/Objective/MultiObjective.lean` defines vector losses,
constraints, Pareto dominance, and weighted scalarization. It proves that
nonnegative weighting preserves Pareto weak dominance and gives a checked
two-objective counterexample showing that a better weighted score does not imply
a Pareto improvement.

`BlckbauxResearch/Objective/Construction.lean` separates generation, raw loss
measurement, aggregation across tasks and randomness, and the observation
exposed to the optimizer. It leaves expectation, quantiles, CVaR, worst-case
aggregation, scalarization, and observation compression as explicit alternatives.

`BlckbauxResearch/Objective/Aggregation.lean` defines explicit finite harness
batches, vector empirical means, and population expectations. It proves that the
empirical vector is unbiased when every sample coordinate is integrable and has
the same expectation. Independence is deliberately not assumed for unbiasedness.

`BlckbauxResearch/Objective/Concentration.lean` adds independence and sub-Gaussian
centered-loss assumptions to obtain a checked one-sided empirical-mean tail
bound. These hypotheses are not retroactively attached to arbitrary harness
rewards; they must be justified by a concrete evaluation construction.

`BlckbauxResearch/Objective/Adaptive.lean` replaces sample independence with an
explicit filtration and conditional sub-Gaussian obligations. It permits the
next task or scoring slice to depend on earlier observations, but it does not
assume that the resulting adaptive centers equal one fixed population objective.
An exact decomposition exposes center drift, and a checked transfer theorem
charges any uniform drift bound directly in the fixed-target tail threshold.

The project also encodes the live two-author research protocol used to develop
the BLCKBAUX optimizer research agenda. Natural-language statements remain readable,
while Lean checks the dialogue structure and formal propositions attached to it.

The protocol distinguishes questions, proposals, paraphrases, confirmations,
arguments, predictions, tests, observations, revisions, and resolutions. It also
keeps assumptions, derivations, Lean proofs, source facts, measurements, external
sources, and conjectures distinct.

Run the research checks:

```sh
./blckbaux research status
./blckbaux research check
./blckbaux research check --via-build
BLCKBAUX_BUILD_SYSTEM=bazel ./blckbaux research check --via-build
./blckbaux research audit
```

`status` runs the checked Lean executable and reports established results,
unresolved obligations, and the central conjecture.
The ordinary check reuses Lake's pinned local dependency cache for interactive
research. `--via-build` routes the Lean library and interpreted target report
through the selected Buck2 or Bazel target, then uses that target's output for
the typed Rust/TOML contract check. It does not rebuild Lean in a second
checkout or link the entire Mathlib closure into an executable. The old
`--isolated` and `--clean` spellings remain accepted aliases.
Both build-system targets currently invoke host Lake, so this path is not a
hermetic Lean toolchain build. The build-system path obtains Mathlib's pinned
precompiled cache in its temporary checkout before compiling BLCKBAUX; it requires
network access or a previously populated Mathlib cache. `audit` performs the
ordinary check, then uses Lean 4.35's `lake check --paranoid` to recheck
exported proofs with the shipped independent kernels. The audit strengthens
confidence in proof checking; it does not discharge source-refinement or
empirical assumptions.
Lean 4.35's audit requires Linux bubblewrap. On macOS the command fails
immediately with that requirement; it does not silently disable the sandbox.

`research check` also enforces the repository naming policy on BLCKBAUX-owned Lean
declarations and file names: at most two semantic words, one underscore, and 24
characters per qualified-name segment. Qualified namespaces carry context, as
in `Run.sameState`; imported Mathlib declarations are not scanned.

`BlckbauxResearch/Agenda.lean` contains the current dialogue. New research moves are
appended rather than replacing earlier claims. Mathematical optimizer claims
belong in dedicated modules imported by `BlckbauxResearch.lean`; the dialogue should
reference those declarations rather than treating prose as a proof.
