BLCKBAUX semantics

This crate is a direct dependency of blckbaux, blckbaux-cuda, and
blckbaux-cuda-device. Its default build has no external dependencies. The cuda
feature adds DeviceCopy implementations for the same repr(C) records; enable
it only in a CUDA build with the required toolkit.

Ownership

graph, tensor, model: operation names, typed dataflow, layouts, lifetimes and
model functions. Validation checks structure and references, not numerical
equivalence or execution of every declared operation.

recurrence: layer-visit order shared by Metal and CUDA dispatchers.

hnet, shape: fixed-patch H-Net dimensions, checked wave construction, sampling
identity, and scalar buffer records. The hnet contraction count is partial;
it is not a latency prediction or a complete FLOP count.
hnet::probe supplies one finite numerical readout fixture to both production
backends, including a scalar reference and exact token-decision checks. It
does not replace the full forward program or prove all-input equivalence.

abi: one field list produces Rust records, CUDA DeviceCopy implementations,
Metal declarations, and Metal compiler checks of Rust sizes and offsets.
Metal compilation must use hnet::metal_source; never redeclare those records
in shaders. A record change requires rebuilding both host and device code.
This includes EncodeShape used by the actual encoder/transpose dispatchers,
not just WaveShape and ShardStats.

PairShape and ObjectiveStats likewise have one ABI, including the octile array.
ObjectiveStats::estimate owns the host reduction interpretation used by both
backends: per-token likelihood, empirical patch-mean variance, aligned paired
token differences and codelength. That empirical variance is not a proved
concentration bound. The objective fixture includes gains below rounded
patch-mean resolution and partial last patches.

CausalPlan owns the preceding-patch boundary and scoreable-label interval.
CorpusSet and CorpusBundle in blckbaux-artifact own corpus sampling, held-out mixture
weights and artifact relocation. CUDA's causal mode uses that same loader;
sampled token predictions are not fed into the scored corpus state.

Initialization owns recurrent initialization scales, dimensionless proposal
scales and static family multipliers. The column initializer owns Gaussian
draw order and FP64 column normalization, while callers supply RNG and FP16
conversion and retain their own allocation/layout. CUDA's tied embedding is a
physical transpose of the readout, not a second logical parameter family.

neighbor: streaming precision-weighted prefix selection used by CPU callers
and CUDA kernels, resident observation capacity and incumbent-protected
retention. neighbor::query is the FP32 host reference for Metal posterior and
acquisition arithmetic. Query-local adaptive k and global likelihood fitting
are independent choices. Total predictive error is not UCB's epistemic error.
Metal still has separate shader arithmetic; finite device fixtures are not a
proof of equivalence on every floating-point input.

random: wrapping SplitMix64 and the host normal-metric stream used for
identity-conditioned Thompson draws. gaussian owns the single Ziggurat table
construction consumed by both Metal source generation and CUDA uploads.
These definitions do not assert statistical independence of different streams.

proposal: the common publication rule for full materialized parameter rows:
reject nonfinite candidates, and reject unchanged candidates unless the
specific perturbation explicitly permits them. FP16 word classification is
shared with CUDA device code. Metal derives flags from its full-row partials;
CUDA fuses flags into Gaussian materialization and reduces them on the device.
Sampled neighbour geometry is not a substitute for this full-row check.

context: cache layout, range checks and shared numerical device fixtures.

muon: existing scalar ABI, coefficients and CPU reference for the empirical
baseline. It neither supplies a model backward pass nor a Lean comparison.

schedule: backend-specific kernel catalogues over the shared operation graph.
Coverage means declared coverage only. These catalogues do not replace the
actual dispatchers and are not evidence of numerical or performance parity.

Backend-independent optimizer state, acceptance, calibration, and radius control
belong here. Surrogate fitting, accelerator allocation, launch geometry, fusion,
synchronization and optimized tensor kernels belong in their owning modules.
Put genuinely shared rules here instead of using cross-crate #[path] imports.
Do not force distinct operators or numerical approximations into one name.

Verification

./blckbaux test includes semantics-unit and the Hegel properties. The latter check
recurrence order, independently recomputed neighbour-prefix decisions, integer
stream fixtures and on-device Metal ABI field reads, alongside optimizer-state
properties. Production H-Net Metal compilation is also a default unit test.
optimizer::SeedReuse is shared by the natural-corpus loops; its Hegel trace
checks the prior successful-root/retry policy independently. CUDA model-check
also exercises corpus-condition replacement, graph reuse and loss-capture
reset with uniform fixture weights. This lifecycle fixture checks no learning.
CUDA compilation and device comparisons require the NVIDIA environment; host
or Metal tests are not substitutes. No general floating-point error bound,
whole-optimizer parity, learning guarantee or achieved latency follows from
this package boundary.
