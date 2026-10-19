# Tests & Quality Gates

Run hermetic test suites, formatting checks, and codebase gates through `./blckbaux`.

## Source Contracts

Read each module being changed in full before editing it, including its local
tests. Follow callers and callees when a contract crosses a module boundary;
search results locate code but do not replace understanding the module.

Owned Rust, Metal, CUDA, and Lean source should document the scientific role of
each module and the non-obvious contract of each function. State the operation,
input/output units and shapes, preconditions, state or buffer changes, completion
and failure behavior, and relevant callers. Include only what applies: avoid
boilerplate that merely repeats a function name. Put the mathematical operation
next to its implementation and identify where configuration selects it.

Distinguish generated proposals from evaluated candidates and accepted updates;
measured objectives from predictions; sampled geometry from full parameter rows;
requested latency from measured latency; and proved properties from assumptions.
For Lean results, name the actual hypotheses and scope rather than describing a
conditional result as a guarantee of the deployed optimizer. Imported/generated
source is not part of this documentation pass.

Keep contracts current in the same change as behavior. A change to a proposal
formula, objective protocol, default, or acceptance/controller policy is an
algorithm change, not a documentation fix or behavior-preserving refactor. State
that change and agree its scope before implementation. Retain the source/config
identity of the comparison run rather than substituting another historical run.

For documentation-only changes, verify that executable source is unchanged and
run the appropriate format/build/test checks. Do not launch training merely to
validate comments. Documentation aids navigation; it does not replace checks of
the implementation or establish a mathematical guarantee.

## Quick Execution

The default Buck2 suite includes the optimizer, runtime, CLI, development tools,
and platform integration tests. Hegel properties live in
`rust/crates/blckbaux/tests/hegel_prop.rs` and run as part of that suite. Use generated
multi-step inputs to test state transitions, configuration round trips, wrapper
agreement, and invariants across failures. Keep concrete regression examples
alongside these properties when they identify a specific defect.

The default suite also builds `blckbaux-semantics` independently. Shared ABI records
generate Metal declarations with compiler-checked sizes and field offsets;
Hegel reads sentinel fields on the GPU. Recurrence and neighbor-prefix properties
use independently constructed expected results. The production H-Net shader is
compiled with that generated preamble. CUDA's shared record implementations
still require a CUDA-toolchain build and NVIDIA device checks; passing Metal
tests does not establish CUDA numerical parity.

```sh
./blckbaux test              # repository verification policy
./blckbaux test --affected   # run only test targets affected by working changes
./blckbaux fmt               # format Rust and Python source
```

## Build Verification

```sh
./blckbaux build --tests                             # verify the native build
./blckbaux build --distribution --tests              # verify wheels across Py 3.12-3.15
```

Default Python version is 3.13. Set `BLCKBAUX_PYTHON_VERSION` to `3.12`, `3.14`, or `3.15` to target another interpreter. Ax and BoTorch integration checks currently run through Python 3.14 because PyTorch has not published CPython 3.15 wheels; the core Blckbaux wheel is built and checked on 3.15.

A `v*` tag runs `.github/workflows/release.yml`. Each supported platform builds
and verifies four ABI-specific wheels, and the publish job attaches the complete
twelve-wheel set to the GitHub Release only after provenance and metadata checks
pass.

## Quality & Radar Gates

Repository quality configuration lives in `.config/`: `kiss.toml` defines KISS
thresholds, `kiss.ignore` lists repository-relative exclusions, and `prek.toml`
defines formatting and policy hooks. `tools/kiss` supplies the KISS configuration
automatically and rejects missing configuration or exclusions. Ruff and pytest
discover their settings in the root `pyproject.toml`.

Use `tools/kiss check .` for the configured architecture check. For Prek commands
outside `./blckbaux`, pass `--config .config/prek.toml`. This explicit config keeps
hook paths and file filters relative to the repository root. If installing Prek
hooks in a checkout, pass that same option to `prek install`; preserve any
existing hooks that enforce the repository's JJ workflow.

```sh
./blckbaux fmt               # apply Rust and Python formatting
./blckbaux radar performance show   # static HNet work, coverage, and goal certificate
```

`radar performance record --run RUN` records p95 whole-round and stage latency only
from a completed run whose hot-path digest matches the current source. After
that, `radar performance gate` passes without device work for unrelated edits and
requires `--run RUN` after a Metal or CUDA hot-path change. Produce that replay
without copying or editing the experiment file:

```sh
./blckbaux tune --prepare performance/configs/hnet-1m-causal-1000.toml
./blckbaux tune performance/configs/hnet-1m-causal-1000.toml
```

For Metal, the measured replay also has to match the separate reviewed
`performance/hnet-1m-round-parity.json` first-round result exactly: model size,
workload truth labels, evaluated tokens, candidate and incumbent objectives and
conditional NLLs, the acceptance decision, and SHA-256 identities of both
decoded outputs. This semantic check is separate from the older latency
certificate and its tolerance. Update it only from a reviewed full 1M run. The
unit tests exercise the comparison without running the 1M workload.
