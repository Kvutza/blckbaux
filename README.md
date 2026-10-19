# BLCKBAUX

Bayesian optimization and kernel execution for resident recurrent architectures. Rust core, Python wheels for 3.12–3.15. Metal and CUDA backends.

## Build

```sh
./blckbaux build                   # native library
./blckbaux build --distribution    # Python 3.12–3.15 wheels
./blckbaux test        # repository verification policy
```

Distribution wheels land in `dist/`. No system Rust, Python, or Clang required.
Version tags publish the CPython 3.12–3.15 wheels for macOS arm64 and
manylinux x86-64/arm64 as assets on the corresponding GitHub Release.

## Optimize & Visualize

```sh
./blckbaux tune examples/tuning/hnet-1m-operator-bo.toml  # runs optimization with live telemetry
```

Experiments automatically export vector SVG figures (`convergence.svg`, `contrastive.svg`, `trust_region.svg`, `latency.svg`) and an interactive `report.html` dashboard under `<RUN_DIR>/plots/`.

## Configuration & Autocompletion

All experiment recipes (`examples/tuning/*.toml`, `performance/configs/*.toml`) are strictly typed under version 4 syntax. Instant autocomplete, parameter docs, and schema validation are provided out-of-the-box for VS Code, Cursor, Zed, Helix, and Neovim via Taplo and [`schemas/tune-spec.schema.json`](schemas/tune-spec.schema.json).

Each authored recipe compiles to a complete resolved run record: workload lives in `data` and `objective`, budgets and latency goals in `execution`, held-out cadence in `evaluation`, and algorithm choices under one closed `optimizer` variant. Authored omissions use typed defaults rather than patching another file. Unknown fields, legacy aliases, and components from a different optimizer variant are rejected.

## Measure

```sh
./blckbaux radar performance show            # H-Net work and kernel coverage
./blckbaux cuda context 1048576              # CUDA PISA cache benchmark
./blckbaux cuda ptx-probe                    # synthesized PTX hardware probe
```

## Codebase

```sh
./blckbaux radar find <pattern>      # AST-scoped symbol search
./blckbaux radar impact <symbol>     # blast-radius across call graph
```

## Hardware

| Backend | Platform |
|---|---|
| Metal | Apple Silicon — unified memory, SIMDgroup primitives |
| CUDA | NVIDIA Turing / Hopper — PTX synthesis via CUDA-Oxide |

## Theory & Publications

- **Preprint:** [*Geometry and Observation Limits in Direct Neural Operator Optimization*](papers/operator_bo.pdf) (arXiv preprint, October 2026).
- [Operator Geometry Theory](docs/operator-bo-theory.md) — formal operator reductions and observable objectives.
- [Black-Box Optimization Program](docs/black-box-optimization-program.md) — normative design, theorem ledger, and proof contracts.

## Docs

- [Documentation Index](docs/README.md)
- [API Reference](docs/api.md)
- [Python Integrations](docs/interop.md)
- [Build System](docs/buck2.md)
- [Testing & Quality](docs/testing.md)
