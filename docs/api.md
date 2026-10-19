# API

| API | Use |
| --- | --- |
| Python `blckbaux` | Fit an `ENN` model or optimize with `create_optimizer` |
| Rust `blckbaux::prelude` | Models, optimizers, acquisition functions, and index types |
| `blckbaux.search` / `blckbaux::search` | Generate and evaluate candidates stored as low-bit parameters |
| `blckbaux_tooling::plot` | Pure-Rust vector SVG charts and interactive HTML report dashboard |
| `blckbaux.experimental` / `blckbaux::experimental` | Quantization, GPU buffers, and experimental model support |
| `blckbaux.botorch`, `blckbaux.optuna`, `blckbaux.ax` | Use BLCKBAUX with Python optimization libraries |

Optimizers use `ask` to propose a trial and `tell` to record its result.
GPU support varies by operation; Rust callers can check `blckbaux::capability::matrix()`.
Experiment telemetry is recorded to `optimization.jsonl` and plotted automatically under `<RUN_DIR>/plots/`.

[API changes](../CHANGELOG.md) · [Python integrations](interop.md)
