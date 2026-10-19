# Python Integrations

BLCKBAUX integrates with BoTorch, Optuna, and Ax through zero-copy adapters.

## Adapters

| Adapter | Purpose |
|---|---|
| `blckbaux.botorch.Model` | Wrap an `ENN` surrogate as a BoTorch model |
| `blckbaux.botorch.Sampler` | Sample joint function values |
| `blckbaux.optuna.Sampler` | Propose continuous trial parameters in Optuna |
| `blckbaux.ax.Node` | Propose trial parameters in Ax |

## Verification

```sh
./blckbaux build --distribution --tests  # verify Python integration across supported wheels
```
