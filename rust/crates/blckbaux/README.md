# BLCKBAUX

Bayesian optimization with epistemic nearest-neighbor models.

```sh
cargo add blckbaux
```

The default features enable Metal on macOS and OpenCL. BPANN is included.

## Reading the resident optimizer

The configuration roles and units are documented in
[optimizer](src/config/spec/optimizer.rs) and
[execution](src/config/spec/execution.rs). Pool identities and the supported
proposal formulas are described in [procedural_pool](src/procedural_pool.rs).

Follow a round through [configuration](src/bf16_metal/configuration.rs),
[proposal submission](src/bf16_metal/propose.rs),
[selection parameters](src/bf16_metal/selection.rs), and
[publication](src/bf16_metal/publication.rs). The
[long-context evaluator](src/fbt_moe/long.rs) defines the corpus objective and
timing scopes. [Observation policies](src/bf16_metal/observations.rs) decide
acceptance; [history](src/bf16_metal/history.rs) commits and archives the trial;
[controller](src/bf16_metal/controller.rs) fits ENN and advances radius control.

Each module documents its assumptions, state changes, and limitations. Historical
`bf16` filenames do not establish the actual storage precision, and `latent`
history does not mean a reduced-dimensional model parameterization. The repository
[source-contract standard](../../../docs/testing.md#source-contracts) applies to
owned code; documentation-only changes must preserve executable behavior.
