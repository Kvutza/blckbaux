# Build System

BLCKBAUX builds through Buck2. `./blckbaux build` bootstraps the native toolchain — no system Rust or Clang needed.

```sh
./blckbaux build                         # build the native library
./blckbaux build --tests                 # build and run the native test suite
./blckbaux build --distribution          # build and smoke-test Python 3.12–3.15 wheels
./blckbaux build --distribution --tests  # run the full Python suite against every wheel
./blckbaux build --distribution --out dist
```

Buck downloads and caches pinned Rust, Clang/LLD, and the Linux sysroot. Build results are reused when inputs haven't changed.

## Toolchain

Rust `nightly-2026-09-29` is pinned in `rust-toolchain.toml`.
Builds consume the repository pin directly. Synchronization is an internal
maintenance operation rather than part of the public BLCKBAUX command surface.

## Python Environments

Pixi manages wheel verification environments in `.pixi/envs/`. They are created automatically on first use. Set `BLCKBAUX_PYTHON_312`, `BLCKBAUX_PYTHON_313`, or `BLCKBAUX_PYTHON_314` to supply your own CPython interpreter instead.

## Rust Dependencies

Cargo manifests declare Rust crates; Reindeer generates their Buck targets in `rust/BUCK`. When changing crate dependencies or features, update the handwritten BUCK target and run `tools/buck2-deps`.

## Bazel

`BLCKBAUX_BUILD_SYSTEM=bazel` routes the CLI through Bazel for an independent build correctness check. `tools/build-parity --sync` regenerates shared target definitions in `build/targets.bzl`.

## Cache Cleanup

```sh
./buck2w clean --stale --dry-run   # inspect stale artifacts
./buck2w clean --stale             # reclaim them
```

## Hosts

| Platform | Notes |
|---|---|
| Apple Silicon macOS | Requires `xcode-select --install` |
| x86\_64 / aarch64 Linux | Sysroot and headers supplied by Buck |

Linux release builds must run on a manylinux 2.28 host. Metal runs on macOS; CUDA and OpenCL require an installed driver.
