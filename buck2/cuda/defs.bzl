def cuda_oxide(name, package, workspace, crates, cuda, parity, click):
    """Build a CUDA-Oxide CPython wheel and its GPU parity action."""
    wheel = name + "-wheel"
    filename = "blckbaux-{}+cuda75-cp312-cp312-manylinux_2_28_x86_64.whl".format(read_config("blckbaux", "release_version", "0.1.1"))
    native.genrule(
        name = wheel,
        srcs = [package, workspace, cuda, click] + crates.values(),
        outs = {"wheel": [filename], "cargo-target": ["target"], "spine": ["target/release/blckbaux-cuda"]},
        default_outs = [filename],
        # Cargo owns this incremental cache; Buck keeps it between local runs.
        no_outputs_cleanup = True,
        cacheable = False,
        labels = ["uses_undeclared_inputs"],
        # CUDA-Oxide pins this compiler; the host workspace uses a newer nightly.
        # Keep CUDA compilation on the toolchain recorded by cuda/rust-toolchain.toml.
        env = {"RUSTUP_TOOLCHAIN": "nightly-2026-08-28"},
        cmd = " ".join([
            "set -euo pipefail;",
            "python3 --version | grep -q '^Python 3\\.12\\.';",
            "OUT_DIR=$PWD/$OUT;",
            "OUT_FILE=$OUT_DIR/{};".format(filename),
            "ROOT=$TMP/repo;",
            "mkdir -p $ROOT/rust/crates $ROOT/cuda;",
            "cp -R $(location {})/. $ROOT/;".format(package),
            "cp -R $(location {})/. $ROOT/;".format(workspace),
        ] + [
            "cp -R $(location {}) $ROOT/rust/crates/{};".format(label, crate)
            for crate, label in crates.items()
        ] + [
            "cp -R $(location {})/. $ROOT/cuda/;".format(cuda),
            "cd $ROOT/cuda;",
            "cargo generate-lockfile --manifest-path $ROOT/Cargo.toml;",
            "cargo oxide build --arch sm_75 --cargo-target-dir $OUT_DIR/target --device-codegen-crate blckbaux_cuda_device -- -p blckbaux-py -p blckbaux-cuda --features cuda --release --locked --manifest-path $ROOT/Cargo.toml;",
            "cd $ROOT;",
            "PYTHONPATH=$(location {}) python3 ops/cuda_wheel.py".format(click),
            "$OUT_DIR/target/release/libblckbaux_rust.so",
            "$TMP/wheel --root $ROOT --version {};".format(read_config("blckbaux", "release_version", "0.1.1")),
            "mkdir -p ${OUT_FILE%/*};",
            "cp $TMP/wheel/blckbaux-{}+cuda75-cp312-cp312-manylinux_2_28_x86_64.whl $OUT_FILE".format(read_config("blckbaux", "release_version", "0.1.1")),
        ]),
        target_compatible_with = [
            "prelude//cpu/constraints:x86_64",
            "prelude//os/constraints:linux",
        ],
        visibility = ["PUBLIC"],
    )
    native.genrule(
        name = name + "-parity",
        srcs = [parity],
        out = "bf16-parity.txt",
        cmd = " ".join([
            "set -euo pipefail;",
            "python3 -m pip install --quiet --no-deps",
            "--target $TMP/site $(location :{});".format(wheel),
            "PYTHONPATH=$TMP/site python3 $SRCDIR/ops/bf16_parity.py > $OUT",
        ]),
        target_compatible_with = [
            "prelude//cpu/constraints:x86_64",
            "prelude//os/constraints:linux",
        ],
        visibility = ["PUBLIC"],
    )
