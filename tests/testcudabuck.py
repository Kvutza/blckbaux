from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_bundle():
    workspace = tomllib.loads((ROOT / "Cargo.toml").read_text())
    build = (ROOT / "BUCK").read_text()
    for member in workspace["workspace"]["members"]:
        if member.startswith("rust/crates/"):
            name = Path(member).name
            assert f'"{name}": "//{member}:' in build
    rule = (ROOT / "buck2/cuda/defs.bzl").read_text()
    assert "srcs = [package, workspace, cuda, click] + crates.values()" in rule
    assert "for crate, label in crates.items()" in rule


def test_atomic():
    source = (ROOT / "buck2/cuda/defs.bzl").read_text(encoding="utf-8")
    lockfile = "cargo generate-lockfile --manifest-path $ROOT/Cargo.toml;"
    build = (
        "cargo oxide build --arch sm_75 --cargo-target-dir $OUT_DIR/target "
        "--device-codegen-crate blckbaux_cuda_device -- -p blckbaux-py -p blckbaux-cuda "
        "--features cuda --release --locked --manifest-path $ROOT/Cargo.toml;"
    )

    assert lockfile in source
    assert build in source
    assert source.index(lockfile) < source.index(build)
    assert '"cargo oxide build",' not in source
    assert '"--arch sm_75",' not in source
    assert "; --arch sm_75" not in " ".join(source.split())
