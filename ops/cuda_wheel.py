"""Package a CUDA-enabled Blckbaux extension as a CPython 3.12 Linux wheel."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import shutil
import sys
import sysconfig
import tempfile
import zipfile
from pathlib import Path

import click

TAG = "cp312-cp312-manylinux_2_28_x86_64"


def _digest(path: Path) -> str:
    value = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest())
    return "sha256=" + value.rstrip(b"=").decode("ascii")


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def build_wheel(root: Path, extension: Path, output: Path, version: str) -> Path:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("CUDA wheel packaging requires CPython 3.12")
    if sys.platform != "linux":
        raise RuntimeError("CUDA wheels are built on Linux")
    if not extension.is_file():
        raise FileNotFoundError(extension)

    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not suffix or "cpython-312" not in suffix:
        raise RuntimeError(f"unexpected extension suffix: {suffix!r}")

    version = version + "+cuda75"
    output.mkdir(parents=True, exist_ok=True)
    wheel = output / f"blckbaux-{version}-{TAG}.whl"
    with tempfile.TemporaryDirectory(prefix="blckbaux-cuda-wheel-") as directory:
        stage = Path(directory)
        package = stage / "blckbaux"
        shutil.copytree(
            root / "src/blckbaux",
            package,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so"),
        )
        shutil.copy2(extension, package / f"blckbaux_rust{suffix}")

        dist = stage / f"blckbaux-{version}.dist-info"
        _write(
            dist / "METADATA",
            "\n".join(
                [
                    "Metadata-Version: 2.3",
                    "Name: blckbaux",
                    f"Version: {version}",
                    "Summary: Epistemic Nearest Neighbors with CUDA-Oxide sm_75 support",
                    "Requires-Python: >=3.12,<3.13",
                    "Requires-Dist: numpy>=2.1",
                    "Requires-Dist: scipy>=1.11",
                    "Requires-Dist: click==8.5.0",
                    "Requires-Dist: botorch>=0.18.1,<0.19",
                    "Requires-Dist: optuna>=5.0,<5.1",
                    "Requires-Dist: ax-platform>=1.3.1,<1.4",
                    "",
                ]
            ),
        )
        _write(
            dist / "WHEEL",
            "\n".join(
                [
                    "Wheel-Version: 1.0",
                    "Generator: blckbaux.cuda_wheel",
                    "Root-Is-Purelib: false",
                    f"Tag: {TAG}",
                    "",
                ]
            ),
        )
        _write(dist / "top_level.txt", "blckbaux\n")
        _write(
            dist / "entry_points.txt",
            "[console_scripts]\nblckbaux-cuda = blckbaux.cuda:main\n",
        )
        license_dir = dist / "licenses"
        license_dir.mkdir(parents=True)
        shutil.copy2(root / "LICENSE", license_dir / "LICENSE")
        shutil.copy2(root / "NOTICE", license_dir / "NOTICE")

        record = dist / "RECORD"
        rows = []
        for path in sorted(stage.rglob("*")):
            if path.is_file() and path != record:
                rows.append(
                    (
                        path.relative_to(stage).as_posix(),
                        _digest(path),
                        str(path.stat().st_size),
                    )
                )
        rows.append((record.relative_to(stage).as_posix(), "", ""))
        stream = io.StringIO(newline="")
        csv.writer(stream, lineterminator="\n").writerows(rows)
        _write(record, stream.getvalue())

        with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(stage).as_posix())
    return wheel


@click.command(help=__doc__)
@click.argument(
    "extension", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument("output", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path(__file__).resolve().parents[1],
    show_default=True,
)
@click.option("--version", required=True)
def main(extension: Path, output: Path, root: Path, version: str) -> None:
    try:
        wheel = build_wheel(root, extension, output, version)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(wheel)


if __name__ == "__main__":
    main()
