"""CUDA execution shipped inside the Blckbaux Python wheel."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from ._rust import cuda as _cuda


def _backend() -> Any:
    if _cuda is None:
        raise RuntimeError(
            "CUDA support requires the Blckbaux CUDA wheel on Linux x86-64"
        )
    return _cuda


def target() -> str:
    """Return the PTX target selected for the current NVIDIA device."""
    return str(_backend().target())


def check() -> None:
    """Run the production CUDA component and recurrent-model checks."""
    _backend().check()


def trace(output: str | Path) -> Path:
    """Write the initialized production-model trace used for backend parity."""
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    _backend().trace(path)
    return path


def read_trace(path: str | Path) -> dict[str, Any]:
    """Read and minimally validate an initialized parity trace."""
    source = Path(path)
    report = json.loads(source.read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or report.get("format") != "blckbaux.initialized-hnet.v1"
    ):
        raise ValueError(f"{source} is not an Blckbaux initialized trace")
    return report


@click.group(help=__doc__)
def main() -> None:
    pass


@main.command("target")
def target_cmd() -> None:
    """Print the selected device target."""
    click.echo(target())


@main.command("check")
def check_cmd() -> None:
    """Run the production CUDA model checks."""
    try:
        check()
    except RuntimeError as error:
        raise click.ClickException(str(error)) from error
    click.echo("ok")


@main.command("trace")
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
def trace_cmd(output: Path) -> None:
    """Write an initialized-model parity trace."""
    try:
        click.echo(trace(output))
    except RuntimeError as error:
        raise click.ClickException(str(error)) from error


if __name__ == "__main__":
    main()
