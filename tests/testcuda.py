from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

import blckbaux.cuda as cuda


class Backend:
    def __init__(self) -> None:
        self.checked = False

    @staticmethod
    def target() -> str:
        return "sm_75"

    def check(self) -> None:
        self.checked = True

    @staticmethod
    def trace(output) -> None:
        output.write_text(
            json.dumps({"format": "blckbaux.initialized-hnet.v1"}), encoding="utf-8"
        )


def test_api(monkeypatch, tmp_path):
    backend = Backend()
    monkeypatch.setattr(cuda, "_cuda", backend)
    assert cuda.target() == "sm_75"
    cuda.check()
    assert backend.checked
    output = cuda.trace(tmp_path / "trace.json")
    assert cuda.read_trace(output)["format"] == "blckbaux.initialized-hnet.v1"


def test_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(cuda, "_cuda", None)
    with pytest.raises(RuntimeError, match="CUDA wheel"):
        cuda.check()
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"format":"other"}', encoding="utf-8")
    with pytest.raises(ValueError, match="initialized trace"):
        cuda.read_trace(invalid)


def test_cli(monkeypatch, tmp_path):
    monkeypatch.setattr(cuda, "_cuda", Backend())
    runner = CliRunner()
    assert runner.invoke(cuda.main, ["target"]).output == "sm_75\n"
    output = tmp_path / "trace.json"
    result = runner.invoke(cuda.main, ["trace", str(output)])
    assert result.exit_code == 0, result.output
    assert output.is_file()
