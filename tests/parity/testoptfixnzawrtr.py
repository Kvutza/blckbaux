from __future__ import annotations

import pytest

from blckbaux.turbo.optimizer_fixtures import load_fixture

try:
    from blckbaux._rust import BayesOptimizer  # noqa: F401

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")


def test_001():
    data = load_fixture("tenzawrseed2")
    lengths = [float(step["tr_length"]) for step in data["steps"]]
    assert all(length == lengths[0] for length in lengths)
    batch_mins = [min(v[0] for v in step["tell_y"]) for step in data["steps"]]
    assert batch_mins[-1] < batch_mins[0]
