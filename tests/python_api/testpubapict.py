from __future__ import annotations

import importlib.util
import inspect

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("blckbaux.blckbaux_rust") is None,
    reason="blckbaux_rust extension unavailable",
)


class TestPublicAPIExports:
    """Verify public API surface is exported correctly from blckbaux package."""

    def test_001(self):
        """All _LAZY_ATTRS entries must be importable."""
        import blckbaux

        for name in blckbaux._LAZY_ATTRS:
            attr = getattr(blckbaux, name)
            assert attr is not None, f"Attribute {name} not found"

    def test_002(self):
        """ENN is a class with expected signature."""
        from blckbaux import ENN

        assert inspect.isclass(ENN)
        # Check constructor signature has required parameters
        sig = inspect.signature(ENN.__init__)
        params = list(sig.parameters.keys())
        assert "train_x" in params
        assert "train_y" in params
        assert "train_yvar" in params

    def test_003(self):
        from blckbaux import ENNFitter

        assert ENNFitter is not None

    def test_004(self):
        """create_optimizer is a callable function."""
        from blckbaux import create_optimizer

        assert callable(create_optimizer)

    def test_optimizer_name(self):
        import blckbaux
        from blckbaux import bo

        assert blckbaux.Optimizer is bo.Optimizer
        assert blckbaux.BayesOptimizer is blckbaux.Optimizer
        assert blckbaux.Optimizer.__name__ == "Optimizer"

    def test_configclasses(self):
        """All config classes are available and constructible."""
        from blckbaux import (
            MorboTrustRegion,
            NoTrustRegion,
            OptimizerConfig,
            TurboTrustRegion,
        )

        # These are dataclasses or similar - should be constructible
        assert inspect.isclass(OptimizerConfig)
        assert inspect.isclass(TurboTrustRegion)
        assert inspect.isclass(MorboTrustRegion)
        assert inspect.isclass(NoTrustRegion)

    def test_005(self):
        """All config factory functions are callable."""
        from blckbaux import (
            lhd_only,
            turbo_enn,
            turbo_one,
            turbo_zero,
        )

        assert callable(turbo_one)
        assert callable(turbo_zero)
        assert callable(turbo_enn)
        assert callable(lhd_only)

    def test_telemetryclass(self):
        """Telemetry class is available."""
        from blckbaux import Telemetry

        assert inspect.isclass(Telemetry)

    def test_006(self):
        """Experimental namespace is importable and exposes low-level symbols."""
        import pytest

        pytest.importorskip("blckbaux.blckbaux_rust")
        import blckbaux.experimental as experimental_mod
        from blckbaux import experimental

        assert inspect.ismodule(experimental)
        assert experimental is experimental_mod
        assert experimental.experimental is experimental
        assert inspect.isclass(experimental.SharingPolicy)
        assert inspect.isclass(experimental.MultiTrustRegionLoop)
        assert callable(experimental.make_region)
        assert inspect.isclass(experimental.BayesOptimizer)
        assert inspect.isclass(experimental.MultiTrustRegion)
        assert inspect.isclass(experimental.ResidentBoSession)
        assert callable(experimental.enn_optimizer)
        assert callable(experimental.quantize_int4)
        assert callable(experimental.quantize_e2m1)
        for name in [
            "ParamBlock",
            "ParameterBuffer",
            "Proposals",
            "SearchState",
            "turbo_enn",
        ]:
            assert name in experimental.__all__
            assert hasattr(experimental, name)
        if experimental.SearchState is not None:
            assert inspect.isclass(experimental.ParamBlock)
            assert inspect.isclass(experimental.ParameterBuffer)
            assert experimental.ParameterBuffer is experimental.ParamBuffer
            assert inspect.isclass(experimental.Proposals)
            assert inspect.isclass(experimental.SearchState)
            assert callable(experimental.turbo_enn)

    def test_enumtypes(self):
        """Enum types are available."""
        from blckbaux import AcquisitionKind, CandidateSampler, InitStrategy

        assert inspect.isclass(CandidateSampler)
        assert inspect.isclass(InitStrategy)
        assert inspect.isclass(AcquisitionKind)


class Test_001:
    """Verify public API does not change unexpectedly."""

    def test_007(self):
        """__all__ must match _LAZY_ATTRS keys."""
        import blckbaux

        assert set(blckbaux.__all__) == set(blckbaux._LAZY_ATTRS.keys())

    def test_008(self):
        import blckbaux

        assert "quantize_int4" not in blckbaux.__all__
        assert "quantize_e2m1" not in blckbaux.__all__

    def test_009(self):
        import blckbaux

        for name in ("quantize_int4", "quantize_e2m1"):
            with pytest.raises(AttributeError):
                getattr(blckbaux, name)
