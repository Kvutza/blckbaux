from __future__ import annotations

import blckbaux
import blckbaux.experimental
import blckbaux.experimental.mtrregn

assert blckbaux.__file__
assert blckbaux.experimental.__file__
assert blckbaux.experimental.mtrregn.__file__

try:
    import blckbaux.blckbaux_rust
except ImportError:
    print("blckbaux wheel consumer: skipped (blckbaux_rust unavailable)")
else:
    assert blckbaux.blckbaux_rust.__file__
    print("blckbaux wheel consumer: ok")
