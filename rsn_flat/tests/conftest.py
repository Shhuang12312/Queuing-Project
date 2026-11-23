"""pytest configuration: put rsn_flat/ and sim/ on sys.path.

Run from the repository root:   python -m pytest rsn_flat/tests
The compiled-engine cross-checks skip themselves when sim/genkern.so is not built
(`cd sim && ./build.sh`)."""
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
FLAT = os.path.dirname(HERE)
SIM = os.path.join(os.path.dirname(FLAT), "sim")
for p in (FLAT, SIM):
    if p not in sys.path:
        sys.path.insert(0, p)

warnings.filterwarnings("ignore", category=UserWarning)
