"""sim/colab_engine.py compiles genkern.c at import (or falls back to Python)."""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

SIM = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "sim")
TOY = ("toy", [[1, 0, 1], [0, 1, 1]], [1, 1, 1], [1, 1, 1], [1, 1, 1], [-0.5] * 3)


def test_import_selects_an_engine():
    import colab_engine as ce
    assert ce.ENGINE in ("c", "python")
    if ce.ENGINE == "c":
        assert ce.SO_PATH and os.path.exists(ce.SO_PATH)
        assert os.environ["RSN_GENKERN_SO"] == ce.SO_PATH


@pytest.mark.skipif(not os.path.exists(os.path.join(SIM, "genkern.so")), reason="genkern.so not built")
def test_c_engine_is_gensim_bit_for_bit():
    import colab_engine as ce
    import gensim
    if ce.ENGINE != "c":
        pytest.skip("no compiler available here")
    a = ce.Net(*TOY).run(100, 5, 0.1, 60, 0.24, 7)
    b = gensim.Net(*TOY).run(100, 5, 0.1, 60, 0.24, 7)
    assert all(a[k] == b[k] for k in ("JE", "level", "hgi_path", "gap", "E_frac", "events"))


def test_python_fallback_has_the_same_interface():
    import colab_engine as ce
    nw = ce.PyNet(*TOY)
    o = nw.run(50, 2, 0.1, 60, 0.24, 1, n_snapshots=100)
    for k in ("JE", "level", "hgi_path", "gap", "E_frac", "JE_1st", "JE_2nd", "L", "U"):
        assert k in o
    assert o["engine"] == "python" and np.isnan(o["hgi_path"]) and o["JE"] > 0
    assert o["gap"] == pytest.approx(o["JE"] - o["level"], abs=2.0)   # snapshot vs time-weighted


def test_forced_python_and_broken_source_fall_back(tmp_path):
    """Subprocesses: RSN_FORCE_PYTHON=1, and a copy whose genkern.c does not compile."""
    env = dict(os.environ, RSN_FORCE_PYTHON="1")
    code = f"import sys; sys.path.insert(0, {SIM!r}); import colab_engine as ce; print(ce.ENGINE)"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.stdout.strip() == "python", out.stderr
    broken = tmp_path / "sim"
    broken.mkdir()
    for f in os.listdir(SIM):
        if f.endswith(".py"):
            shutil.copy(os.path.join(SIM, f), broken / f)
    (broken / "genkern.c").write_text("this is not C\n")
    shutil.copytree(os.path.join(os.path.dirname(SIM), "rsn_flat"), tmp_path / "rsn_flat",
                    ignore=shutil.ignore_patterns("tests", "notebooks", "__pycache__", "*.json"))
    code = f"import sys; sys.path.insert(0, {str(broken)!r}); import colab_engine as ce; print(ce.ENGINE, ce.SO_PATH)"
    env = {k: v for k, v in os.environ.items() if k not in ("RSN_FORCE_PYTHON", "RSN_GENKERN_SO")}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=str(tmp_path))
    assert out.stdout.strip() == "python None", out.stderr
    assert "RuntimeWarning" in out.stderr
