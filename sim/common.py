"""Shared helpers for the experiment scripts: run line, half-window split, batch-means half-width."""
import numpy as np
from scipy.stats import t as tdist

HGI_E_TOY = 2.8093965016743754          # network A in results/multi_bench.json (benchmark with independent extrapolation legs)


def log_run(command, seeds, runtime_s, outputs, note=""):
    """Print one line with the command, its seeds, the wall time and the output files."""
    print(f"run: {command} | seeds: {seeds} | wall time: {runtime_s:.0f} s | outputs: {outputs}" + (f" | {note}" if note else ""))


def halves(x):
    """First-half and second-half averages of a per-batch series (odd counts split the middle batch)."""
    x = np.asarray(x, float); n = len(x); w1 = np.ones(n)
    if n % 2:
        w1[n // 2] = 0.5; w1[n // 2 + 1:] = 0
    else:
        w1[n // 2:] = 0
    w2 = 1 - w1
    return float((x * w1).sum() / w1.sum()), float((x * w2).sum() / w2.sum())


def mean_hw(x, level=0.95):
    """Mean and t-based half-width of a sample (batch means or per-seed values)."""
    x = np.asarray(x, float); n = len(x)
    if n < 2:
        return float(x.mean()), float("nan")
    return float(x.mean()), float(tdist.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / np.sqrt(n))


def se(x):
    """Standard error of the mean of a sample."""
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")
