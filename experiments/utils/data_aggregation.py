from typing import Optional, List, Tuple, Callable

import numpy as np


def _bootstrap_ci(
    x: np.ndarray,
    fn: Callable[[np.ndarray], float] = np.mean,
    alpha: float = 0.05,
    n_boot: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    stat = fn(x)
    if x.size == 1:
        return (stat, stat, stat)
    n = x.size
    boots = np.empty(n_boot, dtype=float)
    idx = rng.integers(0, n, size=(n_boot, n))
    for i in range(n_boot):
        boots[i] = fn(x[idx[i]])
    lo = np.percentile(boots, 100 * (alpha / 2))
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return (stat, lo, hi)

# -------- Aggregate per generation across runs --------------------------------
def aggregate_convergence(
    runs: List[List[float]],
    avg_number_of_generations: int,
    stat: str = "mean",                # "mean" or "median"
    band: str = "ci",                  # "ci" for bootstrap CI, or "iqr" for 25-75%
    alpha: float = 0.05,               # for CI
    n_boot: int = 2000,
    rng: Optional[np.random.Generator] = None,
):
    """
    Aligns runs by generation index and computes central tendency + variability.

    Returns dict with:
      gen: np.ndarray[int]          -> generation indices [0..max_len-1]
      center: np.ndarray[float]     -> mean/median at each generation
      lower: np.ndarray[float]      -> lower band (CI or 25th)
      upper: np.ndarray[float]      -> upper band (CI or 75th)
      n_at_gen: np.ndarray[int]     -> number of runs contributing at each generation
    """
    if rng is None:
        rng = np.random.default_rng()
    max_len = max(len(r) for r in runs)
    gens = np.arange(max_len)
    center, lower, upper, n_at_gen = [], [], [], []

    fn = np.mean if stat == "mean" else np.median

    for g in gens:
        vals = np.array([r[g] for r in runs if len(r) > g], dtype=float)
        n_at_gen.append(vals.size)
        if vals.size == 0:
            center.append(np.nan); lower.append(np.nan); upper.append(np.nan)
            continue

        if band == "ci":
            c, lo, hi = _bootstrap_ci(vals, fn=fn, alpha=alpha, n_boot=n_boot, rng=rng)
        else:  # "iqr" (50% central band)
            c = fn(vals)
            lo, hi = np.percentile(vals, [25, 75])

        center.append(c); lower.append(lo); upper.append(hi)
    return {
        "gen": gens,
        "center": np.array(center),
        "lower": np.array(lower),
        "upper": np.array(upper),
        "n_at_gen": np.array(n_at_gen),
    }