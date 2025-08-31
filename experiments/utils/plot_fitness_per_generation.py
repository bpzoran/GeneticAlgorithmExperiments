from typing import List, Optional, Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np


def fitness_per_generation_plot(min_cost_per_generation, title, color):
    plt.plot(min_cost_per_generation, color=color)
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")

    plt.show()

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

def plot_convergence_curve(
    runs: List[List[float]],
    stat: str = "mean",            # "mean" or "median"
    band: str = "ci",              # "ci" for 95% CI, "iqr" for interquartile (25–75%)
    alpha: float = 0.05,
    n_boot: int = 2000,
    title: str = "GA Convergence (central tendency ± variability)",
    ylabel: str = "Best (minimal) fitness",
    xlabel: str = "Generation",
    annotate_counts: bool = True,  # annotate how many runs contribute at early/mid/late gens
    color: str = "C0",             # color for diagnostic count plot
):
    agg = aggregate_convergence(runs, stat=stat, band=band, alpha=alpha, n_boot=n_boot)

    gen = agg["gen"]
    center = agg["center"]
    low = agg["lower"]
    up = agg["upper"]

    # Optional: mask leading/trailing all-NaN (shouldn't occur, but safe)
    valid = ~np.isnan(center)
    if not valid.any():
        raise ValueError("All aggregated values are NaN. Check your `runs` input.")
    first, last = np.argmax(valid), len(valid) - np.argmax(valid[::-1]) - 1
    gen, center, low, up = gen[first:last+1], center[first:last+1], low[first:last+1], up[first:last+1]

    plt.figure()
    plt.plot(gen, center, label=f"{stat.capitalize()} best fitness")
    plt.fill_between(gen, low, up, alpha=0.2,
                     label=("95% CI" if band == "ci" else "IQR (25–75%)"))
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title); plt.legend()
    plt.tight_layout()
    plt.show()

    if annotate_counts:
        # Quick diagnostic figure showing number of contributing runs across generations
        plt.figure()
        plt.plot(agg["gen"], agg["n_at_gen"], color=color)
        plt.xlabel("Generation"); plt.ylabel("# runs contributing")
        plt.title("Contributing runs per generation (diagnostic)")
        plt.tight_layout()
        plt.show()
