from __future__ import annotations
from typing import Dict, List, Any
import math
import statistics as stats

def transform_function_string(text: str) -> str:
    words = text.split()
    new_words = []
    for word in words:
        if "_func" in word:
            word = word.replace("func", "Function")
        new_words.append(word)
    result = " ".join(new_words).replace("_", " ")
    return result.title()


def get_fitness_range(results: dict, lowest, highest) -> tuple[float, float]:
    experiment_data = results.get("Experiment", {})
    fitness_values = []

    for key, metrics in experiment_data.items():
        if isinstance(metrics, dict) and "Average fitness" in metrics:
            fitness_values.append(float(metrics["Average fitness"]))

    if not fitness_values:
        raise ValueError("No 'Average fitness' values found.")
    min_fitness = min(fitness_values)
    if (min_fitness - lowest) / (highest - lowest) > 0.2 :
        min_fitness = min_fitness  - ((highest - lowest) * 0.1)
    else:
        min_fitness = lowest
    return min_fitness, highest

Run = List[float]
StrategyRuns = Dict[str, List[Run]]

def summarize_ga(
    data: StrategyRuns,
    n_first_gens: int = 10,
    rsd_as_percent: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Summarize GA results per strategy.

    Args:
        data: {strategy: [run1, run2, ...]}, each run = [fitness_gen0, fitness_gen1, ...]
        n_first_gens: 'n' for metrics over the first n generations.
        rsd_as_percent: If True, RSD is returned in percent; otherwise as a ratio.

    Returns:
        {
          strategy: {
            "num_runs": int,
            "avg_min_fitness": float,
            "rsd_min_fitness": float,           # relative std dev of per-run minima
            "avg_generations": float,           # average length of runs (generations)
            "avg_fitness_after_n": float,       # average fitness after first n generations
            "avg_slope_first_n": float          # average slope over first n generations
          },
          ...
        }
    """
    out: Dict[str, Dict[str, Any]] = {}

    for strategy, runs in data.items():
        # Filter out empty runs
        runs = [r for r in runs if r]  # keep only non-empty lists
        num_runs = len(runs)

        if num_runs == 0:
            out[strategy] = {
                "num_runs": 0,
                "avg_min_fitness": math.nan,
                "rsd_min_fitness": math.nan,
                "avg_generations": 0.0,
                "avg_fitness_after_n": math.nan,
                "avg_slope_first_n": math.nan,
            }
            continue

        # Per-run minima (assuming minimization)
        per_run_min = [min(r) for r in runs]
        avg_min = stats.fmean(per_run_min)

        # Relative standard deviation (RSD) of minima
        if num_runs >= 2:
            sd_min = stats.pstdev(per_run_min) if num_runs == 1 else stats.stdev(per_run_min)
        else:
            sd_min = 0.0
        # Avoid division by ~0; report NaN if mean ≈ 0
        rsd = (sd_min / abs(avg_min)) if abs(avg_min) > 1e-12 else math.nan
        if rsd_as_percent and not math.isnan(rsd):
            rsd *= 100.0

        # Average number of generations (run length)
        avg_generations = stats.fmean(len(r) for r in runs)

        # Fitness after first n generations (per run: clamp at last if shorter)
        def value_after_n(r: Run, n: int) -> float:
            k = max(1, n)  # n must be >=1; "after first n" -> index n-1
            idx = min(len(r) - 1, k - 1)
            return r[idx]

        avg_after_n = stats.fmean(value_after_n(runs[i], n_first_gens) for i in range(num_runs))

        # Slope over first n generations (per run: use first and min(n, len))
        def slope_first_n(r: Run, n: int) -> float:
            if len(r) < 2:
                return math.nan
            m = min(n, len(r))
            if m < 2:
                return math.nan
            return (r[m - 1] - r[0]) / (m - 1)

        slopes = [slope_first_n(r, n_first_gens) for r in runs]
        # Filter NaNs from slopes (in case of length-1 runs)
        slopes = [s for s in slopes if not math.isnan(s)]
        avg_slope = stats.fmean(slopes) if slopes else math.nan

        out[strategy] = {
            "num_runs": num_runs,
            "avg_min_fitness": avg_min,
            "rsd_min_fitness": rsd,
            "avg_generations": avg_generations,
            "avg_fitness_after_n": avg_after_n,
            "avg_slope_first_n": avg_slope,
        }

    return out
