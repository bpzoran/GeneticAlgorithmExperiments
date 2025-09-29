from __future__ import annotations

import statistics
from typing import Dict, List, Any
import math
import statistics as stats

from utils.exp_logging import log_message_info


def value_after_n(r: Run, n: int) -> float:
    k = max(1, n)  # n must be >=1; "after first n" -> index n-1
    idx = min(len(r) - 1, k - 1)
    return r[idx]

def relative_early_convergence(r: Run, n: int) -> float:
    val_after_n = value_after_n(r, n)
    val_after_0 = value_after_n(r, 1)
    min_fitness = value_after_n(r, len(r))
    return _compute_relative_early_convergence(val_after_0, val_after_n, min_fitness)

def _compute_relative_early_convergence(f0: float, f_t: float, f_final: float) -> float:
    """
    Compute Relative Early Convergence (REC).

    REC = (f_t - f_final) / (f0 - f_final)

    Parameters
    ----------
    f0 : float
        Fitness at the first generation.
    f_t : float
        Fitness at the early checkpoint generation (e.g., generation 40).
    f_final : float
        Final fitness at the end of the optimization.

    Returns
    -------
    float
        REC value in [0, 1] if f_final < f0 and convergence is monotonic.
        Can be outside [0, 1] if fitness values are non-monotonic.
    """
    denominator = f0 - f_final
    scope = f_t - f_final
    if denominator == 0:
        if scope == 0:
            return 1.0  # No change at all
        else:
            raise Exception("Min fitness cannot rise.")
    return scope / denominator


import numpy as np


def area_under_convergence_curve(fitness_values: Run):
    """
    Compute the Area Under the Convergence Curve (AUCC), normalized to [0, 1] values.
    Lower AUCC = faster/better early convergence.
    Normalized AUCC allows comparing runs of different lengths or fitness scales.

    Parameters
    ----------
    fitness_values : list or np.ndarray
        Sequence of fitness values over generations (length T+1).

    Returns
    -------
    float
       Normalized AUCC value.
    """
    fitness_values = np.array(fitness_values, dtype=float)
    T = len(fitness_values) - 1  # number of intervals

    # Trapezoidal rule
    _area = np.trapz(fitness_values, dx=1)
    if fitness_values.min() == fitness_values.max():
        return 1
    f0, f_final = fitness_values[0], fitness_values[-1]
    denom = T * abs(f0 - f_final)
    if denom != 0:
        area = _area / denom
    else:
        area = float("nan")
    if area > 1:
        area = 1
    return area


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

def minimal_average_generations(data: dict[str, list[list[float]]]) -> int:
    averages = []
    for strategy, runs in data.items():
        for run in runs:
            if run:  # avoid empty lists
                avg = statistics.mean(run)
                averages.append(avg)
    return round(min(averages)) if averages else 0

def minimal_average_length_per_strategy(data: dict[str, list[list[float]]]) -> float | None:
    averages = []
    for runs in data.values():
        if runs:
            avg_len = sum(len(run) for run in runs) / len(runs)
            averages.append(avg_len)
    return min(averages) if averages else 0

def summarize_ga(
    data: StrategyRuns,
    percentage_of_generations_for_performance: float = 10.0,
) -> Dict[str, Dict[str, Any]]:
    """
    Summarize GA results per strategy.

    Args:
        data: {strategy: [run1, run2, ...]}, each run = [fitness_gen0, fitness_gen1, ...]
        percentage_of_generations_for_performance: 'n' for metrics over the first n generations.

    Returns:
        {
          strategy: {
            "num_runs": int,
            "avg_min_fitness": float,
            "sd_min_fitness": float,           # std dev of per-run minima
            "avg_generations_for_performance": float,           # average length of runs (generations)
            "avg_fitness_after_n": float,       # average fitness after first n generations
            "relative_early_convergence": float, # relative early convergence after first n generations
            "avg_slope_first_n": float          # average slope over first n generations
          },
          ...
        }
    """
    out: Dict[str, Dict[str, Any]] = {}
    min_average_generations = minimal_average_length_per_strategy(data)
    avg_generations_for_performance = number_of_generations_for_performance_check(min_average_generations,
                                                                                  percentage_of_generations_for_performance)
    for strategy, runs in data.items():
        # Filter out empty runs
        runs = [r for r in runs if r]  # keep only non-empty lists
        num_runs = len(runs)

        if num_runs == 0:
            out[strategy] = {
                "num_runs": 0,
                "avg_min_fitness": math.nan,
                "sd_min_fitness": math.nan,
                "avg_generations_for_performance": 0.0,
                "avg_generations": 0.0,
                "avg_fitness_after_n": math.nan,
                "relative_early_convergence":  math.nan,
                "avg_slope_first_n": math.nan,
            }
            continue

        # Per-run minima (assuming minimization)
        per_run_min = [min(r) for r in runs]
        avg_min = stats.fmean(per_run_min)

        # Standard deviation (SD) of minima
        if num_runs >= 2:
            sd_min = stats.pstdev(per_run_min) if num_runs == 1 else stats.stdev(per_run_min)
        else:
            sd_min = 0.0

        # Average number of generations (run length)
        number_of_generations = round(stats.fmean(len(r) for r in runs))

        avg_after_n = stats.fmean(value_after_n(runs[i], round(avg_generations_for_performance)) for i in range(num_runs))
        avg_after_first = stats.fmean(value_after_n(runs[i], 1) for i in range(num_runs))
        relative_early_conv = stats.fmean(relative_early_convergence(runs[i], round(avg_generations_for_performance)) for i in range(num_runs))
        areas_under_convergence_curve = []
        for i in range(num_runs):
            areas_under_convergence_curve.append(area_under_convergence_curve(runs[i]))

        areas_under_conv_curve = [v for v in areas_under_convergence_curve if not math.isnan(v)]
        area_under_conv_curve = stats.fmean(areas_under_conv_curve)
        # Fitness after first n generations (per run: clamp at last if shorter)



        # Slope over first n generations (per run: use first and min(n, len))
        def slope_first_n(r: Run, m: int) -> float:
            try:
                if len(r) < 2:
                    return math.nan
                if m == 2:
                    m = 2
                if len(r) < m:
                    m = len(r)
                return (r[m - 1] - r[0]) / (m - 1)
            except Exception as e:
                log_message_info(f"Error calculating slope for run {r} with m={m}: {e}")
                return math.nan


        slopes = [slope_first_n(r, avg_generations_for_performance) for r in runs]
        # Filter NaNs from slopes (in case of length-1 runs)
        slopes = [s for s in slopes if not math.isnan(s)]
        avg_slope = stats.fmean(slopes) if slopes else math.nan

        out[strategy] = {
            "num_runs": num_runs,
            "avg_min_fitness": avg_min,
            "sd_min_fitness": sd_min,
            "avg_generations_for_performance": avg_generations_for_performance,
            "avg_generations": number_of_generations,
            f"avg_fitness_after_n": avg_after_n,
            f"avg_fitness_after_first": avg_after_first,
            f"relative_early_convergence": relative_early_conv,
            "area_under_convergence_curve": area_under_conv_curve,
            f"avg_slope_first_n": avg_slope,
        }

    return out

def number_of_generations_for_performance_check(number_of_iterations: int, percentage_of_generations_for_performance: float) -> int:
    result = round(number_of_iterations * percentage_of_generations_for_performance)
    if result < 2:
        result = 2
    return result



