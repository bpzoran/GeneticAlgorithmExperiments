import csv
import math
import os
from typing import Any, Dict

from settings.experiment_ga_settings import ExperimentGASettings


def aggregated_data_to_csv(data: dict, experiment_name: str = "output.csv"):
    app_settings = ExperimentGASettings()
    file_path = os.path.join(app_settings.csv_path, f"aggregated_data_{experiment_name}.csv")
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "strategy", "gen", "center", "lower", "upper", "n_at_gen"])  # header

        for strategy, metrics in data.items():
            for i in range(len(metrics["gen"])):
                writer.writerow([
                    experiment_name,
                    strategy,
                    metrics["gen"][i],
                    metrics["center"][i] if i < len(metrics["center"]) else "",
                    metrics["lower"][i] if i < len(metrics["lower"]) else "",
                    metrics["upper"][i] if i < len(metrics["upper"]) else "",
                    metrics["n_at_gen"][i] if i < len(metrics["n_at_gen"]) else "",
                ])


def runs_to_csv(data: dict, experiment_name: str) -> str:
    """
    Export optimization results to CSV.

    Args:
        data (dict): Dictionary where key = optimization type,
                     value = list of runs, each run = list of minimal costs per generation.
        directory (str): Directory where CSV should be saved.
        experiment_name (str): Name of the CSV file.

    Returns:
        str: Full file path of the saved CSV.
    """
    app_settings = ExperimentGASettings()
    # Ensure directory exists
    directory = app_settings.csv_path
    os.makedirs(directory, exist_ok=True)

    file_path = os.path.join(directory, f"runs_{experiment_name}.csv")

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "strategy", "run", "generation", "cost"])  # header

        for strategy, runs in data.items():
            for run_idx, run in enumerate(runs, start=1):
                for gen_idx, cost in enumerate(run):
                    writer.writerow([experiment_name, strategy, run_idx, gen_idx, cost])

    return file_path

def results_to_csv(results: dict, experiment_name: str) -> None:
    if not experiment_name.endswith(".csv"):
        experiment_name += ".csv"
    app_settings = ExperimentGASettings()
    # Ensure directory exists
    directory = app_settings.csv_path
    os.makedirs(directory, exist_ok=True)
    full_path = os.path.join(directory, f"final_results_{experiment_name}")

    # Extract the experiment data
    experiment_data = results.get("Experiment", {})
    experiment_name = experiment_data.get("Experiment name", "")
    num_vars = experiment_data.get("Number of variables", "")
    saturation = experiment_data.get("Saturation after generations", "")

    # Define header
    header = [
        "Experiment name",
        "Number of variables",
        "Saturation after generations",
        "Mutation type",
        "Average fitness",
        f"Average fitness after {app_settings.number_of_generations} generations",
        "Average number of generations"
    ]

    with open(full_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for mutation_type, metrics in experiment_data.items():
            if mutation_type in ("Experiment name", "Number of variables", "Saturation after generations"):
                continue  # skip metadata
            writer.writerow([
                experiment_name,
                num_vars,
                saturation,
                mutation_type,
                metrics.get("Average fitness", ""),
                metrics.get(next((k for k in metrics if k.startswith("Average fitness after")), ""), ""),
                metrics.get("Average number of generations", "")
            ])
def export_ga_summary_to_csv(
    summary: Dict[str, Dict[str, Any]],
    experiment_name: str,
    number_of_variables: int,
    saturation_generations: int,
    float_fmt: str = ".6f",
) -> str:
    if not experiment_name.endswith(".csv"):
        experiment_name += ".csv"
    app_settings = ExperimentGASettings()
    # Ensure directory exists
    directory = app_settings.csv_path
    os.makedirs(directory, exist_ok=True)
    full_path = os.path.join(directory, f"final_results_{experiment_name}")

    fieldnames = [
        "experiment",
        "number_of_variables",
        "saturation_generations",
        "strategy",
        "num_runs",
        "avg_min_fitness",
        "rsd_min_fitness",
        "avg_generations",
        "avg_fitness_after_n",
        "avg_slope_first_n",
    ]

    def _fmt(x: Any) -> Any:
        if isinstance(x, float):
            if math.isnan(x) or math.isinf(x):
                return ""
            return format(x, float_fmt)
        return x

    with open(full_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for strategy, metrics in summary.items():
            row = {
                "experiment": experiment_name,
                "number_of_variables": number_of_variables,
                "saturation_generations": saturation_generations,
                "strategy": strategy,
                "num_runs": metrics.get("num_runs"),
                "avg_min_fitness": _fmt(metrics.get("avg_min_fitness")),
                "rsd_min_fitness": _fmt(metrics.get("rsd_min_fitness")),
                "avg_generations": _fmt(metrics.get("avg_generations")),
                "avg_fitness_after_n": _fmt(metrics.get("avg_fitness_after_n")),
                "avg_slope_first_n": _fmt(metrics.get("avg_slope_first_n")),
            }
            writer.writerow(row)

    return full_path