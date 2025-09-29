import csv
import math
import os
from typing import Any, Dict

from settings.experiment_ga_settings import ExperimentGASettings

from pathlib import Path
import csv
import gzip
from typing import Iterable

from utils.experiment_utils import relative_early_convergence


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
        f"Average fitness after {app_settings.percentage_of_generations_for_performance * 100} generations",
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
def get_first_avg_fitness_after_key(data: dict) -> str | None:
    prefix = "avg_fitness_after_"
    for strategy, metrics in data.items():
        for key in metrics.keys():
            if key.startswith(prefix):
                return key[len(prefix):]   # take the part after prefix
    return None

def export_ga_summary_to_csv(
    summary: Dict[str, Dict[str, Any]],
    file_name: str,
    experiment_name: str,
    number_of_variables: int,
    saturation_generations: int,
    float_fmt: str = ".6f",
) -> str:
    if not file_name.endswith(".csv"):
        file_name += ".csv"
    app_settings = ExperimentGASettings()
    # Ensure directory exists
    directory = app_settings.csv_path
    os.makedirs(directory, exist_ok=True)
    full_path = os.path.join(directory, f"final_results_{file_name}")

    fieldnames = [
        "experiment",
        "number_of_variables",
        "saturation_generations",
        "strategy",
        "num_runs",
        "avg_min_fitness",
        "sd_min_fitness",
        "avg_generations_for_performance",
        "avg_generations",
        "avg_fitness_after_first",
        "avg_fitness_after_n",
        "relative_early_convergence",
        "area_under_convergence_curve",
        "avg_slope_first_n",
    ]

    def _fmt(x: Any) -> Any:
        if isinstance(x, float):
            if math.isnan(x) or math.isinf(x):
                return ""
            return format(x, float_fmt)
        return x
    number_of_gens_for_performance = get_first_avg_fitness_after_key(summary)
    f_names = [
        item[:-1] + number_of_gens_for_performance if item.endswith('_n') else item
        for item in fieldnames]


    with open(full_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=f_names)
        writer.writeheader()
        for strategy, metrics in summary.items():
            avg_fitness_after_n = [k for k in metrics.keys() if k.startswith("avg_fitness_after_")][0]
            avg_slope_first_n = [k for k in metrics.keys() if k.startswith("avg_slope_first_")][0]
            row = {
                "experiment": experiment_name,
                "number_of_variables": number_of_variables,
                "saturation_generations": saturation_generations,
                "strategy": strategy,
                "num_runs": metrics.get("num_runs"),
                "avg_min_fitness": _fmt(metrics.get("avg_min_fitness")),
                "sd_min_fitness": _fmt(metrics.get("sd_min_fitness")),
                "avg_generations_for_performance": _fmt(metrics.get("avg_generations_for_performance")),
                "avg_generations": _fmt(metrics.get("avg_generations")),
                avg_fitness_after_n: _fmt(metrics.get(avg_fitness_after_n)),
                "avg_fitness_after_first": _fmt(metrics.get("avg_fitness_after_first")),
                "relative_early_convergence": _fmt(metrics.get("relative_early_convergence")),
                "area_under_convergence_curve": _fmt(metrics.get("area_under_convergence_curve")),
                avg_slope_first_n: _fmt(metrics.get(avg_slope_first_n)),
            }



            writer.writerow(row)

    return full_path



def merge_csvs(
    input_dir: str | Path,
    filename_prefix: str,
    output_file: str | Path,
    *,
    delimiter: str = ",",
    encoding: str = "utf-8",
    recursive: bool = False,
    strict: bool = True,
) -> tuple[int, int]:
    """
    Merge CSV files whose names start with `filename_prefix` from `input_dir`
    into `output_file`.

    - Assumes same header/column order across files.
    - Writes the header once (from the first non-empty file).
    - If `strict=True`, raises if any file has a different row length.
    - If `output_file` ends with '.gz', writes a gzipped CSV.

    Returns: (files_merged, rows_written_excluding_header)
    """
    input_dir = Path(input_dir)
    pattern = f"{filename_prefix}*.csv"
    files: Iterable[Path] = (
        input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
    )
    files = sorted(f for f in files if f.is_file())

    if not files:
        raise FileNotFoundError(f"No CSVs matching '{pattern}' in {input_dir}")

    # Open output (supports .gz)
    open_out = gzip.open if str(output_file).endswith(".gz") else open
    rows_written = 0
    files_merged = 0
    header_ref: list[str] | None = None
    writer = None

    with open_out(output_file, "wt", newline="", encoding=encoding) as out_f:
        for fpath in files:
            with open(fpath, "rt", newline="", encoding=encoding) as in_f:
                reader = csv.reader(in_f, delimiter=delimiter)
                header = next(reader, None)
                if header is None:
                    # empty file; skip
                    continue

                # Initialize writer on first non-empty file
                if writer is None:
                    writer = csv.writer(out_f, delimiter=delimiter)
                    header_ref = header
                    writer.writerow(header_ref)

                for row in reader:
                    if strict and len(row) != len(header_ref):  # type: ignore[arg-type]
                        raise ValueError(
                            f"Row length {len(row)} != expected {len(header_ref)} in {fpath.name}: {row}"
                        )
                    writer.writerow(row)
                    rows_written += 1

                files_merged += 1

    return files_merged, rows_written
