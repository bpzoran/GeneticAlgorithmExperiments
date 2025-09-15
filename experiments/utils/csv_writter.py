import csv
import os

from settings.experiment_ga_settings import ExperimentGASettings


def aggregated_data_to_csv(data: dict, filename: str = "output.csv"):
    app_settings = ExperimentGASettings()
    file_path = os.path.join(app_settings.csv_path, f"aggregated_data_{filename}.csv")
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "gen", "center", "lower", "upper", "n_at_gen"])  # header

        for strategy, metrics in data.items():
            for i in range(len(metrics["gen"])):
                writer.writerow([
                    strategy,
                    metrics["gen"][i],
                    metrics["center"][i] if i < len(metrics["center"]) else "",
                    metrics["lower"][i] if i < len(metrics["lower"]) else "",
                    metrics["upper"][i] if i < len(metrics["upper"]) else "",
                    metrics["n_at_gen"][i] if i < len(metrics["n_at_gen"]) else "",
                ])


def runs_to_csv(data: dict, filename: str) -> str:
    """
    Export optimization results to CSV.

    Args:
        data (dict): Dictionary where key = optimization type,
                     value = list of runs, each run = list of minimal costs per generation.
        directory (str): Directory where CSV should be saved.
        filename (str): Name of the CSV file.

    Returns:
        str: Full file path of the saved CSV.
    """
    app_settings = ExperimentGASettings()
    # Ensure directory exists
    directory = app_settings.csv_path
    os.makedirs(directory, exist_ok=True)

    file_path = os.path.join(directory, f"runs_{filename}.csv")

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "run", "generation", "cost"])  # header

        for strategy, runs in data.items():
            for run_idx, run in enumerate(runs, start=1):
                for gen_idx, cost in enumerate(run):
                    writer.writerow([strategy, run_idx, gen_idx, cost])

    return file_path