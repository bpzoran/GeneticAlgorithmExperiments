import sys
from typing import Callable

from utils.exp_logging import init_logging
from settings.experiment_ga_settings import ExperimentGASettings

def run_experiment(experiment: Callable) -> None:
    app_settings = ExperimentGASettings()
    if app_settings.csv_path is None:
        init_logging(log_to_file=app_settings.log_to_file)

    for a in sys.argv[1:]:
        if a == "plot":
            app_settings.plot_fitness = True
        else:
            try:
                app_settings.num_runs = int(a)
            except ValueError:
                pass

    experiment()
