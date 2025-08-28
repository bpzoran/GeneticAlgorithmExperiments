import sys
from typing import Callable

from exp_logging import init_logging
from settings.experiment_ga_settings import ExperimentGASettings

def run_experiment(experiment: Callable) -> None:
    init_logging(log_to_file=False)
    app_settings = ExperimentGASettings()

    for a in sys.argv[1:]:
        if a == "plot":
            app_settings.plot_fitness = True
        else:
            try:
                app_settings.num_runs = int(a)
            except ValueError:
                pass

    experiment()
