import math

from settings.experiment_ga_settings import ExperimentGASettings
from utils.exp_logging import log_message_info
from runners.experiment import Experiment
from runners.experiment_runner import run_experiment
from functions.moderately_coupled_trigonometric import moderately_coupled_trigonometric_func
TITLE = "Moderately Coupled Trigonometric Function"
ENABLED = False

def execute():
    log_message_info(TITLE)
    args_bounds = [{"low": 0, "high": math.pi, "step": 0.0157},  # arg1
                   {"low": 0, "high": math.pi, "step": 0.0157},  # arg2
                   {"low": 0, "high": 200, "step": 1},  # arg3
                   {"low": 0, "high": math.pi, "step": 0.0157},  # arg4
                   {"low": 0, "high": math.pi, "step": 0.0157},  # arg5
                   {"low": 0, "high": 200, "step": 1},  # arg6
                   {"low": 0, "high": math.pi, "step": 0.0157},  # arg7
                   {"low": 0, "high": 200, "step": 1}  # arg8
                   ]

    experiment = Experiment(moderately_coupled_trigonometric_func, args_bounds=args_bounds)
    experiment.execute_experiment()

def main():
    if not ENABLED:
        log_message_info(f"{TITLE} - Experiment disabled")
        return
    run_experiment(execute)

if __name__ == "__main__":
    main()
