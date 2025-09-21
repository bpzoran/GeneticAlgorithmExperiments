import math

from settings.experiment_ga_settings import ExperimentGASettings
from utils.exp_logging import log_message_info
from runners.experiment import Experiment
from runners.experiment_runner import run_experiment
from functions.moderately_coupled_trigonometric import simple_trig_func

saturation_criterias = [10, 30]

def execute():
    log_message_info("Simple trigonometric function")
    app_settings = ExperimentGASettings()
    app_settings.saturation_criterias = list(set(app_settings.saturation_criterias) & set(saturation_criterias))
    args_bounds = [{"low": 0, "high": math.pi, "step": 0.0157},  # arg1
                   {"low": 0, "high": math.pi, "step": 0.0157},  # arg2
                   {"low": 0, "high": 200, "step": 1},  # arg3
                   {"low": 0, "high": math.pi, "step": 0.0157},  # arg4
                   {"low": 0, "high": math.pi, "step": 0.0157},  # arg5
                   {"low": 0, "high": 200, "step": 1},  # arg6
                   {"low": 0, "high": math.pi, "step": 0.0157},  # arg7
                   {"low": 0, "high": 200, "step": 1}  # arg8
                   ]

    experiment = Experiment(simple_trig_func, args_bounds=args_bounds)
    experiment.execute_experiment()

def main():
    run_experiment(execute)

if __name__ == "__main__":
    main()
