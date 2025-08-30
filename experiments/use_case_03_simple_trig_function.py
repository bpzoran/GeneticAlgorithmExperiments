import math

from exp_logging import log_message_info
from experiment import Experiment
from experiment_runner import run_experiment
from functions.simple_trig import simple_trig_func
from settings.experiment_ga_settings import ExperimentGASettings


def execute_use_case_03_simple_trig_function():
    log_message_info("Simple trigonometric function")
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
    run_experiment(execute_use_case_03_simple_trig_function)

if __name__ == "__main__":
    main()
