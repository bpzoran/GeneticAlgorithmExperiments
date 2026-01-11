from settings.experiment_ga_settings import ExperimentGASettings
from utils.exp_logging import log_message_info
from runners.experiment import Experiment
from runners.experiment_runner import run_experiment
from functions.separated_trig_arithmetic import separated_trig_arithmetic_func
TITLE = "Separated Trig Arithmetic Function"
ENABLED = False
def execute():
    log_message_info(TITLE)
    args_bounds = [{"low": 1.0, "high": 4.0, "step": 0.01},  # arg1
                   {"low": 37.0, "high": 40.0, "step": 0.01},  # arg2
                   {"low": 78, "high": 88.0, "step": 0.1},  # arg3
                   {"low": -5.0, "high": 4.0, "step": 0.1},  # arg4
                   {"low": 0, "high": 100, "step": 1},  # arg5
                   ]

    experiment = Experiment(separated_trig_arithmetic_func, args_bounds=args_bounds)
    experiment.execute_experiment()

def main():
    if not ENABLED:
        log_message_info(f"{TITLE} - Experiment disabled")
        return
    run_experiment(execute)

if __name__ == "__main__":
    main()
