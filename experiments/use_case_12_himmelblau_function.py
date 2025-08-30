import sys

from exp_logging import log_message_info
from experiment import Experiment
from experiment_runner import run_experiment
from functions.himmelblau import himmelblau_func
from settings.experiment_ga_settings import ExperimentGASettings


def execute_use_case_12_himmelblau_function():
    log_message_info("Himmelblau function - 7 variables")
    app_settings = ExperimentGASettings()
    app_settings.saturation_criteria = 30
    experiment = Experiment(himmelblau_func)
    experiment.fill_args_with_same_values(-6, 6, 2)
    experiment.execute_experiment()

def main():
    run_experiment(execute_use_case_12_himmelblau_function)

if __name__ == "__main__":
    main()
