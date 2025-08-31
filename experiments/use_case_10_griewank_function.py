from utils.exp_logging import log_message_info
from runners.experiment import Experiment
from runners.experiment_runner import run_experiment
from functions.griewank import griewank_func
from settings.experiment_ga_settings import ExperimentGASettings


def execute_use_case_10_griewank_function():
    log_message_info("Griewank function - 7 variables")
    app_settings = ExperimentGASettings()
    app_settings.saturation_criteria = 30
    experiment = Experiment(griewank_func)
    experiment.fill_args_with_same_values(-50, 50, 7, 1)
    experiment.execute_experiment()

def main():
    run_experiment(execute_use_case_10_griewank_function)

if __name__ == "__main__":
    main()
