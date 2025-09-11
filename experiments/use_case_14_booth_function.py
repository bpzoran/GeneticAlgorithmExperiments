from utils.exp_logging import log_message_info
from runners.experiment import Experiment
from runners.experiment_runner import run_experiment
from functions.booth import booth_func
from settings.experiment_ga_settings import ExperimentGASettings


def execute_use_case_14_booth_function():
    log_message_info("booth function - 2 variables")
    app_settings = ExperimentGASettings()
    app_settings.saturation_criteria = 10
    app_settings.plot_fitness = True
    app_settings.num_runs = 50
    experiment = Experiment(booth_func)
    experiment.fill_args_with_same_values(-10, 10, 2, 0.1)
    experiment.execute_experiment()

def main():
    run_experiment(execute_use_case_14_booth_function)

if __name__ == "__main__":
    main()
