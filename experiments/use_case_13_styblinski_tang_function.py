from utils.exp_logging import log_message_info
from runners.experiment import Experiment
from runners.experiment_runner import run_experiment
from functions.styblinski_tang import styblinski_tang_func
from settings.experiment_ga_settings import ExperimentGASettings


def execute_use_case_13_styblinski_tang_function():
    log_message_info("styblinski_tang function - 3 variables")
    app_settings = ExperimentGASettings()
    app_settings.saturation_criteria = 30
    app_settings.plot_fitness = True
    app_settings.num_runs = 200
    experiment = Experiment(styblinski_tang_func)
    experiment.fill_args_with_same_values(-5, 5, 3, 0.1)
    experiment.execute_experiment()

def main():
    run_experiment(execute_use_case_13_styblinski_tang_function)

if __name__ == "__main__":
    main()
