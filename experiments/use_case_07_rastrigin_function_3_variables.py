from settings.experiment_ga_settings import ExperimentGASettings
from utils.exp_logging import log_message_info
from runners.experiment import Experiment
from runners.experiment_runner import run_experiment
from functions.rastrigin import rastrigin_func


def execute_use_case_7_ratrigin_function():
    log_message_info("Ratrigin function - 3 variables")
    app_settings = ExperimentGASettings()
    app_settings.saturation_criteria = 30
    experiment = Experiment(rastrigin_func)
    experiment.fill_args_with_same_values(-5.12, 5.12, 3, 0.1024)
    experiment.execute_experiment()

def main():
    run_experiment(execute_use_case_7_ratrigin_function)

if __name__ == "__main__":
    main()
