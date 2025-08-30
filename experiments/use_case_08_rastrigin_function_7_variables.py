from exp_logging import log_message_info
from experiment import Experiment
from experiment_runner import run_experiment
from functions.rastrigin import rastrigin_func
from settings.experiment_ga_settings import ExperimentGASettings


def execute_use_case_8_ratrigin_function():
    log_message_info("Ratrigin function - 7 variables")
    ExperimentGASettings().saturation_criteria = 30
    experiment = Experiment(rastrigin_func)
    experiment.fill_args_with_same_values(-5.12, 5.12, 7, 0.1024)
    experiment.execute_experiment()

def main():
    run_experiment(execute_use_case_8_ratrigin_function)

if __name__ == "__main__":
    main()
