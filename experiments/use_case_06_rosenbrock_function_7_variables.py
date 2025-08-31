from utils.exp_logging import log_message_info
from runners.experiment import Experiment
from runners.experiment_runner import run_experiment
from functions.rosenbrock import rosenbrock_func
from settings.experiment_ga_settings import ExperimentGASettings


def execute_use_case_6_rosenbrock_function_7_variables():
    log_message_info("Rosenbrock function - 7 variables")
    ExperimentGASettings().saturation_criteria = 30
    experiment = Experiment(rosenbrock_func)
    experiment.fill_args_with_same_values(0.5, 1.5, 7, 0.01)
    experiment.execute_experiment()

def main():
    run_experiment(execute_use_case_6_rosenbrock_function_7_variables)

if __name__ == "__main__":
    main()
