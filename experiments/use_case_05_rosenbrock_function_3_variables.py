from exp_logging import log_message_info
from experiment import Experiment
from experiment_runner import run_experiment
from functions.rosenbrock import rosenbrock_func


def execute_use_case_5_rosenbrock_function_3_variables():
    log_message_info("Rosenbrock function - 3 variables")
    experiment = Experiment(rosenbrock_func)
    experiment.fill_args_with_same_values(0.5, 1.5, 0.01, 3)
    experiment.execute_experiment()

def main():
    run_experiment(execute_use_case_5_rosenbrock_function_3_variables)

if __name__ == "__main__":
    main()
