from exp_logging import log_message_info
from experiment import Experiment
from experiment_runner import run_experiment
from functions.sphere import sphere_func


def execute_use_case_4_sphere_function_4_variables():
    log_message_info("Sphere function - 4 variables")
    experiment = Experiment(sphere_func)
    experiment.fill_args_with_same_values(-5.12, 5.12, 4, 0.01)
    experiment.execute_experiment()

def main():
    run_experiment(execute_use_case_4_sphere_function_4_variables)

if __name__ == "__main__":
    main()
