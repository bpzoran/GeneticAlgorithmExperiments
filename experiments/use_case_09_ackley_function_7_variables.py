from exp_logging import log_message_info
from experiment import Experiment
from experiment_runner import run_experiment
from functions.ackley import ackley_func
from settings.experiment_ga_settings import ExperimentGASettings


def execute_use_case_9_ackley_function():
    log_message_info("Ackley function - 7 variables")
    experiment = Experiment(ackley_func)
    experiment.fill_args_with_same_values(-4, 4, 7, 0.08)
    experiment.execute_experiment()

def main():
    run_experiment(execute_use_case_9_ackley_function)

if __name__ == "__main__":
    main()
