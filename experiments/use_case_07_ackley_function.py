from settings.experiment_ga_settings import ExperimentGASettings
from use_case_03_moderately_coupled_trigonometric_function import TITLE
from utils.exp_logging import log_message_info
from runners.experiment import Experiment
from runners.experiment_runner import run_experiment
from functions.ackley import ackley_func
TITLE = "Ackley function"
ENABLED = True
def execute():
    log_message_info(TITLE)
    experiment = Experiment(ackley_func)
    experiment.fill_args_with_same_values(-4, 4, 0, 0.08)
    experiment.execute_experiment()

def main():
    if not ENABLED:
        log_message_info(f"{TITLE} - Experiment disabled")
        return
    run_experiment(execute)

if __name__ == "__main__":
    main()
