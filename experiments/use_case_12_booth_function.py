from utils.exp_logging import log_message_info
from runners.experiment import Experiment
from runners.experiment_runner import run_experiment
from functions.booth import booth_func
from settings.experiment_ga_settings import ExperimentGASettings
TITLE = "Booth function"
ENABLED = True
def execute():
    log_message_info(TITLE)
    experiment = Experiment(booth_func)
    experiment.fill_args_with_same_values(-10, 10, 2, 0.1)
    experiment.execute_experiment()

def main():
    if not ENABLED:
        log_message_info(f"{TITLE} - Experiment disabled")
        return
    run_experiment(execute)

if __name__ == "__main__":
    main()
