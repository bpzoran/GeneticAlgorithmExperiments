from settings.experiment_ga_settings import ExperimentGASettings
from utils.exp_logging import log_message_info
from runners.experiment import Experiment
from runners.experiment_runner import run_experiment
from functions.rastrigin import rastrigin_func
TITLE = "Rastrigin function"
ENABLED = True
variable_numbers = [2, 3, 7]
saturation_criterias = [3, 5, 10, 15, 20, 30]

def execute():
    log_message_info(TITLE)
    experiment = Experiment(rastrigin_func)
    experiment.fill_args_with_same_values(-5.12, 5.12)
    experiment.execute_experiment()

def main():
    if not ENABLED:
        log_message_info(f"{TITLE} - Experiment disabled")
        return
    run_experiment(execute)

if __name__ == "__main__":
    main()
