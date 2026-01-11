from settings.experiment_ga_settings import ExperimentGASettings
from utils.exp_logging import log_message_info
from runners.experiment import Experiment
from runners.experiment_runner import run_experiment
from functions.sphere import sphere_func
TITLE = "Sphere Function"
ENABLED = True
def execute():
    log_message_info(TITLE)
    experiment = Experiment(sphere_func)
    experiment.fill_args_with_same_values(-5.12, 5.12, 0, 0.1024)
    experiment.execute_experiment()

def main():
    if not ENABLED:
        log_message_info(f"{TITLE} - Experiment disabled")
        return
    run_experiment(execute)

if __name__ == "__main__":
    main()
