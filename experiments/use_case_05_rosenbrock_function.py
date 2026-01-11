from settings.experiment_ga_settings import ExperimentGASettings
from utils.exp_logging import log_message_info
from runners.experiment import Experiment
from runners.experiment_runner import run_experiment
from functions.rosenbrock import rosenbrock_func
TITLE = "Rosenbrock function"
ENABLED = True
def execute():
    log_message_info(TITLE)
    experiment = Experiment(rosenbrock_func)
    experiment.fill_args_with_same_values(0.5, 1.5, 0, 0.01)
    experiment.execute_experiment()

def main():
    if not ENABLED:
        log_message_info(f"{TITLE} - Experiment disabled")
        return
    run_experiment(execute)

if __name__ == "__main__":
    main()
