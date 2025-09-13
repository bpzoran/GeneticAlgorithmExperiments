from settings.experiment_ga_settings import ExperimentGASettings
from utils.exp_logging import log_message_info
from runners.experiment import Experiment
from runners.experiment_runner import run_experiment
from functions.rosenbrock import rosenbrock_func

variable_numbers = [3, 7]
saturation_criterias = [10, 30]
def execute():
    log_message_info("Rosenbrock function")
    app_settings = ExperimentGASettings()
    app_settings.saturation_criterias = list(set(app_settings.saturation_criterias) & set(saturation_criterias))
    experiment = Experiment(rosenbrock_func)
    experiment.fill_args_with_same_values(0.5, 1.5, variable_numbers, 0.01)
    experiment.execute_experiment()

def main():
    run_experiment(execute)

if __name__ == "__main__":
    main()
