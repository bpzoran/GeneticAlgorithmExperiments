from settings.experiment_ga_settings import ExperimentGASettings
from utils.exp_logging import log_message_info
from runners.experiment import Experiment
from runners.experiment_runner import run_experiment
from functions.sphere import sphere_func

variable_numbers = [2, 3, 7]
saturation_criterias = [3, 5, 10, 30]
def execute():
    log_message_info("Sphere function")
    app_settings = ExperimentGASettings()
    app_settings.saturation_criterias = list(set(app_settings.saturation_criterias) & set(saturation_criterias))
    experiment = Experiment(sphere_func)
    experiment.fill_args_with_same_values(-5.12, 5.12, variable_numbers, 0.1024)
    experiment.execute_experiment()

def main():
    run_experiment(execute)

if __name__ == "__main__":
    main()
