from utils.exp_logging import log_message_info
from runners.experiment import Experiment
from runners.experiment_runner import run_experiment
from functions.styblinski_tang import styblinski_tang_func
from settings.experiment_ga_settings import ExperimentGASettings
variable_numbers = [2, 3, 7]
saturation_criterias = [3, 5, 10, 30]

def execute():
    log_message_info("styblinski_tang function")
    app_settings = ExperimentGASettings()
    app_settings.saturation_criterias = list(set(app_settings.saturation_criterias) & set(saturation_criterias))
    experiment = Experiment(styblinski_tang_func)
    experiment.fill_args_with_same_values(-5, 5, variable_numbers, 0.1)
    experiment.execute_experiment()

def main():
    run_experiment(execute)

if __name__ == "__main__":
    main()
