from settings.experiment_ga_settings import ExperimentGASettings
from utils.exp_logging import log_message_info
from runners.experiment import Experiment
from runners.experiment_runner import run_experiment
from functions.rastrigin import rastrigin_func

variable_numbers = [2, 7]
saturation_criterias = [10, 30]

def execute():
    log_message_info("Ratrigin function - 2 variables")
    app_settings = ExperimentGASettings()
    app_settings.saturation_criterias = list(set(app_settings.saturation_criterias) & set(saturation_criterias))
    app_settings.plot_fitness = True
    experiment = Experiment(rastrigin_func)
    experiment.fill_args_with_same_values(-5.12, 5.12, variable_numbers, 0.1024)
    experiment.execute_experiment()

def main():
    run_experiment(execute)

if __name__ == "__main__":
    main()
