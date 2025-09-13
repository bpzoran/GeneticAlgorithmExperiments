from utils.exp_logging import log_message_info
from runners.experiment import Experiment
from runners.experiment_runner import run_experiment
from functions.himmelblau import himmelblau_func
from settings.experiment_ga_settings import ExperimentGASettings

saturation_criterias = [10, 30]
def execute():
    log_message_info("Himmelblau function")
    app_settings = ExperimentGASettings()
    app_settings.saturation_criterias = list(set(app_settings.saturation_criterias) & set(saturation_criterias))
    experiment = Experiment(himmelblau_func)
    experiment.fill_args_with_same_values(-6, 6, 2)
    experiment.execute_experiment()

def main():
    run_experiment(execute)

if __name__ == "__main__":
    main()
