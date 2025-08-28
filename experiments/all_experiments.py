import use_case_01
import use_case_02
import use_case_03
import use_case_04_sphere_function
import use_case_05_rosenbrock_function_3_variables
import use_case_06_rosenbrock_function_7_variables
import use_case_07_rastrigin_function_3_variables
import use_case_08_rastrigin_function_7_variables
import use_case_09_ackley_function_7_variables
from experiment_runner import run_experiment
from settings.experiment_ga_settings import ExperimentGASettings


def run_all_use_cases() -> None:
    app_settings = ExperimentGASettings()
    app_settings.num_runs = 50
    use_case_01.main()
    use_case_02.main()
    use_case_03.main()
    use_case_04_sphere_function.main()
    use_case_05_rosenbrock_function_3_variables.main()
    use_case_06_rosenbrock_function_7_variables.main()
    use_case_07_rastrigin_function_3_variables.main()
    use_case_08_rastrigin_function_7_variables.main()
    use_case_09_ackley_function_7_variables.main()

def main() -> None:
    run_experiment(run_all_use_cases)

if __name__ == "__main__":
    main()
