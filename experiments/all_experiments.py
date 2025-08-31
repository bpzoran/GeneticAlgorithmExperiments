import use_case_01_simple_trigonometric_arithmetic_function
import use_case_02_complex_trig_function
import use_case_03_simple_trig_function
import use_case_04_sphere_function
import use_case_05_rosenbrock_function_3_variables
import use_case_06_rosenbrock_function_7_variables
import use_case_07_rastrigin_function_3_variables
import use_case_08_rastrigin_function_7_variables
import use_case_09_ackley_function_7_variables
import use_case_10_griewank_function
import use_case_11_beale_function
import use_case_12_himmelblau_function
from runners.experiment_runner import run_experiment
from settings.experiment_ga_settings import ExperimentGASettings


def run_all_use_cases() -> None:
    app_settings = ExperimentGASettings()
    app_settings.num_runs = 1000
    app_settings.logging_step = 1000
    app_settings.backup_settings()
    use_case_01_simple_trigonometric_arithmetic_function.main()
    app_settings.restore_settings()
    use_case_02_complex_trig_function.main()
    app_settings.restore_settings()
    use_case_03_simple_trig_function.main()
    app_settings.restore_settings()
    use_case_04_sphere_function.main()
    app_settings.restore_settings()
    use_case_05_rosenbrock_function_3_variables.main()
    app_settings.restore_settings()
    use_case_06_rosenbrock_function_7_variables.main()
    app_settings.restore_settings()
    use_case_07_rastrigin_function_3_variables.main()
    app_settings.restore_settings()
    use_case_08_rastrigin_function_7_variables.main()
    app_settings.restore_settings()
    use_case_09_ackley_function_7_variables.main()
    app_settings.restore_settings()
    use_case_10_griewank_function.main()
    app_settings.restore_settings()
    use_case_11_beale_function.main()
    app_settings.restore_settings()
    use_case_12_himmelblau_function.main()
    app_settings.restore_settings()

def main() -> None:
    run_experiment(run_all_use_cases)

if __name__ == "__main__":
    main()
