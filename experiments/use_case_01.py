from exp_logging import log_message_info
from experiment import Experiment
from experiment_runner import run_experiment
from functions.simple_trigonometric_arithmetic import simple_trigonometric_arithmetic_func


def execute_use_case_1_simple_trigonometric_arithmetic_function():
    log_message_info("Simple trigonometric arithmetic function")
    args_bounds = [{"low": 1.0, "high": 4.0, "step": 0.01},  # arg1
                   {"low": 37.0, "high": 40.0, "step": 0.01},  # arg2
                   {"low": 78, "high": 88.0, "step": 0.1},  # arg3
                   {"low": -5.0, "high": 4.0, "step": 0.1},  # arg4
                   {"low": 0, "high": 100, "step": 1},  # arg5
                   ]

    experiment = Experiment(simple_trigonometric_arithmetic_func, args_bounds=args_bounds)
    experiment.execute_experiment()

def main():
    run_experiment(execute_use_case_1_simple_trigonometric_arithmetic_function)

if __name__ == "__main__":
    main()
