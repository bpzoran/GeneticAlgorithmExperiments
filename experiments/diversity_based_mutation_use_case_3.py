import math
import sys

import numpy as np
import pygad

from gadapt.ga import GA

from exp_logging import init_logging, log_message_info
from gadapt_experiment import execute_gadapt_experiment
from pygad_experiment import execute_pygad_experiment

num_runs = 1000
logging_step = 50
number_of_generations = 40
plot_fitness = False


def simple_trig_func(args):
    """
    Arithmetic / trigonometric function computes a mathematical expression using trigonometric
    and arithmetic operations on elements of an input array.

    This function performs the following steps:

    1. Computes the sine of the first element (args[0]).
    2. Computes the cosine of the second element (args[1]).
    3. Adds the square of the third element (args[2]).
    4. Multiplies the sine of the fourth element (args[3]) by the cosine of the fifth element (args[4]) and adds the result.
    5. Adds the value of the sixth element (args[5]).
    6. Multiplies the cosine of the seventh element (args[6]) by the eighth element (args[7]) and adds the result.

    The function ensures that the input list contains exactly 8 elements. If not, it raises a ValueError.

    Each operation in this function is independent of the others, but each can contain one or more local minima,
    making the function useful for optimization experiments.

    Parameters:
    args (list): A list of 8 numerical values used as input for the function.

    Returns:
    float: The result of the mathematical expression.
    """
    if len(args) != 8:
        raise ValueError("Input vector must contain 8 variables.")
    return np.sin(args[0]) + np.cos(args[1]) + args[2] ** 2 + np.sin(args[3]) * np.cos(args[4]) + args[5] + np.cos(
        args[6]) * args[7]


def execute_diversity_based_mutation_use_case_3():
    """
    The execute_diversity_based_mutation_use_case_3 function performs
    optimization using genetic algorithms (GAs) with different mutation strategies.
    It utilizes two libraries, PyGAD and GAdapt, to optimize a trigonometric function over multiple runs,
    comparing the performance of adaptive, random, and diversity mutation strategies.

    Flow
    Define a trigonometric function to optimize.
    Perform optimization using PyGAD with adaptive mutation across multiple runs, collecting and logging results.
    Perform optimization using GAdapt with random mutation, logging results similarly.
    Perform optimization using GAdapt with diversity mutation, again logging results.
    Print final results from all optimization strategies.
    """
    init_logging(log_to_file=True)

    result_list = []

    ##### GADAPT OPTIMIZATION WITH RANDOM MUTATION ###############

    log_message_info("Start optimization with GAdapt, random mutation:")

    ga = GA(cost_function=simple_trig_func,
            population_size=32,
            population_mutation="random",
            chromosome_mutation="random",
            gene_mutation="random",
            percentage_of_mutation_chromosomes=80,
            percentage_of_mutation_genes=40,
            exit_check="min_cost",
            keep_elitism_percentage=50,
            max_attempt_no=10,
            logging=False)

    # Addition of variables with specified ranges and steps
    ga.add(min_value=0.0, max_value=math.pi, step=0.0157)
    ga.add(min_value=0.0, max_value=math.pi, step=0.0157)
    ga.add(min_value=0.0, max_value=200, step=1)
    ga.add(min_value=0.0, max_value=math.pi, step=0.0157)
    ga.add(min_value=0.0, max_value=math.pi, step=0.0157)
    ga.add(min_value=0.0, max_value=200, step=1)
    ga.add(min_value=0.0, max_value=math.pi, step=0.0157)
    ga.add(min_value=0.0, max_value=200, step=1)

    execute_gadapt_experiment(ga, "random mutation", num_runs, number_of_generations, logging_step, plot_fitness, result_list)

    ##### GADAPT OPTIMIZATION WITH DIVERSITY MUTATION ###############

    log_message_info("Start optimization with GAdapt, diversity mutation:")

    ga = GA(cost_function=simple_trig_func,
            population_size=32,
            population_mutation="cost_diversity,parent_diversity,cross_diversity,random",
            chromosome_mutation="cross_diversity,random",
            gene_mutation="cross_diversity,random",
            percentage_of_mutation_chromosomes=100,
            percentage_of_mutation_genes=50,
            keep_elitism_percentage=50,
            exit_check="min_cost",
            max_attempt_no=10,
            logging=False)

    # Addition of variables with specified ranges and steps
    ga.add(min_value=0.0, max_value=math.pi, step=0.0157)
    ga.add(min_value=0.0, max_value=math.pi, step=0.0157)
    ga.add(min_value=0.0, max_value=200, step=1)
    ga.add(min_value=0.0, max_value=math.pi, step=0.0157)
    ga.add(min_value=0.0, max_value=math.pi, step=0.0157)
    ga.add(min_value=0.0, max_value=200, step=1)
    ga.add(min_value=0.0, max_value=math.pi, step=0.0157)
    ga.add(min_value=0.0, max_value=200, step=1)

    execute_gadapt_experiment(ga, "diversity mutation", num_runs, number_of_generations, logging_step, plot_fitness, result_list)

    ##### PYGAD OPTIMIZATION WITH ADAPTIVE MUTATION ###############

    log_message_info("Start optimization with PyGad:")

    # Define fitness function
    def fitness_func(ga_instance, solution, solution_idx):
        return 0 - simple_trig_func(solution)

    # Define min, max values, and steps for each parameter
    args_bounds = [{"low": 0, "high": math.pi, "step": 0.0157},  # arg1
                   {"low": 0, "high": math.pi, "step": 0.0157},  # arg2
                   {"low": 0, "high": 200, "step": 1},  # arg3
                   {"low": 0, "high": math.pi, "step": 0.0157},  # arg4
                   {"low": 0, "high": math.pi, "step": 0.0157},  # arg5
                   {"low": 0, "high": 200, "step": 1},  # arg6
                   {"low": 0, "high": math.pi, "step": 0.0157},  # arg7
                   {"low": 0, "high": 200, "step": 1}  # arg8
                   ]

    def get_ga_instance():
        return pygad.GA(num_generations=10000,
                        num_parents_mating=16,
                        parent_selection_type="sss",
                        sol_per_pop=32,
                        num_genes=8,
                        gene_type=float,
                        gene_space=args_bounds,
                        fitness_func=fitness_func,
                        mutation_percent_genes=[30, 15],
                        mutation_type="adaptive",
                        suppress_warnings=True,
                        keep_elitism=16,
                        stop_criteria="saturate_10"
                        )

    execute_pygad_experiment(get_ga_instance, "adaptive mutation", num_runs, number_of_generations, logging_step, plot_fitness, result_list)

    ######### FINAL RESULTS #############

    log_message_info("************Final results:************")
    for r in result_list:
        log_message_info(r)


if __name__ == "__main__":
    args = []
    i = 1
    while True:
        try:
            args.append(sys.argv[i])
            i += 1
        except IndexError:
            break
    for a in args:
        if a == "plot":
            plot_fitness = True
        else:
            try:
                num_runs = int(a)
            except ValueError:
                pass
    execute_diversity_based_mutation_use_case_3()
