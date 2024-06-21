import math

import numpy as np
import pygad

from gadapt.ga import GA
from gadapt.utils import ga_utils

from exp_logging import init_logging, log_message_info
from gadapt_experiment import execute_gadapt_experiment
from pygad_experiment import execute_pygad_experiment
import sys

num_runs = 1000
logging_step = 50
number_of_generations = 40
plot_fitness = False


def simple_trigonometric_arithmetic_function(args):
    """
    The function simple_trigonometric_arithmetic_function computes the sum of five terms derived from its input list
    args, involving square roots, squares, and trigonometric calculations.
    Calculate the square root of the first element in args.
    Compute the square of the cosine of the second element in args.
    Calculate the sine of the third element in args.
    Square the fourth element in args.
    Calculate the square root of the fifth element in args.
    Sum all the computed values and return the result.
    """
    term1 = math.sqrt(args[0])
    term2 = math.cos(args[1]) ** 2
    term3 = math.sin(args[2])
    term4 = args[3] ** 2
    term5 = math.sqrt(abs(args[4]))

    return term1 + term2 + term3 + term4 + term5


def execute_diversity_based_mutation_use_case_1():
    """
    The function diversity_based_mutation_use_case_2 orchestrates the execution of genetic algorithm optimizations
    using different mutation strategies across multiple runs. It leverages the libraries PyGAD and GAdapt to optimize a
    trigonometric function, comparing the outcomes of adaptive, random, and diversity-based mutations.
    """
    init_logging(log_to_file=True)

    result_list = []

    ##### GADAPT OPTIMIZATION WITH RANDOM MUTATION ###############

    log_message_info("Start optimization with GAdapt, random mutation:")

    ga = GA(cost_function=simple_trigonometric_arithmetic_function,
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
    ga.add(min_value=1.0, max_value=4.0, step=0.01)
    ga.add(min_value=37.0, max_value=40.0, step=0.01)
    ga.add(min_value=78.0, max_value=88.0, step=0.1)
    ga.add(min_value=-5.0, max_value=4.0, step=0.1)
    ga.add(min_value=0, max_value=100, step=1.0)

    execute_gadapt_experiment(ga, "random mutation", num_runs, number_of_generations, logging_step, plot_fitness, result_list)

    ##### GADAPT OPTIMIZATION WITH DIVERSITY MUTATION ###############

    log_message_info("Start optimization with GAdapt, diversity mutation:")

    ga = GA(cost_function=simple_trigonometric_arithmetic_function,
            population_size=32,
            population_mutation="cost_diversity,parent_diversity,random",
            chromosome_mutation="cross_diversity",
            gene_mutation="cross_diversity,random",
            percentage_of_mutation_chromosomes=100,
            percentage_of_mutation_genes=50,
            keep_elitism_percentage=50,
            exit_check="min_cost",
            max_attempt_no=10,
            logging=False
            )

    # Addition of variables with specified ranges and steps
    ga.add(min_value=1.0, max_value=4.0, step=0.01)
    ga.add(min_value=37.0, max_value=40.0, step=0.01)
    ga.add(min_value=78.0, max_value=88.0, step=0.1)
    ga.add(min_value=-5.0, max_value=4.0, step=0.1)
    ga.add(min_value=-0, max_value=100, step=1.0)

    execute_gadapt_experiment(ga, "diversity mutation", num_runs, number_of_generations, logging_step,plot_fitness, result_list)

    ##### PYGAD OPTIMIZATION WITH ADAPTIVE MUTATION ###############

    log_message_info("Start optimization with PyGad:")

    # Define fitness function
    def fitness_func(ga_instance, solution, solution_idx):
        return 0 - simple_trigonometric_arithmetic_function(solution)

    # Define min, max values, and steps for each parameter
    args_bounds = [{"low": 1.0, "high": 4.0, "step": 0.01},  # arg1
                   {"low": 37.0, "high": 40.0, "step": 0.01},  # arg2
                   {"low": 78, "high": 88.0, "step": 0.1},  # arg3
                   {"low": -5.0, "high": 4.0, "step": 0.1},  # arg4
                   {"low": 0, "high": 100, "step": 1},  # arg5
                   ]
    best_fitness_list = []
    fitness_per_generation = []
    generations_completed = []

    def get_ga_instance():
        return pygad.GA(num_generations=10000,
                        num_parents_mating=16,
                        parent_selection_type="sss",
                        sol_per_pop=32,
                        num_genes=5,
                        gene_type=float,
                        gene_space=args_bounds,
                        fitness_func=fitness_func,
                        mutation_percent_genes=[30, 15],
                        mutation_type="adaptive",
                        suppress_warnings=True,
                        keep_elitism=16,
                        stop_criteria="saturate_10"
                        )

    execute_pygad_experiment(get_ga_instance, "adaptive mutation", num_runs, number_of_generations, logging_step,plot_fitness, result_list)

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
    execute_diversity_based_mutation_use_case_1()
