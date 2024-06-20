import math
import sys

import pygad

from gadapt.ga import GA

from exp_logging import log_message_info, init_logging
from gadapt_experiment import execute_gadapt_experiment
from pygad_experiment import execute_pygad_experiment

num_runs = 100
logging_step = 50
number_of_generations = 40
plot_fitness = False


def complex_trig_func(args):
    """
    The complex_trig_func function computes a complex mathematical expression involving trigonometric and
    algebraic operations on elements of an input list args.

    This function performs the following steps:

    1. Calculates the square root of the absolute value of the cosine of the first element (args[0]).
    2. Computes the square of the cosine of the second element (args[1]).
    3. Adds the sine of the third element (args[2]).
    4. Adds the square of the fourth element (args[3]).
    5. Adds the square root of the fifth element (args[4]).
    6. Adds the cosine of the sixth element (args[5]) twice, with one instance being subtracted later in the expression.
    7. Subtracts a complex expression involving the seventh element (args[6]), its sine, and its cube.
    8. Adds a division operation involving the sine of the first element (args[0]), the seventh element (args[6]), and the fifth element (args[4]).

    The function is designed to be complex for optimization purposes, containing many local minima and non-optimal
    solutions bordering areas with lower values. This complexity is intended to challenge optimization algorithms.
    Several arguments are used in multiple operations, and some operations depend on the results of others.

    Parameters:
    args (list): A list of numerical values (length >= 7) used as input for the function.

    Returns:
    float: The result of the complex mathematical expression.
    """
    if len(args) != 7:
        raise ValueError("Input vector must contain 7 variables.")
    return (math.sqrt(abs(math.cos(args[0]))) +
            math.pow(math.cos(args[1]), 2) +
            math.sin(args[2]) +
            math.pow(args[3], 2) +
            math.sqrt(args[4]) +
            math.cos(args[5]) -
            (args[6] * math.sin(pow(args[6], 3)) + 1) +
            math.sin(args[0]) / (math.sqrt(args[0]) / 3 +
                                 (args[6] * math.sin(pow(args[6], 3)) + 1)) / math.sqrt(
                args[4]) +
            math.cos(args[5]))


def diversity_based_mutation_use_case_2():
    """
    The diversity_based_mutation_use_case_2 function orchestrates a series of optimizations using genetic algorithms (GAs)
    with different mutation strategies. It logs the performance of each strategy,
    including fitness and generation metrics, and aggregates the results for comparison.
    Flow:
    Performs optimization using PyGAD with adaptive mutation.
    Performs optimization using GAdapt with random mutation.
    Performs optimization using GAdapt with diversity mutation.
    """
    init_logging(log_to_file=True)
    result_list = []

    ##### GADAPT OPTIMIZATION WITH RANDOM MUTATION ###############

    log_message_info("Start optimization with GAdapt, random mutation:")

    ga = GA(cost_function=complex_trig_func,
            population_size=32,
            parent_selection="from_top_to_bottom",
            population_mutation="random",
            chromosome_mutation="random",
            gene_mutation="random",
            percentage_of_mutation_chromosomes=80,  # Up to 80% of chromosomes will be mutated
            percentage_of_mutation_genes=30,  # Up to 30% of genes in mutated chromosome will be mutated
            keep_elitism_percentage=50,
            exit_check="min_cost",
            max_attempt_no=10,
            logging=False
            )

    # Addition of variables with specified ranges and steps
    ga.add(1.0, 4.0, 0.01)
    ga.add(37.0, 40.0, 0.01)
    ga.add(78.0, 88.0, 0.1)
    ga.add(-5.0, 4.0, 0.1)
    ga.add(1.0, 100.0, 1)
    ga.add(1.0, 4.0, 0.01)
    ga.add(-1, -0.01, 0.005)

    execute_gadapt_experiment(ga, "random mutation", num_runs, number_of_generations, logging_step, plot_fitness, result_list)

    ##### GADAPT OPTIMIZATION WITH DIVERSITY MUTATION ###############

    log_message_info("Start optimization with GAdapt, diversity mutation:")

    ga = GA(cost_function=complex_trig_func,
            population_size=32,
            population_mutation="parent_diversity,cross_diversity,random",
            chromosome_mutation="cross_diversity",
            gene_mutation="cross_diversity,random",
            percentage_of_mutation_chromosomes=100,
            percentage_of_mutation_genes=50,
            parent_selection="from_top_to_bottom",
            keep_elitism_percentage=50,
            number_of_crossover_parents=16,
            max_attempt_no=10,
            exit_check="min_cost",
            logging=False
            )

    # Addition of variables with specified ranges and steps
    ga.add(1.0, 4.0, 0.01)
    ga.add(37.0, 40.0, 0.01)
    ga.add(78.0, 88.0, 0.1)
    ga.add(-5.0, 4.0, 0.1)
    ga.add(1.0, 100.0, 1)
    ga.add(1.0, 4.0, 0.01)
    ga.add(-1, -0.01, 0.005)

    execute_gadapt_experiment(ga, "diversity mutation", num_runs, number_of_generations, logging_step, plot_fitness, result_list)

    ##### PYGAD OPTIMIZATION WITH ADAPTIVE MUTATION ###############

    log_message_info("Start optimization with PyGad:")

    # Define fitness function
    def fitness_func(ga_instance, solution, solution_idx):
        return 0 - complex_trig_func(solution)

    # Define min, max values, and steps for each parameter
    args_bounds = [{"low": 1.0, "high": 4.0, "step": 0.01},  # arg1
                   {"low": 37.0, "high": 40.0, "step": 0.01},  # arg2
                   {"low": 78.0, "high": 88.0, "step": 0.1},  # arg3
                   {"low": -5.0, "high": 4.0, "step": 0.1},  # arg4
                   {"low": 1.0, "high": 100.0, "step": 1},  # arg5
                   {"low": 1.0, "high": 4.0, "step": 0.01},  # arg6
                   {"low": -1, "high": 0.01, "step": 0.005},  # arg7
                   ]

    def get_ga_instance():
        return pygad.GA(num_generations=10000,
                        num_parents_mating=16,
                        parent_selection_type="sss",
                        sol_per_pop=32,
                        num_genes=7,
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

    log_message_info("Final results:")
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
    diversity_based_mutation_use_case_2()
