import math

import numpy as np
import pygad

from gadapt.ga import GA
from gadapt.utils import ga_utils

from exp_logging import init_logging, log_message_info

num_runs = 1000
logging_step = 50


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
            max_attempt_no=10)

    # Addition of variables with specified ranges and steps
    ga.add(min_value=0.0, max_value=math.pi, step=0.0157)
    ga.add(min_value=0.0, max_value=math.pi, step=0.0157)
    ga.add(min_value=0.0, max_value=200, step=1)
    ga.add(min_value=0.0, max_value=math.pi, step=0.0157)
    ga.add(min_value=0.0, max_value=math.pi, step=0.0157)
    ga.add(min_value=0.0, max_value=200, step=1)
    ga.add(min_value=0.0, max_value=math.pi, step=0.0157)
    ga.add(min_value=0.0, max_value=200, step=1)

    cost_values = []
    iteration_numbers = []
    is_succ = True
    for i in range(num_runs):
        results = ga.execute()
        if not results.success:
            is_succ = False
            break
        cost_values.append(results.min_cost)
        iteration_numbers.append(float(results.number_of_iterations))
        if i % logging_step == 0:
            final_min_cost = ga_utils.average(cost_values)
            avg_num_of_it = ga_utils.average(iteration_numbers)
            log_message_info(f"GAdapt - random mutation - Optimization number {i}.")
            log_message_info(f"GAdapt - random mutation - Average best fitness: {final_min_cost}")
            log_message_info(f"GAdapt - random mutation - Average generations completed: {avg_num_of_it}")
    if is_succ:
        final_min_cost = ga_utils.average(cost_values)
        avg_num_of_it = ga_utils.average(iteration_numbers)

        gadapt_avg_fitness = f"GAdapt - random mutation - Final average best fitness: {final_min_cost}"
        gadapt_avg_generation_number = f"GAdapt - random mutation - Final average generations completed: {avg_num_of_it}"

        log_message_info(gadapt_avg_fitness)
        log_message_info(gadapt_avg_generation_number)
        result_list.append("***********GADAPT - RANDOM MUTATION***********")
        result_list.append(gadapt_avg_fitness)
        result_list.append(gadapt_avg_generation_number)

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
            max_attempt_no=10)

    # Addition of variables with specified ranges and steps
    ga.add(min_value=0.0, max_value=math.pi, step=0.0157)
    ga.add(min_value=0.0, max_value=math.pi, step=0.0157)
    ga.add(min_value=0.0, max_value=200, step=1)
    ga.add(min_value=0.0, max_value=math.pi, step=0.0157)
    ga.add(min_value=0.0, max_value=math.pi, step=0.0157)
    ga.add(min_value=0.0, max_value=200, step=1)
    ga.add(min_value=0.0, max_value=math.pi, step=0.0157)
    ga.add(min_value=0.0, max_value=200, step=1)

    cost_values = []
    iteration_numbers = []
    is_succ = True
    for i in range(num_runs):
        results = ga.execute()
        if not results.success:
            is_succ = False
            break
        cost_values.append(results.min_cost)
        iteration_numbers.append(float(results.number_of_iterations))
        if i % logging_step == 0:
            final_min_cost = ga_utils.average(cost_values)
            avg_num_of_it = ga_utils.average(iteration_numbers)
            log_message_info(f"GAdapt - diversity mutation - Optimization number {i}.")
            log_message_info(f"GAdapt - diversity mutation - Average best fitness: {final_min_cost}")
            log_message_info(f"GAdapt - diversity mutation - Average generations completed: {avg_num_of_it}")
    if is_succ:
        final_min_cost = ga_utils.average(cost_values)
        avg_num_of_it = ga_utils.average(iteration_numbers)
        gadapt_avg_fitness = f"GAdapt - diversity mutation - Final average best fitness: {final_min_cost}"
        gadapt_avg_generation_number = f"GAdapt - diversity mutation - Final average generations completed: {avg_num_of_it}"

        log_message_info(gadapt_avg_fitness)
        log_message_info(gadapt_avg_generation_number)
        result_list.append("**********GADAPT - DIVERSITY MUTATION**********")
        result_list.append(gadapt_avg_fitness)
        result_list.append(gadapt_avg_generation_number)

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
        best_fitnesses = []
        generations_completed = []
        for i in range(num_runs):
            # Create genetic algorithm optimizer
            ga_instance = pygad.GA(num_generations=10000,
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

            # Run the genetic algorithm
            ga_instance.run()

            # Get the best solution
            best_solution, best_solution_fitness, best_match_index = ga_instance.best_solution()
            best_fitnesses.append(best_solution_fitness)
            generations_completed.append(ga_instance.generations_completed)
            if i % logging_step == 0:
                log_message_info(f"PyGAD - adaptive mutation - Optimization number {i}.")
                log_message_info(f"PyGAD - adaptive mutation - Average best fitness: {-np.mean(best_fitnesses)}")
                log_message_info(f"PyGAD - adaptive mutation - Average generations completed: {np.mean(generations_completed)}")
        pygad_avg_fitness = f"PyGAD - adaptive mutation - Final average best fitness: {-np.mean(best_fitnesses)}"
        pygad_avg_generation_number = f"PyGAD - adaptive mutation - Final average generations completed: {np.mean(generations_completed)}"

        log_message_info(pygad_avg_fitness)
        log_message_info(pygad_avg_generation_number)
        result_list.append("***********PYGAD - ADAPTIVE MUTATION***********")
        result_list.append(pygad_avg_fitness)
        result_list.append(pygad_avg_generation_number)

    ######### FINAL RESULTS #############

    log_message_info("************Final results:************")
    for r in result_list:
        log_message_info(r)


if __name__ == "__main__":
    execute_diversity_based_mutation_use_case_3()