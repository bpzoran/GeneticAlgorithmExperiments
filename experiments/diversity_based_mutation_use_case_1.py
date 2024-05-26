import math

import numpy as np
import pygad

from gadapt.ga import GA
from gadapt.utils import ga_utils

from exp_logging import init_logging, log_message_info

num_runs = 1000
logging_step = 50
number_of_generations = 30


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
            max_attempt_no=10)

    # Addition of variables with specified ranges and steps
    ga.add(min_value=1.0, max_value=4.0, step=0.01)
    ga.add(min_value=37.0, max_value=40.0, step=0.01)
    ga.add(min_value=78.0, max_value=88.0, step=0.1)
    ga.add(min_value=-5.0, max_value=4.0, step=0.1)
    ga.add(min_value=0, max_value=100, step=1.0)

    cost_values = []
    iteration_numbers = []
    fitness_per_generation = []
    is_succ = True
    for i in range(num_runs):
        results = ga.execute()
        if not results.success:
            is_succ = False
            break
        cost_values.append(results.min_cost)
        iteration_numbers.append(float(results.number_of_iterations))
        num_of_generations = number_of_generations
        if num_of_generations > results.number_of_iterations:
            num_of_generations = results.number_of_iterations
        fitness_per_generation.append(float(results.min_cost_per_generation[num_of_generations - 1]))
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
        gadapt_avg_fitness_after_n_generations = f"GAdapt - random mutation - Final average fitness after {number_of_generations} generations: {np.mean(fitness_per_generation)}"

        log_message_info(gadapt_avg_fitness)
        log_message_info(gadapt_avg_generation_number)
        result_list.append("***********GADAPT - RANDOM MUTATION***********")
        result_list.append(gadapt_avg_fitness)
        result_list.append(gadapt_avg_generation_number)
        result_list.append(gadapt_avg_fitness_after_n_generations)

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
            max_attempt_no=10)

    # Addition of variables with specified ranges and steps
    ga.add(min_value=1.0, max_value=4.0, step=0.01)
    ga.add(min_value=37.0, max_value=40.0, step=0.01)
    ga.add(min_value=78.0, max_value=88.0, step=0.1)
    ga.add(min_value=-5.0, max_value=4.0, step=0.1)
    ga.add(min_value=-0, max_value=100, step=1.0)

    cost_values = []
    iteration_numbers = []
    fitness_per_generation = []
    is_succ = True
    for i in range(num_runs):
        results = ga.execute()
        if not results.success:
            is_succ = False
            break
        cost_values.append(results.min_cost)
        iteration_numbers.append(float(results.number_of_iterations))
        num_of_generations = number_of_generations
        if num_of_generations > results.number_of_iterations:
            num_of_generations = results.number_of_iterations
        fitness_per_generation.append(float(results.min_cost_per_generation[num_of_generations - 1]))
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
        gadapt_avg_fitness_after_n_generations = f"GAdapt - diversity mutation - Final average fitness after {number_of_generations} generations: {np.mean(fitness_per_generation)}"

        log_message_info(gadapt_avg_fitness)
        log_message_info(gadapt_avg_generation_number)
        result_list.append("**********GADAPT - DIVERSITY MUTATION**********")
        result_list.append(gadapt_avg_fitness)
        result_list.append(gadapt_avg_generation_number)
        result_list.append(gadapt_avg_fitness_after_n_generations)

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
        for i in range(num_runs):
            # Create genetic algorithm optimizer
            ga_instance = pygad.GA(num_generations=10000,
                                   num_parents_mating=16,
                                   parent_selection_type="sss",
                                   sol_per_pop=32,
                                   num_genes=5,
                                   gene_type=float,
                                   gene_space=args_bounds,
                                   fitness_func=fitness_func,
                                   mutation_percent_genes=20,
                                   mutation_type="random",
                                   suppress_warnings=True,
                                   keep_elitism=16,
                                   stop_criteria="saturate_10"
                                   )

            # Run the genetic algorithm
            ga_instance.run()

            # Get the best solution
            best_solution, best_solution_fitness, best_match_index = ga_instance.best_solution()
            best_fitness_list.append(best_solution_fitness)
            generations_completed.append(ga_instance.generations_completed)
            num_of_generations = number_of_generations
            if num_of_generations > ga_instance.generations_completed:
                num_of_generations = ga_instance.generations_completed
            fitness_per_generation.append(-ga_instance.best_solutions_fitness[num_of_generations - 1])
            if i % logging_step == 0:
                log_message_info(f"PyGAD - adaptive mutation - Optimization number {i}.")
                log_message_info(f"PyGAD - adaptive mutation - Average best fitness: {-np.mean(best_fitness_list)}")
                log_message_info(f"PyGAD - adaptive mutation - Average generations completed: {np.mean(generations_completed)}")
        pygad_avg_fitness = f"PyGAD - adaptive mutation - Final average best fitness: {-np.mean(best_fitness_list)}"
        pygad_avg_generation_number = f"PyGAD - adaptive mutation - Final average generations completed: {np.mean(generations_completed)}"
        pygad_avg_fitness_after_n_generations = f"PyGAD - adaptive mutation - Final average fitness after {number_of_generations} generations: {np.mean(fitness_per_generation)}"

        log_message_info(pygad_avg_fitness)
        log_message_info(pygad_avg_generation_number)
        result_list.append("***********PYGAD - ADAPTIVE MUTATION***********")
        result_list.append(pygad_avg_fitness)
        result_list.append(pygad_avg_generation_number)
        result_list.append(pygad_avg_fitness_after_n_generations)

    ######### FINAL RESULTS #############

    log_message_info("************Final results:************")
    for r in result_list:
        log_message_info(r)


if __name__ == "__main__":
    execute_diversity_based_mutation_use_case_1()
