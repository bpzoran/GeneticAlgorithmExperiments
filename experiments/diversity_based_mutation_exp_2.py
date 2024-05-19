import math

import numpy as np
import pygad

from gadapt.ga import GA
from gadapt.utils import ga_utils

from exp_logging import log_message_info, init_logging

num_runs = 100
logging_step = 50


def execute_experiment_2():
    def complex_trig_func(args):
        return (math.sqrt(abs(math.cos(args[0]))) +
                math.pow(math.cos(args[1]), 2) +
                math.sin(args[2]) +
                math.pow(args[3], 2) +
                math.sqrt(args[4]) +
                math.cos(args[5]) -
                (args[6] * math.sin(pow(args[6], 3)) + 1) +
                math.sin(args[0]) / (math.sqrt(args[0]) / 3 + (args[6] * math.sin(pow(args[6], 3)) + 1)) / math.sqrt(
                    args[4]) +
                math.cos(args[5]))

    init_logging(log_to_file=True)
    result_list = []

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
    best_fitnesses = []
    generations_completed = []
    for i in range(num_runs):
        # Create genetic algorithm optimizer
        ga_instance = pygad.GA(num_generations=10000,
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

        # Run the genetic algorithm
        ga_instance.run()

        # Get the best solution
        best_solution, best_solution_fitness, best_match_index = ga_instance.best_solution()
        best_fitnesses.append(best_solution_fitness)
        generations_completed.append(ga_instance.generations_completed)
        if i % logging_step == 0:
            log_message_info(f"PyGAD - Optimization number {i}.")
            log_message_info(f"PyGAD - Average best fitness: {-np.mean(best_fitnesses)}")
            log_message_info(f"PyGAD - Average generations completed: {np.mean(generations_completed)}")
    pygad_avg_fitness = f"PyGAD - Final average best fitness: {-np.mean(best_fitnesses)}"
    pygad_avg_generation_number = f"PyGAD - Final average generations completed: {np.mean(generations_completed)}"

    log_message_info(pygad_avg_fitness)
    log_message_info(pygad_avg_generation_number)
    result_list.append("***********PYGAD - ADAPTIVE MUTATION***********")
    result_list.append(pygad_avg_fitness)
    result_list.append(pygad_avg_generation_number)

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
            )

    # Addition of variables with specified ranges and steps
    ga.add(1.0, 4.0, 0.01)
    ga.add(37.0, 40.0, 0.01)
    ga.add(78.0, 88.0, 0.1)
    ga.add(-5.0, 4.0, 0.1)
    ga.add(1.0, 100.0, 1)
    ga.add(1.0, 4.0, 0.01)
    ga.add(-1, -0.01, 0.005)

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
            exit_check="min_cost"
            )

    # Addition of variables with specified ranges and steps
    ga.add(1.0, 4.0, 0.01)
    ga.add(37.0, 40.0, 0.01)
    ga.add(78.0, 88.0, 0.1)
    ga.add(-5.0, 4.0, 0.1)
    ga.add(1.0, 100.0, 1)
    ga.add(1.0, 4.0, 0.01)
    ga.add(-1, -0.01, 0.005)

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

    ######### FINAL RESULTS #############

    log_message_info("Final results:")
    for r in result_list:
        log_message_info(r)


if __name__ == "__main__":
    execute_experiment_2()
