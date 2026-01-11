import logging
import math
from typing import Tuple

import numpy as np

from utils.exp_logging import log_message_info
from settings.experiment_ga_settings import ExperimentGASettings
from utils.experiment_utils import number_of_generations_for_performance_check

log_message_info("Start optimization with PyGad:")
logger = logging.getLogger(__name__)

def find_fitness_for_first_generation(func, population: np.ndarray) -> float:
    fitness = 0.0
    costs = []
    for individual in population:
        costs.append(func(None, individual, None))
    return max(costs)

# Define fitness function
def execute_pygad_experiment(pygad_creator,
                             optimization_name: str = "",
                             result_list=None) -> Tuple[list[list[float]], float, float, int]:
    log_message_info(f"Start optimization with PyGAD, {optimization_name}:")
    app_settings = ExperimentGASettings()
    best_fitness_list = []
    fitness_per_generation = []
    generations_completed = []
    min_cost_per_generations_per_run = []
    final_min_cost = math.nan
    for i in range(app_settings.num_runs):
        # Create genetic algorithm optimizer
        ga_instance = pygad_creator()
        fitness_values = []
        fitness_values.append(find_fitness_for_first_generation(ga_instance.fitness_func, ga_instance.initial_population))
        def callback_generation(ga_inst):
            fitness_values.append(ga_instance.best_solution()[1])

        ga_instance.on_generation = callback_generation
        # Run the genetic algorithm
        ga_instance.run()

        # Get the best solution
        best_fitness_list.append(fitness_values[-1])
        generations_completed.append(ga_instance.generations_completed + 1)
        num_of_generations = number_of_generations_for_performance_check(ga_instance.generations_completed + 1, app_settings.percentage_of_generations_for_performance)


        if len(ga_instance.best_solutions_fitness) == 0:
            logger.warning(f"No minimum cost per generation! PyGAD - {optimization_name}, i = {i}")
            continue
        if num_of_generations > len(ga_instance.best_solutions_fitness):
            num_of_generations_old = num_of_generations
            num_of_generations = len(ga_instance.best_solutions_fitness)
            logger.warning(
                f"num_of_generations ({num_of_generations_old}) was > len(results.min_cost_per_generation) ({len(ga_instance.best_solutions_fitness)})! - {optimization_name}, i = {i}")
        fitness_per_generation.append(-ga_instance.best_solutions_fitness[num_of_generations - 1])
        if (i != 0) and (i % app_settings.logging_step == 0):
            log_message_info(f"PyGAD - {optimization_name} - Optimization number {i}.")
            log_message_info(f"PyGAD - {optimization_name} - Average best fitness: {round(-np.mean(best_fitness_list), 10):.10f}")
            log_message_info(
                f"PyGAD - {optimization_name} - Average generations completed: {round(np.mean(generations_completed), 10):.10f}")
        min_cost_per_generations_per_run.append([-fv for fv in fitness_values])
        final_min_cost = round(-np.mean(best_fitness_list), 10)
    pygad_avg_fitness = f"PyGAD - {optimization_name} - Final average best fitness: {final_min_cost:.10f}"
    final_average_generations_completed = round(np.mean(generations_completed), 10)
    pygad_avg_generation_number = f"PyGAD - {optimization_name} - Final average generations completed: {final_average_generations_completed:.10f}"
    mean_fitness_per_generation = round(np.mean(fitness_per_generation), 10)
    pygad_avg_fitness_after_n_generations = f"PyGAD - {optimization_name} - Final average fitness after {app_settings.percentage_of_generations_for_performance} generations: {mean_fitness_per_generation:.10f}"

    log_message_info(pygad_avg_fitness)
    log_message_info(pygad_avg_generation_number)
    result_list.append(f"***********PYGAD - {optimization_name.upper()}***********")
    result_list.append(pygad_avg_fitness)
    result_list.append(pygad_avg_generation_number)
    result_list.append(pygad_avg_fitness_after_n_generations)

    return min_cost_per_generations_per_run, final_min_cost, mean_fitness_per_generation, final_average_generations_completed