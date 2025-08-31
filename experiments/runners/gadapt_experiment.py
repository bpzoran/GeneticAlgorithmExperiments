import numpy as np
from gadapt.utils import ga_utils

from utils.exp_logging import log_message_info
from utils.plot_fitness_per_generation import fitness_per_generation_plot
from settings.experiment_ga_settings import ExperimentGASettings


def execute_gadapt_experiment(ga,
                              optimization_name: str = "",
                              result_list=None):
    if result_list is None:
        result_list = []
    log_message_info(f"Start optimization with GAdapt, {optimization_name}:")
    app_settings = ExperimentGASettings()
    cost_values = []
    iteration_numbers = []
    fitness_per_generation = []
    is_succ = True
    min_cost_per_generation = []
    for i in range(app_settings.num_runs):
        results = ga.execute()
        if not results.success:
            is_succ = False
            break
        min_cost_per_generation = results.min_cost_per_generation
        cost_values.append(results.min_cost)
        iteration_numbers.append(float(results.number_of_iterations))
        num_of_generations = app_settings.number_of_generations
        if num_of_generations > results.number_of_iterations:
            num_of_generations = results.number_of_iterations
        fitness_per_generation.append(float(results.min_cost_per_generation[num_of_generations - 1]))
        if i % app_settings.logging_step == 0:
            final_min_cost = ga_utils.average(cost_values)
            avg_num_of_it = ga_utils.average(iteration_numbers)
            log_message_info(f"GAdapt - {optimization_name} - Optimization number {i}.")
            log_message_info(f"GAdapt - {optimization_name} - Average best fitness: {round(final_min_cost, 10):.10f}")
            log_message_info(f"GAdapt - {optimization_name} - Average generations completed: {round(avg_num_of_it, 10):.10f}")
    if is_succ:
        final_min_cost = ga_utils.average(cost_values)
        avg_num_of_it = ga_utils.average(iteration_numbers)

        gadapt_avg_fitness = f"GAdapt - {optimization_name} - Final average best fitness: {round(final_min_cost, 10):.10f}"
        gadapt_avg_generation_number = f"GAdapt - {optimization_name} - Final average generations completed: {round(avg_num_of_it, 10):.10f}"
        gadapt_avg_fitness_after_n_generations = f"GAdapt - {optimization_name} - Final average fitness after {app_settings.number_of_generations} generations: {round(np.mean(fitness_per_generation), 10):.10f}"

        log_message_info(gadapt_avg_fitness)
        log_message_info(gadapt_avg_generation_number)
        result_list.append(f"***********GADAPT - {optimization_name.upper()}***********")
        result_list.append(gadapt_avg_fitness)
        result_list.append(gadapt_avg_generation_number)
        result_list.append(gadapt_avg_fitness_after_n_generations)
    if app_settings.plot_fitness:
        fitness_per_generation_plot(min_cost_per_generation, f"GAdapt - {optimization_name}", "red")
