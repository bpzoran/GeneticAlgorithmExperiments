import numpy as np

from exp_logging import log_message_info
from plot_fitness_per_generation import fitness_per_generation_plot

log_message_info("Start optimization with PyGad:")

fitness_values = []
# Define fitness function
def execute_pygad_experiment(pygad_creator,
                             optimization_name: str = "",
                             num_runs=-1,
                             number_of_generations=100,
                             logging_step=50,
                             plot_fitness=False,
                             result_list=None):
    best_fitness_list = []
    fitness_per_generation = []
    generations_completed = []
    fitness_values = []
    for i in range(num_runs):
        # Create genetic algorithm optimizer
        ga_instance = pygad_creator()
        fitness_values = []

        def callback_generation(ga_inst):
            fitness_values.append(ga_instance.best_solution()[1])

        ga_instance.on_generation = callback_generation
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
            log_message_info(f"PyGAD - {optimization_name} - Optimization number {i}.")
            log_message_info(f"PyGAD - {optimization_name}- Average best fitness: {-np.mean(best_fitness_list)}")
            log_message_info(
                f"PyGAD - {optimization_name} - Average generations completed: {np.mean(generations_completed)}")
    pygad_avg_fitness = f"PyGAD - {optimization_name}- Final average best fitness: {-np.mean(best_fitness_list)}"
    pygad_avg_generation_number = f"PyGAD - {optimization_name}- Final average generations completed: {np.mean(generations_completed)}"
    pygad_avg_fitness_after_n_generations = f"PyGAD - {optimization_name}- Final average fitness after {number_of_generations} generations: {np.mean(fitness_per_generation)}"

    log_message_info(pygad_avg_fitness)
    log_message_info(pygad_avg_generation_number)
    result_list.append("***********PYGAD - ADAPTIVE MUTATION***********")
    result_list.append(pygad_avg_fitness)
    result_list.append(pygad_avg_generation_number)
    result_list.append(pygad_avg_fitness_after_n_generations)

    if plot_fitness:
        f_v = [-fv for fv in fitness_values]
        fitness_per_generation_plot(f_v, f"GAdapt - {optimization_name}", "blue")
