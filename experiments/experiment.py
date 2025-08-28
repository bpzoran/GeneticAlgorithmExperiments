from collections.abc import Callable
import pygad

from gadapt.ga import GA

from exp_logging import init_logging, log_message_info
from gadapt_experiment import execute_gadapt_experiment
from pygad_experiment import execute_pygad_experiment
from settings.experiment_ga_settings import ExperimentGASettings


class Experiment:

    def __init__(self, f: Callable, args_bounds: list[dict[str, float]] = None) -> None:
        self.f = f
        if args_bounds is None:
            args_bounds = []
        self.args_bounds = args_bounds
        self.app_settings = ExperimentGASettings()

    def fill_args_with_same_values(self, low: float, high: float, step: float, number_of_args: int) -> None:
        self.args_bounds.clear()
        for _ in range(number_of_args):
            self.args_bounds.append({"low": low, "high": high, "step": step})


    def fill_gadapt_with_args(self, ga: GA) -> None:
        for ab in self.args_bounds:
            ga.add(min_value=ab["low"], max_value=ab["high"], step=ab["step"])


    # Define fitness function
    def fitness_func(self, ga_instance, solution, solution_idx):
        return 0 - self.f(solution)

    def execute_experiment(self):
        result_list = []
        if self.app_settings.pygad_random_mutation_enabled:
            ##### PyGAD OPTIMIZATION WITH RANDOM MUTATION ###############
            # Define min, max values, and steps for each parameter
            def get_ga_instance():
                 return pygad.GA(num_generations=1000,
                                num_parents_mating=16,
                                parent_selection_type="sss",
                                sol_per_pop=64,
                                num_genes=len(self.args_bounds),
                                gene_type=float,
                                gene_space=self.args_bounds,
                                fitness_func=self.fitness_func,
                                mutation_percent_genes=50,
                                mutation_type="random",
                                suppress_warnings=True,
                                keep_elitism=32,
                                stop_criteria=f"saturate_{self.app_settings.saturation_criteria}"
                                )



            execute_pygad_experiment(get_ga_instance, "random mutation", result_list)
        if self.app_settings.gadapt_random_mutation_enabled:
            ##### GADAPT OPTIMIZATION WITH RANDOM MUTATION ###############

            ga = GA(cost_function=self.f,
                    population_size=64,
                    population_mutation="random",
                    chromosome_mutation="random",
                    gene_mutation="random",
                    percentage_of_mutation_chromosomes=80,
                    percentage_of_mutation_genes=50,
                    exit_check="min_cost",
                    keep_elitism_percentage=50,
                    max_attempt_no=self.app_settings.saturation_criteria,
                    logging=False)

            # Addition of variables with specified ranges and steps
            self.fill_gadapt_with_args(ga)

            execute_gadapt_experiment(ga, "random mutation",
                                      result_list)
        if self.app_settings.gadapt_diversity_mutation_enabled:
            ##### GADAPT OPTIMIZATION WITH DIVERSITY MUTATION ###############

            ga = GA(cost_function=self.f,
                    population_size=64,
                    population_mutation="cost_diversity, cross_diversity, parent_diversity, random",
                    chromosome_mutation="cross_diversity,random",
                    gene_mutation="cross_diversity,random",
                    percentage_of_mutation_chromosomes=80,
                    percentage_of_mutation_genes=50,
                    exit_check="min_cost",
                    keep_elitism_percentage=50,
                    max_attempt_no=self.app_settings.saturation_criteria,
                    logging=False)

            # Addition of variables with specified ranges and steps
            self.fill_gadapt_with_args(ga)

            execute_gadapt_experiment(ga, "diversity mutation", result_list)

        if self.app_settings.pygad_adaptive_mutation_enabled:
            ##### PYGAD OPTIMIZATION WITH ADAPTIVE MUTATION ###############

            def get_ga_instance():
                return pygad.GA(num_generations=1000,
                                num_parents_mating=16,
                                parent_selection_type="sss",
                                sol_per_pop=64,
                                num_genes=len(self.args_bounds),
                                gene_type=float,
                                gene_space=self.args_bounds,
                                fitness_func=self.fitness_func,
                                mutation_percent_genes=[60, 40],
                                mutation_type="adaptive",
                                suppress_warnings=True,
                                keep_elitism=32,
                                stop_criteria=f"saturate_{self.app_settings.saturation_criteria}"
                                )


            execute_pygad_experiment(get_ga_instance, "adaptive mutation", result_list)

        ######### FINAL RESULTS #############

        log_message_info("************Final results:************")
        for r in result_list:
            log_message_info(r)
