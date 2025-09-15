import math
import sys
from collections.abc import Callable
from typing import Optional

import numpy as np
import pygad

from gadapt.ga import GA

from plugins.pygad.pygad_blending_crossover import blend_crossover_pygad
from utils import csv_writter
from utils.data_aggregation import aggregate_convergence
from utils.exp_logging import log_message_info
from runners.gadapt_experiment import execute_gadapt_experiment
from runners.pygad_experiment import execute_pygad_experiment
from settings.experiment_ga_settings import ExperimentGASettings
from utils.plot_fitness_per_generation import plot_convergence_curve


class Experiment:

    def __init__(self, f: Callable, args_bounds: list[dict[str, float]] = None, experiment_name = None) -> None:
        self._args_bounds = None
        self.args_bounds_list = []
        self._arg_bounds = None
        self.f = f
        if args_bounds is None:
            args_bounds = []
        self.args_bounds = args_bounds
        self.app_settings = ExperimentGASettings()
        if experiment_name is None or experiment_name == "":
            experiment_name = f.__name__
        self._experiment_name = experiment_name

    @property
    def experiment_name(self) -> str:
        return self._experiment_name + f" ({len(self._args_bounds)} variables, saturation = {self.app_settings.saturation_criteria})"

    @property
    def args_bounds(self) -> list:
        return self._arg_bounds

    @args_bounds.setter
    def args_bounds(self, value: list):
        self.args_bounds_list = []
        if value not in self.args_bounds_list:
            self.args_bounds_list.append(value)

    def fill_args_with_same_values(self, low: float, high: float,  number_of_args: int | list[int], step: Optional[float] = None) -> None:
        if isinstance(number_of_args, int):
            bounds = {"low": low, "high": high, **({"step": step} if step is not None else {})}
            self.args_bounds = [bounds.copy() for _ in range(number_of_args)]
        elif isinstance(number_of_args, list):
            number_of_args = list(set(number_of_args) & set(self.app_settings.variable_numbers))
            if len(number_of_args) == 0:
                return
            self.args_bounds_list = []
            for num_of_args in number_of_args:
                bounds = {"low": low, "high": high, **({"step": step} if step is not None else {})}
                arg_bounds = [bounds.copy() for _ in range(num_of_args)]
                self.args_bounds_list.append(arg_bounds)


    def fill_gadapt_with_args(self, ga: GA) -> None:
        for ab in self._args_bounds:
            step = ab.get("step")
            if step is None:
                step = sys.float_info.min
            low = ab.get("low")
            high = ab.get("high")
            ga.add(min_value=low, max_value=high, step=step)


    # Define fitness function
    def fitness_func(self, ga_instance, solution, solution_idx):
        return 0 - self.f(solution)

    def execute_experiment(self):
        for args_bounds in self.args_bounds_list:
            self._args_bounds = args_bounds
            for saturation_criteria in self.app_settings.saturation_criterias:
                self.app_settings.saturation_criteria = saturation_criteria
                log_message_info(f"Executing experiment: {self.experiment_name} with args bounds: {args_bounds} and saturation criteria: {saturation_criteria}")
                self._execute_single_experiment()


    def _execute_single_experiment(self):
        result_list = []
        runs: dict = {}
        min_cost_per_generations_per_run_per_mutation_type: dict = {}
        average_generations: dict = {}
        app_settings = ExperimentGASettings()
        if self.app_settings.pygad_random_mutation_enabled:
            ##### PyGAD OPTIMIZATION WITH RANDOM MUTATION ###############
            # Define min, max values, and steps for each parameter
            def get_ga_instance():
                 return pygad.GA(num_generations=1000,
                                num_parents_mating=round((self.app_settings.keep_elitism_percentage / 100) * self.app_settings.population_size),
                                parent_selection_type="sss",
                                sol_per_pop=self.app_settings.population_size,
                                num_genes=len(self._args_bounds),
                                gene_type=float,
                                gene_space=self._args_bounds,
                                fitness_func=self.fitness_func,
                                mutation_percent_genes=self.app_settings.percentage_of_mutation_genes,
                                mutation_type="random",
                                suppress_warnings=True,
                                keep_elitism=round((self.app_settings.keep_elitism_percentage / 100) * self.app_settings.population_size),
                                stop_criteria=f"saturate_{self.app_settings.saturation_criteria}",
                                crossover_type=blend_crossover_pygad
                                )
            mutation_type = "random mutation"
            optimization_name = f"{self.experiment_name} - {mutation_type}"
            min_cost_per_generations_per_run, average_number_of_generations = execute_pygad_experiment(get_ga_instance, optimization_name, result_list)
            runs[mutation_type] = aggregate_convergence(min_cost_per_generations_per_run, stat=self.app_settings.plot_stat, band=self.app_settings.plot_band)
            min_cost_per_generations_per_run_per_mutation_type[mutation_type] = min_cost_per_generations_per_run
            average_generations[mutation_type] = average_number_of_generations
            average_generations[mutation_type] = average_number_of_generations
        if self.app_settings.gadapt_random_mutation_enabled:
            ##### GADAPT OPTIMIZATION WITH RANDOM MUTATION ###############

            ga = GA(cost_function=self.f,
                    population_size=self.app_settings.population_size,
                    population_mutation="random",
                    chromosome_mutation="random",
                    gene_mutation="random",
                    percentage_of_mutation_chromosomes=self.app_settings.percentage_of_mutation_chromosomes,
                    percentage_of_mutation_genes=self.app_settings.percentage_of_mutation_genes,
                    exit_check="min_cost",
                    keep_elitism_percentage=self.app_settings.keep_elitism_percentage,
                    max_attempt_no=self.app_settings.saturation_criteria,
                    parent_selection="from_top_to_bottom",
                    logging=False)

            # Addition of variables with specified ranges and steps
            self.fill_gadapt_with_args(ga)
            mutation_type = "random mutation"
            optimization_name = f"{self.experiment_name} - {mutation_type}"
            min_cost_per_generations_per_run, average_number_of_generations = execute_gadapt_experiment(ga, optimization_name,
                                      result_list)
            runs[mutation_type] = aggregate_convergence(min_cost_per_generations_per_run, stat=self.app_settings.plot_stat, band=self.app_settings.plot_band)
            min_cost_per_generations_per_run_per_mutation_type[mutation_type] = min_cost_per_generations_per_run
            average_generations[mutation_type] = average_number_of_generations
        if self.app_settings.gadapt_diversity_mutation_enabled:
            ##### GADAPT OPTIMIZATION WITH DIVERSITY MUTATION ###############

            ga = GA(cost_function=self.f,
                    population_size=self.app_settings.population_size,
                    population_mutation="cost_diversity, cross_diversity, parent_diversity, random",
                    chromosome_mutation="cross_diversity,random",
                    gene_mutation="cross_diversity,random",
                    percentage_of_mutation_chromosomes=self.app_settings.percentage_of_mutation_chromosomes,
                    percentage_of_mutation_genes=self.app_settings.percentage_of_mutation_genes,
                    exit_check="min_cost",
                    keep_elitism_percentage=self.app_settings.keep_elitism_percentage,
                    max_attempt_no=self.app_settings.saturation_criteria,
                    parent_selection="from_top_to_bottom",
                    logging=False)

            # Addition of variables with specified ranges and steps
            self.fill_gadapt_with_args(ga)
            mutation_type = "diversity mutation"
            optimization_name = f"{self.experiment_name} - {mutation_type}"
            min_cost_per_generations_per_run, average_number_of_generations = execute_gadapt_experiment(ga,
                                                                                                        optimization_name,
                                                                                                        result_list)
            runs[mutation_type] = aggregate_convergence(min_cost_per_generations_per_run, stat=self.app_settings.plot_stat, band=self.app_settings.plot_band)
            min_cost_per_generations_per_run_per_mutation_type[mutation_type] = min_cost_per_generations_per_run
            average_generations[mutation_type] = average_number_of_generations
        if self.app_settings.pygad_adaptive_mutation_enabled:
            ##### PYGAD OPTIMIZATION WITH ADAPTIVE MUTATION ###############

            def get_ga_instance():
                return pygad.GA(num_generations=1000,
                                num_parents_mating=round((self.app_settings.keep_elitism_percentage / 100) * self.app_settings.population_size),
                                parent_selection_type="sss",
                                sol_per_pop=self.app_settings.population_size,
                                num_genes=len(self._args_bounds),
                                gene_type=float,
                                gene_space=self._args_bounds,
                                fitness_func=self.fitness_func,
                                mutation_percent_genes=[60, 40],
                                mutation_type="adaptive",
                                suppress_warnings=True,
                                keep_elitism=round((self.app_settings.keep_elitism_percentage / 100) * self.app_settings.population_size),
                                stop_criteria=f"saturate_{self.app_settings.saturation_criteria}",
                                crossover_type=blend_crossover_pygad
                                )

            mutation_type = "adaptive mutation"
            optimization_name = f"{self.experiment_name} - {mutation_type}"
            min_cost_per_generations_per_run, average_number_of_generations = execute_pygad_experiment(get_ga_instance,
                                                                                                       optimization_name,
                                                                                                       result_list)
            runs[mutation_type] = aggregate_convergence(min_cost_per_generations_per_run, stat=self.app_settings.plot_stat, band=self.app_settings.plot_band)
            min_cost_per_generations_per_run_per_mutation_type[mutation_type] = min_cost_per_generations_per_run
            average_generations[mutation_type] = average_number_of_generations
        ######### FINAL RESULTS #############

        log_message_info("************Final results:************")
        for r in result_list:
            log_message_info(r)
        if app_settings.log_to_file:
            csv_writter.aggregated_data_to_csv(runs, experiment_name=self.experiment_name)
            csv_writter.runs_to_csv(min_cost_per_generations_per_run_per_mutation_type, experiment_name=self.experiment_name)
        if self.app_settings.plot_fitness:
            lowest, highest, max_len = analyze_runs(runs)
            plot_convergence_curve(agg=runs,
                                   x0=average_generations,
                                   lowest=lowest,
                                   highest=highest,
                                   max_len=max_len,
                                   stat=self.app_settings.plot_stat,
                                   band=self.app_settings.plot_band,
                                   description=self.experiment_name,
                                   outdir=app_settings.plot_path,
                                   save=app_settings.log_to_file,
                                   basename=self.experiment_name,)


def analyze_runs(runs: dict[str, dict[str, np.ndarray]]):
    all_uppers = np.concatenate([
        data["upper"] for data in runs.values() if len(data["upper"]) > 0
    ])
    all_lowers = np.concatenate([
        data["lower"] for data in runs.values() if len(data["lower"]) > 0
    ])
    lowest = np.min(all_lowers)
    highest = np.max(all_uppers)
    max_len = max(len(data["center"]) for data in runs.values())
    return lowest, highest, max_len
