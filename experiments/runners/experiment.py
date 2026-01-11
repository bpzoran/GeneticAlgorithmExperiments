import math
import sys
from collections.abc import Callable
from typing import Optional

import numpy as np
import pygad

from gadapt.ga import GA
from gadapt.utils.ga_utils import average

from plugins.pygad.pygad_blending_crossover import blend_crossover_pygad
from utils import csv_writter
from utils.data_aggregation import aggregate_convergence
from utils.exp_logging import log_message_info
from runners.gadapt_experiment import execute_gadapt_experiment
from runners.pygad_experiment import execute_pygad_experiment
from settings.experiment_ga_settings import ExperimentGASettings
from utils.experiment_utils import transform_function_string, get_fitness_range, summarize_ga, \
    number_of_generations_for_performance_check
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
        if value not in self.args_bounds_list: # Only add if not already present
            self.args_bounds_list.append(value)

    def fill_args_with_same_values(self, low: float, high: float,  number_of_args: int | list[int] = 0, step: Optional[float] = None) -> None:
        self.args_bounds_list = []
        if number_of_args == 0 or number_of_args is None:
            number_of_args = ExperimentGASettings().variable_numbers
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
        final_results: dict = {}
        min_cost_per_generations_per_run_per_mutation_type: dict = {}
        average_generations: dict = {}
        app_settings = ExperimentGASettings()
        pygad_mutation_percentage_up = round(self.app_settings.percentage_of_mutation_chromosomes * (
                    self.app_settings.percentage_of_mutation_genes / 100))
        pygad_mutation_percentage_down = round(pygad_mutation_percentage_up * self.app_settings.mutation_ratio)
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
                                mutation_type="random",
                                suppress_warnings=True,
                                keep_elitism=round((self.app_settings.keep_elitism_percentage / 100) * self.app_settings.population_size),
                                stop_criteria=f"saturate_{self.app_settings.saturation_criteria}",
                                crossover_type=blend_crossover_pygad,
                                mutation_probability=pygad_mutation_percentage_down / 100
                                )
            mutation_type = "random mutation"
            optimization_name = f"{self.experiment_name} - {mutation_type}"
            min_cost_per_generations_per_run, final_min_cost, mean_fitness_per_generation, average_number_of_generations = execute_pygad_experiment(get_ga_instance, optimization_name, result_list)
            final_results.setdefault("Experiment", {})
            final_results["Experiment"]["Experiment name"] = transform_function_string(self._experiment_name)
            final_results["Experiment"]["Number of variables"] = len(self._args_bounds)
            final_results["Experiment"]["Saturation after generations"] = self.app_settings.saturation_criteria
            avg_number_of_generations_for_performance = number_of_generations_for_performance_check(average_number_of_generations, self.app_settings.percentage_of_generations_for_performance)
            final_results["Experiment"][mutation_type] = {
                "Average fitness": f"{final_min_cost:.10f}",
                f"Average fitness after {avg_number_of_generations_for_performance} generations": f"{mean_fitness_per_generation:.10f}",
                "Average number of generations": f"{average_number_of_generations:.10f}"
            }
            runs[mutation_type] = aggregate_convergence(min_cost_per_generations_per_run,
                                                        stat=self.app_settings.plot_stat,
                                                        band=self.app_settings.plot_band)
            min_cost_per_generations_per_run_per_mutation_type[mutation_type] = min_cost_per_generations_per_run
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
            min_cost_per_generations_per_run, final_min_cost, mean_fitness_per_generation, average_number_of_generations = execute_gadapt_experiment(
                ga, optimization_name,
                result_list)
            final_results.setdefault("Experiment", {})
            final_results["Experiment"]["Experiment name"] = transform_function_string(self._experiment_name)
            final_results["Experiment"]["Number of variables"] = len(self._args_bounds)
            final_results["Experiment"]["Saturation after generations"] = self.app_settings.saturation_criteria
            avg_number_of_generations = number_of_generations_for_performance_check(average_number_of_generations,
                                                                                    self.app_settings.percentage_of_generations_for_performance)
            final_results["Experiment"][mutation_type] = {
                "Average fitness": f"{final_min_cost:.10f}",
                f"Average fitness after {avg_number_of_generations} generations": f"{mean_fitness_per_generation:.10f}",
                "Average number of generations": f"{average_number_of_generations:.10f}"
            }
            runs[mutation_type] = aggregate_convergence(min_cost_per_generations_per_run,
                                                        stat=self.app_settings.plot_stat,
                                                        band=self.app_settings.plot_band)
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
            min_cost_per_generations_per_run, final_min_cost, mean_fitness_per_generation, average_number_of_generations = execute_gadapt_experiment(ga,
                                                                                                        optimization_name,
                                                                                               result_list)
            final_results.setdefault("Experiment", {})
            final_results["Experiment"]["Experiment name"] = transform_function_string(self._experiment_name)
            final_results["Experiment"]["Number of variables"] = len(self._args_bounds)
            final_results["Experiment"]["Saturation after generations"] = self.app_settings.saturation_criteria
            average_number_of_generations_for_performance = number_of_generations_for_performance_check(average_number_of_generations,
                                                                                                        self.app_settings.percentage_of_generations_for_performance)
            final_results["Experiment"][mutation_type] = {
                "Average fitness": f"{final_min_cost:.10f}",
                f"Average fitness after {average_number_of_generations_for_performance} generations": f"{mean_fitness_per_generation:.10f}",
                "Average number of generations": f"{average_number_of_generations:.10f}"
            }
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
                                #mutation_percent_genes=[pygad_mutation_percentage_up, pygad_mutation_percentage_down],
                                mutation_probability=[self.app_settings.percentage_of_mutation_genes / 100, 0.01],  # start high, end low
                                mutation_percent_genes=[pygad_mutation_percentage_up, pygad_mutation_percentage_down],
                                mutation_type="adaptive",
                                suppress_warnings=True,
                                keep_elitism=round((self.app_settings.keep_elitism_percentage / 100) * self.app_settings.population_size),
                                stop_criteria=f"saturate_{self.app_settings.saturation_criteria}",
                                crossover_type=blend_crossover_pygad
                                )

            mutation_type = "adaptive mutation"
            optimization_name = f"{self.experiment_name} - {mutation_type}"
            min_cost_per_generations_per_run, final_min_cost, mean_fitness_per_generation, average_number_of_generations = execute_pygad_experiment(get_ga_instance,
                                                                                                       optimization_name,
                                                                                                       result_list)
            final_results.setdefault("Experiment", {})
            final_results["Experiment"]["Experiment name"] = transform_function_string(self._experiment_name)
            final_results["Experiment"]["Number of variables"] = len(self._args_bounds)
            final_results["Experiment"]["Saturation after generations"] = self.app_settings.saturation_criteria
            average_number_of_generations_for_performance = number_of_generations_for_performance_check(
                average_number_of_generations,
                self.app_settings.percentage_of_generations_for_performance)
            final_results["Experiment"][mutation_type] = {
                "Average fitness": f"{final_min_cost:.10f}",
                f"Average fitness after {average_number_of_generations_for_performance} generations": f"{mean_fitness_per_generation:.10f}",
                "Average number of generations": f"{average_number_of_generations:.10f}"
            }
            runs[mutation_type] = aggregate_convergence(min_cost_per_generations_per_run, stat=self.app_settings.plot_stat, band=self.app_settings.plot_band)
            min_cost_per_generations_per_run_per_mutation_type[mutation_type] = min_cost_per_generations_per_run
            average_generations[mutation_type] = average_number_of_generations
        ######### FINAL RESULTS #############

        log_message_info("************Final results:************")
        for r in result_list:
            log_message_info(r)
        ga_summary = summarize_ga(min_cost_per_generations_per_run_per_mutation_type,
                                  app_settings.percentage_of_generations_for_performance)
        if app_settings.log_to_file:
            csv_writter.aggregated_data_to_csv(runs, experiment_name=transform_function_string(self.experiment_name))
            csv_writter.runs_to_csv(min_cost_per_generations_per_run_per_mutation_type, experiment_name=transform_function_string(self.experiment_name))
            csv_writter.export_ga_summary_to_csv(ga_summary, file_name=transform_function_string(self.experiment_name), experiment_name=transform_function_string(self._experiment_name), number_of_variables=len(self._args_bounds), saturation_generations=self.app_settings.saturation_criteria)
        if self.app_settings.plot_fitness:
            lowest, highest, max_len = analyze_runs(runs)
            fitness_range = get_fitness_range(final_results, lowest, highest)
            plot_convergence_curve(agg=runs,
                                   x0=average_generations,
                                   lowest=fitness_range[0],
                                   highest=fitness_range[1],
                                   max_len=max_len,
                                   description=self.experiment_name,
                                   outdir=app_settings.plot_path,
                                   save=app_settings.log_to_file,
                                   basename=self.experiment_name,
                                   metrics_by_strategy=ga_summary)


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
