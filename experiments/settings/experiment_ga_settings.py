class ExperimentGASettings:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ExperimentGASettings, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        population_size: int = 64,
        percentage_of_mutation_chromosomes: float = 50,
        percentage_of_mutation_genes: float = 50.0,
        mutation_ratio: float = 0.1,
        keep_elitism_percentage: float = 50.0,
        num_runs: int = 1000,
        logging_step: int = 50,
        percentage_of_generations_for_performance: float = 0.25,
        plot_fitness: bool = True,
        saturation_criteria: int = 10,
        gadapt_random_mutation_enabled: bool = False,
        pygad_random_mutation_enabled: bool = True,
        gadapt_diversity_mutation_enabled: bool = True,
        pygad_adaptive_mutation_enabled: bool = True,
        log_to_file: bool = True,
        plot_stat = "mean",
        plot_band = "ci",
            variable_numbers=None,
            saturation_criterias=None,
        csv_path: str = None,
        plot_path: str = None
    ):
        # Prevent reinitialization for the singleton
        if getattr(self, "_initialized", False):
            return
        if variable_numbers is None:
            variable_numbers = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        if saturation_criterias is None:
            saturation_criterias = [3, 5, 10, 20, 30, 50, 100]

        self._population_size_ = None
        self._percentage_of_mutation_chromosomes_ = None
        self._percentage_of_mutation_genes_ = None
        self._mutation_ratio_ = None
        self._keep_elitism_percentage_ = None
        self._pygad_adaptive_mutation_enabled_ = None
        self._log_to_file_ = None
        self._gadapt_diversity_mutation_enabled_ = None
        self._pygad_random_mutation_enabled_ = None
        self._gadapt_random_mutation_enabled_ = None
        self._saturation_criteria_ = None
        self._plot_fitness_ = None
        self._percentage_of_generations_for_performance_ = None
        self._logging_step_ = None
        self._num_runs_ = None
        self._population_size = population_size
        self._percentage_of_mutation_chromosomes = percentage_of_mutation_chromosomes
        self._percentage_of_mutation_genes = percentage_of_mutation_genes
        self._mutation_ratio = mutation_ratio
        self._keep_elitism_percentage = keep_elitism_percentage
        self._num_runs = num_runs
        self._logging_step = logging_step
        self._percentage_of_generations_for_performance = percentage_of_generations_for_performance
        self._plot_fitness = plot_fitness
        self._saturation_criteria = saturation_criteria
        self._plot_stat = plot_stat
        self._plot_band = plot_band
        self._csv_path = csv_path
        self._plot_path = plot_path
        self._gadapt_random_mutation_enabled = gadapt_random_mutation_enabled
        self._pygad_random_mutation_enabled = pygad_random_mutation_enabled
        self._gadapt_diversity_mutation_enabled = gadapt_diversity_mutation_enabled
        self._pygad_adaptive_mutation_enabled = pygad_adaptive_mutation_enabled
        self._log_to_file = log_to_file
        self._variable_numbers = variable_numbers
        self._saturation_criterias = saturation_criterias
        self.backup_settings()

        self._initialized = True

    def backup_settings(self):
        self._population_size_ = self.population_size
        self._percentage_of_mutation_chromosomes_ = self.percentage_of_mutation_chromosomes
        self._percentage_of_mutation_genes_ = self.percentage_of_mutation_genes
        self._mutation_ratio_ = self.mutation_ratio
        self._keep_elitism_percentage_ = self.keep_elitism_percentage
        self._num_runs_ = self.num_runs
        self._logging_step_ = self.logging_step
        self._percentage_of_generations_for_performance_ = self.percentage_of_generations_for_performance
        self._plot_fitness_ = self.plot_fitness
        self._saturation_criteria_ = self.saturation_criteria

        self._gadapt_random_mutation_enabled_ = self.gadapt_random_mutation_enabled
        self._pygad_random_mutation_enabled_ = self.pygad_random_mutation_enabled
        self._gadapt_diversity_mutation_enabled_ = self.gadapt_diversity_mutation_enabled
        self._pygad_adaptive_mutation_enabled_ = self.pygad_adaptive_mutation_enabled
        self._log_to_file_ = self.log_to_file
        self._plot_stat_ = self.plot_stat
        self._plot_band_ = self.plot_band
        self._csv_path_ = self.csv_path
        self._plot_path_ = self.plot_path
        self._variable_numbers_ = self._variable_numbers
        self._saturation_criterias_ = self._saturation_criterias
    def restore_settings(self):
        self.population_size = self._population_size_
        self.percentage_of_mutation_chromosomes = self._percentage_of_mutation_chromosomes_
        self.percentage_of_mutation_genes = self._percentage_of_mutation_genes_
        self.mutation_ratio = self._mutation_ratio_
        self.keep_elitism_percentage = self._keep_elitism_percentage_
        self.num_runs = self._num_runs_
        self.logging_step = self._logging_step_
        self.percentage_of_generations_for_performance = self._percentage_of_generations_for_performance_
        self.plot_fitness = self._plot_fitness_
        self.saturation_criteria = self._saturation_criteria_

        self.gadapt_random_mutation_enabled = self._gadapt_random_mutation_enabled_
        self.pygad_random_mutation_enabled = self._pygad_random_mutation_enabled_
        self.gadapt_diversity_mutation_enabled = self._gadapt_diversity_mutation_enabled_
        self.pygad_adaptive_mutation_enabled = self._pygad_adaptive_mutation_enabled_
        self.log_to_file = self._log_to_file_
        self.plot_stat = self._plot_stat_
        self.plot_band = self._plot_band_
        self.csv_path = self._csv_path_
        self.plot_path = self._plot_path_
        self.variable_numbers = self._variable_numbers_
        self.saturation_criterias = self._saturation_criterias_

    @property
    def population_size(self) -> int:
        return self._population_size

    @population_size.setter
    def population_size(self, value: int):
        if value <= 0:
            raise ValueError("population_size must be positive")
        self._population_size = value

    @property
    def percentage_of_mutation_chromosomes(self) -> float:
        return self._percentage_of_mutation_chromosomes

    @percentage_of_mutation_chromosomes.setter
    def percentage_of_mutation_chromosomes(self, value: float):
        if not (0.0 <= value <= 100.0):
            raise ValueError("percentage_of_mutation_chromosomes must be between 0 and 100")
        self._percentage_of_mutation_chromosomes = value

    @property
    def percentage_of_mutation_genes(self) -> float:
        return self._percentage_of_mutation_genes

    @percentage_of_mutation_genes.setter
    def percentage_of_mutation_genes(self, value: float):
        if not (0.0 <= value <= 100.0):
            raise ValueError("percentage_of_mutation_genes must be between 0 and 100")
        self._percentage_of_mutation_genes = value

    @property
    def mutation_ratio(self) -> float:
        return self._mutation_ratio

    @mutation_ratio.setter
    def mutation_ratio(self, value: float):
        if not (0.0 <= value <= 100.0):
            raise ValueError("mutation_ratio must be between 0 and 100")
        self._mutation_ratio = value

    @property
    def keep_elitism_percentage(self) -> float:
        return self._keep_elitism_percentage

    @keep_elitism_percentage.setter
    def keep_elitism_percentage(self, value: float):
        if not (0.0 <= value <= 100.0):
            raise ValueError("keep_elitism_percentage must be between 0 and 100")
        self._keep_elitism_percentage = value

    @property
    def num_runs(self) -> int:
        return self._num_runs

    @num_runs.setter
    def num_runs(self, value: int):
        if value <= 0:
            raise ValueError("num_runs must be positive")
        self._num_runs = value

    @property
    def logging_step(self) -> int:
        return self._logging_step

    @logging_step.setter
    def logging_step(self, value: int):
        if value <= 0:
            raise ValueError("logging_step must be positive")
        self._logging_step = value

    @property
    def percentage_of_generations_for_performance(self) -> float:
        return self._percentage_of_generations_for_performance

    @percentage_of_generations_for_performance.setter
    def percentage_of_generations_for_performance(self, value: float):
        if value <= 0:
            raise ValueError("percentage_of_generations_for_performance must be positive")
        self._percentage_of_generations_for_performance = value

    @property
    def plot_fitness(self) -> bool:
        return self._plot_fitness

    @plot_fitness.setter
    def plot_fitness(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("plot_fitness must be a boolean")
        self._plot_fitness = value

    @property
    def saturation_criteria(self) -> int:
        return self._saturation_criteria

    @saturation_criteria.setter
    def saturation_criteria(self, value: int):
        if value <= 0:
            raise ValueError("saturation_criteria must be positive")
        self._saturation_criteria = value

    # --- new properties ---

    @property
    def gadapt_random_mutation_enabled(self) -> bool:
        return self._gadapt_random_mutation_enabled

    @gadapt_random_mutation_enabled.setter
    def gadapt_random_mutation_enabled(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("gadapt_random_mutation_enabled must be a boolean")
        self._gadapt_random_mutation_enabled = value

    @property
    def pygad_random_mutation_enabled(self) -> bool:
        return self._pygad_random_mutation_enabled

    @pygad_random_mutation_enabled.setter
    def pygad_random_mutation_enabled(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("pygad_random_mutation_enabled must be a boolean")
        self._pygad_random_mutation_enabled = value

    @property
    def gadapt_diversity_mutation_enabled(self) -> bool:
        return self._gadapt_diversity_mutation_enabled

    @gadapt_diversity_mutation_enabled.setter
    def gadapt_diversity_mutation_enabled(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("gadapt_diversity_mutation_enabled must be a boolean")
        self._gadapt_diversity_mutation_enabled = value

    @property
    def pygad_adaptive_mutation_enabled(self) -> bool:
        return self._pygad_adaptive_mutation_enabled

    @pygad_adaptive_mutation_enabled.setter
    def pygad_adaptive_mutation_enabled(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("pygad_adaptive_mutation_enabled must be a boolean")
        self._pygad_adaptive_mutation_enabled = value

    @property
    def log_to_file(self) -> bool:
        return self._log_to_file

    @log_to_file.setter
    def log_to_file(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("log_to_file must be a boolean")
        self._log_to_file = value

    @property
    def plot_stat(self) -> str:
        return self._plot_stat

    @plot_stat.setter
    def plot_stat(self, value: str):
        if not isinstance(value, str):
            raise ValueError("plot_stat must be a string")
        self._plot_stat = value

    @property
    def plot_band(self) -> str:
        return self._plot_band

    @property
    def csv_path(self) -> str:
        return self._csv_path

    @property
    def plot_path(self) -> str:
        return self._plot_path

    @property
    def variable_numbers(self) -> list[int]:
        return self._variable_numbers

    @variable_numbers.setter
    def variable_numbers(self, value: list[int]):
        if not isinstance(value, list):
            raise ValueError("variable_numbers must be a list")
        self._variable_numbers = value

    @property
    def saturation_criterias(self) -> list[int]:
        return self._saturation_criterias

    @saturation_criterias.setter
    def saturation_criterias(self, value: list[int]):
        if not isinstance(value, list):
            raise ValueError("saturation_criterias must be a list")
        self._saturation_criterias = value

    @plot_band.setter
    def plot_band(self, value: str):
        if not isinstance(value, str):
            raise ValueError("plot_band must be a string")
        self._plot_band = value

    @csv_path.setter
    def csv_path(self, value: str):
        if not isinstance(value, str):
            raise ValueError("csv_path must be a string")
        self._csv_path = value

    @plot_path.setter
    def plot_path(self, value: str):
        if not isinstance(value, str):
            raise ValueError("plot_path must be a string")
        self._plot_path = value

    def __repr__(self):
        return (
            f"ExperimentGASettings("
            f"population_size={self.population_size}, "
            f"percentage_of_mutation_chromosomes={self.percentage_of_mutation_chromosomes}, "
            f"percentage_of_mutation_genes={self.percentage_of_mutation_genes}, "
            f"mutation_ratio={self.mutation_ratio}, "
            f"keep_elitism_percentage={self.keep_elitism_percentage}, "
            f"num_runs={self.num_runs}, "
            f"logging_step={self.logging_step}, "
            f"percentage_of_generations_for_performance={self.percentage_of_generations_for_performance}, "
            f"plot_fitness={self.plot_fitness}, "
            f"saturation_criteria={self.saturation_criteria}, "
            f"gadapt_random_mutation_enabled={self.gadapt_random_mutation_enabled}, "
            f"pygad_random_mutation_enabled={self.pygad_random_mutation_enabled}, "
            f"gadapt_diversity_mutation_enabled={self.gadapt_diversity_mutation_enabled}, "
            f"pygad_adaptive_mutation_enabled={self.pygad_adaptive_mutation_enabled}, "
            f"log_to_file={self.log_to_file}, "
            f"plot_stat={self.plot_stat}"
            f"plot_band={self.plot_band}"
            f"csv_path={self.csv_path}"
            f"plot_path={self.plot_path}"
            f"variable_numbers={self.variable_numbers}"
            f"saturation_criterias={self.saturation_criterias}"
            f")"
        )
