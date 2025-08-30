class ExperimentGASettings:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ExperimentGASettings, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        population_size: int = 64,
        percentage_of_mutation_chromosomes: float = 80.0,
        percentage_of_mutation_genes: float = 50.0,
        keep_elitism_percentage: float = 50.0,
        num_runs: int = 1000,
        logging_step: int = 50,
        number_of_generations: int = 40,
        plot_fitness: bool = False,
        saturation_criteria: int = 10,
        gadapt_random_mutation_enabled: bool = False,
        pygad_random_mutation_enabled: bool = True,
        gadapt_diversity_mutation_enabled: bool = True,
        pygad_adaptive_mutation_enabled: bool = True,
    ):
        # Prevent reinitialization for the singleton
        if getattr(self, "_initialized", False):
            return
        self._population_size_ = None
        self._percentage_of_mutation_chromosomes_ = None
        self._percentage_of_mutation_genes_ = None
        self._keep_elitism_percentage_ = None
        self._pygad_adaptive_mutation_enabled_ = None
        self._gadapt_diversity_mutation_enabled_ = None
        self._pygad_random_mutation_enabled_ = None
        self._gadapt_random_mutation_enabled_ = None
        self._saturation_criteria_ = None
        self._plot_fitness_ = None
        self._number_of_generations_ = None
        self._logging_step_ = None
        self._num_runs_ = None
        self._population_size = population_size
        self._percentage_of_mutation_chromosomes = percentage_of_mutation_chromosomes
        self._percentage_of_mutation_genes = percentage_of_mutation_genes
        self._keep_elitism_percentage = keep_elitism_percentage
        self._num_runs = num_runs
        self._logging_step = logging_step
        self._number_of_generations = number_of_generations
        self._plot_fitness = plot_fitness
        self._saturation_criteria = saturation_criteria

        self._gadapt_random_mutation_enabled = gadapt_random_mutation_enabled
        self._pygad_random_mutation_enabled = pygad_random_mutation_enabled
        self._gadapt_diversity_mutation_enabled = gadapt_diversity_mutation_enabled
        self._pygad_adaptive_mutation_enabled = pygad_adaptive_mutation_enabled
        self.backup_settings()

        self._initialized = True

    def backup_settings(self):
        self._population_size_ = self.population_size
        self._percentage_of_mutation_chromosomes_ = self.percentage_of_mutation_chromosomes
        self._percentage_of_mutation_genes_ = self.percentage_of_mutation_genes
        self._keep_elitism_percentage_ = self.keep_elitism_percentage
        self._num_runs_ = self.num_runs
        self._logging_step_ = self.logging_step
        self._number_of_generations_ = self.number_of_generations
        self._plot_fitness_ = self.plot_fitness
        self._saturation_criteria_ = self.saturation_criteria

        self._gadapt_random_mutation_enabled_ = self.gadapt_random_mutation_enabled
        self._pygad_random_mutation_enabled_ = self.pygad_random_mutation_enabled
        self._gadapt_diversity_mutation_enabled_ = self.gadapt_diversity_mutation_enabled
        self._pygad_adaptive_mutation_enabled_ = self.pygad_adaptive_mutation_enabled
    def restore_settings(self):
        self.population_size = self._population_size_
        self.percentage_of_mutation_chromosomes = self._percentage_of_mutation_chromosomes_
        self.percentage_of_mutation_genes = self._percentage_of_mutation_genes_
        self.keep_elitism_percentage = self._keep_elitism_percentage_
        self.num_runs = self._num_runs_
        self.logging_step = self._logging_step_
        self.number_of_generations = self._number_of_generations_
        self.plot_fitness = self._plot_fitness_
        self.saturation_criteria = self._saturation_criteria_

        self.gadapt_random_mutation_enabled = self._gadapt_random_mutation_enabled_
        self.pygad_random_mutation_enabled = self._pygad_random_mutation_enabled_
        self.gadapt_diversity_mutation_enabled = self._gadapt_diversity_mutation_enabled_
        self.pygad_adaptive_mutation_enabled = self._pygad_adaptive_mutation_enabled_

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
    def number_of_generations(self) -> int:
        return self._number_of_generations

    @number_of_generations.setter
    def number_of_generations(self, value: int):
        if value <= 0:
            raise ValueError("number_of_generations must be positive")
        self._number_of_generations = value

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

    def __repr__(self):
        return (
            "ExperimentGASettings("
            f"num_runs={self.num_runs}, "
            f"logging_step={self.logging_step}, "
            f"number_of_generations={self.number_of_generations}, "
            f"plot_fitness={self.plot_fitness}, "
            f"saturation_criteria={self.saturation_criteria}, "
            f"gadapt_random_mutation_enabled={self.gadapt_random_mutation_enabled}, "
            f"pygad_random_mutation_enabled={self.pygad_random_mutation_enabled}, "
            f"gadapt_diversity_mutation_enabled={self.gadapt_diversity_mutation_enabled}, "
            f"pygad_adaptive_mutation_enabled={self.pygad_adaptive_mutation_enabled}"
            ")"
        )
