
# Genetic Algorithm Optimization Experiments

This repository contains a Python project for conducting genetic algorithm optimization experiments. The experiments focus on comparing different mutation strategies, namely adaptive mutation, random mutation, and diversity mutation, using the PyGAD and GAdapt libraries.

## Project Structure

The project consists of the following modules:
```
genetic-algorithm-optimization/
├── experiments/
│   ├── use_case_01_separated_trig_arithmetic_function.py
│   ├── use_case_02_highly_coupled_trigonometric_function.py
│   ├── use_case_03_moderately_coupled_trigonometric_function.py
│   ├── use_case_04_sphere_function.py
│   ├── use_case_05_rosenbrock_function.py
│   ├── use_case_06_rastrigin_function.py
│   ├── use_case_07_ackley_function.py
│   ├── use_case_08_griewank_function.py
│   ├── use_case_09_beale_function.py
│   ├── use_case_10_himmelblau_function.py
│   ├── use_case_11_styblinski_tang_function.py
│   ├── use_case_12_booth_function.py
│   ├── all_experiments.py
│   ├── __init__.py
│   ├── utils/
│   │   ├── exp_logging.py
│   │   ├── plot_fitness_per_generation.py
│   │   └── __init__.py
│   ├── runners/
│   │   ├── experiment.py
│   │   ├── experiment_runner.py
│   │   ├── gadapt_experiment.py
│   │   ├── pygad_experiment.py
│   │   └── __init__.py
│   ├── figures/
│   │   └── (generated figures go here)
│   └── settings/
│       └── experiment_ga_settings.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```
Explanation:

`experiments/`: Contains Python scripts for running different use cases.

`experiments/use_case_01_separated_trig_arithmetic_function.py` performes optimization for the separated trigonometric arithmetic function.
    
`experiments/use_case_02_highly_coupled_trigonometric_function.py` performs optimization for the highly coupled trigonometric function.
    
`experiments/use_case_03_moderately_coupled_trigonometric_function.py` performs optimization for the moderately coupled trigonometric function.
    
`experiments/use_case_04_sphere_function.py` performs optimization for the sphere function.
    
`experiments/use_case_05_rosenbrock_function.py` performs optimization for the Rosenbrock function.
    
`experiments/use_case_06_rastrigin_function.py` performs optimization for the Rastrigin function.
    
`experiments/use_case_07_ackley_function.py` performs optimization for the Ackley function.
    
`experiments/use_case_08_griewank_function.py` performs optimization for the Griewank function.
    
`experiments/use_case_09_beale_function.py` performs optimization for the Beale function.
    
`experiments/use_case_10_himmelblau_function.py` performs optimization for the Himmelblau function.
    
`experiments/use_case_11_styblinski_tang_function.py` performs optimization for the Styblinski-Tang function.
    
`experiments/use_case_12_booth_function.py` performs optimization for the Booth function.

`experiments/all_experiments.py`: Runs all experiments sequentially.

`experiments/__init__.py`: Initializes the experiments

`experiments/runners/`: Contains modules for running experiments with different GA libraries.

`experiments/runners/experiment.py`: Defines the base experiment class.

`experiments/runners/experiment_runner.py`: Manages the execution of experiments.

`experiments/runners/gadapt_experiment.py`: Implements experiments using the GAdapt library.

`experiments/runners/pygad_experiment.py`: Implements experiments using the PyGAD library.

`experiments/settings/`: Contains configuration settings for experiments.

`experiments/settings/experiment_ga_settings.py`: Defines GA settings for experiments.

`experiments/utils/`: Contains utility modules.

`experiments/utils/exp_logging.py`: Initializes logging for experiments.

`experiments/utils/plot_fitness_per_generation.py`: Plots fitness per generation.

`requirements.txt`: Lists dependencies for the project.

`README.md`: Provides an overview of the project.

`LICENSE`: Specifies the project license.

`.gitignore`: Excludes unnecessary files (e.g., __pycache__/, logs, etc.) from version control.

## Experiment Details

This project investigates the effectiveness of different genetic algorithm mutation strategies on a variety of benchmark optimization problems. Each experiment applies the following strategies:

- **PyGAD with Random Mutation**: Utilizes the standard random mutation operator provided by the PyGAD library.
- **GAdapt with Diversity Mutation**: Employs the diversity-based mutation operator from the GAdapt library, which aims to maintain population diversity and avoid premature convergence.
- **PyGAD with Adaptive Mutation**: Uses PyGAD's adaptive mutation, where mutation rates are adjusted dynamically based on population fitness.

The experiments are run across multiple functions, including trigonometric, sphere, Rosenbrock, Rastrigin, Ackley, Griewank, Beale, and Himmelblau functions. For each strategy, performance metrics such as best fitness, convergence speed, and population diversity are logged and analyzed. Results are visualized to compare the strengths and weaknesses of each mutation approach.

## Installation
To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Running the Experiments
To run the experiments, execute the following commands from the `experiments` folder:
```bash
python use_case_01_separated_trig_arithmetic_function.py
python use_case_02_highly_coupled_trigonometric_function.py
python use_case_03_moderately_coupled_trigonometric_function.py
python use_case_04_sphere_function.py
python use_case_05_rosenbrock_function.py
python use_case_06_rastrigin_function.py
python use_case_07_ackley_function.py
python use_case_08_griewank_function.py
python use_case_09_beale_function.py
python use_case_10_himmelblau_function.py
python use_case_11_styblinski_tang_function.py
python use_case_12_booth_function.py
```

### Running All Experiments

You can run all experiments sequentially using the `all_experiments.py` script. This script automatically discovers all use case modules in the `experiments` directory and executes them one by one. It also merges the results into a single CSV file and generates a summary report.

To run all experiments, execute the following command from the `experiments` folder:
```bash
python all_experiments.py
```

### Command-Line Arguments

You can customize the experiments using the following command-line arguments:

**Integers:**
- `--population_size`: Population size (default: 64)
- `--num_runs`: Number of runs (default: 100)
- `--logging_step`: Logging step (default: 50)
- `--saturation_criteria`: Saturation criteria (default: 15)

**Floats:**
- `--percentage_of_mutation_chromosomes`: Percentage of mutation chromosomes (default: 60)
- `--percentage_of_mutation_genes`: Percentage of mutation genes (default: 60)
- `--mutation_ratio`: Mutation ratio (default: 0.1)
- `--keep_elitism_percentage`: Keep elitism percentage (default: 50.0)
- `--percentage_of_generations_for_performance`: Percentage of generations for performance (default: 0.25)

**Booleans (accepts true/false, yes/no, 1/0):**
- `--plot_fitness`: Plot fitness (default: True)
- `--gadapt_random_mutation_enabled`: Gadapt random mutation enabled (default: False)
- `--pygad_random_mutation_enabled`: Pygad random mutation enabled (default: True)
- `--gadapt_diversity_mutation_enabled`: Gadapt diversity mutation enabled (default: True)
- `--pygad_adaptive_mutation_enabled`: Pygad adaptive mutation enabled (default: True)
- `--log_to_file`: Log to file (default: True)

**Strings:**
- `--plot_stat`: Plot stat (default: "mean")
- `--plot_band`: Plot band (default: "ci")
- `--csv_path`: CSV path (default: None)
- `--results_path`: Results path (default: None)
- `--plot_path`: Plot path (default: None)

**Lists:**
- `--variable_numbers`: Variable numbers (space-separated integers) (default: [2, 7])
- `--saturation_criterias`: Saturation criterias (space-separated integers) (default: [15])

### Examples

Run an experiment with 100 runs:
```bash
python use_case_05_rosenbrock_function.py --num_runs 100
```

Run an experiment and plot fitness:
```bash
python use_case_10_himmelblau_function.py --plot_fitness true
```

Run an experiment with custom population size and mutation ratio:
```bash
python use_case_09_beale_function.py --population_size 128 --mutation_ratio 0.2
```

Run all experiments with 50 runs each:
```bash
python all_experiments.py --num_runs 50
```

## Requirements
- Python 3.12 or higher
- gadapt==0.4.23
- pygad==3.5.0

The required libraries are listed in `requirements.txt`.

## Logging
Logs are generated in the `log` directory with timestamped filenames. The logs include detailed information about each optimization run, including fitness values and the number of generations completed.

## Figures
The figures used in the research paper related to the GAdapt library and diversity-based mutation are stored in the `figures` folder.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
