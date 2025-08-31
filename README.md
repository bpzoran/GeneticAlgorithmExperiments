
# Genetic Algorithm Optimization Experiments

This repository contains a Python project for conducting genetic algorithm optimization experiments. The experiments focus on comparing different mutation strategies, namely adaptive mutation, random mutation, and diversity mutation, using the PyGAD and GAdapt libraries.

## Project Structure

The project consists of the following modules:
```
genetic-algorithm-optimization/
├── experiments/
│   ├── use_case_01_simple_trigonometric_arithmetic_function.py
│   ├── use_case_02_complex_trig_function.py
│   ├── use_case_03_simple_trig_function.py
│   ├── use_case_04_sphere_function.py
│   ├── use_case_05_rosenbrock_function_3_variables.py
│   ├── use_case_06_rosenbrock_function_7_variables.py
│   ├── use_case_07_rastrigin_function_3_variables.py
│   ├── use_case_08_rastrigin_function_7_variables.py
│   ├── use_case_09_ackley_function_7_variables.py
│   ├── use_case_10_griewank_function.py
│   ├── use_case_11_beale_function.py
│   ├── use_case_12_himmelblau_function.py
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

`experiments/use_case_01_simple_trigonometric_arithmetic_function.py` performes optimization for the simple trigonometric arithmetic function.
    
`experiments/use_case_02_complex_trig_function.py` performs optimization for the complex trigonometric function.
    
`experiments/use_case_03_simple_trig_function.py` performs optimization for the simple trigonometric function.
    
`experiments/use_case_04_sphere_function.py` performs optimization for the sphere function.
    
`experiments/use_case_05_rosenbrock_function_3_variables.py` performs optimization for the Rosenbrock function with 3 variables.
    
`experiments/use_case_06_rosenbrock_function_7_variables.py` performs optimization for the Rosenbrock function with 7 variables.
    
`experiments/use_case_07_rastrigin_function_3_variables.py` performs optimization for the Rastrigin function with 3 variables.
    
`experiments/use_case_08_rastrigin_function_7_variables.py` performs optimization for the Rastrigin function with 7 variables.
    
`experiments/use_case_09_ackley_function_7_variables.py` performs optimization for the Ackley function with 7 variables.
    
`experiments/use_case_10_griewank_function.py` performs optimization for the Griewank function.
    
`experiments/use_case_11_beale_function.py` performs optimization for the Beale function.
    
`experiments/use_case_12_himmelblau_function.py` performs optimization for the Himmelblau function.

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
python use_case_01_simple_trigonometric_arithmetic_function.py
python use_case_02_complex_trig_function.py
python use_case_03_simple_trig_function.py
python use_case_04_sphere_function.py
python use_case_05_rosenbrock_function_3_variables.py
python use_case_06_rosenbrock_function_7_variables.py
python use_case_07_rastrigin_function_3_variables.py
python use_case_08_rastrigin_function_7_variables.py
python use_case_09_ackley_function_7_variables.py
python use_case_10_griewank_function.py
python use_case_11_beale_function.py
python use_case_12_himmelblau_function.py
```
The default number of runs for each tested strategy is 1000. To customize the number of runs, add the desired number as a parameter to the command. For example:

```bash
python use_case_06_rosenbrock_function_7_variables.py 100
```
This will execute each strategy 100 times.

To plot fitness per generation, add the `plot` parameter to the command. For example:
```bash
python use_case_12_himmelblau_function.py plot
```
This will plot fitness per generation for the final run of each strategy.

You can combine custom run numbers and the `plot` parameter. For example:
```bash
python use_case_11_beale_function.py plot 20
```
This will execute each strategy 20 times and plot the fitness per generation for the final run of each strategy.

All experiments can be run in parallel by executing the following command from the `experiments` folder:
```python run_all_experiments.py
```

Number of runs and plotting can be customized in the same way as described above.

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
