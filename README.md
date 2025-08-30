
# Genetic Algorithm Optimization Experiments

This repository contains a Python project for conducting genetic algorithm optimization experiments. The experiments focus on comparing different mutation strategies, namely adaptive mutation, random mutation, and diversity mutation, using the PyGAD and GAdapt libraries.

## Project Structure

The project consists of the following modules:

- `diversity_based_mutation_use_case_1.py`: Optimizes a simple trigonometric function using different mutation strategies.
- `diversity_based_mutation_use_case_2.py`: Optimizes a complex trigonometric function using different mutation strategies.
- `diversity_based_mutation_use_case_3.py`: Optimizes an arithmetic/trigonometric function using different mutation strategies.
- `exp_logging.py`: Provides logging utilities to record the experiments' progress and results.

## Experiment Details

### diversity_based_mutation_use_case_1.py
This module performs optimization on a simple trigonometric function using GAs with different mutation strategies. The function being optimized is:

```python
def simple_trigonometric_arithmetic_function(args):
    term1 = math.sqrt(args[0])
    term2 = math.cos(args[1]) ** 2
    term3 = math.sin(args[2])
    term4 = args[3] ** 2
    term5 = math.sqrt(abs(args[4]))

    return term1 + term2 + term3 + term4 + term5
```

Optimization strategies used:

1. PyGAD with Adaptive Mutation
2. GAdapt with Random Mutation
3. GAdapt with Diversity Mutation

### diversity_based_mutation_use_case_2.py
This module performs optimization on a complex trigonometric function using GAs with different mutation strategies. The function being optimized is:

```python
def complex_trig_func(args):
    if len(args) != 7:
        raise ValueError("Input vector must contain 7 variables.")
    return (math.sqrt(abs(math.cos(args[0]))) +
            math.pow(math.cos(args[1]), 2) +
            math.sin(args[2]) +
            math.pow(args[3], 2) +
            math.sqrt(args[4]) +
            math.cos(args[5]) -
            (args[6] * math.sin(pow(args[6], 3)) + 1) +
            math.sin(args[0]) / (math.sqrt(args[0]) / 3 + (args[6] * math.sin(pow(args[6], 3)) + 1)) / math.sqrt(args[4]) +
            math.cos(args[5]))
```

Optimization strategies used:

1. PyGAD with Adaptive Mutation
2. GAdapt with Random Mutation
3. GAdapt with Diversity Mutation

### diversity_based_mutation_use_case_3.py

This module performs optimization on a trigonometric function using GAs with different mutation strategies. The function being optimized is:

```python
def simple_trig_func(args):
    if len(args) != 8:
        raise ValueError("Input vector must contain 8 variables.")
    return np.sin(args[0]) + np.cos(args[1]) + args[2] ** 2 + np.sin(args[3]) * np.cos(args[4]) + args[5] + np.cos(args[6]) * args[7]
```

Optimization strategies used:

1. PyGAD with Adaptive Mutation
2. GAdapt with Random Mutation
3. GAdapt with Diversity Mutation

### exp_logging.py

This module initializes logging for the genetic algorithm experiments and provides utility functions to log messages.

## Installation
To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Running the Experiments
To run the experiments, execute the following commands from the `experiments` folder:
```bash
python diversity_based_mutation_use_case_1.py
python diversity_based_mutation_use_case_2.py
python diversity_based_mutation_use_case_3.py
```

The default number of runs for each tested strategy is 1000. To customize the number of runs, add the desired number as a parameter to the command. For example:

```bash
python use_case_01_simple_trigonometric_arithmetic_function.py 100
```
This will execute each strategy 100 times.

To plot fitness per generation, add the `plot` parameter to the command. For example:
```bash
python use_case_02_complex_trig_function.py plot
```
This will plot fitness per generation for the final run of each strategy.

You can combine custom run numbers and the `plot` parameter. For example:
```bash
python use_case_03_simple_trig_function.py plot 20
```
This will execute each strategy 20 times and plot the fitness per generation for the final run of each strategy.

## Requirements
- Python 3.6 or higher
- gadapt==0.4.18
- pygad==3.3.1

The required libraries are listed in `requirements.txt`.

## Logging
Logs are generated in the `log` directory with timestamped filenames. The logs include detailed information about each optimization run, including fitness values and the number of generations completed.

## Figures
The figures used in the research paper related to the GAdapt library and diversity-based mutation are stored in the `figures` folder.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
