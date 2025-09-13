# all_experiments.py
from importlib import import_module
from pathlib import Path
from typing import Callable, List, Optional

from runners.experiment_runner import run_experiment
from settings.experiment_ga_settings import ExperimentGASettings


def _discover_use_case_modules() -> List[str]:
    """
    Find all Python modules in the same directory as this file whose names
    start with 'use_case', excluding this file and __init__.py. Returns a
    list of module *stems* sorted alphabetically.
    """
    here = Path(__file__).resolve().parent
    this_stem = Path(__file__).resolve().stem

    candidates = []
    for p in here.iterdir():
        if (
            p.is_file()
            and p.suffix == ".py"
            and p.stem.startswith("use_case")
            and p.stem not in {this_stem, "__init__"}
        ):
            candidates.append(p.stem)

    candidates.sort()  # sort by module name
    return candidates


def _import_use_case(stem: str):
    """
    Import a module by stem from the current package (relative import if packaged),
    otherwise as a top-level module.
    """
    if __package__:
        return import_module(f".{stem}", package=__package__)
    return import_module(stem)


def _get_main(mod) -> Optional[Callable[[], None]]:
    """
    Return mod.main if present and callable, else None.
    """
    fn = getattr(mod, "main", None)
    return fn if callable(fn) else None


def run_all_use_cases() -> None:
    # must start exactly like this:
    app_settings = ExperimentGASettings()
    app_settings.num_runs = 1000
    app_settings.plot_fitness = True
    app_settings.backup_settings()

    for stem in _discover_use_case_modules():
        mod = _import_use_case(stem)
        main_fn = _get_main(mod)
        if main_fn is None:
            # Skip modules without a main()
            continue

        # restore settings before each module's main()
        app_settings.restore_settings()
        main_fn()


def main() -> None:
    run_experiment(run_all_use_cases)


if __name__ == "__main__":
    main()
