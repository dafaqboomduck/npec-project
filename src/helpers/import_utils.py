import importlib
from src.config.packages import PACKAGES

def attempt_imports(package_names=PACKAGES):
    modules = {}

    for pkg in package_names:
        try:
            module = importlib.import_module(pkg.value)
            modules[pkg.name] = module
        except ImportError as e:
            raise ImportError(
                f"Failed to import {pkg.value}. "
                "Make sure it is installed in the current Python environment."
            ) from e

    return modules