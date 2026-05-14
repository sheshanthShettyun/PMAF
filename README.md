# PMAF Topic Code Collection

This folder gathers the PMAF assignment code in normal Python files.

## Folders

- `pmafdhuv_converted_notebooks/`: code cells exported from the `pmafdhuv` Jupyter notebooks and renamed by topic.
- `current_codebase_scripts/`: existing Python scripts from `Pmaf Lab` copied as-is.

## Virtual environment

Use the existing Pythonproject virtual environment:

```bash
cd /home/sriyaan/Documents/visualstudioNOToff/Pythonproject/PMAF_Topic_Code_Collection
source ../.venv/bin/activate
```

Some missing packages were installed afterward into `../.venv`.

## Dependency status

Installed into `../.venv`: `mlxtend`, `statsmodels`, and `flask`.

`tensorflow` was not installed because pip reported no compatible build for this Python 3.14 virtual environment. The deep-learning script is still included in `current_codebase_scripts/`, but it will need a TensorFlow-compatible Python version to run.
