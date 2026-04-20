## Installation Guide

This repository now ships as two Python packages under src:
- npec_cv: computer vision pipeline
- npec_robotics: robotics pipeline (depends on npec_cv)

Distribution names on PyPI/TestPyPI:
- npec_cv
- npec_robotics

Install npec-cv (default):

```bash
pip install -e src/npec_cv
```

Install npec-cv with CUDA extras:

```bash
pip install -e src/npec_cv[cuda]
```

Install npec-robotics (after npec-cv is installed locally):

```bash
pip install -e src/npec_robotics
```

Install from package index after publishing:

```bash
pip install npec_cv
pip install npec_robotics
```

## Build And Publish Two Different Wheels

Build npec_cv wheel:

```bash
mkdir -p build dist
TMPDIR=$PWD/build python -m build --outdir dist src/npec_cv
```

Build npec_robotics wheel:

```bash
TMPDIR=$PWD/build python -m build --outdir dist src/npec_robotics
```

Upload each dist directory separately with twine (TestPyPI example):

```bash
python -m twine upload --repository testpypi dist/*
```


## DEV set up

Set up for developers: pre-commit, formatting, linting, and import sorting.

```bash
pip install -r requirements-dev.txt
pre-commit install
```