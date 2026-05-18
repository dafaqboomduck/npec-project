## Documentation

Full documentation is published automatically to GitHub Pages on every push to `main`:

**https://BredaUniversityADSAI.github.io/2025-26d-fai2-adsai-group-computervision5/**

To build the docs locally:

```bash
pip install -r requirements/docs.txt
sphinx-build -b html docs/ docs/_build/html
# then open docs/_build/html/index.html
```

## Installation Guide

This repository now ships as a Python package under src:
- npec_cv: computer vision pipeline

Distribution names on TestPyPI:
- npec_cv

Install npec-cv (default):

```bash
pip install -e src/npec_cv
```

Install npec-cv with CUDA extras:

```bash
pip install -e src/npec_cv[cuda]
```

Install from package index after publishing:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ npec-cv
```

## Build And Publish Wheels

Build npec_cv wheel:

```bash
mkdir -p build dist
TMPDIR=$PWD/build python -m build --outdir dist src/npec_cv
```

Or more simply:

```bash
python -m build
```

Upload each dist directory separately with twine (TestPyPI example):

```bash
python -m twine upload --repository testpypi dist/*
```

## Dev Setup

### Backend
Install dependencies and activate pre-commit hooks. Required once per machine after cloning.

```bash
pip install -r requirements/dev.txt
pre-commit install
```

Hooks run automatically on every `git commit` — black, ruff, and isort will fix or flag issues before the commit goes through. To run manually across the whole codebase:

```bash
pre-commit run --all-files
```

### Frontend
Install Node dependencies once after cloning:

```bash
cd frontend
npm install
```

Before pushing, run lint and formatting checks manually:

```bash
npm run lint
npx prettier --check .
```

To auto-fix formatting:

```bash
npx prettier --write .
```

Frontend linting is also enforced automatically via GitHub Actions on every pull request that touches `frontend/`.