# Installation

This page covers installing PyGPLA for (a) use, (b) running simulations, (c) building the
documentation, and (d) development/testing.

## Requirements

- Python **3.10+**
- NumPy **≥ 1.21**

Optional extras:

- SciPy (`.[sim]`) for simulations and typical signal-processing steps (bandpass + Hilbert)
- Sphinx + MyST (`.[docs]`) for documentation builds
- PyTest (`.[tests]`) for running the test suite

## Recommended: use a virtual environment

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Or using Conda:

```bash
conda create -n pygpla python=3.12
conda activate pygpla
```

## Install PyGPLA

### From PyPI (placeholder)

Once published:

```bash
pip install pygpla
```

### From source (recommended for now)

```bash
git clone https://github.com/CMC-lab/PyGPLA.git
cd PyGPLA
pip install -e .
```

### Install optional extras

```bash
# simulations / signal processing
pip install -e ".[sim]"

# docs build
pip install -e ".[docs]"

# tests
pip install -e ".[tests]"
```

You can combine extras:

```bash
pip install -e ".[sim,docs,tests]"
```

## Verify the installation

```{code-block} python
import pygpla
print(pygpla.__version__)
```

## Build the docs

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html`.

## Run the tests

```bash
pip install -e ".[tests]"
pytest
```

## Troubleshooting

### SciPy install issues

If `pip install -e ".[sim]"` fails, try upgrading pip first:

```bash
python -m pip install --upgrade pip
```

On some systems, you may prefer installing SciPy via Conda:

```bash
conda install scipy
```

### Docs theme import error

If Sphinx errors with `No module named 'sphinx_rtd_theme'`, ensure you installed the docs
extras:

```bash
pip install -e ".[docs]"
```

## Next steps

- Quickstart: :doc:`quickstart`
- Tutorials (Figure 2 walkthrough): :doc:`tutorials`
- Usage notes and configuration knobs: :doc:`usage`
