(contributing)=
# Contributing

Thanks for your interest in contributing to **PyGPLA**.

Contributions of all sizes are welcome: bug reports, documentation improvements, examples,
tests, performance fixes, and new features. If you’re unsure where to start, open an issue
with your question or proposal.

## TL;DR

1. Fork the repo and clone your fork locally.
2. Create a feature branch.
3. Install dev dependencies.
4. Make changes + add tests/docs as needed.
5. Run `pytest` and `ruff`.
6. Open a pull request.

## Reporting issues

Before opening a new issue, please search existing issues to avoid duplicates.

When you open an issue, include:

- What you expected vs what happened
- Minimal reproduction steps (code snippets are ideal)
- Environment details: OS, Python version, `pygpla` version/commit, and relevant deps
  (NumPy/SciPy/Matplotlib)

## Feature requests

Feature proposals are easiest to review when they include:

- The problem you’re trying to solve
- A concrete use case (what the user would write/do)
- Any relevant references (e.g., GPLA paper section, MATLAB parity requirements)
- Optional: implementation sketch (API shape, module placement)

If it’s a larger change, opening an issue first is usually better than starting with a big PR.

## Development setup

### Clone + environment

```bash
git clone https://github.com/CMC-unit/PyGPLA.git
cd PyGPLA
python -m venv .venv
source .venv/bin/activate
```

### Install in editable mode

Core install:

```bash
pip install -e .
```

Recommended dev extras:

```bash
pip install -e ".[tests,docs,sim]"
```

- `.[tests]` installs `pytest`
- `.[docs]` installs Sphinx + MyST (and the Sphinx theme)
- `.[sim]` installs SciPy (needed for some simulations and real-data style preprocessing)

## Running tests

```bash
pytest
```

If you changed numerical code, add tests that:

- validate shapes/dtypes
- cover edge cases (e.g., zero-spike units, NaNs, whitening on/off)
- are robust to small numeric differences (`np.testing.assert_allclose`)

## Linting / style

This repo uses Ruff for linting (configured in `pyproject.toml`).

```bash
ruff check .
```

If you use Ruff formatting in your environment:

```bash
ruff format .
```

Style expectations:

- Prefer small, well-scoped PRs over large refactors.
- Keep APIs consistent with `pygpla.api.gpla` and the existing config/dataclass patterns.
- Add docstrings for public functions/classes (Sphinx builds API docs from them).

## Building the docs

Install doc dependencies and build:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

The rendered docs start at `docs/_build/html/index.html`.

:::{note}
Docs are written in Markdown using MyST (`myst_parser`). Prefer MyST-friendly features:
`{py:func}` roles, `{code-block}` directives when you want line numbers, and `:::` fenced
admonitions.
:::

## Submitting a pull request

### Branching

Create a branch from `main`:

```bash
git checkout -b my-change
```

### Commit messages

Keep commit messages descriptive (what/why), not just “fix” or “update”.

### PR checklist

- Code changes include tests when appropriate
- Public API changes include doc updates (Quickstart/Tutorials/API docs)
- `pytest` passes
- `ruff check .` passes

## Licensing and attribution

By submitting a contribution, you confirm you have the right to submit it and you agree it
may be redistributed under this project’s license terms.
