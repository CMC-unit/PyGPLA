# Development Notes

This directory preserves internal planning documents created while migrating the earlier
PyGPLA development code into the current Python package. They are retained as a historical
record of the project’s design and migration process, not as current user documentation.
For current information, see the repository `README.md` and the documentation under `docs/`.

## Implemented

The planned `src/pygpla` package structure, layered core/preprocessing/statistics API,
configuration objects, supported simulations, Hatchling packaging, tests, CI, Sphinx
documentation, and PyPI distribution were implemented.

## Deviations

Unsupported or unused planned components—including multifrequency simulations, utility and
statistics-summary modules, bundled datasets, and a CLI—were not retained. Sphinx was chosen
instead of MkDocs, Python support starts at 3.10, and automated regression fixtures against
MATLAB reference outputs remain future work.
