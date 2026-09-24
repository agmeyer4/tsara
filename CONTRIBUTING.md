# Contributing to TSARA

This file is for someone about to change the package. It says how to set up,
which gates a change has to pass, how the code is laid out and why, and what
a change is expected to bring with it. The science lives in
[`docs/METHODS.md`](docs/METHODS.md); the reader-facing tour is the
[README](README.md).

## Set up

Python 3.11 or newer, in an environment of your own (never the system
interpreter):

```bash
git clone https://github.com/agmeyer4/tsara.git
cd tsara
pip install -e ".[dev,viz]"
```

`dev` is the test and notebook toolchain; `viz` is matplotlib alone. The
install is editable, so source changes are live; reinstall only when
`pyproject.toml`'s dependencies change.

## The gates

Every change passes all four before it is proposed, and continuous
integration (`.github/workflows/ci.yml`) runs the same four on every pull
request:

```bash
ruff check . && ruff format --check .   # lint (pycodestyle, pyflakes, isort, pyupgrade,
                                        # NumPy docstrings) and formatting, notebooks included
mypy --strict src tests                 # types, strict, tests included
pytest --cov=tsara --cov-branch         # the suite; 100 % line AND branch coverage
                                        # is a floor that fails the run
```

Plain `pytest` runs the suite without the coverage floor, which is what you
want while iterating on one test file; the floor applies only to a full run
with the coverage flags. Two opt-in gates need data or time that CI does not
have:

```bash
TSARA_NOTEBOOKS=1 pytest tests/test_notebooks.py    # executes notebook 04 (about a minute)
TSARA_ARCHIVE=/path/to/Data pytest tests/test_notebooks.py
                                                    # executes notebook 04b against the
                                                    # campaign archive and requires every
                                                    # ledger row to reproduce
```

Run both before a pull request that touches alignment or the notebooks.

## How the package is laid out

There is no ecosystem standard for the order inside a Python module, so TSARA
fixes one house rule and applies it everywhere. The checkable parts are held
by `tests/test_architecture.py`; the rest is this section.

**Four layers, and imports point down only.**

```
core/        a leaf: imports nothing else from tsara
config/      every schema plus the one YAML door: imports only core
synthetic/  ingest/  align/   the stages: import core, config and themselves,
             and NEVER each other. They hand each other xarray Datasets whose
             vocabulary lives in core/naming.py. That is what lets a new stage
             be inserted anywhere.
pipeline, cli   (future) on top
```

`tsara/_version.py` is the one place the version number is written; anything
may import it, and nothing imports the package root. A new package needs a row
in the `LAYERS` table of the architecture test before its imports are policed.

**Every stage package has one skeleton.** An `__init__.py` whose docstring
maps every module, in the order the stage runs, and re-exports the public
API. One module per operation, named for it (`binning.py`, `pairing.py`,
`campaign.py`). Helpers that two siblings need live in a module named for what
it holds (`ingest/base.py`, `align/variables.py`, `align/cells.py`), never
inside an operation module. A `bundle.py` for saving and loading the stage's
product.

**Every module reads from the top.** In order: the module docstring (what it
does, where it sits, which METHODS section explains it), imports, `logger`,
`__all__`, then constants, types and the error class, then the public
functions with the entry point first, then each private helper after the
function that first calls it, in the order that caller reaches for it. A
private class is a type and sits with the constants. A long module carries
`# ---` section headers that match its docstring's outline. Python does not
care about any of this; a reader opening any file does.

**Tests mirror the source.** `tests/<package>/test_<module>.py` for every
module, plus `tests/test_architecture.py` (the shape of the package),
`tests/test_examples.py` (every example config validates and every manifest
field is demonstrated) and `tests/test_notebooks.py` (the opt-in executions).
A test that enumerates its own subjects by hand is treated as a bug: discover
the subjects by walking the tree, and guard the discovery against returning
nothing, because a test parametrized over zero cases passes.

## What a change brings with it

- **Tests**, written with the change. Anything whose job is to catch a
  mistake is mutation-tested: plant the mistake, see the test fail, restore
  the file. A test that has never failed has proven nothing.
- **A section in `docs/METHODS.md`** for any algorithm, estimator or rule of
  arithmetic, in the same change, never later. Swappable estimators are
  registered by name and every registered name has a section. Rejected
  alternatives are written down there too, with what rejected them.
- **Save and load** for any new stage product, in the change that introduces
  the product.
- **Documented attributes.** Every `tsara_*` or `uncertainty_*` attribute a
  product carries is named in METHODS; the architecture test finds them by
  reading string constants out of the code and fails on one it cannot find in
  the document.
- **Measured claims.** A number in a docstring, a commit message or METHODS
  states the rule it was measured under and its denominator, so that it can
  be re-run. A number that cannot be re-run is not quoted. Measure before
  proposing: several technically correct fixes have turned out to be inert on
  the real campaign archive, and a measurement is what makes that
  conversation short.
- **A pure move proven as one.** Renaming, splitting or reordering code is
  done so that every function body is byte-identical afterwards, and checked
  by comparing the syntax tree of each definition against a snapshot, not by
  reading the diff.

## Vocabulary

One word has one meaning across code, attributes, documents and notebooks.
The table at the top of `docs/METHODS.md` is the authority. Two to know
before writing anything: a *source* is an emission source and nothing else
(the input side of a join is the *readings*; where a number came from is its
*provenance*); a *cell* is the interval of air a value describes, and *time*
is its midpoint.

## Real data

No real measurement, file path or archive-derived table enters the
repository except as a documented measurement in METHODS. Tests and notebooks
that read the campaign archive are gated on `TSARA_ARCHIVE` or
`TSARA_REAL_DATA` and skip without them; the real-data notebook is committed
without outputs. The synthetic generator exists so that every other test has
ground truth to score against.

## Notebooks

The walkthrough notebooks under `examples/notebooks/` are deliverables,
committed with their outputs so they read on GitHub. They are linted and
formatted by ruff like any source, and executed by the opt-in test above.
After editing one, re-execute it end to end, look at every figure, and keep
scratch paths and per-cell timings out of the committed outputs.

## Proposing a change

One feature branch per phase, one pull request per branch, and a walkthrough
of the phase before the pull request opens, because a transformation of data
comes back looking reasonable when it is wrong and the reviewer has to
understand the operation rather than trust that it ran. Commit messages say
what was measured and what the evidence was.
