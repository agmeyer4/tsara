"""The walkthrough notebooks still execute, and every check in them still holds.

Notebook ``04b`` is committed WITHOUT outputs, because its outputs are the
campaign archive's data. That is the right call for the repository and it has
a cost: nothing in the suite ran the notebook, so its section 9 raised on
every execution for a day and a half (from ``0b30aa8`` until Phase 4.6) and
nobody saw it. Notebook ``04`` is committed with outputs, but a re-execution
that fails half-way still leaves the old outputs in place. These tests close
that gap by executing a *copy* of each notebook the way ``jupyter nbconvert``
does, and reading the result back.

Both are opt-in, because each takes longer than the rest of the suite:

* ``04b`` runs when ``TSARA_ARCHIVE`` names the directory holding the
  archive's ``2024/`` and ``2026/`` trees (about half a minute; it is the
  notebook's own gate). Its closing cell prints a ledger of every number
  ``docs/METHODS.md`` quotes, re-measured from the files, and a count of the
  checks made from the definitions; the test requires no ledger entry to
  differ and every check to hold.
* ``04`` runs when ``TSARA_NOTEBOOKS`` is set to anything (about a minute;
  generated data only). Its scoreboard cell prints ``N of N checks
  hold``, and the test requires exactly that.

Neither test compares outputs with the committed ones: log timestamps and
section 7's machine timings change on every run by design.

Execution goes through the ``jupyter nbconvert`` command rather than the
``nbconvert`` API, for two reasons: it is exactly what the notebook builders
run, so a difference between the two could not hide here; and it keeps the
kernel in a subprocess, where a hung cell hits the timeout instead of the
test process.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterator

NOTEBOOKS = Path(__file__).resolve().parent.parent / "examples" / "notebooks"

ARCHIVE_ENV = "TSARA_ARCHIVE"
NOTEBOOKS_ENV = "TSARA_NOTEBOOKS"

#: Seconds a single cell may run. Notebook 04's Monte Carlo cells take tens of
#: seconds; nothing legitimate takes ten minutes.
CELL_TIMEOUT_S = 600

requires_archive = pytest.mark.skipif(
    not os.environ.get(ARCHIVE_ENV),
    reason=f"Set {ARCHIVE_ENV} to the directory holding the archive's 2024/ and 2026/ trees "
    "to execute notebook 04b against it.",
)
requires_opt_in = pytest.mark.skipif(
    not os.environ.get(NOTEBOOKS_ENV),
    reason=f"Set {NOTEBOOKS_ENV}=1 to execute notebook 04 (about a minute, generated data only).",
)


def execute(notebook: Path, out_dir: Path, *, env: dict[str, str]) -> dict[str, Any]:
    """Execute ``notebook`` into ``out_dir`` and return the executed notebook's JSON.

    The source notebook is never written to. ``jupyter nbconvert`` exits
    non-zero when a cell raises, and its stderr then ends with the traceback,
    which is the most useful thing a failing test can show.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        f"--ExecutePreprocessor.timeout={CELL_TIMEOUT_S}",
        "--output-dir",
        str(out_dir),
        "--output",
        notebook.name,
        str(notebook),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, check=False, env=env, cwd=NOTEBOOKS
    )
    assert result.returncode == 0, f"{notebook.name} failed to execute:\n{result.stderr[-4000:]}"
    executed = out_dir / notebook.name
    loaded: dict[str, Any] = json.loads(executed.read_text())
    return loaded


def stream_text(nb: dict[str, Any]) -> str:
    """Concatenate every stream output in cell order."""
    return "".join(
        "".join(output["text"])
        for cell in nb["cells"]
        if cell["cell_type"] == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )


def error_outputs(nb: dict[str, Any]) -> Iterator[str]:
    """Yield ``ExceptionName: message`` for every error output, in cell order."""
    for cell in nb["cells"]:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                yield f"{output.get('ename')}: {output.get('evalue')}"


def clean_env() -> dict[str, str]:
    """The environment the notebooks are executed in.

    ``TSARA_REAL_DATA`` is dropped so notebook 01's profiling cell would take
    its fallback branch if it were ever executed here, matching the committed
    outputs; ``TSARA_ARCHIVE`` is passed through only to the test that needs it.
    """
    return {k: v for k, v in os.environ.items() if k not in ("TSARA_REAL_DATA",)}


@requires_archive
def test_notebook_04b_executes_and_its_ledger_agrees(tmp_path: Path) -> None:
    nb = execute(NOTEBOOKS / "04b_alignment_real_data.ipynb", tmp_path, env=clean_env())
    errors = list(error_outputs(nb))
    assert not errors, f"error outputs: {errors}"
    text = stream_text(nb)

    # The summaries are the LAST lines the notebook prints; take the last match,
    # so an earlier line in the same shape cannot stand in for them.
    ledgers = re.findall(r"ledger: (\d+) agree, (\d+) differ, (\d+) not compared, of (\d+)", text)
    assert ledgers, "the ledger summary line was not printed"
    agree, differ, not_compared, total = (int(g) for g in ledgers[-1])
    assert differ == 0, f"{differ} ledger entries differ from docs/METHODS.md"
    assert agree + not_compared == total

    checks = re.findall(r"checks: (\d+) of (\d+) hold", text)
    assert checks, "the checks summary line was not printed"
    assert checks[-1][0] == checks[-1][1], f"checks: {checks[-1][0]} of {checks[-1][1]} hold"
    failed = [line.strip() for line in text.splitlines() if "✘" in line]
    assert not failed, f"checks that did not hold: {failed}"


@requires_opt_in
def test_notebook_04_executes_and_every_check_holds(tmp_path: Path) -> None:
    env = clean_env()
    env.pop(ARCHIVE_ENV, None)  # notebook 04 reads no real data
    nb = execute(NOTEBOOKS / "04_alignment_walkthrough.ipynb", tmp_path, env=env)
    errors = list(error_outputs(nb))
    assert not errors, f"error outputs: {errors}"
    text = stream_text(nb)

    scoreboards = re.findall(r"(\d+) of (\d+) checks hold", text)
    assert scoreboards, "the scoreboard line was not printed"
    assert scoreboards[-1][0] == scoreboards[-1][1], f"scoreboard: {scoreboards[-1]}"
    failed = [line.strip() for line in text.splitlines() if "✘" in line]
    assert not failed, f"checks that did not hold: {failed}"
