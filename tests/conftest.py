"""Shared fixtures for the acceptance suite.

The repository has no packaging metadata on purpose (`src/` layout, no
`pyproject.toml`), so `semg` is importable only after `src/` is on `sys.path`.
That bootstrap must happen here, before any test module imports `semg`.

`reporting._emitted` is a module global guarding "one RESULT_JSON per process".
It must be reset between tests or the second test that emits a report fails for
a reason that has nothing to do with the code under test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def cfg():
    from semg import config as config_mod

    return config_mod.load(ROOT / "configs")


@pytest.fixture(scope="session")
def results_dir(root: Path) -> Path:
    return root / "results"


@pytest.fixture(scope="session")
def manifest_dir(root: Path) -> Path:
    return root / "data" / "manifest"


@pytest.fixture(autouse=True)
def _reset_result_emitter():
    """Each test gets a fresh RESULT_JSON emitter."""
    from semg import reporting

    reporting.reset_emitter()
    yield
    reporting.reset_emitter()


def require(path: Path, what: str = "artifact") -> Path:
    """Skip (not fail) when a derived artifact is absent.

    A clean checkout has code and configs but no `data/` or `results/`; the tests
    that read produced artifacts cannot run there. Skipping keeps the structural
    part of the suite usable on a fresh clone while still asserting hard once the
    pipeline has actually run.
    """
    if not path.exists():
        pytest.skip(f"{what} not present: {path}")
    return path
