"""The fast examples run top to bottom (quickstart.py runs in the install smoke job; the VMEC, Knyazev and benchmark ones are too slow)."""

import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize("script", ["ctok_boozmn", "koenies_tae_drive"])
def test_fast_examples_run(script, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(Path(__file__).parents[1] / "examples" / f"{script}.py"))
    assert script != "koenies_tae_drive" or (tmp_path / "koenies_tae_drive.png").stat().st_size > 0
