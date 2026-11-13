"""The fast examples run top to bottom and write their figure.

quickstart.py runs in the install smoke job; the VMEC, Knyazev and benchmark scripts are too slow for CI.
"""

import runpy
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize("script, figure", [("ctok_boozmn", "ctok_boozmn.png"), ("koenies_tae_drive", "koenies_tae_drive.png")])
def test_fast_examples_write_their_figure(script, figure, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(EXAMPLES / f"{script}.py"), run_name="example")
    assert (tmp_path / figure).stat().st_size < 60_000  # palette PNG
    assert "wrote" in capsys.readouterr().out
