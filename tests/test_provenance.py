"""``tests/data/PROVENANCE.md`` lists every fixture with its size and SHA-256: fixtures cannot drift or appear silently."""

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_matches_the_fixture_files():
    rows = re.findall(r"^\| `([^`]+)` \| (\d+) \| `([0-9a-f]{64})` \|", (ROOT / "tests/data/PROVENANCE.md").read_text(), re.M)
    files = [p for d in ("tests/data", "examples/data") for p in (ROOT / d).iterdir() if p.suffix != ".py" and p.name != "PROVENANCE.md"]
    assert sorted(r[0] for r in rows) == sorted(str(p.relative_to(ROOT)) for p in files)
    for name, size, sha in rows:
        data = (ROOT / name).read_bytes()
        assert (len(data), hashlib.sha256(data).hexdigest()) == (int(size), sha), name
