import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from racing.track import load_track  # noqa: E402

# only the built-in tracks: tracks painted with the editor can be as hard as the user likes
TRACK_FILES = [ROOT / "tracks" / f"{n}.track" for n in ("oval", "serpentine", "technical")]


@pytest.fixture(scope="session")
def tracks():
    return {p.stem: load_track(p) for p in TRACK_FILES}


@pytest.fixture(scope="session")
def serp(tracks):
    return tracks["serpentine"]


@pytest.fixture(scope="session")
def oval(tracks):
    return tracks["oval"]
