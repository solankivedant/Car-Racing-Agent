import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.world import load_map  # noqa: E402

MAPS = ROOT / "maps"


@pytest.fixture(scope="session")
def main_map():
    return load_map(MAPS / "map_main.txt")


@pytest.fixture(scope="session")
def track_map():
    return load_map(MAPS / "track_main.txt")


def load_test_map(name):
    return load_map(MAPS / "tests" / f"{name}.txt")


def all_map_paths():
    return [MAPS / "map_main.txt", MAPS / "track_main.txt", *sorted((MAPS / "tests").glob("*.txt"))]
