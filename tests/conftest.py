import os
from pathlib import Path

import pytest

GOLDEN = Path(__file__).parent / "golden"


def dota_dir() -> Path:
    return Path(os.environ.get("DOTASIM_DOTA_DIR", "~/dota2")).expanduser()


def has_game() -> bool:
    return (dota_dir() / "game" / "bin" / "linuxsteamrt64" / "dota2").exists()


def pytest_collection_modifyitems(config, items):
    if has_game():
        return
    skip = pytest.mark.skip(reason="no Dota 2 install at DOTASIM_DOTA_DIR")
    for item in items:
        if "game" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def golden_ws_bytes() -> bytes:
    return (GOLDEN / "worldstate_6965.bin").read_bytes()
