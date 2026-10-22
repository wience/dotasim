import asyncio
import json

import numpy as np
import pytest

from dotasim.data.record import load_episode, record
from tests.conftest import dota_dir

pytestmark = [pytest.mark.game, pytest.mark.slow]


def test_record_two_games_both_perspectives(tmp_path):
    out = tmp_path / "ds"
    summary = asyncio.run(asyncio.wait_for(
        record(games=2, instances=2, frames=5, difficulty=4, cap_s=120, out=out, dota_dir=dota_dir()), 900))
    files = sorted(out.glob("ep*_t*.npz"))
    assert len(files) == 4, files
    meta = json.loads((out / "meta.json").read_text())
    assert meta["frames"] == 5 and meta["difficulty"] == 4
    for f in files:
        steps = load_episode(f)
        assert len(steps) > 300, (f, len(steps))
        dt = np.diff([ws.dota_time for ws in steps])
        assert np.mean(np.abs(dt - 5 / 30) < 0.005) > 0.99, (f, dt[:10])  # 5 frames = 5 ticks at 30/s
        team = int(f.stem[-1])
        assert steps[-1].hero_of(team).last_hits >= 0
    assert summary["episodes"] == 2
