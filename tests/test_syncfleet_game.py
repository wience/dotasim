import asyncio
import json
import os
from pathlib import Path

import pytest

from dotasim.env.env import Matchup
from dotasim.train.policy import NetPolicy
from dotasim.train.syncfleet import SyncFleet
from tests.conftest import dota_dir

SF = "npc_dota_hero_nevermore"
CKPT = Path(__file__).resolve().parents[1] / "runs" / "bc_v1" / "policy.pt"
pytestmark = [pytest.mark.game, pytest.mark.slow]


@pytest.mark.skipif(not CKPT.exists(), reason="no bc_v1 checkpoint")
def test_twelve_selfplay_servers_batched(tmp_path):
    live = NetPolicy(CKPT, greedy=False)
    matchup = Matchup(heroes={0: SF, 5: SF}, roles={0: "controlled", 5: "controlled"}, frames=5, cap_s=120)
    fleet = SyncFleet(12, lambda i: matchup, lambda i, team: live, tmp_path, dota_dir=dota_dir())
    totals = asyncio.run(asyncio.wait_for(fleet.run(episodes_total=12), 1500))
    assert totals["episodes"] == 12, totals
    # one env finishing (post-game detection waits up to 3 s) stalls the others' Lua waits once; allow 0.1%
    assert totals["lua_timeouts"] <= max(1, totals["steps"] // 1000), totals
    print(totals)
    assert totals["speed_x_mean"] >= 3, totals   # measured 3.2x with 12 servers; python featurize is the bottleneck (queued)
    assert totals["act_ms_mean"] < 25, totals     # measured 18.6 ms per batched half-step incl. featurize
    status = json.loads((tmp_path / "status.json").read_text())
    assert len(status["instances"]) == 12
    replays = Path(os.path.expanduser("~/dota2/game/dota/replays"))
    assert len(list(replays.glob("*.dem"))) <= 30
