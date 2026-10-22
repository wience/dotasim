import asyncio
import json

import pytest

from dotasim.env.env import Matchup
from dotasim.train.policy import RandomPolicy, ValveOnlyPolicy
from dotasim.train.rollout import Fleet
from tests.conftest import dota_dir

SF = "npc_dota_hero_nevermore"
pytestmark = pytest.mark.game


@pytest.mark.slow
def test_twelve_valve_games_end_to_end(tmp_path):
    fleet = Fleet(12, Matchup(heroes={0: SF, 5: SF}, roles={0: "valve", 5: "valve"}), ValveOnlyPolicy(), tmp_path,
                  dota_dir=dota_dir())
    totals = asyncio.run(asyncio.wait_for(fleet.run(episodes_per_instance=1), 1800))
    status = json.loads((tmp_path / "status.json").read_text())
    assert len(status["instances"]) == 12
    assert all(s["phase"] in ("postgame", "idle") for s in status["instances"]), status["instances"]
    assert totals["lua_timeouts"] == 0
    assert totals["episodes"] == 12
    assert totals["games_per_hour"] >= 40, totals


def test_two_random_instances_inference_budget(tmp_path):
    fleet = Fleet(2, Matchup(heroes={0: SF, 5: SF}, roles={0: "controlled", 5: "valve"}, cap_s=120), RandomPolicy(seed=1),
                  tmp_path, dota_dir=dota_dir())
    totals = asyncio.run(asyncio.wait_for(fleet.run(episodes_per_instance=1, max_steps=200), 600))
    assert totals["steps"] >= 400, totals
    assert totals["infer_ms_p95"] < 5, totals
    assert totals["lua_timeouts"] == 0, totals
    status = json.loads((tmp_path / "status.json").read_text())
    assert len(status["instances"]) == 2
    assert (tmp_path / "ws_0.json").exists()
