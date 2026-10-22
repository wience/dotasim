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


class ExplodingPolicy(RandomPolicy):
    """Raises once, on its third decision, the way a buggy model would."""

    def __init__(self):
        super().__init__(seed=2)
        self.calls = 0

    def act(self, obs):
        self.calls += 1
        if self.calls == 3:
            raise ValueError("boom")
        return super().act(obs)


def test_one_policy_exception_does_not_kill_the_fleet(tmp_path):
    fleet = Fleet(2, Matchup(heroes={0: SF, 5: SF}, roles={0: "controlled", 5: "valve"}, cap_s=120), ExplodingPolicy(),
                  tmp_path, dota_dir=dota_dir(), stagger_s=0.0)
    totals = asyncio.run(asyncio.wait_for(fleet.run(episodes_per_instance=1, max_steps=100), 600))
    assert totals["episodes"] == 1, totals
    assert len(totals["errors"]) == 1 and "boom" in totals["errors"][0], totals
    status = json.loads((tmp_path / "status.json").read_text())
    phases = sorted(s["phase"] for s in status["instances"])
    assert phases == ["dead", "idle"], phases  # the survivor stopped at max_steps and closed normally


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
