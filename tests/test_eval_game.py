import asyncio
import json
from pathlib import Path

import pytest

from dotasim.eval.eval import evaluate
from tests.conftest import dota_dir

pytestmark = [pytest.mark.game, pytest.mark.slow]
CKPT = Path(__file__).resolve().parents[1] / "runs" / "bc_v1" / "policy.pt"


@pytest.mark.skipif(not CKPT.exists(), reason="no bc_v1 checkpoint")
def test_bc_agent_lanes_against_passive_bots(tmp_path):
    res = asyncio.run(asyncio.wait_for(
        evaluate(checkpoint=CKPT, games=3, instances=3, difficulty=0, cap_s=300, out=tmp_path / "eval",
                 dota_dir=dota_dir()), 1500))
    assert len(res["games"]) == 3, res
    for g in res["games"]:
        assert g["lua_timeouts"] == 0, g
        assert g["ended"] in ("cap", "post_game"), g
    assert res["median"]["last_hits"] >= 5, res["median"]
    saved = json.loads((tmp_path / "eval" / "eval.json").read_text())
    assert saved["median"] == res["median"]


def test_random_baseline_runs(tmp_path):
    res = asyncio.run(asyncio.wait_for(
        evaluate(checkpoint=None, games=1, instances=1, difficulty=0, cap_s=120, out=tmp_path / "rnd",
                 dota_dir=dota_dir()), 600))
    assert len(res["games"]) == 1 and res["games"][0]["lua_timeouts"] == 0
    assert "last_hits" in res["median"] and "valve_last_hits" in res["games"][0]
