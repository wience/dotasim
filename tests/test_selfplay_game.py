import asyncio
import json
from pathlib import Path

import pytest

from dotasim.train.selfplay import SelfPlayConfig, train_selfplay
from tests.conftest import dota_dir

CKPT = Path(__file__).resolve().parents[1] / "runs" / "bc_v1" / "policy.pt"
pytestmark = [pytest.mark.game, pytest.mark.slow]


@pytest.mark.skipif(not CKPT.exists(), reason="no bc_v1 checkpoint")
def test_two_iterations_of_selfplay(tmp_path):
    cfg = SelfPlayConfig(init=CKPT, instances=2, iterations=2, cap_s=120, eval_every=1, eval_games=2, out=tmp_path / "sp",
                         pool_every=1, dota_dir=dota_dir())
    summary = asyncio.run(asyncio.wait_for(train_selfplay(cfg), 2400))
    lines = [json.loads(l) for l in (tmp_path / "sp" / "log.jsonl").read_text().splitlines()]
    assert len(lines) == 2, lines
    for l in lines:
        assert all(abs(l[k]) < 1e6 for k in ("loss_pi", "loss_v", "entropy", "kl_anchor")), l
        assert l["steps"] > 500 and l["lua_timeouts"] == 0, l
    assert (tmp_path / "sp" / "policy.pt").exists()
    assert len(list((tmp_path / "sp" / "pool").glob("*.pt"))) >= 1
    assert summary["iterations"] == 2 and "eval" in lines[-1], summary
    assert "win_rate_unfair" in lines[-1]["eval"] or "win_rate_passive" in lines[-1]["eval"]
