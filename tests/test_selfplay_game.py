import asyncio
import json
from pathlib import Path

import pytest
import torch

from dotasim.train.league import League
from dotasim.train.ppo import PPOConfig
from dotasim.train.reward import GAMMA, step_reward
from dotasim.train.selfplay import SelfPlayConfig, run_iteration, train_selfplay
from dotasim.train.policy import NetPolicy
from tests.conftest import dota_dir

CKPT = Path(__file__).resolve().parents[1] / "runs" / "bc_v1" / "policy_final.pt"
pytestmark = [pytest.mark.game, pytest.mark.slow]


class _SelfOnlyLeague(League):
    def sample(self):
        return "self", None


@pytest.mark.skipif(not CKPT.exists(), reason="no bc_v1 checkpoint")
def test_both_teams_fill_the_buffer_with_aligned_rewards(tmp_path):
    """One self-play episode on one server: both teams' trajectories reach the buffer, each stored reward is
    gamma*phi(next) - phi(decision state) for consecutive worldstates of that team, and the episode is closed
    for both teams with the same terminal handling."""
    cfg = SelfPlayConfig(init=CKPT, instances=1, iterations=1, cap_s=90, out=tmp_path / "sp", dota_dir=dota_dir(),
                         ppo=PPOConfig(gamma=GAMMA))
    live = NetPolicy(CKPT, greedy=False)
    league = _SelfOnlyLeague(tmp_path / "pool", frames=5)
    trace: list = []
    buf, totals, outcomes = asyncio.run(asyncio.wait_for(run_iteration(cfg, live, league, version=3, it=0, trace=trace), 600))
    assert totals["episodes"] == 1 and totals["errors"] == [], totals
    teams = {(t.env, t.team) for t in buf.finished if not t.error}
    assert teams == {(0, 2), (0, 3)}, teams
    n2, n3 = (len(buf.trajectory(0, t).steps) for t in (2, 3))
    assert abs(n2 - n3) <= 2 and n2 > 300, (n2, n3)
    assert buf.steps() == n2 + n3
    for i, team, prev, nxt, r in trace:
        assert nxt.dota_time > prev.dota_time, (team, prev.dota_time, nxt.dota_time)
        assert abs(r - step_reward(prev, nxt, team, GAMMA)) < 1e-9
    # consecutive decision states of one team are consecutive worldstates of that team (no step skipped or doubled)
    t2 = [x for x in trace if x[1] == 2]
    assert all(a[3] is b[2] for a, b in zip(t2[:-1], t2[1:]))
    for t in (2, 3):
        traj = buf.trajectory(0, t)
        assert traj.capped is True and traj.terminal == 0.0 and traj.last_value != 0.0, (traj.capped, traj.terminal, traj.last_value)
    assert sum(sum(v) for v in outcomes.values()) == 1
    assert all(w.version == 3 for w in buf.windows(GAMMA, 0.95))


def test_league_refuses_other_frames(tmp_path):
    ck = tmp_path / "other.pt"
    torch.save({"state_dict": {}, "frames": 10}, ck)
    league = League(tmp_path / "pool", frames=5)
    with pytest.raises(ValueError):
        league.add(ck, "bad")
    assert league.pool == []


@pytest.mark.skipif(not CKPT.exists(), reason="no bc_v1 checkpoint")
def test_two_iterations_of_selfplay(tmp_path):
    cfg = SelfPlayConfig(init=CKPT, instances=2, iterations=2, cap_s=120, eval_every=1, eval_games=2, out=tmp_path / "sp",
                         pool_every=1, dota_dir=dota_dir())
    summary = asyncio.run(asyncio.wait_for(train_selfplay(cfg), 2400))
    lines = [json.loads(l) for l in (tmp_path / "sp" / "log.jsonl").read_text().splitlines()]
    assert len(lines) == 2, lines
    for l in lines:
        assert "error" not in l, l
        assert all(abs(l[k]) < 1e6 for k in ("loss_pi", "loss_v", "entropy", "kl_anchor")), l
        assert l["trajectories"] >= 3 and l["steps"] > 1500, l   # 2 episodes, at least 3 of 4 team trajectories
    assert (tmp_path / "sp" / "policy.pt").exists()
    assert len(list((tmp_path / "sp" / "pool").glob("*.pt"))) >= 1
    assert summary["iterations"] == 2 and "eval" in lines[-1], summary
    assert "win_rate_unfair" in lines[-1]["eval"] or "win_rate_passive" in lines[-1]["eval"]
