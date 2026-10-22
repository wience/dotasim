import time
from pathlib import Path

import numpy as np
import pytest
import torch

from dotasim.model.features import featurize
from dotasim.model.net import Policy, masked_log_probs, sample
from dotasim.train.buffer import Buffer, Step
from dotasim.train.ppo import PPOConfig, evaluate_windows, joint_log_prob, ppo_update
from dotasim.train.reward import GAMMA, step_reward

DATA = Path(__file__).resolve().parents[1] / "data" / "sf_mirror_v1"
CKPT = Path(__file__).resolve().parents[1] / "runs" / "bc_v1" / "policy.pt"


def _net() -> Policy:
    net = Policy()
    if CKPT.exists():
        net.load_state_dict(torch.load(CKPT, map_location="cpu")["state_dict"])
    return net


def _rollout(net, steps, team, buf, env, version, max_steps=400):
    """Teacher-free rollout over a recorded episode: the net picks actions, rewards come from the recording."""
    h = net.initial_state(1)
    for i in range(min(max_steps, len(steps) - 1)):
        f = featurize(steps[i], team)
        with torch.inference_mode():
            logits, h_next = net(torch.from_numpy(f.units[None]), torch.from_numpy(f.unit_mask[None]),
                                 torch.from_numpy(f.hero[None]), h)
            tm = torch.from_numpy(f.type_mask[None])
            a = sample(logits, tm, greedy=False)
            logp = float(joint_log_prob(masked_log_probs(logits, tm), {k: torch.tensor([v]) for k, v in a.items()})[0])
            value = float(logits["value"][0])
        buf.add(env, team, Step(f.units, f.unit_mask, f.hero, f.type_mask, a, logp, value, h[0].numpy(), version,
                                step_reward(steps[i], steps[i + 1], team, GAMMA)))
        h = h_next
    return float(value)


@pytest.mark.skipif(not DATA.exists(), reason="dataset not recorded")
def test_buffer_gae_and_ppo_update():
    from dotasim.data.record import load_episode

    torch.manual_seed(0)
    net = _net()
    anchor = Policy()
    anchor.load_state_dict(net.state_dict())
    buf = Buffer(window=32)
    files = sorted(DATA.glob("ep*_t2.npz"))[:3]
    eps = [load_episode(f) for f in files]
    last_v0 = _rollout(net, eps[0], 2, buf, env=0, version=7)
    buf.finish(0, 2, terminal_reward=1.0, error=False, last_value=0.0)          # won
    last_v1 = _rollout(net, eps[1], 2, buf, env=1, version=7)
    buf.finish(1, 2, terminal_reward=0.0, error=False, last_value=last_v1)      # capped: bootstrap
    _rollout(net, eps[2], 2, buf, env=2, version=7, max_steps=100)
    buf.finish(2, 2, terminal_reward=0.0, error=True, last_value=0.0)           # crashed: dropped

    cfg = PPOConfig(gamma=GAMMA, lam=0.95)
    wins = buf.windows(cfg.gamma, cfg.lam)
    assert all(w.version == 7 for w in wins) and all(w.env != 2 for w in wins)
    assert sum(int(w.valid.sum()) for w in wins) == 400 + 400
    assert all(torch.isfinite(w.adv).all() and torch.isfinite(w.ret).all() for w in wins)
    # capped trajectory: the last delta bootstraps from last_value, not zero
    traj = buf.trajectory(1, 2)
    last = traj.steps[-1]
    expected_adv = last.reward + cfg.gamma * last_v1 - last.value
    assert abs(float(traj.adv[-1]) - expected_adv) < 1e-5, (float(traj.adv[-1]), expected_adv)
    # ratio == 1 before any update
    batch = buf.collate(wins[:4])
    new_logp, _, _, _ = evaluate_windows(net, batch)
    ratio = torch.exp(new_logp - batch["logp"])[batch["valid"]]
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-4), (ratio.min(), ratio.max())
    with pytest.raises(ValueError):
        buf.collate(wins[:2], expect_version=8)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    stats = ppo_update(net, anchor, wins, cfg, opt)
    for k in ("loss_pi", "loss_v", "entropy", "kl_anchor", "clipfrac", "approx_kl"):
        assert np.isfinite(stats[k]), stats


@pytest.mark.slow
@pytest.mark.skipif(not DATA.exists(), reason="dataset not recorded")
def test_full_size_update_budget():
    """An iteration's update (about 86k steps, 3 epochs, minibatch 64) must fit the 3-minute budget on 8 threads."""
    torch.set_num_threads(8)
    net, anchor = Policy(), Policy()
    buf = Buffer(window=32)
    K, UD, HD = 48, 16, 24
    T = 86_000 // 32 * 32
    rng = np.random.default_rng(0)
    for i in range(T):
        buf.add(0, 2, Step(rng.standard_normal((K, UD), dtype=np.float32), np.ones(K, bool), rng.standard_normal(HD, dtype=np.float32),
                           np.ones(7, bool), {"type": 1, "dir": 3, "dist": 1, "target": 0}, -1.5, 0.0, np.zeros(256, np.float32), 1, 0.0))
    buf.finish(0, 2, 0.0, False, 0.0)
    cfg = PPOConfig(gamma=GAMMA, lam=0.95, epochs=3, minibatch=64)
    wins = buf.windows(cfg.gamma, cfg.lam)
    t = time.perf_counter()
    stats = ppo_update(net, anchor, wins, cfg, torch.optim.Adam(net.parameters(), lr=cfg.lr))
    wall = time.perf_counter() - t
    print({"update_wall_s": round(wall, 1), "windows": len(wins)})
    assert wall < 180, wall
