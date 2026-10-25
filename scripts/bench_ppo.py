"""Time one PPO update on synthetic windows shaped like a real iteration (default 1400 windows of 32 steps).
Usage: python scripts/bench_ppo.py [cpu|cuda] [n_windows]"""
import sys, time
import numpy as np, torch
from dotasim.model.features import HERO_DIM, K, N_TYPES, UNIT_DIM
from dotasim.model.net import Policy
from dotasim.train.buffer import Window
from dotasim.train.ppo import PPOConfig, ppo_update

dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 1400
W = 32
rng = np.random.default_rng(0)
wins = []
for i in range(n):
    umask = rng.random((W, K)) < 0.6; umask[:, 0] = True
    tmask = np.ones((W, N_TYPES), bool); tmask[:, 3:] = rng.random((W, N_TYPES - 3)) < 0.3
    a_type = rng.integers(0, 3, W); a_target = np.array([rng.choice(np.nonzero(m)[0]) for m in umask])
    ten = {"units": torch.from_numpy(rng.standard_normal((W, K, UNIT_DIM)).astype(np.float32)),
           "unit_mask": torch.from_numpy(umask), "hero": torch.from_numpy(rng.standard_normal((W, HERO_DIM)).astype(np.float32)),
           "type_mask": torch.from_numpy(tmask), "logp": torch.from_numpy((-rng.random(W) * 3).astype(np.float32)),
           "adv": torch.from_numpy(rng.standard_normal(W).astype(np.float32)), "ret": torch.from_numpy(rng.standard_normal(W).astype(np.float32)),
           "valid": torch.ones(W, dtype=torch.bool), "a_type": torch.from_numpy(a_type), "a_dir": torch.from_numpy(rng.integers(0, 16, W)),
           "a_dist": torch.from_numpy(rng.integers(0, 4, W)), "a_target": torch.from_numpy(a_target)}
    wins.append(Window(i % 12, 2, 0, 0, ten, torch.zeros(256)))
torch.set_num_threads(8)
net, anchor = Policy().to(dev), Policy().to(dev)
anchor.load_state_dict(net.state_dict())
opt = torch.optim.Adam(net.parameters(), lr=1e-4)
cfg = PPOConfig()
t = time.time()
stats = ppo_update(net, anchor, wins, cfg, opt, version=0)
if dev != "cpu":
    torch.cuda.synchronize()
print({"device": dev, "windows": n, "epochs": cfg.epochs, "update_s": round(time.time() - t, 1), "loss_pi": round(stats["loss_pi"], 4)})
