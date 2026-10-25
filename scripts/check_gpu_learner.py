"""End-to-end check of the separate-process learner: synthetic windows -> ROCm python subprocess -> new weights."""
import sys, time
import numpy as np, torch
from dotasim.model.features import HERO_DIM, K, N_TYPES, UNIT_DIM
from dotasim.model.net import Policy
from dotasim.train.buffer import Window
from dotasim.train.ppo import PPOConfig
from dotasim.train.selfplay import SelfPlayConfig, _update_in_subprocess
from pathlib import Path

rng = np.random.default_rng(0); W = 32
def win(i):
    umask = rng.random((W, K)) < 0.6; umask[:, 0] = True
    tmask = np.ones((W, N_TYPES), bool)
    ten = {"units": torch.from_numpy(rng.standard_normal((W, K, UNIT_DIM)).astype(np.float32)), "unit_mask": torch.from_numpy(umask),
           "hero": torch.from_numpy(rng.standard_normal((W, HERO_DIM)).astype(np.float32)), "type_mask": torch.from_numpy(tmask),
           "logp": torch.from_numpy((-rng.random(W) * 3).astype(np.float32)), "adv": torch.from_numpy(rng.standard_normal(W).astype(np.float32)),
           "ret": torch.from_numpy(rng.standard_normal(W).astype(np.float32)), "valid": torch.ones(W, dtype=torch.bool),
           "a_type": torch.from_numpy(rng.integers(0, 3, W)), "a_dir": torch.from_numpy(rng.integers(0, 16, W)),
           "a_dist": torch.from_numpy(rng.integers(0, 4, W)), "a_target": torch.from_numpy(np.array([rng.choice(np.nonzero(m)[0]) for m in umask]))}
    return Window(i % 12, 2, 7, 0, ten, torch.zeros(256))
cfg = SelfPlayConfig(init=Path("x"), out=Path("runs/check_gpu_learner"), device="cuda", learner_python=sys.argv[1], ppo=PPOConfig())
net, anchor = Policy(), Policy(); anchor.load_state_dict(net.state_dict())
before = {k: v.clone() for k, v in net.state_dict().items()}
t = time.time()
stats, opt_state = _update_in_subprocess(cfg, net, anchor, [win(i) for i in range(200)], 7, None)
stats2, opt_state = _update_in_subprocess(cfg, net, anchor, [win(i) for i in range(200)], 7, opt_state)   # second call reuses Adam state
changed = sum(float((before[k] - v).abs().sum()) for k, v in net.state_dict().items())
print({"two_updates_wall_s": round(time.time() - t, 1), "weights_changed": changed > 0, "adam_state_entries": len(opt_state["state"]),
       "stats": {k: round(v, 4) if isinstance(v, float) else v for k, v in stats2.items()}})
