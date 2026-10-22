"""Diagnose a cloning checkpoint. offline: greedy predictions vs labels over a recorded episode, per 300-step bucket.
live: one short game, logging action kinds and the hero's position every 50 steps.
usage: python scripts/diag_bc.py offline runs/bc_v1/policy_final.pt | live runs/bc_v1/policy_final.pt [greedy|sample] [cap]
"""
import asyncio
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from dotasim.model.features import featurize
from dotasim.model.labels import LOOKAHEAD, label
from dotasim.model.net import masked_log_probs
from dotasim.train.policy import NetPolicy

NAMES = ["none", "move", "attack", "r1", "r2", "r3", "ult"]


def offline(ckpt: str):
    from dotasim.data.record import load_episode
    steps = load_episode(sorted(Path("data/sf_mirror_v1").glob("ep*_t2.npz"))[0])
    pol = NetPolicy(ckpt, greedy=True)
    h = pol.net.initial_state(1)
    rows = []
    for i in range(len(steps) - 1):
        f = featurize(steps[i], 2)
        y = label(steps[i], steps[i + 1], f, 2, steps[min(i + LOOKAHEAD, len(steps) - 1)])
        with torch.inference_mode():
            logits, h = pol.net(torch.from_numpy(f.units[None]), torch.from_numpy(f.unit_mask[None]), torch.from_numpy(f.hero[None]), h)
            lp = masked_log_probs(logits, torch.from_numpy(f.type_mask[None]))
        rows.append((steps[i].dota_time, y.action_type, int(lp["type"].argmax()), float(lp["type"][0, 1].exp())))
    for b in range(0, min(len(rows), 2400), 300):
        chunk = rows[b:b + 300]
        yl = Counter(NAMES[r[1]] for r in chunk)
        yp = Counter(NAMES[r[2]] for r in chunk)
        pm = np.mean([r[3] for r in chunk])
        print(f"t={chunk[0][0]:6.1f}..{chunk[-1][0]:6.1f}  labels={dict(yl)}  greedy={dict(yp)}  mean p(move)={pm:.2f}")


async def live(ckpt: str, mode: str, cap: float):
    from dotasim.env.channel import install_scripts
    from dotasim.env.env import DotaEnv, Matchup
    from dotasim.env.instance import InstanceConfig
    import os
    dota = Path(os.environ.get("DOTASIM_DOTA_DIR", "~/dota2")).expanduser()
    env = DotaEnv(InstanceConfig(index=0, dota_dir=dota), install_scripts(dota))
    pol = NetPolicy(ckpt, greedy=(mode == "greedy"))
    SF = "npc_dota_hero_nevermore"
    obs = await env.reset(Matchup(heroes={0: SF, 5: SF}, roles={0: "controlled", 5: "valve"}, frames=pol.frames, difficulty=0, cap_s=cap))
    kinds, n = Counter(), 0
    try:
        done = False
        while not done:
            acts = pol.act(obs)
            kinds[acts[2].kind] += 1
            obs, done, info = await env.step(acts)
            n += 1
            if n % 150 == 0:
                me = obs[2].hero_of(2)
                print(f"step {n} t={obs[2].dota_time:6.1f} pos=({me.x:.0f},{me.y:.0f}) lvl={me.level} lh={me.last_hits} kinds={dict(kinds)} mismatch={info['lua_mismatch']}")
        print("final:", info)
    finally:
        await env.close()


if __name__ == "__main__":
    if sys.argv[1] == "offline":
        offline(sys.argv[2])
    else:
        asyncio.run(live(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "greedy", float(sys.argv[4]) if len(sys.argv) > 4 else 150))
