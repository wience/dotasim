"""Probe lockstep timeouts: (a) Fleet, 1 server, both teams controlled, random policy; (b) SyncFleet same;
(c) SyncFleet, team 2 only. Prints steps, lua timeouts, wall per case."""
import asyncio
import os
import sys
import time
from pathlib import Path

from dotasim.env.env import Matchup
from dotasim.train.policy import NetPolicy, RandomPolicy
from dotasim.train.rollout import Fleet
from dotasim.train.syncfleet import SyncFleet

SF = "npc_dota_hero_nevermore"
DOTA = Path(os.environ.get("DOTASIM_DOTA_DIR", "~/dota2")).expanduser()
CKPT = "runs/bc_v1/policy_final.pt"


class BatchRandom(RandomPolicy):
    def act_batch(self, items):
        return [self.act({t: ws})[t] for _, t, ws in items]


async def fleet_both():
    m = Matchup(heroes={0: SF, 5: SF}, roles={0: "controlled", 5: "controlled"}, frames=5, cap_s=90)
    f = Fleet(1, m, RandomPolicy(seed=1), Path("runs/probe_sync/a"), dota_dir=DOTA)
    return await f.run(1)


async def sync_case(both: bool, net: bool):
    roles = {0: "controlled", 5: "controlled" if both else "valve"}
    m = Matchup(heroes={0: SF, 5: SF}, roles=roles, frames=5, cap_s=90)
    pol = NetPolicy(CKPT, greedy=False) if net else BatchRandom(seed=1)
    f = SyncFleet(1, lambda i: m, lambda i, t: pol if (t == 2 or both) else None, Path("runs/probe_sync/b"), dota_dir=DOTA)
    return await f.run(1)


async def main():
    which = sys.argv[1:] or ["a", "b", "c", "d"]
    for w in which:
        t = time.time()
        if w == "a":
            r = await fleet_both()
        elif w == "b":
            r = await sync_case(both=True, net=False)
        elif w == "c":
            r = await sync_case(both=False, net=False)
        else:
            r = await sync_case(both=True, net=True)
        print(w, {k: r[k] for k in ("episodes", "steps", "lua_timeouts")}, "wall", round(time.time() - t, 1), flush=True)


asyncio.run(main())
