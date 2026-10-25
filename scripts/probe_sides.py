"""Mirror games with one fixed checkpoint on both sides: who wins, by side? 50/50 means no protocol bias.
Usage: python scripts/probe_sides.py [checkpoint] [games] [instances] [cap_s]"""
import asyncio, os, sys, time
from collections import Counter
from pathlib import Path
from dotasim.env.env import Matchup
from dotasim.train.policy import NetPolicy
from dotasim.train.syncfleet import SyncFleet

SF = "npc_dota_hero_nevermore"
DOTA = Path(os.environ.get("DOTASIM_DOTA_DIR", "~/dota2")).expanduser()
ck = sys.argv[1] if len(sys.argv) > 1 else "runs/bc_v2/policy.pt"
games = int(sys.argv[2]) if len(sys.argv) > 2 else 24
n = int(sys.argv[3]) if len(sys.argv) > 3 else 12
cap = float(sys.argv[4]) if len(sys.argv) > 4 else 300

async def main():
    m = Matchup(heroes={0: SF, 5: SF}, roles={0: "controlled", 5: "controlled"}, frames=5, cap_s=cap)
    pol = NetPolicy(ck, greedy=False)
    f = SyncFleet(n, lambda i: m, lambda i, t: pol, Path("runs/probe_sides"), dota_dir=DOTA, tv=False)
    wins = Counter(); ends = Counter(); t0 = time.time()
    def on_step(i, acted, obs, done, info):
        if done:
            wins[{2: "Radiant", 3: "Dire", None: "draw"}.get(info.get("winner"))] += 1
            ends[info.get("end")] += 1
    r = await f.run(games, on_step=on_step)
    print({"checkpoint": ck, "games": r["episodes"], "winners": dict(wins), "ends": dict(ends), "steps": r["steps"],
           "lua_timeouts": r["lua_timeouts"], "wall_s": round(time.time() - t0)})

asyncio.run(main())
