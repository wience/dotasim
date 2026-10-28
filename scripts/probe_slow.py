"""Does a slow policy make the game end in pre-game? Reproduces the "cold wave" (sp_v14: 278 of 684 episodes ended
at dota_time < 0 with the server saying 'Terminating after game'). Runs N servers with a policy that sleeps per
decision; prints each episode's end reason and game time.
Usage: python scripts/probe_slow.py [sleep_s=0.4] [instances=2] [cap_s=150]"""
import asyncio, os, sys, time
from pathlib import Path
from dotasim.env.env import Matchup
from dotasim.train.policy import NetPolicy
from dotasim.train.syncfleet import SyncFleet

SF = "npc_dota_hero_nevermore"
DOTA = Path(os.environ.get("DOTASIM_DOTA_DIR", "~/dota2")).expanduser()
SLEEP = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2
CAP = float(sys.argv[3]) if len(sys.argv) > 3 else 150


class Slow:
    """The fleet calls act_batch; everything else is delegated to the real policy."""
    def __init__(self, inner):
        self.inner = inner

    def act_batch(self, items):
        time.sleep(SLEEP)
        return self.inner.act_batch(items)

    def __getattr__(self, name):
        return getattr(self.inner, name)


async def main():
    m = Matchup(heroes={0: SF, 5: SF}, roles={0: "controlled", 5: "controlled"}, frames=5, cap_s=CAP)
    pol = Slow(NetPolicy("runs/bc_v2/policy.pt", greedy=False))
    out = Path("runs/probe_slow")
    f = SyncFleet(N, lambda i: m, lambda i, t: pol, out, dota_dir=DOTA, tv=False)
    t0 = time.time()

    def on_step(i, acted, obs, done, info):
        if done:
            ws = f.envs[i].last_ws
            print(f"env {i} done wall {time.time() - t0:.0f}s dota_time {ws.dota_time if ws else None} end {info.get('end')} "
                  f"post_game {info.get('post_game')} error {info.get('error')}", flush=True)
    r = await f.run(1, on_step=on_step)
    print({k: r[k] for k in ("episodes", "steps", "lua_timeouts", "ends")}, "wall", round(time.time() - t0, 1))
    print("full pre-game logs:", sorted(p.name for p in (out / "end_logs").glob("pregame_*")))

asyncio.run(main())
