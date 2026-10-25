"""One server, both teams controlled by a checkpoint: print why and when the episode ends."""
import asyncio, os, sys, time
from pathlib import Path
from dotasim.env.env import Matchup
from dotasim.train.policy import NetPolicy
from dotasim.train.syncfleet import SyncFleet

SF = "npc_dota_hero_nevermore"
DOTA = Path(os.environ.get("DOTASIM_DOTA_DIR", "~/dota2")).expanduser()
CKPT = sys.argv[1] if len(sys.argv) > 1 else "runs/bc_v2/policy.pt"
CAP = float(sys.argv[2]) if len(sys.argv) > 2 else 120

async def main():
    m = Matchup(heroes={0: SF, 5: SF}, roles={0: "controlled", 5: "controlled"}, frames=5, cap_s=CAP)
    pol = NetPolicy(CKPT, greedy=False)
    f = SyncFleet(1, lambda i: m, lambda i, t: pol, Path("runs/probe_end"), dota_dir=DOTA)
    t0 = time.time()
    n = [0]
    def on_step(i, acted, obs, done, info):
        env = f.envs[i]
        n[0] += 1
        ws = env.last_ws
        if n[0] % 100 == 0 or done:
            print(f"step {n[0]} wall {time.time()-t0:.0f}s dota_time {ws.dota_time if ws else None} state {ws.game_state if ws else None} "
                  f"done {done} phase {env.phase} post_game_seen {env.instance.post_game_seen} info {{k: v for k, v in info.items() if k != 'tail'}}", flush=True)
        if done:
            tail = [l for l in env.instance.console_tail_lines()] if hasattr(env.instance, "console_tail_lines") else list(env.instance._tail)
            for l in tail[-12:]:
                print("   |", l[:160])
    r = await f.run(1, on_step=on_step)
    print({k: r[k] for k in ("episodes", "steps", "lua_timeouts")}, "wall", round(time.time() - t0, 1), "errors", r.get("errors", [])[:3])

asyncio.run(main())
