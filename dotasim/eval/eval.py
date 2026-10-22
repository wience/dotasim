"""Evaluate a checkpoint (or the random baseline) as the controlled Shadow Fiend against Valve's SF."""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from dotasim.env.env import Matchup
from dotasim.train.policy import NetPolicy, RandomPolicy
from dotasim.train.reward import winner
from dotasim.train.rollout import Fleet

SF = "npc_dota_hero_nevermore"
KEYS = ("last_hits", "denies", "kills", "deaths", "level", "valve_last_hits", "lua_mismatch")


async def evaluate(checkpoint: Path | None, games: int, instances: int, difficulty: int, cap_s: float, out: Path,
                   dota_dir: Path | None = None, frames: int = 5, timescale: float = 10, greedy: bool = True) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    policy = NetPolicy(checkpoint, greedy=greedy) if checkpoint else RandomPolicy(seed=0)
    frames = getattr(policy, "frames", frames)
    matchup = Matchup(heroes={0: SF, 5: SF}, roles={0: "controlled", 5: "valve"}, frames=frames,
                      difficulty=difficulty, cap_s=cap_s)
    n = min(instances, games)
    fleet = Fleet(n, matchup, policy, out / "run", timescale=timescale, dota_dir=dota_dir)
    results: list[dict] = []
    last_obs: dict[int, dict] = {}

    def on_step(i, obs, done, info):
        if 2 in obs:
            last_obs[i] = obs
        if done:
            env = fleet.envs[i]
            ws = last_obs.get(i, {}).get(2)
            me = ws.hero_of(2) if ws else None
            foe = env.instance.lua_ally().get(5, {})
            w = winner(ws) if (ws is not None and info.get("post_game")) else None
            results.append({
                "won": w == 2, "draw": w is None,
                "instance": i, "game_time": round(info.get("game_time", 0.0), 1),
                "last_hits": me.last_hits if me else -1, "denies": me.denies if me else -1,
                "kills": me.kills if me else 0, "deaths": me.deaths if me else 0, "level": me.level if me else 0,
                "valve_last_hits": foe.get("last_hits", -1), "lua_timeouts": info.get("lua_timeouts", 0),
                "lua_mismatch": info.get("lua_mismatch", 0),
                "ended": "error" if "error" in info else ("post_game" if info.get("post_game") else "cap"),
            })

    fleet.on_step = on_step
    totals = await fleet.run(episodes_per_instance=-(-games // n))
    results = results[:games]
    median = {k: statistics.median([g[k] for g in results]) if results else -1 for k in KEYS}
    median["win_rate"] = sum(g["won"] for g in results) / len(results) if results else 0.0
    median["draw_rate"] = sum(g["draw"] for g in results) / len(results) if results else 0.0
    summary = {"checkpoint": str(checkpoint) if checkpoint else "random", "difficulty": difficulty, "cap_s": cap_s,
               "frames": frames, "games": results, "median": median, "fleet": totals}
    (out / "eval.json").write_text(json.dumps(summary, indent=1))
    w = SummaryWriter(str(out / "tb"))
    for k, v in median.items():
        w.add_scalar(f"eval/{k}_median", v, 0)
    w.close()
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a policy vs Valve bots in the SF mirror.")
    p.add_argument("--checkpoint", default=None, help="policy.pt; omit for the random baseline")
    p.add_argument("--games", type=int, default=3)
    p.add_argument("--instances", type=int, default=3)
    p.add_argument("--difficulty", type=int, default=0)
    p.add_argument("--cap", type=float, default=300)
    p.add_argument("--frames", type=int, default=5)
    p.add_argument("--sample", action="store_true", help="sample actions instead of greedy")
    p.add_argument("--out", default="runs/eval")
    a = p.parse_args(argv)
    res = asyncio.run(evaluate(Path(a.checkpoint) if a.checkpoint else None, a.games, a.instances, a.difficulty, a.cap,
                               Path(a.out), frames=a.frames, greedy=not a.sample))
    print(json.dumps({"median": res["median"], "games": res["games"]}, indent=1))


if __name__ == "__main__":
    main()
