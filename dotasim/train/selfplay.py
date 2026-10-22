"""Self-play PPO loop: league matchups on a synchronized fleet, one PPO update per iteration, periodic eval."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from dotasim.env.env import Matchup
from dotasim.eval.eval import evaluate
from dotasim.model.net import Policy
from dotasim.train.buffer import Buffer, Step
from dotasim.train.league import League
from dotasim.train.policy import NetPolicy
from dotasim.train.ppo import PPOConfig, ppo_update
from dotasim.train.reward import step_reward, terminal_reward, winner
from dotasim.train.syncfleet import SyncFleet

SF = "npc_dota_hero_nevermore"


@dataclass
class SelfPlayConfig:
    init: Path
    out: Path
    instances: int = 12
    iterations: int = 150
    cap_s: float = 600
    frames: int = 5
    difficulty: int = 4
    eval_every: int = 10
    eval_games: int = 6
    pool_every: int = 5
    anchor_every: int = 20
    timescale: float = 10
    dota_dir: Path | None = None
    ppo: PPOConfig = field(default_factory=PPOConfig)


def _save(net: Policy, path: Path, frames: int, extra: dict) -> None:
    torch.save({"state_dict": net.state_dict(), "frames": frames, "hero_pool": [SF], "config": extra}, path)


async def run_iteration(cfg: SelfPlayConfig, live: NetPolicy, league: League, version: int, it: int) -> tuple[Buffer, dict, dict]:
    kinds: dict[int, str] = {}
    opponents: dict[int, NetPolicy | None] = {}
    for i in range(cfg.instances):
        kinds[i], opponents[i] = league.sample()

    def matchup_for_env(i: int) -> Matchup:
        roles = {0: "controlled", 5: "valve" if kinds[i] == "valve" else "controlled"}
        return Matchup(heroes={0: SF, 5: SF}, roles=roles, frames=cfg.frames, difficulty=cfg.difficulty, cap_s=cfg.cap_s)

    def policy_for(i: int, team: int):
        if team == 2 or kinds[i] == "self":
            return live
        return opponents[i]   # pool opponent, or None for Valve

    buf = Buffer()
    prev: dict[tuple[int, int], object] = {}
    outcomes = {"self": [0, 0, 0], "pool": [0, 0, 0], "valve": [0, 0, 0]}   # win, draw, loss for team 2

    def on_step(i, team, ws, detail, done, info):
        if policy_for(i, team) is not live or detail is None:
            return
        key = (i, team)
        r = step_reward(prev[key], ws, team) if key in prev else 0.0
        prev[key] = ws
        f = detail["feats"]
        buf.add(i, team, Step(f.units, f.unit_mask, f.hero, f.type_mask, detail["a"], detail["logp"], detail["value"],
                              detail["h0"], version, r))
        if done:
            err = "error" in info
            w = winner(ws) if info.get("post_game") else None
            term = terminal_reward(team, w) if w is not None else 0.0
            buf.finish(i, team, term, err, last_value=0.0 if (w is not None or err) else detail["value"])
            prev.pop(key, None)
            if team == 2 and not err:
                outcomes[kinds[i]][0 if w == 2 else (1 if w is None else 2)] += 1

    fleet = SyncFleet(cfg.instances, matchup_for_env, policy_for, cfg.out / "run", dota_dir=cfg.dota_dir, timescale=cfg.timescale)
    totals = await fleet.run(episodes_total=cfg.instances, on_step=on_step)
    return buf, totals, outcomes


async def train_selfplay(cfg: SelfPlayConfig) -> dict:
    cfg.out = Path(cfg.out)
    cfg.out.mkdir(parents=True, exist_ok=True)
    live = NetPolicy(cfg.init, greedy=False)
    anchor = Policy()
    anchor.load_state_dict(live.net.state_dict())
    league = League(cfg.out / "pool", cfg.frames)
    opt = torch.optim.Adam(live.net.parameters(), lr=cfg.ppo.lr)
    writer = SummaryWriter(str(cfg.out / "tb"))
    log = open(cfg.out / "log.jsonl", "a")
    version = 0
    t_start = time.time()
    last_eval: dict = {}
    for it in range(cfg.iterations):
        t0 = time.time()
        buf, totals, outcomes = await run_iteration(cfg, live, league, version, it)
        windows = buf.windows(cfg.ppo.gamma, cfg.ppo.lam)
        stats = ppo_update(live.net, anchor, windows, cfg.ppo, opt, version=version) if windows else {}
        version += 1
        _save(live.net, cfg.out / "policy.pt", cfg.frames, {"iteration": it + 1, "version": version})
        if cfg.pool_every and (it + 1) % cfg.pool_every == 0:
            league.add(cfg.out / "policy.pt", f"it{it + 1:04d}")
        if cfg.anchor_every and (it + 1) % cfg.anchor_every == 0:
            anchor.load_state_dict(live.net.state_dict())
        line = {"iteration": it + 1, "version": version, "steps": buf.steps(), "episodes": totals["episodes"],
                "lua_timeouts": totals["lua_timeouts"], "errors": len(totals["errors"]), "speed_x": totals["speed_x_mean"],
                "act_ms": totals["act_ms_mean"], "outcomes": outcomes, "wall_s": round(time.time() - t0, 1), **stats}
        for k in ("loss_pi", "loss_v", "entropy", "kl_anchor", "clipfrac", "approx_kl"):
            if k in stats:
                writer.add_scalar(f"train/{k}", stats[k], it + 1)
        writer.add_scalar("train/steps", line["steps"], it + 1)
        writer.add_scalar("train/speed_x", line["speed_x"], it + 1)
        wins = sum(o[0] for o in outcomes.values()); games = sum(sum(o) for o in outcomes.values())
        writer.add_scalar("train/win_rate_team2", wins / max(games, 1), it + 1)
        if cfg.eval_every and (it + 1) % cfg.eval_every == 0:
            last_eval = {}
            for name, diff in (("unfair", 4), ("passive", 0)):
                res = await evaluate(cfg.out / "policy.pt", cfg.eval_games, min(cfg.eval_games, cfg.instances), diff,
                                     min(cfg.cap_s, 600), cfg.out / f"eval_it{it + 1:04d}_{name}", dota_dir=cfg.dota_dir, greedy=True)
                last_eval[f"win_rate_{name}"] = res["median"].get("win_rate", 0.0)
                last_eval[f"last_hits_{name}"] = res["median"]["last_hits"]
                writer.add_scalar(f"eval/win_rate_{name}", last_eval[f"win_rate_{name}"], it + 1)
                writer.add_scalar(f"eval/last_hits_{name}", last_eval[f"last_hits_{name}"], it + 1)
            line["eval"] = last_eval
        log.write(json.dumps(line) + "\n")
        log.flush()
        print(json.dumps(line), flush=True)
    log.close()
    writer.close()
    return {"iterations": cfg.iterations, "version": version, "wall_s": round(time.time() - t_start, 1), "last_eval": last_eval}


def main(argv=None):
    p = argparse.ArgumentParser(description="Self-play PPO for the SF mirror.")
    p.add_argument("--init", required=True)
    p.add_argument("--out", default="runs/sp_v1")
    p.add_argument("--instances", type=int, default=12)
    p.add_argument("--iterations", type=int, default=150)
    p.add_argument("--cap", type=float, default=600)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-games", type=int, default=6)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    a = p.parse_args(argv)
    cfg = SelfPlayConfig(init=Path(a.init), out=Path(a.out), instances=a.instances, iterations=a.iterations, cap_s=a.cap,
                         eval_every=a.eval_every, eval_games=a.eval_games, ppo=PPOConfig(lr=a.lr, epochs=a.epochs))
    print(json.dumps(asyncio.run(train_selfplay(cfg)), indent=1))


if __name__ == "__main__":
    main()
