"""Self-play PPO loop: league matchups on a synchronized fleet, one PPO update per iteration, periodic eval."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from dotasim.env.env import Matchup
from dotasim.env.worldstate import WorldState
from dotasim.eval.eval import evaluate
from dotasim.model.net import Policy
from dotasim.train.buffer import Buffer, Step
from dotasim.train.league import League
from dotasim.train.policy import NetPolicy
from dotasim.train.ppo import PPOConfig, ppo_update
from dotasim.train.reward import step_reward, terminal_reward, winner
from dotasim.train.syncfleet import SyncFleet

SF = "npc_dota_hero_nevermore"
ROLLOUT_THREADS, UPDATE_THREADS = 4, 8


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


class _Collector:
    """Turns SyncFleet step callbacks into buffer steps with correctly aligned rewards.

    A decision taken on worldstate s_n (detail) is held as pending until that team's next worldstate s_n+1
    arrives; the step is then stored with reward gamma*phi(s_n+1) - phi(s_n). At episode end every live team's
    pending decision is closed against the newest worldstate of the episode, the winner is read from that
    newest state, and a capped episode bootstraps from V(newest state).
    """

    def __init__(self, live: NetPolicy, policy_for, kinds: dict[int, str], gamma: float, trace: list | None = None):
        self.live, self.policy_for, self.kinds, self.gamma = live, policy_for, kinds, gamma
        self.buf = Buffer()
        self.pending: dict[tuple[int, int], tuple[dict, WorldState]] = {}
        self.last_ws: dict[tuple[int, int], WorldState] = {}
        self.outcomes = {"self": [0, 0, 0], "pool": [0, 0, 0], "valve": [0, 0, 0]}   # win, draw, loss for team 2
        self.trace = trace
        self.version = 0

    def _emit(self, i: int, team: int, nxt: WorldState) -> None:
        detail, prev = self.pending.pop((i, team))
        r = step_reward(prev, nxt, team, self.gamma)
        f = detail["feats"]
        self.buf.add(i, team, Step(f.units, f.unit_mask, f.hero, f.type_mask, detail["a"], detail["logp"],
                                   detail["value"], detail["h0"], self.version, r))
        if self.trace is not None:
            self.trace.append((i, team, prev, nxt, r))

    def __call__(self, i: int, acted: dict, arrived: dict, done: bool, info: dict) -> None:
        live_teams = [t for t in (2, 3) if self.policy_for(i, t) is self.live]
        for team, ws in arrived.items():
            if team in live_teams and (i, team) in self.pending:
                self._emit(i, team, ws)
            self.last_ws[(i, team)] = ws
        for team, detail in acted.items():
            if team in live_teams and detail is not None:
                self.pending[(i, team)] = (detail, self.last_ws[(i, team)])
        if not done:
            return
        err = "error" in info
        newest = max((ws for (e, _), ws in self.last_ws.items() if e == i), key=lambda w: w.dota_time, default=None)
        w = winner(newest) if (newest is not None and info.get("post_game")) else None
        capped = bool(info.get("capped")) and not err
        for team in live_teams:
            if (i, team) in self.pending and newest is not None:
                self._emit(i, team, newest)
            last_value = self.live.value_of(i, team, newest) if (capped and newest is not None) else 0.0
            self.buf.finish(i, team, terminal_reward(team, w), err, last_value, capped=capped)
        if not err:
            self.outcomes[self.kinds[i]][0 if w == 2 else (1 if w is None else 2)] += 1
        for key in [k for k in self.last_ws if k[0] == i]:
            del self.last_ws[key]
        for key in [k for k in self.pending if k[0] == i]:
            del self.pending[key]


async def run_iteration(cfg: SelfPlayConfig, live: NetPolicy, league: League, version: int, it: int,
                        trace: list | None = None, instances: int | None = None):
    n = instances or cfg.instances
    kinds: dict[int, str] = {}
    opponents: dict[int, NetPolicy | None] = {}
    for i in range(n):
        kinds[i], opponents[i] = league.sample()

    def matchup_for_env(i: int) -> Matchup:
        roles = {0: "controlled", 5: "valve" if kinds[i] == "valve" else "controlled"}
        return Matchup(heroes={0: SF, 5: SF}, roles=roles, frames=cfg.frames, difficulty=cfg.difficulty, cap_s=cfg.cap_s)

    def policy_for(i: int, team: int):
        if team == 2 or kinds[i] == "self":
            return live
        return opponents[i]   # pool opponent, or None for Valve

    col = _Collector(live, policy_for, kinds, cfg.ppo.gamma, trace)
    col.version = version
    fleet = SyncFleet(n, matchup_for_env, policy_for, cfg.out / "run", dota_dir=cfg.dota_dir, timescale=cfg.timescale)
    totals = await fleet.run(episodes_total=n, on_step=col)
    return col.buf, totals, col.outcomes


async def wait_servers_exit(max_s: float = 40.0) -> int:
    """Block until no game server process is left from the previous iteration (they exit a few seconds after SIGTERM)."""
    import psutil
    t0 = time.time()
    while time.time() - t0 < max_s:
        alive = [p for p in psutil.process_iter(["cmdline"]) if any("linuxsteamrt64/dota2" in (c or "") for c in (p.info["cmdline"] or []))]
        if not alive:
            return 0
        await asyncio.sleep(1.0)
    for p in alive:
        try:
            p.kill()
        except psutil.Error:
            pass
    return len(alive)


async def train_selfplay(cfg: SelfPlayConfig) -> dict:
    cfg.out = Path(cfg.out)
    cfg.out.mkdir(parents=True, exist_ok=True)
    live = NetPolicy(cfg.init, greedy=False, threads=ROLLOUT_THREADS)   # 8 threads next to 12 servers spin-waits and 30x slower decisions
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
        try:
            await wait_servers_exit()   # launching 12 new servers while 12 old ones exit starves the decision loop
            buf, totals, outcomes = await run_iteration(cfg, live, league, version, it)
            windows = buf.windows(cfg.ppo.gamma, cfg.ppo.lam)
            torch.set_num_threads(UPDATE_THREADS)
            stats = ppo_update(live.net, anchor, windows, cfg.ppo, opt, version=version) if windows else {}
            torch.set_num_threads(ROLLOUT_THREADS)
        except Exception as e:  # noqa: BLE001 - one bad iteration must not end an unattended run
            line = {"iteration": it + 1, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:],
                    "wall_s": round(time.time() - t0, 1)}
            log.write(json.dumps(line) + "\n"); log.flush()
            print(json.dumps({k: v for k, v in line.items() if k != "traceback"}), flush=True)
            continue
        version += 1
        _save(live.net, cfg.out / "policy.pt", cfg.frames, {"iteration": it + 1, "version": version})
        if cfg.pool_every and (it + 1) % cfg.pool_every == 0:
            league.add(cfg.out / "policy.pt", f"it{it + 1:04d}")
        if cfg.anchor_every and (it + 1) % cfg.anchor_every == 0:
            anchor.load_state_dict(live.net.state_dict())
        line = {"iteration": it + 1, "version": version, "steps": buf.steps(), "trajectories": len([t for t in buf.finished if not t.error]),
                "episodes": totals["episodes"], "failed": totals.get("failed", 0), "lua_timeouts": totals["lua_timeouts"],
                "errors": len(totals["errors"]), "error_samples": totals["errors"][:3], "speed_x": totals["speed_x_mean"], "act_ms": totals["act_ms_mean"],
                "outcomes": outcomes, "wall_s": round(time.time() - t0, 1), **stats}
        for k in ("loss_pi", "loss_v", "entropy", "kl_anchor", "clipfrac", "approx_kl"):
            if k in stats:
                writer.add_scalar(f"train/{k}", stats[k], it + 1)
        writer.add_scalar("train/steps", line["steps"], it + 1)
        writer.add_scalar("train/speed_x", line["speed_x"], it + 1)
        for kind, (w_, d_, l_) in outcomes.items():
            if w_ + d_ + l_:
                writer.add_scalar(f"train/win_rate_vs_{kind}", w_ / (w_ + d_ + l_), it + 1)
                writer.add_scalar(f"train/draw_rate_vs_{kind}", d_ / (w_ + d_ + l_), it + 1)
        if cfg.eval_every and (it + 1) % cfg.eval_every == 0:
            last_eval = {}
            for name, diff in (("unfair", 4), ("passive", 0)):
                try:
                    res = await evaluate(cfg.out / "policy.pt", cfg.eval_games, min(cfg.eval_games, cfg.instances), diff,
                                         min(cfg.cap_s, 600), cfg.out / f"eval_it{it + 1:04d}_{name}", dota_dir=cfg.dota_dir, greedy=False)
                except Exception as e:  # noqa: BLE001
                    last_eval[f"error_{name}"] = str(e)
                    continue
                last_eval[f"win_rate_{name}"] = res["median"].get("win_rate", 0.0)
                last_eval[f"draw_rate_{name}"] = res["median"].get("draw_rate", 0.0)
                last_eval[f"last_hits_{name}"] = res["median"]["last_hits"]
                last_eval[f"level_{name}"] = res["median"]["level"]
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
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    a = p.parse_args(argv)
    cfg = SelfPlayConfig(init=Path(a.init), out=Path(a.out), instances=a.instances, iterations=a.iterations, cap_s=a.cap,
                         eval_every=a.eval_every, eval_games=a.eval_games, ppo=PPOConfig(lr=a.lr, epochs=a.epochs))
    print(json.dumps(asyncio.run(train_selfplay(cfg)), indent=1))


if __name__ == "__main__":
    main()
