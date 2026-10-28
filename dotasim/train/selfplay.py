"""Self-play PPO loop: league matchups on a synchronized fleet, one PPO update per iteration, periodic eval."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
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
MIN_STEPS_PER_ENV = 1000   # a healthy 300 s wave gives about 3,000 per env; a cold-WSL wave dies in pre-game with a few hundred
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
    device: str = "cpu"   # learner device ("cuda" also names ROCm); rollouts always run the CPU copy
    learner_python: str | None = None   # run each update in this interpreter (the ROCm venv) as a separate process:
                                         # ROCm's runtime busy-waits two cores even idle, which starves the servers
    p_pool: float = 0.2    # share of waves vs a past checkpoint; the rest after p_valve is self-play
    p_valve: float = 0.0   # share of waves vs Valve bots at `difficulty` (stage 1 of the curriculum: 1.0 at difficulty 0)
    overlap: bool = False  # run the GPU update while the next wave plays (one update of policy lag; needs learner_python)


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
        self.outcomes = {"self": [0, 0, 0], "pool": [0, 0, 0], "valve": [0, 0, 0]}   # win, draw, loss: live side (team 2 for self)
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
        w = info.get("winner") or (winner(newest) if (newest is not None and info.get("post_game")) else None)
        capped = bool(info.get("capped")) and not err
        for team in live_teams:
            if (i, team) in self.pending and newest is not None:
                self._emit(i, team, newest)
            last_value = self.live.value_of(i, team, newest) if (capped and newest is not None) else 0.0
            self.buf.finish(i, team, terminal_reward(team, w), err, last_value, capped=capped)
        if not err:
            view = 2 if self.kinds[i] == "self" else live_teams[0]
            self.outcomes[self.kinds[i]][0 if w == view else (1 if w is None else 2)] += 1
        for key in [k for k in self.last_ws if k[0] == i]:
            del self.last_ws[key]
        for key in [k for k in self.pending if k[0] == i]:
            del self.pending[key]


async def run_iteration(cfg: SelfPlayConfig, live: NetPolicy, league: League, version: int, it: int,
                        trace: list | None = None, instances: int | None = None):
    n = instances or cfg.instances
    # One matchup kind per wave: bots/config.lua is shared by every server, so mixing kinds in a wave gives the
    # later-launched servers the wrong roles (a team controlled in Lua that Python never feeds stalls its game).
    kind, opponent = league.sample()
    kinds: dict[int, str] = {i: kind for i in range(n)}
    opponents: dict[int, NetPolicy | None] = {i: opponent for i in range(n)}

    def live_team(i: int) -> int:
        # Dire wins most mirrors on this map, so the live side alternates: per env against pool checkpoints, per wave
        # against Valve (the Valve role lives in the shared config.lua, so it cannot differ within a wave).
        return 2 if (it if kind == "valve" else i) % 2 == 0 else 3

    def matchup_for_env(i: int) -> Matchup:
        roles = {0: "controlled", 5: "controlled"}
        if kinds[i] == "valve":
            roles[5 if live_team(i) == 2 else 0] = "valve"
        return Matchup(heroes={0: SF, 5: SF}, roles=roles, frames=cfg.frames, difficulty=cfg.difficulty, cap_s=cfg.cap_s)

    def policy_for(i: int, team: int):
        if kinds[i] == "self" or team == live_team(i):
            return live
        return opponents[i]   # pool opponent, or None for Valve

    col = _Collector(live, policy_for, kinds, cfg.ppo.gamma, trace)
    col.version = version
    fleet = SyncFleet(n, matchup_for_env, policy_for, cfg.out / "run", dota_dir=cfg.dota_dir, timescale=cfg.timescale,
                      tv=False)   # replays only for evaluation games
    opp_name = getattr(opponent, "name", None) or ("valve unfair" if kind == "valve" else ("self (same network)" if kind == "self" else "pool"))
    fleet.labels = {i: {"kind": kinds[i], "live_side": "both" if kinds[i] == "self" else live_team(i), "opponent": opp_name} for i in range(n)}
    fleet.meta = {"iteration": it + 1, "version": version, "kind": kind}
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


def _launch_learner(cfg: SelfPlayConfig, net: Policy, anchor: Policy, windows: list, version: int,
                    opt_state: dict | None) -> dict:
    """Start ppo_update in `cfg.learner_python` (see gpu_learner.py); payload via /tmp (native disk, not /mnt/c).
    Returns a handle for _await_learner; with cfg.overlap the next wave plays while it runs (one update of lag)."""
    import subprocess
    import tempfile
    d = Path(tempfile.gettempdir())
    inp, outp = d / f"dotasim_learner_{cfg.out.name}_in.pt", d / f"dotasim_learner_{cfg.out.name}_out.pt"
    torch.save({"state_dict": net.state_dict(), "anchor": anchor.state_dict(), "opt_state": opt_state,
                "windows": windows, "cfg": cfg.ppo, "version": version, "device": cfg.device}, inp)
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    proc = subprocess.Popen([cfg.learner_python, "-m", "dotasim.train.gpu_learner", str(inp), str(outp)], env=env)
    return {"proc": proc, "out": outp, "t0": time.time()}


async def _await_learner(h: dict, net: Policy) -> tuple[dict, dict]:
    while h["proc"].poll() is None:
        await asyncio.sleep(0.5)
    if h["proc"].returncode:
        raise RuntimeError(f"learner exited {h['proc'].returncode}")
    res = torch.load(h["out"], map_location="cpu", weights_only=False)
    net.load_state_dict(res["state_dict"])
    res["stats"]["update_wall_s"] = round(time.time() - h["t0"], 1)
    return res["stats"], res["opt_state"]


def _write_config(cfg: SelfPlayConfig) -> None:
    """Everything the status page needs to say what kind of training this is."""
    import dataclasses
    import subprocess
    from dotasim.train import reward as Rw
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()
    except OSError:
        sha = ""
    d = {k: (str(v) if isinstance(v, Path) else v) for k, v in dataclasses.asdict(cfg).items()}
    d["reward"] = {"W_LH": Rw.W_LH, "W_DN": Rw.W_DN, "W_LVL": Rw.W_LVL, "W_NW": Rw.W_NW, "W_HP": Rw.W_HP, "gamma": Rw.GAMMA,
                   "terminal": "+1 win / -1 loss (fort destroyed), 0 at the cap", "shaping": "own progress, absolute; hp difference relative"}
    d["league"] = {"p_self": round(1 - cfg.p_pool - cfg.p_valve, 3), "p_pool": cfg.p_pool, "p_valve": cfg.p_valve, "valve_difficulty": cfg.difficulty,
                   "one_kind_per_wave": True, "live_side": "alternates vs pool and Valve"}
    d["threads"] = {"rollout": ROLLOUT_THREADS, "update": UPDATE_THREADS}
    d["git"] = sha
    d["started"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (cfg.out / "config.json").write_text(json.dumps(d, indent=1))


async def train_selfplay(cfg: SelfPlayConfig) -> dict:
    cfg.out = Path(cfg.out)
    cfg.out.mkdir(parents=True, exist_ok=True)
    live = NetPolicy(cfg.init, greedy=False, threads=ROLLOUT_THREADS)   # 8 threads next to 12 servers spin-waits and 30x slower decisions
    learner = live.net if cfg.device == "cpu" or cfg.learner_python else Policy().to(cfg.device)   # the update runs here; weights copied both ways around it
    opt_state = None   # Adam state round-trips through the learner subprocess
    anchor = Policy().to(cfg.device if not cfg.learner_python else "cpu")
    anchor.load_state_dict(live.net.state_dict())
    league = League(cfg.out / "pool", cfg.frames, p_pool=cfg.p_pool, p_valve=cfg.p_valve)
    opt = torch.optim.Adam(learner.parameters(), lr=cfg.ppo.lr)
    writer = SummaryWriter(str(cfg.out / "tb"))
    log = open(cfg.out / "log.jsonl", "a")
    _write_config(cfg)
    version = 0
    t_start = time.time()
    last_eval: dict = {}
    async def finish(it, buf, totals, outcomes, stats, t0):
        """Everything after a wave's update: checkpoint, pool, anchor, the log line, the evaluation."""
        nonlocal version, last_eval
        version += 1
        _save(live.net, cfg.out / "policy.pt", cfg.frames, {"iteration": it + 1, "version": version})
        if cfg.pool_every and (it + 1) % cfg.pool_every == 0:
            league.add(cfg.out / "policy.pt", f"it{it + 1:04d}")
        if cfg.anchor_every and (it + 1) % cfg.anchor_every == 0:
            anchor.load_state_dict(live.net.state_dict())   # strict shapes; .to(device) happened at construction
        line = {"iteration": it + 1, "version": version, "steps": buf.steps(), "trajectories": len([t for t in buf.finished if not t.error]),
                "episodes": totals["episodes"], "failed": totals.get("failed", 0), "lua_timeouts": totals["lua_timeouts"], "ends": totals.get("ends", {}),
                "errors": len(totals["errors"]), "error_samples": totals["errors"][:3], "speed_x": totals["speed_x_mean"], "act_ms": totals["act_ms_mean"],
                "outcomes": outcomes, "wall_s": round(time.time() - t0, 1), **stats}
        fin = [t for t in buf.finished if not t.error and t.steps]
        if fin:   # does the agent lane more over iterations? (shaping telescopes, so the return is about the end potential)
            line["mean_return"] = round(float(np.mean([sum(s.reward for s in t.steps) for t in fin])), 4)
            line["mean_last_hits"] = round(float(np.mean([t.steps[-1].hero[4] * 100 for t in fin])), 2)
            line["mean_denies"] = round(float(np.mean([t.steps[-1].hero[5] * 50 for t in fin])), 2)
            for side, name in ((2, "radiant"), (3, "dire")):   # the two sides see different absolute maps; a mean can hide one side idling
                ts = [t for t in fin if t.team == side]
                if ts:
                    line[f"last_hits_{name}"] = round(float(np.mean([t.steps[-1].hero[4] * 100 for t in ts])), 2)
            writer.add_scalar("train/mean_return", line["mean_return"], it + 1)
            writer.add_scalar("train/mean_last_hits", line["mean_last_hits"], it + 1)
            writer.add_scalar("train/mean_denies", line["mean_denies"], it + 1)
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
                    await wait_servers_exit()   # eval servers must not launch into the exiting rollout servers' logs
                    res = await evaluate(cfg.out / "policy.pt", cfg.eval_games, min(cfg.eval_games, cfg.instances), diff,
                                         min(cfg.cap_s, 600), cfg.out / f"eval_it{it + 1:04d}_{name}", dota_dir=cfg.dota_dir, greedy=False)
                except Exception as e:  # noqa: BLE001
                    last_eval[f"error_{name}"] = str(e)
                    continue
                last_eval[f"win_rate_{name}"] = res["median"].get("win_rate", 0.0)
                last_eval[f"draw_rate_{name}"] = res["median"].get("draw_rate", 0.0)
                last_eval[f"last_hits_{name}"] = res["median"]["last_hits"]
                last_eval[f"denies_{name}"] = res["median"]["denies"]
                last_eval[f"level_{name}"] = res["median"]["level"]
                writer.add_scalar(f"eval/win_rate_{name}", last_eval[f"win_rate_{name}"], it + 1)
                writer.add_scalar(f"eval/last_hits_{name}", last_eval[f"last_hits_{name}"], it + 1)
            line["eval"] = last_eval
        log.write(json.dumps(line) + "\n")
        log.flush()
        print(json.dumps(line), flush=True)

    pending = None   # a learner process still running while the next wave plays (cfg.overlap)
    for it in range(cfg.iterations):
        t0 = time.time()
        eval_due = bool(cfg.eval_every and (it + 1) % cfg.eval_every == 0)
        try:
            await wait_servers_exit()   # launching 12 new servers while 12 old ones exit starves the decision loop
            buf, totals, outcomes = await run_iteration(cfg, live, league, version, it)
            if pending is not None:     # the previous wave's update ran during this wave; land it before deciding anything
                h, meta = pending
                pending = None
                stats, opt_state = await _await_learner(h, live.net)
                await finish(*meta[:4], stats, meta[4])   # meta = (it, buf, totals, outcomes, t0)
            if buf.steps() < MIN_STEPS_PER_ENV * cfg.instances:
                # right after a WSL restart the first waves stall on disk and the servers quit before the horn:
                # a few thousand pre-game steps are not worth an update (sp_v5 and sp_v9 first launches)
                line = {"iteration": it + 1, "skipped": "cold wave", "steps": buf.steps(), "ends": totals.get("ends", {}),
                        "wall_s": round(time.time() - t0, 1)}
                log.write(json.dumps(line) + "\n"); log.flush(); print(json.dumps(line), flush=True)
                continue
            windows = buf.windows(cfg.ppo.gamma, cfg.ppo.lam)
            torch.set_num_threads(UPDATE_THREADS)
            if cfg.learner_python and windows:
                h = _launch_learner(cfg, live.net, anchor, windows, version, opt_state)
                if cfg.overlap and not eval_due and it + 1 < cfg.iterations:
                    pending = (h, (it, buf, totals, outcomes, t0))   # next wave plays on the current weights meanwhile
                    torch.set_num_threads(ROLLOUT_THREADS)
                    continue
                stats, opt_state = await _await_learner(h, live.net)
            else:
                if learner is not live.net:
                    learner.load_state_dict(live.net.state_dict())
                stats = ppo_update(learner, anchor, windows, cfg.ppo, opt, version=version) if windows else {}
                if learner is not live.net:
                    live.net.load_state_dict({k: v.detach().cpu() for k, v in learner.state_dict().items()})
            torch.set_num_threads(ROLLOUT_THREADS)
        except Exception as e:  # noqa: BLE001 - one bad iteration must not end an unattended run
            pending = None
            line = {"iteration": it + 1, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:],
                    "wall_s": round(time.time() - t0, 1)}
            log.write(json.dumps(line) + "\n"); log.flush()
            print(json.dumps({k: v for k, v in line.items() if k != "traceback"}), flush=True)
            continue
        await finish(it, buf, totals, outcomes, stats, t0)
    if pending is not None:   # the run ended on an overlapped update
        h, meta = pending
        stats, opt_state = await _await_learner(h, live.net)
        await finish(*meta[:4], stats, meta[4])
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
    p.add_argument("--device", default="cpu", help="learner device: cpu, or cuda for a ROCm/CUDA GPU (rollouts stay on CPU)")
    p.add_argument("--learner-python", default=None, help="interpreter for a separate learner process, e.g. ~/dotasim-rocm-venv/bin/python")
    p.add_argument("--p-pool", type=float, default=0.2, help="share of waves vs a past checkpoint")
    p.add_argument("--p-valve", type=float, default=0.0, help="share of waves vs Valve bots (1.0 = curriculum stage 1)")
    p.add_argument("--difficulty", type=int, default=4, help="Valve bot difficulty for Valve waves: 0 passive ... 4 unfair")
    p.add_argument("--overlap", action="store_true", help="GPU update runs while the next wave plays (needs --learner-python)")
    a = p.parse_args(argv)
    cfg = SelfPlayConfig(init=Path(a.init), out=Path(a.out), instances=a.instances, iterations=a.iterations, cap_s=a.cap,
                         eval_every=a.eval_every, eval_games=a.eval_games, ppo=PPOConfig(lr=a.lr, epochs=a.epochs), device=a.device,
                         learner_python=(str(Path(a.learner_python).expanduser()) if a.learner_python else None),
                         p_pool=a.p_pool, p_valve=a.p_valve, difficulty=a.difficulty, overlap=a.overlap)
    print(json.dumps(asyncio.run(train_selfplay(cfg)), indent=1))


if __name__ == "__main__":
    main()
