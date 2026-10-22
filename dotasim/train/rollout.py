"""Fleet: N DotaEnv instances driven by one policy, with a status.json heartbeat for the UI."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from dotasim.env.channel import install_scripts
from dotasim.env.env import DotaEnv, Matchup
from dotasim.env.instance import InstanceConfig, InstanceError
from dotasim.train.policy import Policy, RandomPolicy, ValveOnlyPolicy

SF = "npc_dota_hero_nevermore"


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


class Fleet:
    def __init__(self, n: int, matchup: Matchup, policy: Policy, run_dir: Path, timescale: float = 10,
                 dota_dir: Path | None = None, stagger_s: float = 3.0):
        dota = Path(dota_dir or os.environ.get("DOTASIM_DOTA_DIR", "~/dota2")).expanduser()
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        bots = install_scripts(dota)
        self.matchup, self.policy, self.stagger_s = matchup, policy, stagger_s
        pf = getattr(policy, "frames", None)
        if pf is not None and pf != matchup.frames:
            raise ValueError(f"policy was trained at frames={pf} but the matchup uses frames={matchup.frames}")
        self.envs = [DotaEnv(InstanceConfig(index=i, dota_dir=dota, timescale=timescale, frames=matchup.frames,
                                            difficulty=matchup.difficulty, run_dir=self.run_dir), bots) for i in range(n)]
        self.infer_ms: list[float] = []
        self.on_step = None   # optional callback(env_index, obs_by_team, done, info) after every step
        self.episodes = self.steps = self.lua_timeouts = 0
        self.errors: list[str] = []
        self._t0 = 0.0

    # ---- status ----------------------------------------------------------
    def status(self) -> dict:
        wall_h = max(time.monotonic() - self._t0, 1e-6) / 3600
        return {
            "run": self.run_dir.name, "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "instances": [e.status for e in self.envs],
            "totals": {"episodes": self.episodes, "steps": self.steps, "games_per_hour": round(self.episodes / wall_h, 1),
                       "infer_ms_mean": round(statistics.fmean(self.infer_ms), 3) if self.infer_ms else 0.0,
                       "lua_timeouts": self.lua_timeouts, "errors": len(self.errors)},
        }

    async def _heartbeat(self) -> None:
        tick = 0
        while True:
            if tick % 3 == 0:
                _atomic_write(self.run_dir / "status.json", json.dumps(self.status()))
            ws = self.envs[0].last_ws
            if ws is not None:
                _atomic_write(self.run_dir / "ws_0.json", json.dumps(ws.reduced()))
            tick += 1
            await asyncio.sleep(0.3)

    # ---- driving ---------------------------------------------------------
    async def _run_env(self, i: int, episodes: int, max_steps: int | None) -> None:
        env = self.envs[i]
        await asyncio.sleep(i * self.stagger_s)
        try:
            for _ in range(episodes):
                try:
                    obs = await env.reset(self.matchup)
                except InstanceError as e:
                    self.errors.append(str(e))
                    env.last_error, env.phase = str(e), "dead"
                    return
                steps, info = 0, {}
                try:
                    while True:
                        t0 = time.perf_counter()
                        actions = self.policy.act(obs) if obs else {}
                        self.infer_ms.append((time.perf_counter() - t0) * 1000)
                        obs, done, info = await env.step(actions)
                        steps += 1
                        self.steps += 1
                        if max_steps and steps >= max_steps:
                            done = True
                        if self.on_step is not None:
                            self.on_step(i, obs, done, info)
                        if done:
                            break
                except Exception as e:  # a policy or parse bug on one instance must not take the fleet down
                    msg = f"instance {i}: {type(e).__name__}: {e}"
                    self.errors.append(msg)
                    env.last_error, env.phase = msg, "dead"
                    return
                self.episodes += 1
                self.lua_timeouts += info.get("lua_timeouts", 0)
                if "error" in info:
                    self.errors.append(info["error"])
        finally:
            await env.close()

    async def run(self, episodes_per_instance: int, max_steps: int | None = None) -> dict:
        self._t0 = time.monotonic()
        hb = asyncio.create_task(self._heartbeat())
        try:
            await asyncio.gather(*(self._run_env(i, episodes_per_instance, max_steps) for i in range(len(self.envs))))
        finally:
            hb.cancel()
            _atomic_write(self.run_dir / "status.json", json.dumps(self.status()))
        wall = time.monotonic() - self._t0
        ms = sorted(self.infer_ms)
        return {
            "episodes": self.episodes, "steps": self.steps, "wall_s": round(wall, 1),
            "games_per_hour": round(self.episodes / (wall / 3600), 1) if wall else 0.0,
            "infer_ms_mean": round(statistics.fmean(ms), 3) if ms else 0.0,
            "infer_ms_p95": round(ms[int(0.95 * (len(ms) - 1))], 3) if ms else 0.0,
            "lua_timeouts": self.lua_timeouts, "errors": self.errors,
        }


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="Run a fleet of headless Dota 2 1v1 servers.")
    p.add_argument("--instances", type=int, default=1)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--policy", choices=["random", "valve"], default="valve")
    p.add_argument("--timescale", type=float, default=10)
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--difficulty", type=int, default=4)
    p.add_argument("--cap", type=float, default=600, help="episode cap in game seconds")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--run", default=f"runs/{datetime.now():%Y%m%d-%H%M%S}")
    a = p.parse_args(argv)
    roles = {0: "controlled", 5: "valve"} if a.policy == "random" else {0: "valve", 5: "valve"}
    matchup = Matchup(heroes={0: SF, 5: SF}, roles=roles, frames=a.frames, difficulty=a.difficulty, cap_s=a.cap)
    policy = RandomPolicy() if a.policy == "random" else ValveOnlyPolicy()
    fleet = Fleet(a.instances, matchup, policy, Path(a.run), timescale=a.timescale)
    totals = asyncio.run(fleet.run(a.episodes, a.max_steps))
    print(json.dumps(totals, indent=1))
    return totals


if __name__ == "__main__":
    main()
