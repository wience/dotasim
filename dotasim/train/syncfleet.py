"""SyncFleet: N servers stepped in lockstep with one batched forward pass per policy per step.

Each controlled team of each server is routed to a policy by policy_for(env_index, team); items for the
same policy object are batched together. Teams routed to None are Valve bots (role "valve" in the matchup).
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from dotasim.env.channel import Action, install_scripts
from dotasim.env.env import DotaEnv, Matchup
from dotasim.env.instance import InstanceConfig, InstanceError
from dotasim.env.worldstate import WorldState


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


class SyncFleet:
    def __init__(self, n: int, matchup_for_env: Callable[[int], Matchup], policy_for: Callable[[int, int], object | None],
                 run_dir: Path, dota_dir: Path | None = None, timescale: float = 10, stagger_s: float = 2.0):
        dota = Path(dota_dir or os.environ.get("DOTASIM_DOTA_DIR", "~/dota2")).expanduser()
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        bots = install_scripts(dota)
        self.matchup_for_env, self.policy_for, self.stagger_s = matchup_for_env, policy_for, stagger_s
        self.envs = [DotaEnv(InstanceConfig(index=i, dota_dir=dota, timescale=timescale, run_dir=self.run_dir), bots)
                     for i in range(n)]
        self.obs: dict[int, dict[int, WorldState]] = {}
        self.matchups: dict[int, Matchup] = {}
        self.resets: dict[int, asyncio.Task] = {}
        self.episodes = self.steps = self.lua_timeouts = 0
        self.errors: list[str] = []
        self.act_ms: list[float] = []
        self._t0 = 0.0

    # ---- status ----------------------------------------------------------
    def status(self) -> dict:
        wall_h = max(time.monotonic() - self._t0, 1e-6) / 3600
        return {"run": self.run_dir.name, "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "instances": [e.status for e in self.envs],
                "totals": {"episodes": self.episodes, "steps": self.steps, "games_per_hour": round(self.episodes / wall_h, 1),
                           "infer_ms_mean": round(statistics.fmean(self.act_ms), 3) if self.act_ms else 0.0,
                           "lua_timeouts": self.lua_timeouts, "errors": len(self.errors)}}

    def _write_status(self) -> None:
        _atomic_write(self.run_dir / "status.json", json.dumps(self.status()))
        ws = self.envs[0].last_ws
        if ws is not None:
            _atomic_write(self.run_dir / "ws_0.json", json.dumps(ws.reduced()))

    # ---- episodes --------------------------------------------------------
    async def _reset(self, i: int, delay: float) -> None:
        await asyncio.sleep(delay)
        env = self.envs[i]
        m = self.matchup_for_env(i)
        self.matchups[i] = m
        for team in (2, 3):
            pol = self.policy_for(i, team)
            if pol is not None and hasattr(pol, "reset_env"):
                pol.reset_env(i)
        self.obs[i] = await env.reset(m)

    def _start_reset(self, i: int, delay: float = 0.0) -> None:
        self.resets[i] = asyncio.create_task(self._reset(i, delay))

    async def run(self, episodes_total: int, on_step: Callable | None = None, max_steps: int | None = None) -> dict:
        """on_step(env_index, team, obs_ws, detail_or_None, done, info) is called per controlled team per step."""
        self._t0 = time.monotonic()
        started = 0
        for i in range(len(self.envs)):
            if started < episodes_total:
                self._start_reset(i, i * self.stagger_s)
                started += 1
        steps_in_ep = {i: 0 for i in range(len(self.envs))}
        last_hb = 0.0
        speeds: list[float] = []
        try:
            while self.episodes < episodes_total:
                # 1. adopt finished resets
                for i, task in list(self.resets.items()):
                    if task.done():
                        del self.resets[i]
                        try:
                            task.result()
                            steps_in_ep[i] = 0
                        except InstanceError as e:
                            self.errors.append(str(e))
                            self.envs[i].phase, self.envs[i].last_error = "dead", str(e)
                            self.obs.pop(i, None)
                playing = [i for i in self.obs if i not in self.resets]
                if not playing:
                    await asyncio.sleep(0.2)
                    continue
                # 2. batched decisions, grouped by policy object
                groups: dict[int, list[tuple[int, int, WorldState]]] = {}
                pol_by_id: dict[int, object] = {}
                for i in playing:
                    for team, ws in self.obs[i].items():
                        pol = self.policy_for(i, team)
                        if pol is None:
                            continue
                        groups.setdefault(id(pol), []).append((i, team, ws))
                        pol_by_id[id(pol)] = pol
                actions: dict[int, dict[int, Action]] = {i: {} for i in playing}
                details: dict[tuple[int, int], dict | None] = {}
                t0 = time.perf_counter()
                for pid, items in groups.items():
                    pol = pol_by_id[pid]
                    if hasattr(pol, "act_batch_full"):
                        res = pol.act_batch_full(items)
                        for (i, team, _), d in zip(items, res):
                            actions[i][team] = d["action"]
                            details[(i, team)] = d
                    else:
                        res = pol.act_batch(items)
                        for (i, team, _), a in zip(items, res):
                            actions[i][team] = a
                            details[(i, team)] = None
                self.act_ms.append((time.perf_counter() - t0) * 1000)
                # 3. step everyone that is playing
                results = await asyncio.gather(*(self.envs[i].step(actions[i]) for i in playing))
                for i, (obs, done, info) in zip(playing, results):
                    self.steps += 1
                    steps_in_ep[i] += 1
                    if max_steps and steps_in_ep[i] >= max_steps:
                        done = True
                    if on_step is not None:
                        for team, ws in self.obs[i].items():
                            if self.policy_for(i, team) is not None:
                                on_step(i, team, obs.get(team, ws), details.get((i, team)), done, info)
                    if done:
                        self.episodes += 1
                        self.lua_timeouts += info.get("lua_timeouts", 0)
                        if "error" in info:
                            self.errors.append(info["error"])
                        speeds.append(info.get("speed_x", 0.0))
                        del self.obs[i]
                        if started < episodes_total:
                            self._start_reset(i)
                            started += 1
                    else:
                        self.obs[i] = obs
                if time.monotonic() - last_hb > 1.0:
                    self._write_status()
                    last_hb = time.monotonic()
        finally:
            for task in self.resets.values():
                task.cancel()
            await asyncio.gather(*(e.close() for e in self.envs), return_exceptions=True)
            self._write_status()
        wall = time.monotonic() - self._t0
        return {"episodes": self.episodes, "steps": self.steps, "wall_s": round(wall, 1),
                "games_per_hour": round(self.episodes / (wall / 3600), 1) if wall else 0.0,
                "act_ms_mean": round(statistics.fmean(self.act_ms), 2) if self.act_ms else 0.0,
                "speed_x_mean": round(statistics.fmean(speeds), 2) if speeds else 0.0,
                "lua_timeouts": self.lua_timeouts, "errors": self.errors}
