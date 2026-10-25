"""SyncFleet: N servers stepped in lockstep with one batched forward pass per policy per step.

Each controlled team of each server is routed to a policy by policy_for(env_index, team); items for the
same policy object are batched together. Teams routed to None are Valve bots (role "valve" in the matchup).

DotaEnv.step returns whichever team's worldstate arrives next, so with two controlled teams the loop
alternates: decide for the teams whose worldstate just arrived, step, receive the other team's state.
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

MAX_RESET_FAILURES = 3   # per env per run before the fleet gives up on that env


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


class SyncFleet:
    def __init__(self, n: int, matchup_for_env: Callable[[int], Matchup], policy_for: Callable[[int, int], object | None],
                 run_dir: Path, dota_dir: Path | None = None, timescale: float = 10, stagger_s: float = 2.0,
                 tv: bool = True):
        dota = Path(dota_dir or os.environ.get("DOTASIM_DOTA_DIR", "~/dota2")).expanduser()
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        bots = install_scripts(dota)
        self.matchup_for_env, self.policy_for, self.stagger_s = matchup_for_env, policy_for, stagger_s
        self.envs = [DotaEnv(InstanceConfig(index=i, dota_dir=dota, timescale=timescale, run_dir=self.run_dir, tv=tv), bots)
                     for i in range(n)]
        self.obs: dict[int, dict[int, WorldState]] = {}       # env -> teams whose worldstate awaits a decision
        self.resets: dict[int, asyncio.Task] = {}
        self.reset_failures: dict[int, int] = {}
        self.episodes = self.failed = self.steps = self.lua_timeouts = 0
        self.ends: dict[str, int] = {}   # episode end reasons, from info['end']
        self.errors: list[str] = []
        self.act_ms: list[float] = []
        self._t0 = 0.0
        self.labels: dict[int, dict] = {}   # per instance, set by the caller: kind, live_side, opponent (for the status page)
        self.meta: dict = {}                # run-level, set by the caller: iteration, version, kind
        self._status_n = 0

    # ---- status ----------------------------------------------------------
    def status(self) -> dict:
        wall_h = max(time.monotonic() - self._t0, 1e-6) / 3600
        return {"run": self.run_dir.name, "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "instances": [{**e.status, **self.labels.get(i, {})} for i, e in enumerate(self.envs)], "meta": self.meta,
                "totals": {"episodes": self.episodes, "steps": self.steps, "games_per_hour": round(self.episodes / wall_h, 1),
                           "infer_ms_mean": round(statistics.fmean(self.act_ms), 3) if self.act_ms else 0.0,
                           "lua_timeouts": self.lua_timeouts, "errors": len(self.errors)}}

    def _write_status(self) -> None:
        _atomic_write(self.run_dir / "status.json", json.dumps(self.status()))
        self._status_n += 1
        for i, env in enumerate(self.envs):   # instance 0 every second, the others every third second (the run dir is on /mnt/c)
            ws = env.last_ws
            if ws is not None and (i == 0 or self._status_n % 3 == 0):
                _atomic_write(self.run_dir / f"ws_{i}.json", json.dumps(ws.reduced()))

    # ---- episodes --------------------------------------------------------
    async def _reset(self, i: int, delay: float) -> None:
        await asyncio.sleep(delay)
        env = self.envs[i]
        m = self.matchup_for_env(i)
        for team in (2, 3):
            pol = self.policy_for(i, team)
            if pol is not None and hasattr(pol, "reset_env"):
                pol.reset_env(i)
        self.obs[i] = await env.reset(m)

    def _start_reset(self, i: int, delay: float = 0.0) -> None:
        self.resets[i] = asyncio.create_task(self._reset(i, delay))

    def _routed_teams(self, i: int) -> list[int]:
        return [t for t in (2, 3) if self.policy_for(i, t) is not None]

    async def run(self, episodes_total: int, on_step: Callable | None = None, max_steps: int | None = None) -> dict:
        """on_step(env_index, acted, arrived, done, info) after every step of every playing env:
        acted = {team: detail} for the teams that were given an action this step (detail from act_batch_full, else None),
        arrived = {team: WorldState} returned by the step (the next decision states), done/info from the env."""
        self._t0 = time.monotonic()
        started = 0
        for i in range(len(self.envs)):
            if started < episodes_total:
                self._start_reset(i, i * self.stagger_s)
                started += 1
        steps_in_ep = {i: 0 for i in range(len(self.envs))}
        first_wave = set(range(min(len(self.envs), episodes_total)))
        last_hb = 0.0
        speeds: list[float] = []
        try:
            while self.episodes + self.failed < episodes_total:
                # 1. adopt finished resets
                for i, task in list(self.resets.items()):
                    if task.done():
                        del self.resets[i]
                        try:
                            task.result()
                            steps_in_ep[i] = 0
                            if on_step is not None:   # the first decision states (from reset) are delivered as arrivals
                                on_step(i, {}, self.obs[i], False, {})
                        except Exception as e:  # noqa: BLE001 - a failed launch must not stop the fleet
                            self.errors.append(f"instance {i} reset: {e}")
                            self.reset_failures[i] = self.reset_failures.get(i, 0) + 1
                            self.obs.pop(i, None)
                            if self.reset_failures[i] < MAX_RESET_FAILURES:
                                self._start_reset(i, 5.0)          # retry the same episode slot
                            else:
                                self.failed += 1                    # give this episode up
                                self.envs[i].phase, self.envs[i].last_error = "dead", str(e)
                playing = [i for i in self.obs if i not in self.resets]
                # barrier: nobody plays until the whole first wave has launched; a server stepping while eleven
                # others load maps gets starved past the read timeout and loses its episode
                if not playing or (first_wave and any(i in self.resets for i in first_wave)):
                    await asyncio.sleep(0.2)
                    continue
                first_wave = set()
                # 2. batched decisions for the teams awaiting one, grouped by policy object
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
                acted: dict[int, dict[int, dict | None]] = {i: {} for i in playing}
                t0 = time.perf_counter()
                for pid, items in groups.items():
                    pol = pol_by_id[pid]
                    if hasattr(pol, "act_batch_full"):
                        for (i, team, _), d in zip(items, pol.act_batch_full(items)):
                            actions[i][team], acted[i][team] = d["action"], d
                    else:
                        for (i, team, _), a in zip(items, pol.act_batch(items)):
                            actions[i][team], acted[i][team] = a, None
                self.act_ms.append((time.perf_counter() - t0) * 1000)
                # 3. step everyone that is playing; an exception in one env is that env's error, not the fleet's
                results = await asyncio.gather(*(self.envs[i].step(actions[i]) for i in playing), return_exceptions=True)
                for i, res in zip(playing, results):
                    if isinstance(res, BaseException):
                        obs, done, info = {}, True, {"error": f"{type(res).__name__}: {res}", "lua_timeouts": 0}
                        self.envs[i].phase, self.envs[i].last_error = "dead", info["error"]
                    else:
                        obs, done, info = res
                    self.steps += 1
                    steps_in_ep[i] += 1
                    if max_steps and steps_in_ep[i] >= max_steps and not done:
                        done, info = True, {**info, "capped": True, "post_game": False}
                    if on_step is not None:
                        on_step(i, acted[i], obs, done, info)
                    if done:
                        self.episodes += 1
                        self.lua_timeouts += info.get("lua_timeouts", 0)
                        self.ends[str(info.get("end"))] = self.ends.get(str(info.get("end")), 0) + 1
                        if "error" in info:
                            self.errors.append(info["error"])
                        speeds.append(info.get("speed_x", 0.0))
                        self.obs.pop(i, None)
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
        return {"episodes": self.episodes, "failed": self.failed, "steps": self.steps, "wall_s": round(wall, 1),
                "games_per_hour": round(self.episodes / (wall / 3600), 1) if wall else 0.0,
                "act_ms_mean": round(statistics.fmean(self.act_ms), 2) if self.act_ms else 0.0,
                "speed_x_mean": round(statistics.fmean(speeds), 2) if speeds else 0.0,
                "lua_timeouts": self.lua_timeouts, "errors": self.errors, "ends": self.ends}
