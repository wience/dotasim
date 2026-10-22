"""Record Valve-vs-Valve games as raw worldstates (both teams' views) plus observer stats.

Layout: <out>/meta.json, <out>/ep<NNNN>_t<team>.npz with
  raw        object[T]        zlib-compressed CMsgBotWorldState bytes
  dota_time  float32[T]
  ally_stats int32[T, 10, 5]  per player id: last_hits, denies, gold, net_worth, level (-1 unknown)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from dotasim.env.env import Matchup
from dotasim.env.worldstate import WorldState, parse
from dotasim.train.policy import ValveOnlyPolicy
from dotasim.train.rollout import Fleet

SF = "npc_dota_hero_nevermore"
STAT_KEYS = ("last_hits", "denies", "gold", "net_worth", "level")
PROTO_PIN = Path(__file__).resolve().parents[1] / "env" / "proto" / "PIN"


class _Recorder:
    def __init__(self, fleet: Fleet, out: Path, teams: tuple[int, ...]):
        self.fleet, self.out, self.teams = fleet, out, teams
        self.buf: dict[int, dict[int, dict[str, list]]] = {}   # env index -> team -> columns
        self.episode = 0
        self.steps = 0
        # zlib releases the GIL: compress in a pool so the event loop keeps draining 12 x 2 sockets
        self.pool = ThreadPoolExecutor(max_workers=4)

    def _fresh(self) -> dict[int, dict[str, list]]:
        return {t: {"raw": [], "dota_time": [], "ally": []} for t in self.teams}

    def __call__(self, i: int, obs: dict[int, WorldState], done: bool, info: dict) -> None:
        env = self.fleet.envs[i]
        b = self.buf.setdefault(i, self._fresh())
        ally = env.instance.lua_ally()
        row = np.full((10, 5), -1, dtype=np.int32)
        for pid, a in ally.items():
            if 0 <= pid < 10:
                row[pid] = [a[k] for k in STAT_KEYS]
        for t in self.teams:
            if t in obs:
                b[t]["raw"].append(self.pool.submit(zlib.compress, env.last_raw(t), 6))
                b[t]["dota_time"].append(obs[t].dota_time)
                b[t]["ally"].append(row)
                self.steps += 1
        if done:
            if "error" in info:
                self.buf[i] = self._fresh()   # drop a broken episode
                return
            ep = self.episode
            self.episode += 1
            for t in self.teams:
                cols = b[t]
                if cols["raw"]:
                    # the frames are already zlib'd: plain savez, and off the event loop so no socket stalls
                    self.pool.submit(_save, self.out / f"ep{ep:04d}_t{t}.npz", cols)
            self.buf[i] = self._fresh()

    def close(self) -> None:
        self.pool.shutdown(wait=True)


def _save(path: Path, cols: dict[str, list]) -> None:
    raw = np.empty(len(cols["raw"]), dtype=object)
    raw[:] = [f.result() for f in cols["raw"]]
    np.savez(path, raw=raw, dota_time=np.array(cols["dota_time"], dtype=np.float32),
             ally_stats=np.stack(cols["ally"]).astype(np.int32))


async def record(games: int, instances: int, frames: int, difficulty: int, cap_s: float, out: Path,
                 dota_dir: Path | None = None, timescale: float = 10, heroes: dict[int, str] | None = None) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    heroes = heroes or {0: SF, 5: SF}
    matchup = Matchup(heroes=heroes, roles={pid: "valve" for pid in heroes}, frames=frames, difficulty=difficulty,
                      cap_s=cap_s, watch_teams=(2, 3))
    (out / "meta.json").write_text(json.dumps({
        "frames": frames, "difficulty": difficulty, "heroes": heroes, "cap_s": cap_s, "build": 6965,
        "proto_pin": PROTO_PIN.read_text().strip(), "date": datetime.now().isoformat(timespec="seconds"),
        "games_requested": games,
    }, indent=1))
    n = min(instances, games)
    fleet = Fleet(n, matchup, ValveOnlyPolicy(), out / "run", timescale=timescale, dota_dir=dota_dir)
    rec = _Recorder(fleet, out, (2, 3))
    fleet.on_step = rec
    totals = await fleet.run(episodes_per_instance=math.ceil(games / n))
    rec.close()
    totals["episodes_recorded"] = rec.episode
    totals["steps_recorded"] = rec.steps
    return totals


def load_episode(path: Path) -> list[WorldState]:
    d = np.load(path, allow_pickle=True)
    steps: list[WorldState] = []
    hp_cache: dict[int, float] = {0: 1.0, 5: 1.0}   # unseen heroes are assumed at full hp (same as DotaEnv)
    for raw, stats in zip(d["raw"], d["ally_stats"]):
        ws = parse(zlib.decompress(raw))
        ws.stats = {pid: dict(zip(STAT_KEYS, (int(v) for v in stats[pid]))) for pid in range(10) if stats[pid, 0] >= 0}
        for pid, hero in ws.heroes.items():   # same last-known hp carry as DotaEnv
            hp_cache[pid] = hero.hp / hero.hp_max if (hero.alive and hero.hp_max) else 0.0
        for pid, frac in hp_cache.items():
            ws.stats.setdefault(pid, {})["hp_frac"] = frac
        for pid, hero in ws.heroes.items():
            if 0 <= pid < 10 and stats[pid, 0] >= 0:
                hero.last_hits, hero.denies, hero.gold, hero.net_worth = (int(v) for v in stats[pid, :4])
                hero.level = max(hero.level, int(stats[pid, 4]))
        steps.append(ws)
    return steps


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description="Record Valve-vs-Valve 1v1 games as worldstate sequences.")
    p.add_argument("--games", type=int, default=24)
    p.add_argument("--instances", type=int, default=12)
    p.add_argument("--frames", type=int, default=5)
    p.add_argument("--difficulty", type=int, default=4)
    p.add_argument("--cap", type=float, default=600)
    p.add_argument("--out", default="data/sf_mirror_v1")
    a = p.parse_args(argv)
    totals = asyncio.run(record(a.games, a.instances, a.frames, a.difficulty, a.cap, Path(a.out)))
    print(json.dumps(totals, indent=1))
    return totals


if __name__ == "__main__":
    main()
