"""DotaEnv: one 1v1 server driven in lockstep through reset()/step()."""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from dotasim.env.channel import Action, Channel
from dotasim.env.instance import Instance, InstanceConfig, InstanceError
from dotasim.env.worldstate import WorldState, parse, summary

PRE_GAME, GAME_IN_PROGRESS, POST_GAME = 4, 5, 6
STEP_TIMEOUT_S = 5.0      # wall time to wait for the next worldstate before declaring the instance dead
CONNECT_TIMEOUT_S = 120.0  # wall time for the first worldstate after launch (map load + socket open)
TICKS_PER_S = 30


class StreamTimeout(InstanceError):
    """No worldstate arrived within the step timeout (the server may be stalled or at post-game)."""


@dataclass
class Matchup:
    heroes: dict[int, str]
    roles: dict[int, str]          # player_id -> controlled | valve ; others are fillers
    frames: int = 10
    difficulty: int = 4
    cap_s: float = 600


@dataclass
class _Team:
    team: int
    token: str
    controlled: bool
    stream: object = None          # async generator of raw worldstate bytes
    ws_idx: int = -1               # tick-derived index of the latest worldstate since calibration (calib ws = 0)
    calib_t: float = 0.0           # dota_time of the calibration worldstate
    last: WorldState | None = None


class DotaEnv:
    def __init__(self, cfg: InstanceConfig, bots_dir: Path):
        self.cfg = cfg
        self.channel = Channel(bots_dir)
        self.instance = Instance(cfg)
        self.episode = 0
        self.phase = "idle"
        self.last_error: str | None = None
        self.matchup: Matchup | None = None
        self._teams: dict[int, _Team] = {}
        self._hist: deque[tuple[float, float]] = deque(maxlen=30)
        self._steps = 0

    # ---- lifecycle -------------------------------------------------------
    async def reset(self, matchup: Matchup) -> dict[int, WorldState]:
        await self.close()
        self.matchup = matchup
        self.episode += 1
        self._steps = 0
        self._hist.clear()
        self.last_error = None
        self.cfg.frames, self.cfg.difficulty = matchup.frames, matchup.difficulty
        self.channel.write_config(matchup.frames, matchup.heroes, matchup.roles)
        self.phase = "launching"
        await self.instance.launch()
        self.phase = "handshake"
        tokens = await self.instance.handshake()
        roles = self.instance.lua.roles
        self._teams = {t: _Team(t, tok, roles.get(tok) == "controlled") for t, tok in tokens.items()}
        watch = [tm for tm in self._teams.values() if tm.controlled] or [self._teams[2]]
        for tm in watch:
            tm.stream = self.instance.worldstates(tm.team)
        self.phase = "pregame"
        obs: dict[int, WorldState] = {}
        for tm in watch:
            ws = parse(await self._next_raw(tm, CONNECT_TIMEOUT_S))  # first message: includes the socket connect
            while ws.game_state != PRE_GAME:
                ws = parse(await self._next_raw(tm))
            tm.ws_idx, tm.calib_t, tm.last = 0, ws.dota_time, ws
            if tm.controlled:
                self.channel.write_calib(tm.token, ws.dota_time, matchup.frames)
                obs[tm.team] = ws
        self.phase = "playing"
        self._hist.append((time.monotonic(), ws.dota_time))
        return obs

    async def _next_raw(self, tm: _Team, timeout: float = STEP_TIMEOUT_S) -> bytes:
        try:
            return await asyncio.wait_for(tm.stream.__anext__(), timeout)
        except StopAsyncIteration:
            raise InstanceError(f"instance {self.cfg.index}: worldstate stream for team {tm.team} closed")
        except asyncio.TimeoutError:
            raise StreamTimeout(f"instance {self.cfg.index}: no worldstate for team {tm.team} in {timeout}s "
                                f"(alive={self.instance.alive()})")

    def ws_index(self, team: int) -> int:
        return self._teams[team].ws_idx

    async def step(self, actions: dict[int, Action]) -> tuple[dict[int, WorldState], bool, dict]:
        watch = [tm for tm in self._teams.values() if tm.stream is not None]
        for tm in watch:
            if tm.controlled:
                self.channel.write_action(tm.token, tm.ws_idx, actions.get(tm.team, Action()))
        obs: dict[int, WorldState] = {}
        done, error = False, None
        results = await asyncio.gather(*(self._next_raw(tm) for tm in watch), return_exceptions=True)
        failure = next((r for r in results if isinstance(r, BaseException)), None)
        if failure is not None:
            done, raws = True, []
            # On a read timeout the log already had STEP_TIMEOUT_S to catch up; on EOF give it a moment.
            if await self._post_game_reached(wait_s=0.0 if isinstance(failure, StreamTimeout) else 3.0):
                self.phase = "postgame"  # the server stops streaming at POST_GAME and later closes: a clean finish
            else:
                error = str(failure)
        else:
            raws = results
        for tm, raw in zip(watch, raws):
            ws = parse(raw)
            # same formula as Lua: index = ticks since calibration / frames, so a dropped message cannot desync us
            tm.ws_idx = round((ws.dota_time - tm.calib_t) * TICKS_PER_S / self.matchup.frames)
            tm.last = ws
            self._merge_lua_obs(tm, ws)
            if tm.controlled:
                obs[tm.team] = ws
        self._steps += 1
        if raws:
            ws0 = watch[0].last
            self._hist.append((time.monotonic(), ws0.dota_time))
            if ws0.game_state == POST_GAME or ws0.dota_time >= self.matchup.cap_s:
                done = True
        if error:
            self.phase, self.last_error = "dead", error
        elif done:
            self.phase = "postgame"
        info = self._info(error)
        if done and self.phase == "postgame":
            info["post_game"] = True
        return obs, done, info

    async def _post_game_reached(self, wait_s: float = 3.0) -> bool:
        """The console log can trail the socket by a moment; give the POST_GAME line a few seconds to land."""
        self.instance.console_tail()
        if self.instance.post_game_seen:
            return True
        if not self.instance.alive():
            return False
        for _ in range(int(wait_s / 0.2)):
            await asyncio.sleep(0.2)
            self.instance.console_tail()
            if self.instance.post_game_seen:
                return True
        return False

    def _merge_lua_obs(self, tm: _Team, ws: WorldState) -> None:
        if not tm.controlled:
            return
        o = self.instance.lua_obs().get(tm.token)
        hero = ws.hero_of(tm.team)
        if o and hero:
            hero.last_hits, hero.denies, hero.gold, hero.xp = o["last_hits"], o["denies"], o["gold"], o["xp"]

    async def close(self) -> None:
        try:
            for tm in self._teams.values():
                if tm.stream is not None:
                    try:
                        await tm.stream.aclose()
                    except (RuntimeError, OSError):
                        pass  # a generator still mid-read after a failed gather; the kill below ends it
                    tm.stream = None
        finally:
            await self.instance.kill()
        if self.phase not in ("idle", "dead", "postgame"):
            self.phase = "idle"

    # ---- reporting -------------------------------------------------------
    def _lua_totals(self) -> tuple[int, int]:
        obs = self.instance.lua_obs()
        to = sum(obs[tm.token]["timeouts"] for tm in self._teams.values() if tm.controlled and tm.token in obs)
        mm = sum(obs[tm.token]["mismatch"] for tm in self._teams.values() if tm.controlled and tm.token in obs)
        return to, mm

    def speed_x(self) -> float:
        if len(self._hist) < 2:
            return 0.0
        (w0, t0), (w1, t1) = self._hist[0], self._hist[-1]
        return (t1 - t0) / (w1 - w0) if w1 > w0 else 0.0

    def _info(self, error: str | None) -> dict:
        ws = next((tm.last for tm in self._teams.values() if tm.last is not None), None)
        info = summary(ws) if ws else {}
        to, mm = self._lua_totals()
        info.update({"episode_idx": self.episode, "game_time": ws.dota_time if ws else 0.0, "speed_x": round(self.speed_x(), 2),
                     "lua_timeouts": to, "lua_mismatch": mm, "steps": self._steps})
        if error:
            info["error"] = error
        return info

    @property
    def last_ws(self) -> WorldState | None:
        return next((tm.last for tm in self._teams.values() if tm.last is not None), None)

    @property
    def status(self) -> dict:
        ws = next((tm.last for tm in self._teams.values() if tm.last is not None), None)
        st = self.instance.stats()
        heroes, kills, last_hits = {}, {}, {}
        if ws:
            for t in (2, 3):
                h = ws.hero_of(t)
                heroes[t] = h.name.removeprefix("npc_dota_hero_") if h else ""
                kills[t] = h.kills if h else 0
                last_hits[t] = h.last_hits if h else -1
        to, _ = self._lua_totals()
        return {"index": self.cfg.index, "pid": st["pid"], "phase": self.phase, "game_time": ws.dota_time if ws else 0.0,
                "speed_x": round(self.speed_x(), 2), "heroes": heroes, "kills": kills, "last_hits": last_hits,
                "episode": self.episode, "lua_timeouts": to, "last_error": self.last_error,
                "cpu": st["cpu"], "rss_mb": round(st["rss_mb"])}
