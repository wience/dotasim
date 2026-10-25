"""One headless Dota 2 server: ports, launch, console-log tail, Lua token handshake, worldstate stream."""
from __future__ import annotations

import asyncio
import os
import re
import signal
import socket
import struct
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

import psutil

LUARDY = re.compile(r"LUARDY token=(\w+) team=(\d) pid=(\d+) role=(\w+)")
FORT = re.compile(r"Building: npc_dota_(goodguys|badguys)_fort destroyed")   # how a 1v1 ends: the engine razes the loser's fort
LUASTAT = re.compile(r"LUASTAT token=(\w+) waits=(\d+) exec=(\d+) timeouts=(\d+) mismatch=(\d+)")
LUAOBS = re.compile(r"LUAOBS token=(\w+) idx=(\d+) lh=(-?\d+) dn=(-?\d+) gold=(-?\d+) xp=(-?\d+) lvl=(-?\d+) to=(\d+) mm=(\d+)")
LUAERR = re.compile(r"LUAERR token=(\w+) (.*)")
LUAALLY = re.compile(r"LUAALLY token=(\w+) team=(\d) pid=(\d+) lh=(-?\d+) dn=(-?\d+) gold=(-?\d+) nw=(-?\d+) lvl=(-?\d+)")


class InstanceError(RuntimeError):
    pass


@dataclass
class InstanceConfig:
    index: int
    dota_dir: Path
    timescale: float = 10
    frames: int = 10
    difficulty: int = 4
    run_dir: Path = Path("runs/current")
    tv: bool = True   # SourceTV recording: a third of a server's CPU (85% -> 56% measured); off for training rollouts
    extra_args: list[str] = field(default_factory=list)  # appended verbatim to the launch line

    @property
    def game_dir(self) -> Path:
        return Path(self.dota_dir).expanduser() / "game"

    @property
    def bots_dir(self) -> Path:
        return self.game_dir / "dota" / "scripts" / "vscripts" / "bots"

    @property
    def server_port(self) -> int:
        return 27015 + self.index

    def ws_port(self, team: int) -> int:
        return 12120 + 2 * self.index + (team - 2)

    @property
    def tv_port(self) -> int:
        return 28000 + self.index

    @property
    def console_log(self) -> Path:
        return self.bots_dir / f"console_{self.index}.log"

    def args(self) -> list[str]:
        return [
            str(self.game_dir / "bin" / "linuxsteamrt64" / "dota2"),
            "-dedicated", "-insecure", "-console", "-condebug",
            "-con_logfile", f"scripts/vscripts/bots/console_{self.index}.log",
            "-novid", "-nosound", "-nowatchdog",
            "+dota_force_gamemode", "21", "+dota_1v1_skip_strategy", "1",
            "+host_timescale", str(self.timescale),
            "+sv_lan", "1", "+sv_cheats", "1", "+sv_hibernate_when_empty", "0",
            "+dota_bot_set_difficulty", str(self.difficulty),
            "-port", str(self.server_port),
            "+map", "start", "gamemode", "21", *([] if os.environ.get("DOTASIM_NO_FILL") else ["-fill_with_bots"]),
            "-botworldstatetosocket_radiant", str(self.ws_port(2)),
            "-botworldstatetosocket_dire", str(self.ws_port(3)),
            "-botworldstatetosocket_frames", str(self.frames), "-botworldstatesocket_threaded",
            "+dota_auto_surrender_all_disconnected_timeout", "180",
            # SourceTV: lets a real client spectate (connect <ip>:<tv_port>) and records replays
            "+tv_enable", "1" if self.tv else "0", "+tv_port", str(self.tv_port), "+tv_delay", "0",
            "+tv_autorecord", "1" if self.tv else "0",
            *self.extra_args,
        ]


def _port_in_use(port: int, kind: int = socket.SOCK_STREAM) -> bool:
    with socket.socket(socket.AF_INET, kind) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
            return False
        except OSError:
            return True


def _prune_replays(replays: Path, keep: int, min_age_s: float = 600.0) -> None:
    """SourceTV writes about 8 MB per 10-minute game; keep the newest `keep` demos.

    Never touch a demo modified in the last min_age_s: a server launched seconds earlier in the same wave is still
    recording into it, and deleting a live recording ends that server's game.
    """
    try:
        now = time.time()
        demos = sorted(replays.glob("*.dem"), key=lambda p: p.stat().st_mtime)
        for p in demos[:-keep] if keep else demos:
            if now - p.stat().st_mtime > min_age_s:
                p.unlink()
    except OSError:
        pass


@dataclass
class _LuaState:
    tokens: dict[int, str] = field(default_factory=dict)      # team -> token (controlled|valve)
    roles: dict[str, str] = field(default_factory=dict)       # token -> role
    stats: dict[str, dict] = field(default_factory=dict)      # token -> last LUASTAT
    obs: dict[str, dict] = field(default_factory=dict)        # token -> last LUAOBS
    ally: dict[int, dict] = field(default_factory=dict)       # player_id -> last LUAALLY (observer-reported stats)
    errors: list[str] = field(default_factory=list)


class Instance:
    def __init__(self, cfg: InstanceConfig):
        self.cfg = cfg
        self.proc: asyncio.subprocess.Process | None = None
        self._ps: psutil.Process | None = None
        self._log_pos = 0
        self._tail: deque[str] = deque(maxlen=200)
        self.lua = _LuaState()
        self._out = None
        self.post_game_seen = False
        self.winner: int | None = None   # 2 or 3 once the console says whose fort fell

    # ---- lifecycle -------------------------------------------------------
    async def launch(self) -> None:
        # the game server and SourceTV speak UDP; the worldstate ports are TCP
        checks = [(self.cfg.server_port, socket.SOCK_DGRAM), (self.cfg.tv_port, socket.SOCK_DGRAM),
                  (self.cfg.ws_port(2), socket.SOCK_STREAM), (self.cfg.ws_port(3), socket.SOCK_STREAM)]
        for port, kind in checks:
            if _port_in_use(port, kind):
                raise InstanceError(f"port {port} in use (instance {self.cfg.index})")
        self.cfg.console_log.parent.mkdir(parents=True, exist_ok=True)
        self.cfg.console_log.write_text("")
        _prune_replays(self.cfg.game_dir / "dota" / "replays", keep=30)
        self._log_pos = 0
        self._tail.clear()
        self.lua = _LuaState()
        self.post_game_seen = False
        self.winner: int | None = None   # 2 or 3 once the console says whose fort fell
        self.cfg.run_dir.mkdir(parents=True, exist_ok=True)
        self._out = open(self.cfg.run_dir / f"instance_{self.cfg.index}.out", "ab")
        env = dict(os.environ, LD_LIBRARY_PATH=str(self.cfg.game_dir / "bin" / "linuxsteamrt64"), ENABLE_PATHMATCH="1")
        try:
            self.proc = await asyncio.create_subprocess_exec(
                *self.cfg.args(), cwd=str(self.cfg.game_dir), env=env,
                stdout=self._out, stderr=asyncio.subprocess.STDOUT, start_new_session=True,
            )
        except Exception:
            self._out.close()
            self._out = None
            raise
        self._ps = psutil.Process(self.proc.pid)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None and self._ps is not None and self._ps.is_running() \
            and self._ps.status() != psutil.STATUS_ZOMBIE

    async def kill(self) -> None:
        if self.proc is None:
            return
        procs = []
        try:
            procs = [self._ps] + self._ps.children(recursive=True)
        except psutil.Error:
            pass
        if self.proc.returncode is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self.proc.wait(), 5)
            except asyncio.TimeoutError:
                for p in procs:
                    try:
                        p.kill()
                    except psutil.Error:
                        pass
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await self.proc.wait()
        if self._out:
            self._out.close()
            self._out = None

    # ---- console log -----------------------------------------------------
    def _poll_log(self) -> None:
        try:
            with open(self.cfg.console_log, "rb") as f:
                f.seek(self._log_pos)
                chunk = f.read()
        except OSError:
            return
        # the game may be mid-write: only consume up to the last complete line
        nl = chunk.rfind(b"\n")
        if nl < 0:
            return
        chunk = chunk[:nl + 1]
        self._log_pos += len(chunk)
        for line in chunk.decode(errors="replace").splitlines():
            self._tail.append(line)
            if "DOTA_GAMERULES_STATE_POST_GAME" in line and self.lua.tokens:
                # only after our own bots reported in: a previous server exiting can still write into this file
                self.post_game_seen = True
            if self.lua.tokens and (fm := FORT.search(line)):
                self.winner = 2 if fm.group(1) == "badguys" else 3
            if m := LUARDY.search(line):
                token, team, pid, role = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
                self.lua.roles[token] = role
                if role in ("controlled", "valve"):
                    self.lua.tokens[team] = token
            elif m := LUAOBS.search(line):
                self.lua.obs[m.group(1)] = {"idx": int(m.group(2)), "last_hits": int(m.group(3)), "denies": int(m.group(4)),
                                            "gold": int(m.group(5)), "xp": int(m.group(6)), "level": int(m.group(7)),
                                            "timeouts": int(m.group(8)), "mismatch": int(m.group(9))}
            elif m := LUAALLY.search(line):
                self.lua.ally[int(m.group(3))] = {"last_hits": int(m.group(4)), "denies": int(m.group(5)),
                                                  "gold": int(m.group(6)), "net_worth": int(m.group(7)), "level": int(m.group(8))}
            elif m := LUASTAT.search(line):
                self.lua.stats[m.group(1)] = {"waits": int(m.group(2)), "exec": int(m.group(3)),
                                              "timeouts": int(m.group(4)), "mismatch": int(m.group(5))}
            elif m := LUAERR.search(line):
                self.lua.errors.append(m.group(2))

    async def handshake(self, timeout: float = 90) -> dict[int, str]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._poll_log()
            if 2 in self.lua.tokens and 3 in self.lua.tokens:
                return dict(self.lua.tokens)
            if not self.alive():
                raise InstanceError(f"instance {self.cfg.index} died before handshake:\n" + "\n".join(self.console_tail()))
            await asyncio.sleep(0.2)
        raise InstanceError(f"instance {self.cfg.index}: no LUARDY for both teams within {timeout}s; "
                            f"tokens seen {self.lua.tokens}, roles {self.lua.roles}; console tail:\n" + "\n".join(self.console_tail()))

    def console_tail(self, n: int = 20) -> list[str]:
        self._poll_log()
        return list(self._tail)[-n:]

    def lua_stats(self) -> dict[str, dict]:
        self._poll_log()
        return dict(self.lua.stats)

    def lua_obs(self) -> dict[str, dict]:
        self._poll_log()
        return dict(self.lua.obs)

    def lua_ally(self) -> dict[int, dict]:
        self._poll_log()
        return dict(self.lua.ally)

    # ---- worldstate ------------------------------------------------------
    async def worldstates(self, team: int, connect_timeout: float = 120) -> AsyncIterator[bytes]:
        port = self.cfg.ws_port(team)
        deadline = time.monotonic() + connect_timeout
        reader = None
        while reader is None:
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
            except OSError:
                if time.monotonic() > deadline or not self.alive():
                    raise InstanceError(f"instance {self.cfg.index}: worldstate port {port} never accepted")
                await asyncio.sleep(0.5)
        try:
            while True:
                try:
                    hdr = await reader.readexactly(4)
                    (n,) = struct.unpack("<I", hdr)
                    yield await reader.readexactly(n)
                except (asyncio.IncompleteReadError, ConnectionError):
                    return
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (OSError, asyncio.CancelledError):
                pass

    # ---- resources -------------------------------------------------------
    def stats(self) -> dict:
        if not self.alive():
            return {"pid": self.proc.pid if self.proc else 0, "cpu": 0.0, "rss_mb": 0.0}
        try:
            return {"pid": self._ps.pid, "cpu": self._ps.cpu_percent(None), "rss_mb": self._ps.memory_info().rss / 2**20}
        except psutil.Error:
            return {"pid": self._ps.pid, "cpu": 0.0, "rss_mb": 0.0}
