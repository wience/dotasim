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
LUASTAT = re.compile(r"LUASTAT token=(\w+) waits=(\d+) exec=(\d+) timeouts=(\d+) mismatch=(\d+)")
LUAOBS = re.compile(r"LUAOBS token=(\w+) idx=(\d+) lh=(-?\d+) dn=(-?\d+) gold=(-?\d+) xp=(-?\d+) lvl=(-?\d+) to=(\d+) mm=(\d+)")
LUAERR = re.compile(r"LUAERR token=(\w+) (.*)")


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
            "+map", "start", "gamemode", "21", "-fill_with_bots",
            "-botworldstatetosocket_radiant", str(self.ws_port(2)),
            "-botworldstatetosocket_dire", str(self.ws_port(3)),
            "-botworldstatetosocket_frames", str(self.frames), "-botworldstatesocket_threaded",
            "+dota_auto_surrender_all_disconnected_timeout", "180",
        ]


def _port_in_use(port: int) -> bool:
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
            return False
        except OSError:
            return True


@dataclass
class _LuaState:
    tokens: dict[int, str] = field(default_factory=dict)      # team -> token (controlled|valve)
    roles: dict[str, str] = field(default_factory=dict)       # token -> role
    stats: dict[str, dict] = field(default_factory=dict)      # token -> last LUASTAT
    obs: dict[str, dict] = field(default_factory=dict)        # token -> last LUAOBS
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

    # ---- lifecycle -------------------------------------------------------
    async def launch(self) -> None:
        for port in (self.cfg.server_port, self.cfg.ws_port(2), self.cfg.ws_port(3)):
            if _port_in_use(port):
                raise InstanceError(f"port {port} in use (instance {self.cfg.index})")
        self.cfg.console_log.parent.mkdir(parents=True, exist_ok=True)
        self.cfg.console_log.write_text("")
        self._log_pos = 0
        self.lua = _LuaState()
        self.post_game_seen = False
        self.cfg.run_dir.mkdir(parents=True, exist_ok=True)
        self._out = open(self.cfg.run_dir / f"instance_{self.cfg.index}.out", "ab")
        env = dict(os.environ, LD_LIBRARY_PATH=str(self.cfg.game_dir / "bin" / "linuxsteamrt64"), ENABLE_PATHMATCH="1")
        self.proc = await asyncio.create_subprocess_exec(
            *self.cfg.args(), cwd=str(self.cfg.game_dir), env=env,
            stdout=self._out, stderr=asyncio.subprocess.STDOUT, start_new_session=True,
        )
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
                self._log_pos += len(chunk)
        except OSError:
            return
        for line in chunk.decode(errors="replace").splitlines():
            self._tail.append(line)
            if "DOTA_GAMERULES_STATE_POST_GAME" in line:
                self.post_game_seen = True
            if m := LUARDY.search(line):
                token, team, pid, role = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
                self.lua.roles[token] = role
                if role in ("controlled", "valve"):
                    self.lua.tokens[team] = token
            elif m := LUAOBS.search(line):
                self.lua.obs[m.group(1)] = {"idx": int(m.group(2)), "last_hits": int(m.group(3)), "denies": int(m.group(4)),
                                            "gold": int(m.group(5)), "xp": int(m.group(6)), "level": int(m.group(7)),
                                            "timeouts": int(m.group(8)), "mismatch": int(m.group(9))}
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
        raise InstanceError(f"instance {self.cfg.index}: no LUARDY within {timeout}s:\n" + "\n".join(self.console_tail()))

    def console_tail(self, n: int = 20) -> list[str]:
        self._poll_log()
        return list(self._tail)[-n:]

    def lua_stats(self) -> dict[str, dict]:
        self._poll_log()
        return dict(self.lua.stats)

    def lua_obs(self) -> dict[str, dict]:
        self._poll_log()
        return dict(self.lua.obs)

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

    # ---- resources -------------------------------------------------------
    def stats(self) -> dict:
        if not self.alive():
            return {"pid": self.proc.pid if self.proc else 0, "cpu": 0.0, "rss_mb": 0.0}
        try:
            return {"pid": self._ps.pid, "cpu": self._ps.cpu_percent(None), "rss_mb": self._ps.memory_info().rss / 2**20}
        except psutil.Error:
            return {"pid": self._ps.pid, "cpu": 0.0, "rss_mb": 0.0}
