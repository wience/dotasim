"""Python -> Lua file channel: config, calibration and per-index action files in the shared bots/ folder."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

LUA_SRC = Path(__file__).parent / "lua"
KEEP_BACK = 5  # action files older than idx-5 are deleted


@dataclass(frozen=True)
class Action:
    kind: str = "none"  # none | move | attack
    x: float = 0
    y: float = 0
    name: str = ""


def lua_literal(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, (list, tuple)):
        return "{" + ", ".join(lua_literal(x) for x in v) + "}"
    if isinstance(v, dict):
        int_keys = sorted(k for k in v if isinstance(k, int))
        str_keys = [k for k in v if isinstance(k, str)]
        parts = [f"{k}={lua_literal(v[k])}" for k in str_keys] + [f"[{k}]={lua_literal(v[k])}" for k in int_keys]
        return "{" + ", ".join(parts) + "}"
    raise TypeError(f"no Lua literal for {type(v).__name__}")


class Channel:
    def __init__(self, bots_dir: Path):
        self.dir = Path(bots_dir)

    def _write(self, name: str, body: str) -> None:
        tmp = self.dir / (name + ".tmp")
        tmp.write_text(body)
        os.replace(tmp, self.dir / name)

    def write_config(self, frames: int, heroes: dict[int, str], roles: dict[int, str]) -> None:
        self._write("config.lua", "return " + lua_literal({"frames": frames, "heroes": heroes, "roles": roles}))

    def write_calib(self, token: str, dota_time: float, frames: int) -> None:
        self._write(f"calib_{token}.lua", "return " + lua_literal({"t": dota_time, "n": frames}))

    def write_action(self, token: str, idx: int, action: Action) -> None:
        body = "return " + lua_literal({"idx": idx, "kind": action.kind, "x": action.x, "y": action.y, "name": action.name})
        self._write(f"actions_{token}_{idx}.lua", body)
        old = self.dir / f"actions_{token}_{idx - KEEP_BACK}.lua"
        try:
            old.unlink()
        except OSError:
            pass

    def clean(self) -> int:
        n = 0
        for p in self.dir.iterdir():
            if p.name.startswith(("actions_", "calib_", "console_")):
                p.unlink()
                n += 1
        return n


def install_scripts(dota_dir: Path) -> Path:
    bots = Path(dota_dir).expanduser() / "game" / "dota" / "scripts" / "vscripts" / "bots"
    bots.mkdir(parents=True, exist_ok=True)
    for src in LUA_SRC.glob("*.lua"):
        shutil.copy(src, bots / src.name)
    Channel(bots).clean()
    return bots
