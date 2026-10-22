# Phase 1: Harness and Fleet Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Python package that runs N headless Dota 2 1v1 servers inside WSL2 in lockstep with a policy, exposes each as `DotaEnv`, and shows the fleet on a local status page.

**Architecture:** One `Instance` per server process (ports, launch, console-log tail, token handshake, worldstate sockets). `DotaEnv` wraps an `Instance` with the action file channel and typed worldstates. `rollout` owns a fleet of envs, drives them with a policy, and writes `status.json` once a second; a static HTML page renders it. Lua side: per-hero stubs delegate to one `core.lua` that reads `config.lua`, picks a role, mints a token, and runs the lockstep loop.

**Tech Stack:** Python 3.12 (uv), asyncio, protobuf (native backend), psutil, pytest; Lua 5.1 bot scripts; Dota 2 Linux build at `~/dota2` in WSL2 Ubuntu.

**Spec:** `docs/superpowers/specs/2026-10-22-dotasim-design.md`

## Global Constraints

- Python 3.12 via uv; all Python runs inside WSL2 (`wsl.exe -d Ubuntu`). Repo lives at `/mnt/c/Users/ADMIN/Documents/GitHub/dotasim`; the venv lives at `~/dotasim-venv` (not on `/mnt/c`, for speed).
- Dota root is `$DOTASIM_DOTA_DIR`, default `~/dota2`. Only this install is ever modified. The Windows Steam install is never touched.
- Launch the binary directly: `cwd=~/dota2/game`, `LD_LIBRARY_PATH=$cwd/bin/linuxsteamrt64`, `ENABLE_PATHMATCH=1`, `bin/linuxsteamrt64/dota2`. Never `dota.sh`.
- Launch flags (verbatim, per instance `i`): `-dedicated -insecure -console -condebug -con_logfile scripts/vscripts/bots/console_<i>.log -novid -nosound -nowatchdog +dota_force_gamemode 21 +dota_1v1_skip_strategy 1 +host_timescale <ts> +sv_lan 1 +sv_cheats 1 +sv_hibernate_when_empty 0 +dota_bot_set_difficulty <d> -port <27015+i> +map start gamemode 21 -fill_with_bots -botworldstatetosocket_radiant <12120+2i> -botworldstatetosocket_dire <12121+2i> -botworldstatetosocket_frames <frames> -botworldstatesocket_threaded +dota_auto_surrender_all_disconnected_timeout 180`.
- Worldstate framing: 4-byte little-endian length, then `CMsgBotWorldState`. The game hosts the ports; Python connects with retries.
- Lua loads files only via `package.loaded[m] = nil; require(m)`. One action file per index. Files: `bots/config.lua`, `bots/calib_<token>.lua`, `bots/actions_<token>_<idx>.lua`.
- Team ids: radiant 2, dire 3. Game states: PRE_GAME 4, GAME_IN_PROGRESS 5, POST_GAME 6, TEAM_SHOWCASE 8.
- Default frames 10, default host_timescale 10, episode cap 600 game seconds, Lua wait timeout 1.0 s real time.
- Tests: end-to-end, integration or golden only. No unit tests, nothing asserting on mock calls. Tests that need the game are marked `@pytest.mark.game` and skipped unless `DOTASIM_DOTA_DIR` resolves to an install.
- Proto pinned to SteamDatabase/GameTracking-Dota2 commit recorded in `dotasim/env/proto/PIN`.

## Review Focus

1. Server process dies mid-episode (crash, OOM, killed from the terminal): `DotaEnv.step` must return `done=True` with `info["error"]` within 5 s, never hang. Pinned in Task 6.
2. Worldstate or server port already bound by a stale server from a previous run: `Instance.launch` must fail fast with the port number in the error, not wait forever for a connection. Pinned in Task 5.
3. Policy slower than the decision budget: Lua times out, the game advances without an action, and the count must surface in `info["lua_timeouts"]` and on the status page rather than silently degrading. Pinned in Task 6.
4. Stale `actions_*`, `calib_*` and `console_*` files from a crashed run in the shared `bots/` folder: `install_scripts` must remove them before a wave launches, or a new instance can adopt an old index. Pinned in Task 4.
5. `LUARDY` never appears (Lua error at load, wrong hero name in config): the handshake must time out at 90 s with the tail of that instance's console log in the error. Pinned in Task 5.

---

### Task 1: Project skeleton, pinned proto, golden worldstate

**Files:**
- Create: `pyproject.toml`, `README.md`, `dotasim/__init__.py`, `dotasim/env/__init__.py`, `dotasim/env/proto/PIN`, `dotasim/env/proto/bot_script.proto`, `dotasim/env/proto/valveextensions.proto`, `dotasim/env/proto/gen.py`, `dotasim/env/proto/bot_script_pb2.py` (generated, committed), `tests/conftest.py`, `tests/golden/worldstate_6965.bin`
- Test: `tests/test_proto_golden.py`

**Interfaces:**
- Produces: `from dotasim.env.proto import bot_script_pb2 as pb`; `pb.CMsgBotWorldState`.
- Produces: `tests/golden/worldstate_6965.bin` (the 46 KB radiant message saved by the spike at `~/dotaspike/ws_sample.bin` in WSL2).
- Produces: `tests/conftest.py` with fixture `golden_ws_bytes` and marker `game` (skips unless `Path(os.environ.get("DOTASIM_DOTA_DIR", "~/dota2")).expanduser()/"game/bin/linuxsteamrt64/dota2"` exists).

- [ ] **Step 1: Write `pyproject.toml`** with `[project] name="dotasim" requires-python=">=3.12"`, dependencies `protobuf>=5`, `psutil>=6`; optional group `dev`: `pytest>=8`, `grpcio-tools>=1.60`; `[tool.pytest.ini_options] markers = ["game: needs a Dota 2 install"]`.

- [ ] **Step 2: Create the venv in WSL2 and install**

Run: `uv python install 3.12 && uv venv ~/dotasim-venv -p 3.12 && uv pip install -p ~/dotasim-venv/bin/python -e ".[dev]"` (install uv first with `curl -LsSf https://astral.sh/uv/install.sh | sh` if missing).
Expected: `~/dotasim-venv/bin/python -c "import google.protobuf, psutil"` prints nothing.

- [ ] **Step 3: Vendor the protos and record the pin**

Fetch `Protobufs/dota_gcmessages_common_bot_script.proto` and `Protobufs/valveextensions.proto` from `https://raw.githubusercontent.com/SteamDatabase/GameTracking-Dota2/<commit>/...`, where `<commit>` is the current master SHA from `https://api.github.com/repos/SteamDatabase/GameTracking-Dota2/commits/master`. Write the SHA into `dotasim/env/proto/PIN`. Save the bot script proto as `bot_script.proto`.

- [ ] **Step 4: Write `gen.py`**: `def generate(proto_dir: Path) -> Path` strips, with regex, ` [boxed_type = "..."]`, `, boxed_type = "..."`, `, synthetic_default = "..."` and lines starting `option additional_includes`, writes the stripped copies to a temp dir, runs `grpc_tools.protoc -I<tmp> --python_out=<proto_dir>`, and returns the `_pb2.py` path. `if __name__ == "__main__": generate(Path(__file__).parent)`.

- [ ] **Step 5: Run it and commit the generated module**

Run: `~/dotasim-venv/bin/python -m dotasim.env.proto.gen && ls dotasim/env/proto/*_pb2.py`
Expected: `bot_script_pb2.py` and `valveextensions_pb2.py` exist.

- [ ] **Step 6: Copy the golden message**

Run: `cp ~/dotaspike/ws_sample.bin tests/golden/worldstate_6965.bin && ls -l tests/golden/`
Expected: a file of about 46,000 bytes.

- [ ] **Step 7: Write the failing golden test**

```python
def test_golden_worldstate_parses(golden_ws_bytes):
    ws = pb.CMsgBotWorldState(); ws.ParseFromString(golden_ws_bytes)
    assert ws.team_id == 2 and ws.game_state == 5
    heroes = [u for u in ws.units if u.unit_type == pb.CMsgBotWorldState.UnitType.Value("HERO")]
    assert len(heroes) == 10 and all(u.name == "npc_dota_hero_nevermore" for u in heroes)
    assert ws.units[0].HasField("action_type")
```

- [ ] **Step 8: Run the test**

Run: `~/dotasim-venv/bin/python -m pytest tests/test_proto_golden.py -v`
Expected: PASS (if `HasField("action_type")` fails, drop that assertion and note it in the task commit message; it only informs phase 2).

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml README.md dotasim tests && git commit -m "feat: project skeleton, pinned worldstate proto, golden message"
```

---

### Task 2: Typed worldstate

**Files:**
- Create: `dotasim/env/worldstate.py`
- Test: `tests/test_worldstate_golden.py`, `tests/golden/worldstate_6965.json`

**Interfaces:**
- Produces:
  ```python
  @dataclass(slots=True) class Unit: handle:int; name:str; kind:str  # "hero"|"creep"|"tower"|"other"
      team:int; x:float; y:float; facing:float; hp:int; hp_max:int; mana:float; mana_max:float
      level:int; alive:bool; attack_damage:int; action_type:int; attack_target:str
  @dataclass(slots=True) class Hero(Unit): player_id:int; last_hits:int; denies:int; gold:int; xp:int; kills:int; deaths:int
  @dataclass(slots=True) class WorldState: team:int; dota_time:float; game_time:float; game_state:int
      units:list[Unit]; heroes:dict[int,Hero]  # keyed by player_id
      def hero_of(self, team:int) -> Hero | None   # lowest player_id hero on that team
      def reduced(self) -> dict                    # {"t":dota_time,"units":[[x,y,team,kind,hp_frac],...],"heroes":{pid:{...}}} for the UI
  def parse(buf: bytes) -> WorldState
  def summary(ws: WorldState) -> dict   # {"dota_time","game_state","kills":{2:..,3:..},"last_hits":{2:..,3:..}}
  ```
- `kind` mapping from `unit_type`: HERO→hero, LANE_CREEP→creep, TOWER→tower, else other. Hero fields `last_hits`, `denies`, `gold`, `xp` come from the Unit message (`last_hits`, `denies`, `unreliable_gold+reliable_gold`, `xp_needed_to_level` is not xp: use `net_worth` if `xp` is absent and note it); `kills`/`deaths` come from `players`.

- [ ] **Step 1: Write the golden test**: `parse(golden_ws_bytes)`, then assert `ws.team == 2`, `len(ws.heroes) == 10`, `ws.hero_of(2).player_id == 0`, `ws.hero_of(3).player_id == 5`, and `summary(ws) == json.load(open("tests/golden/worldstate_6965.json"))`. Generate the JSON once from the first passing run and commit it; the test then pins it.

- [ ] **Step 2: Run to verify it fails**

Run: `~/dotasim-venv/bin/python -m pytest tests/test_worldstate_golden.py -v`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Implement `worldstate.py`** per the Interfaces block.

- [ ] **Step 4: Run, write the golden JSON from the output, run again**

Expected: PASS twice (second run against the committed JSON).

- [ ] **Step 5: Commit** `git add dotasim/env/worldstate.py tests && git commit -m "feat: typed worldstate with UI reduction"`

---

### Task 3: Lua scripts

**Files:**
- Create: `dotasim/env/lua/core.lua`, `dotasim/env/lua/hero_selection.lua`, `dotasim/env/lua/bot_nevermore.lua`, `dotasim/env/lua/bot_puck.lua`, `dotasim/env/lua/bot_queenofpain.lua`

**Interfaces:**
- Consumes `bots/config.lua`:
  ```lua
  return { frames = 10, heroes = { [0]="npc_dota_hero_nevermore", [5]="npc_dota_hero_nevermore" },
           roles = { [0]="controlled", [5]="valve" } }   -- player ids absent from roles are fillers
  ```
- Consumes `bots/calib_<token>.lua`: `return { t = -89.867, n = 10 }`.
- Consumes `bots/actions_<token>_<idx>.lua`: `return { idx = 12, kind = "move", x = -600, y = -600 }` | `{ idx, kind = "attack", name = "npc_dota_creep_badguys_melee", x, y }` | `{ idx, kind = "none" }`.
- Produces console line at bot load: `LUARDY token=<hex> team=<2|3> pid=<n> role=<controlled|valve|filler>`.
- Produces console line every 50 waits and at episode end: `LUASTAT token=<hex> waits=<n> exec=<n> timeouts=<n> mismatch=<n>`.
- Produces: hero stub files are three lines each: `local core = require("bots/core")` then `core.install(GetBot())` — `install` sets the global `Think` for controlled (lockstep loop) and filler (no-op) roles and sets nothing for valve.
- Token: `string.format("%x%s", math.floor(RealTime()*1000) % 0xFFFFFF, tostring({}):match("0x(%x+)"))`.
- Lockstep in `core.lua` is the spike's loop (`m = floor((DotaTime()-t)/(1/30)+0.5)`, block when `m % n == 1`, want `(m-1)/n`, 1.0 s timeout). Attack resolves the target by `name` among `GetNearbyCreeps(1600,true)`, `GetNearbyHeroes(1600,true,BOT_MODE_NONE)`, `GetNearbyTowers(1600,true)` picking the nearest to `(x,y)`; if none matches, `mismatch` increments and the bot does nothing.
- `hero_selection.lua`: `Think()` calls `SelectHero(pid, name)` for every `pid` in `config.heroes` on this team that is a bot with no hero yet; fillers with no entry get `npc_dota_hero_nevermore`. `UpdateLaneAssignments()` returns LANE_MID for all.

- [ ] **Step 1: Write the five Lua files** per the Interfaces block. Wrap the controlled Think body in `pcall` and print `LUAERR token=<hex> <err>` at most 5 times.

- [ ] **Step 2: Syntax-check with the game's own Lua**: there is no local luac, so the check is Task 6's end-to-end test. Commit now.

```bash
git add dotasim/env/lua && git commit -m "feat: lua core with roles, token handshake and lockstep loop"
```

---

### Task 4: Action channel and script install

**Files:**
- Create: `dotasim/env/channel.py`
- Test: `tests/test_channel_golden.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True) class Action: kind:str="none"; x:float=0; y:float=0; name:str=""
  class Channel:
      def __init__(self, bots_dir: Path): ...
      def write_config(self, frames:int, heroes:dict[int,str], roles:dict[int,str]) -> None   # bots/config.lua
      def write_calib(self, token:str, dota_time:float, frames:int) -> None                   # bots/calib_<token>.lua
      def write_action(self, token:str, idx:int, action:Action) -> None                      # bots/actions_<token>_<idx>.lua, atomic via .tmp + os.replace, deletes idx-5
      def clean(self) -> int        # removes actions_*, calib_*, console_* ; returns count
  def install_scripts(dota_dir: Path) -> Path   # copies dotasim/env/lua/*.lua into <dota_dir>/game/dota/scripts/vscripts/bots/, calls Channel.clean, returns bots dir
  def lua_literal(v) -> str          # Python str/int/float/bool/dict/list -> Lua literal; dict keys int or str
  ```

- [ ] **Step 1: Write the golden test** in a `tmp_path`: `write_action(tok, 3, Action("attack", -600, -600, "npc_dota_creep_badguys_melee"))` produces exactly `return {idx=3, kind="attack", x=-600, y=-600, name="npc_dota_creep_badguys_melee"}`; `write_config(10, {0:"npc_dota_hero_nevermore",5:"npc_dota_hero_puck"}, {0:"controlled",5:"valve"})` produces the config literal from Task 3 with keys in ascending order; writing indices 0..7 leaves only files 3..7; `clean()` on a dir with two `actions_*` and one `console_0.log` returns 3.

- [ ] **Step 2: Run to verify it fails**, then **Step 3: Implement**, then **Step 4: Run to verify it passes**

Run: `~/dotasim-venv/bin/python -m pytest tests/test_channel_golden.py -v`

- [ ] **Step 5: Commit** `git add dotasim/env/channel.py tests/test_channel_golden.py && git commit -m "feat: action file channel and script install"`

---

### Task 5: Instance

**Files:**
- Create: `dotasim/env/instance.py`
- Test: `tests/test_instance_game.py` (marked `game`)

**Interfaces:**
- Produces:
  ```python
  @dataclass class InstanceConfig: index:int; dota_dir:Path; timescale:float=10; frames:int=10; difficulty:int=4
      @property server_port -> 27015+index; ws_port(team) -> 12120+2*index+(team-2); console_log -> bots/console_<index>.log
  class InstanceError(RuntimeError)
  class Instance:
      def __init__(self, cfg: InstanceConfig): ...
      async def launch(self) -> None            # raises InstanceError("port 12120 in use") if any of its 3 ports is bound; starts process; truncates console log
      async def handshake(self, timeout=90) -> dict[int,str]   # tails console log for LUARDY lines; returns {team: token} for roles controlled|valve; raises InstanceError with last 20 log lines on timeout
      async def worldstates(self, team:int) -> AsyncIterator[bytes]   # connects to ws_port(team) with 0.5 s retries for up to 120 s, yields raw messages, returns on EOF
      def alive(self) -> bool
      def stats(self) -> dict       # {"pid","cpu","rss_mb"} via psutil, zeros if dead
      def console_tail(self, n=20) -> list[str]
      def lua_stats(self) -> dict   # last LUASTAT per token: {"waits","exec","timeouts","mismatch"}
      async def kill(self) -> None  # SIGTERM then SIGKILL after 5 s; also kills children
  ```
- Launch builds the Global Constraints flag list, `cwd=dota_dir/"game"`, env with `LD_LIBRARY_PATH` and `ENABLE_PATHMATCH=1`, stdout/stderr to `runs/current/instance_<i>.out`.

- [ ] **Step 1: Write the end-to-end test** (`@pytest.mark.game`): `install_scripts`, `Channel.write_config(10, {0:"npc_dota_hero_nevermore",5:"npc_dota_hero_nevermore"}, {0:"valve",5:"valve"})`, launch index 0, `handshake()` returns tokens for teams 2 and 3, the first 30 messages from `worldstates(2)` parse with `worldstate.parse`, `stats()["rss_mb"] > 100`, then `kill()` and `alive()` is False. Second test: bind port 12120 with a plain socket, assert `launch()` raises `InstanceError` mentioning `12120` within 2 s.

- [ ] **Step 2: Run to verify it fails**

Run (in WSL2): `~/dotasim-venv/bin/python -m pytest tests/test_instance_game.py -v -m game`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Implement `instance.py`** per the Interfaces block. Console tail: poll the file every 0.2 s from the last offset (the game appends; no inotify needed).

- [ ] **Step 4: Run to verify it passes**

Expected: both tests PASS; total under 120 s.

- [ ] **Step 5: Commit** `git add dotasim/env/instance.py tests/test_instance_game.py && git commit -m "feat: dota server instance with handshake and worldstate stream"`

---

### Task 6: DotaEnv

**Files:**
- Create: `dotasim/env/env.py`
- Test: `tests/test_env_game.py` (marked `game`)

**Interfaces:**
- Consumes Task 2 `parse`, Task 4 `Channel`/`Action`, Task 5 `Instance`.
- Produces:
  ```python
  @dataclass class Matchup: heroes:dict[int,str]; roles:dict[int,str]; frames:int=10; difficulty:int=4; cap_s:float=600
  class DotaEnv:
      def __init__(self, cfg: InstanceConfig, bots_dir: Path): ...
      async def reset(self, matchup: Matchup) -> dict[int, WorldState]   # kills any running instance, writes config, launches, handshakes, waits for the first PRE_GAME worldstate per controlled team, writes calib, returns obs for controlled teams
      async def step(self, actions: dict[int, Action]) -> tuple[dict[int, WorldState], bool, dict]
          # writes one action file per controlled team (idx increments per team), awaits the next worldstate per controlled team (for valve-only matchups, awaits team 2's stream), done when game_state==6, any stream hits EOF, dota_time >= cap_s, or the process died; info = summary(ws) | {"episode_idx", "game_time", "speed_x", "lua_timeouts", "lua_mismatch", "error"}
      async def close(self) -> None
      @property status(self) -> dict   # {"index","pid","phase","game_time","speed_x","heroes":{2,3},"kills","last_hits","episode","lua_timeouts","last_error"} for the status page
  ```
- `speed_x` is dota_time advanced over wall time, over the last 30 steps.
- Phase strings: `launching`, `handshake`, `pregame`, `playing`, `postgame`, `dead`.

- [ ] **Step 1: Write the end-to-end tests** (`@pytest.mark.game`):
  - `test_random_policy_episode`: matchup roles `{0:"controlled",5:"valve"}`, `reset`, then up to 300 steps of `Action("move", hero.x+random(-400,400), hero.y+random(-400,400))` alternating with `Action("attack", name=<nearest enemy creep name>, x, y)`; assert every step returns an obs for team 2 with increasing `dota_time`, `info["lua_timeouts"] == 0` at the end, and the hero moved more than 1000 units from spawn.
  - `test_process_death_is_done`: after `reset`, `os.kill(pid, SIGKILL)`; the next `step` returns `done=True` with `"error" in info` within 5 s.
  - `test_slow_policy_is_counted`: after `reset`, sleep 1.5 s before one step; `info["lua_timeouts"] >= 1` afterwards, and the next 20 steps still return obs.

- [ ] **Step 2: Run to verify they fail**, **Step 3: Implement `env.py`**, **Step 4: Run to verify they pass**

Run: `~/dotasim-venv/bin/python -m pytest tests/test_env_game.py -v -m game`
Expected: 3 PASS, total under 5 minutes.

- [ ] **Step 5: Commit** `git add dotasim/env/env.py tests/test_env_game.py && git commit -m "feat: DotaEnv reset/step with lockstep and failure reporting"`

---

### Task 7: Rollout fleet and status writer

**Files:**
- Create: `dotasim/train/__init__.py`, `dotasim/train/rollout.py`, `dotasim/train/policy.py`
- Test: `tests/test_fleet_game.py` (marked `game`, marked `slow`)

**Interfaces:**
- Produces:
  ```python
  class Policy(Protocol): def act(self, obs: dict[int, WorldState]) -> dict[int, Action]
  class RandomPolicy(Policy)          # Task 6's move/attack alternation, per controlled team
  class ValveOnlyPolicy(Policy)       # returns {} ; for valve-vs-valve waves
  class Fleet:
      def __init__(self, n:int, matchup: Matchup, policy: Policy, run_dir: Path, timescale:float=10): ...
      async def run(self, episodes_per_instance:int) -> dict   # launches staggered 3 s apart, steps all envs concurrently (asyncio.gather per step), writes status.json every 1.0 s and ws_<i>.json for instance 0 every 0.3 s, returns {"episodes","steps","wall_s","infer_ms_mean","infer_ms_p95","lua_timeouts"}
  ```
- CLI: `python -m dotasim.train.rollout --instances 12 --episodes 1 --policy random|valve --timescale 10 --frames 10 --run runs/<name>`; prints the returned dict as JSON.
- `status.json` schema: `{"run": str, "updated": iso8601, "instances": [DotaEnv.status, ...], "totals": {"episodes","steps","games_per_hour","infer_ms_mean","lua_timeouts"}}`.
- Inference timing wraps `policy.act` only.

- [ ] **Step 1: Write the end-to-end test** (`@pytest.mark.game @pytest.mark.slow`): `Fleet(12, Matchup(heroes={0:SF,5:SF}, roles={0:"valve",5:"valve"}), ValveOnlyPolicy(), tmp_run)`, `run(1)`; assert every instance reached `postgame` or the cap, `lua_timeouts == 0`, `status.json` has 12 entries, and `games_per_hour >= 40`. Second test: 2 instances with `RandomPolicy`, 200 steps each; assert `infer_ms_p95 < 5`.

- [ ] **Step 2: Raise the WSL2 memory cap** if 12 instances exceed `free -g`: write `%USERPROFILE%\.wslconfig` with `[wsl2]\nmemory=24GB` and `wsl --shutdown` (the user confirms this step; it restarts their WSL distro).

- [ ] **Step 3: Run to verify it fails**, **Step 4: Implement**, **Step 5: Run to verify it passes**

Run: `~/dotasim-venv/bin/python -m pytest tests/test_fleet_game.py -v -m "game and slow"`
Expected: PASS; record the printed totals in `docs/experiments.md` under "Phase 1 fleet baseline".

- [ ] **Step 6: Commit** `git add dotasim/train tests/test_fleet_game.py docs/experiments.md && git commit -m "feat: rollout fleet with status writer and fleet baseline"`

---

### Task 8: Status page

**Files:**
- Create: `dotasim/ui/status.html`, `dotasim/ui/__init__.py`, `dotasim/ui/serve.py`

**Interfaces:**
- `serve.py`: `python -m dotasim.ui.serve runs/<name> --port 8765` copies `status.html` into the run dir and runs `http.server` there.
- `status.html`: no dependencies; fetches `status.json` every 1 s; renders the totals line and a table with columns index, phase, game time (m:ss), speed (x), heroes, kills, last hits, episode, Lua timeouts, last error; rows with `phase == "dead"` or `lua_timeouts > 0` get a red left border; a `<canvas>` below draws `ws_0.json` every 300 ms: map bounds -8000..8000, radiant units green, dire red, creeps 3 px, heroes 7 px with an hp arc, towers squares. Dark background `#111`, text `#ddd`, system font.

- [ ] **Step 1: Write the two files.**

- [ ] **Step 2: Verify in the browser** while Task 7's fleet test runs (or against its saved `runs/` dir): start `serve.py`, open `http://localhost:8765/status.html` in the built-in browser, confirm 12 rows, live game time, and dots on the minimap; take a screenshot for the experiment log.

- [ ] **Step 3: Commit** `git add dotasim/ui && git commit -m "feat: static fleet status page with minimap"`

---

### Task 9: Spectate from the Windows client (verification only)

- [ ] **Step 1:** With one instance running in WSL2, get its IP with `wsl.exe -d Ubuntu -- hostname -I`, open the Windows Dota 2 client console and run `connect <ip>:27015`.
- [ ] **Step 2:** Record the outcome (spectator view works / connection refused / needs a firewall or port-proxy rule) in `docs/experiments.md` under "Live view". No code changes in this task; the result decides whether phase 3 builds the minimap further or drops it.

---

### Task 10: Phase 1 exit

- [ ] **Step 1:** Run the full suite: `~/dotasim-venv/bin/python -m pytest -v -m "not slow"` then `-m slow`. Expected: all PASS.
- [ ] **Step 2:** Update `README.md` with: WSL2 setup (SteamCMD lines, `steamclient.so` link, uv venv), the three commands (fleet, serve, tests), and the phase 1 baseline numbers.
- [ ] **Step 3:** Commit and tag: `git commit -am "docs: phase 1 exit" && git tag phase1`.
