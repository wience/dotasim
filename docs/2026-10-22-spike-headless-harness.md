# Spike: headless Dota 2 harness (2026-10-22)

Throwaway probe before writing the spec. Question: can the current Dota 2 build run headless,
stream world state to Python, take actions back from Python in lockstep, and run faster than real time?
Everything under `spike/` is throwaway reference material, not project code.

Build tested: Dota 2 client 6965, both the Windows Steam install and the Linux build inside WSL2 (Ubuntu 26.04), 1v1 Solo Mid (game mode 21).

## Pass criteria

| # | Criterion | Result |
|---|-----------|--------|
| 1 | Linux dedicated server starts headless in WSL2 | **Pass.** Linux build 6965 downloaded with SteamCMD (73 GB). Ran a complete 1v1 game to POST_GAME at 10x with the same flags, 1,940 worldstates per team, 1,850 lockstep steps, 0 timeouts, 27 last hits. No Steam client, no X server, no Steam runtime container needed. |
| 2 | Worldstate arrives on TCP and parses against the pinned proto | **Pass.** 1,700 messages per team per minute, 0 parse failures. |
| 3 | Lua bot executes an action supplied from Python | **Pass.** 4,450 lockstep steps at a 5-tick interval, 0 timeouts. Move orders walked the hero from the fountain to the river; attack orders on nearby creeps gave 32 last hits and two levels in 11 game-minutes. |
| 4 | host_timescale >= 5 holds | **Pass.** 5 gives 5.0x, 10 gives 10.0x, 20 gives 10.0x with or without lockstep, so 10x is an engine cap, not the file round-trip. |
| 5 | CPU and RAM per instance recorded | **Pass.** See numbers. |

## Numbers (one headless instance, 10 bots on the map, 2 controlled, steady state after map load)

| host_timescale | frames per worldstate | lockstep | achieved speed | msgs/s per team | dota2.exe CPU (100 = one core) | private bytes |
|---|---|---|---|---|---|---|
| 5 | 10 | yes | 5.0x | 15.0 | ~27% avg, 44% max | not sampled |
| 10 | 10 | yes | 10.0x | 30.0 | ~55% avg, 70% max | not sampled |
| 10 | 5 | yes | 10.0x | 57.1 | ~53% avg, 77% max | ~930 MB |
| 10 | 10 | no | 10.0x | 30.0 | ~43% avg, 56% max | ~840 MB |
| 20 | 10 | yes | 10.0x | 30.0 | ~53% avg | not sampled |
| 20 | 10 | no | 10.0x | 30.0 | ~50% avg, 75% max | ~910 MB |
| 10 (Linux, WSL2) | 10 | yes | 10.0x | 30.0 | ~51% avg, 127% max | ~965 MB (USS) |

- Cadence follows `-botworldstatetosocket_frames`: 10 frames gives one message per 0.333 s of game time, 5 frames gives one per ~0.175 s (57 msgs/s at 10x, i.e. about 5.25 ticks; the string is no longer in server.dll but the flag works). The decision interval is a parameter.
- Lockstep adds no measurable throughput cost when Python answers within a tick; the engine caps at 10x either way.
- Memory: the working set (RSS) sits around 210 to 260 MB because Windows trims idle pages, but private bytes are 0.84 to 0.93 GB per instance, with a ~1 GB spike during map load. Budget 1 GB per instance.
- Message size 35 to 60 KB, 76 to 130 units.
- Rough budget for the 7800X3D (8 cores, 32 GB): 8 to 12 instances at 10x, ~100x real time aggregate, before the dual-boot. WSL2's default 16 GB allotment caps that at ~12 by memory.
- The Lua wait is a busy loop, so an instance burns its core while the policy is thinking; keep inference fast or sleep-poll.

## How it works

### Launch (Windows; Steam must be running, the gameserver logs on to Steam)

```
dota2.exe -dedicated -insecure -console -condebug -novid -nosound -nowatchdog -dev
  +dota_force_gamemode 21 +dota_1v1_skip_strategy 1 +host_timescale 10
  +sv_lan 1 +sv_cheats 1 +sv_hibernate_when_empty 0
  +map start gamemode 21 -fill_with_bots
  -botworldstatetosocket_radiant 12120 -botworldstatetosocket_dire 12121
  -botworldstatetosocket_frames 10 -botworldstatesocket_threaded
  +dota_auto_surrender_all_disconnected_timeout 180
```

- `-nowatchdog` is required: the watchdog kills the game when a Lua Think blocks for more than a few seconds.
- `+dota_force_gamemode 21` together with `+map start gamemode 21` as separate tokens gives mode 21 (`GetGameMode()` reports 21, 1v1 rules apply). They were not isolated from each other. `+map "dota gamemode 21"` as one token fails to load any map.
- Cvars go before `+map` so they apply to the map load.
- `-condebug` writes `game/dota/console.log`; Lua `print` shows up there as `[VScript] ...`.

### Worldstate (game -> Python)

- The game **hosts** TCP ports 12120 (radiant) and 12121 (dire). Python connects (retry while the game boots). One client per port.
- Framing: 4-byte little-endian length, then a `CMsgBotWorldState` protobuf.
- Proto: `Protobufs/dota_gcmessages_common_bot_script.proto` + `valveextensions.proto` from SteamDatabase/GameTracking-Dota2.
  Strip `boxed_type = "..."`, `synthetic_default = "..."` and `option additional_includes` before protoc; standard protoc rejects them.
- Stream starts in TEAM_SHOWCASE at dota_time about -15.8, then resets to -90 at PRE_GAME. Calibrate on the first PRE_GAME message.

### Actions (Python -> Lua)

- Scripts live in `game/dota/scripts/vscripts/bots/`. `bot_<heroname>.lua` with a global `Think()` is a complete takeover: the default AI is off, Think runs every tick (verified with `+dota_bot_long_frame_limit 0.001`, which logs every call).
- `hero_selection.lua` picks heroes with `SelectHero(pid, name)` inside its `Think()`.
- File channel: Python writes `bots/actions_<team>_<idx>.lua` containing `return {idx=..., kind=..., ...}`.
  Lua loads it with `package.loaded[mod] = nil; require("bots/actions_<team>_<idx>")`.
  `loadfile` returns a function that yields nothing, `dofile` wants the name without `.lua`, `io` and `os` do not exist in the bot VM.
- One file per index. Rewriting one file while the game reads it raises `PermissionError` on Windows.
- Lockstep: Python writes `calib_<team>.lua` with the dota_time of the first PRE_GAME worldstate (index 0). Lua computes
  `m = round((DotaTime() - calib_t) * 30)` each tick and blocks on `m % 10 == 1` for action index `(m - 1) / 10`,
  with a 1 s real-time timeout as a safety net. The game cannot advance while Lua blocks, which is the virtual clock.

### Gotchas

- `-fill_with_bots` seats 10 bots even in mode 21 (`GetGameMode()` reports 21, 1v1 rules apply). Control the lowest player id per team; the other 8 idle at the fountain.
- Scripts in `vscripts/bots/` override the bots in the owner's own local lobbies. Remove the folder when not training.
- Only `Action_MoveToLocation` and `Action_AttackUnit` (target from `GetNearbyCreeps`) were exercised. Ability casts, items and courier orders are unverified.
- **Open for the spec:** the worldstate identifies units by `handle`, but bot-API unit objects expose no handle or entity index (`GetHandle`, `GetUnitHandle`, `GetEntityIndex`, `GetIndex` are all nil; `GetUnitName` and `GetPlayerID` work). A target-unit head has to be resolved in Lua by matching name plus location against `GetNearbyCreeps` / `GetNearbyHeroes` / `GetUnitList`, or by player id for heroes.
- Dota 2 is not deterministic across runs. Record replays with SourceTV (`+tv_enable 1 +tv_autorecord 1`) for the writeup.
- The old dotaservice harness (2018) still describes the right mechanism but its file-polling, Python 3.7 and gRPC layers are not needed.

## Linux (WSL2) recipe, verified

Done once on this machine (Ubuntu 26.04 under WSL2, user `wince`):

```
sudo dpkg --add-architecture i386 && sudo apt update && sudo apt install -y lib32gcc-s1 lib32stdc++6
mkdir -p ~/steamcmd && cd ~/steamcmd && curl -sqL https://steamcdn-a.akamaihd.net/client/installer/steamcmd_linux.tar.gz | tar zxvf -
./steamcmd.sh +@sSteamCmdForcePlatformType linux +force_install_dir /home/wince/dota2 +login <steam_username> +app_update 570 validate +quit
mkdir -p ~/.steam/sdk64 && ln -sf ~/steamcmd/linux64/steamclient.so ~/.steam/sdk64/steamclient.so
```

- Pass the username on the command line. Typing it at a `read` prompt through `wsl.exe` delivered an empty name to SteamCMD (Windows CRLF through the pipe).
- The download is 73 GB on disk (`~/dota2`).
- Without the `steamclient.so` link the server dies at map load with `FATAL ERROR: Failed to initialize Steamworks SDK for gameserver`. SteamCMD's copy works; no Steam client is needed.
- Do not use `dota.sh`: it refuses to run outside Valve's "sniper" container. Launch the binary directly from `~/dota2/game`:

```
cd ~/dota2/game
LD_LIBRARY_PATH=$PWD/bin/linuxsteamrt64 ENABLE_PATHMATCH=1 bin/linuxsteamrt64/dota2 <same flags as Windows>
```

- Console log lands at `~/dota2/game/dota/console.log`, Lua scripts at `~/dota2/game/dota/scripts/vscripts/bots/`, same as Windows.
- WSL2's Ubuntu ships Python 3.14 without pip or ensurepip; `python3 -m venv --without-pip ~/venv` plus `get-pip.py` gives a working venv without sudo.
- `spike/run_spike_linux.py` is the Windows runner with only the paths, env and memory sampling changed.

## Design implications for the spec

1. Harness is ~150 lines of Python plus two Lua files. No gRPC, no Docker needed on Windows; a container is optional on Linux.
2. Decision interval is a launch parameter (frames per worldstate); the model sees one worldstate per decision. Measure 5 vs 10 frames on last-hit rate before fixing it.
3. Instance manager: unique ports per instance, Steam running, one `bots/` folder shared, per-instance action file prefix.
4. The observation carries what WarcraftSim's unit tokens needed (per-unit type, position, health, mana, abilities, cooldowns, items, player stats, damage and ability events). Mapping a chosen target back to a Lua unit object is the one open plumbing question (see Gotchas).
