# dotasim

Self-play reinforcement learning for Dota 2 1v1 (Solo Mid) on one desktop, in the spirit of
[WarcraftSim](https://warlock.ai/blog/warcraftsim-1/). The real game runs headless and faster than
real time; a Python harness drives it in lockstep through Valve's bot scripting API.

Design: `docs/superpowers/specs/2026-10-22-dotasim-design.md`. Current phase plan: `docs/plans/`.
Experiment log: `docs/experiments.md`. Spike notes: `docs/2026-10-22-spike-headless-harness.md`.

Phase 1 (done): harness and fleet page. 12 headless servers in WSL2 at 10x real time, 0 lockstep
timeouts, about 72x real time aggregate, 1 GB and half a core per server.

## Setup (Windows host, WSL2 Ubuntu)

1. Dota 2 Linux build, once, with your own Steam account (Dota 2 is free; the download is 73 GB):

```bash
sudo dpkg --add-architecture i386 && sudo apt update && sudo apt install -y lib32gcc-s1 lib32stdc++6
mkdir -p ~/steamcmd && cd ~/steamcmd && curl -sqL https://steamcdn-a.akamaihd.net/client/installer/steamcmd_linux.tar.gz | tar zxvf -
./steamcmd.sh +@sSteamCmdForcePlatformType linux +force_install_dir $HOME/dota2 +login YOUR_STEAM_USERNAME +app_update 570 validate +quit
mkdir -p ~/.steam/sdk64 && ln -sf ~/steamcmd/linux64/steamclient.so ~/.steam/sdk64/steamclient.so
```

2. Python (uv, 3.12, venv at `~/dotasim-venv`):

```bash
bash scripts/setup_wsl.sh
```

Everything else runs through `scripts/wsl.sh`, which puts the venv on PATH and cds to the repo.
From Windows: `wsl.exe -d Ubuntu -- bash /mnt/c/.../dotasim/scripts/wsl.sh <command>`.

## Commands

Run a fleet (Valve bots vs Valve bots, one 10-minute game per instance):

```bash
python -m dotasim.train.rollout --instances 12 --episodes 1 --policy valve --run runs/demo
```

Serve the status page for that run, then open http://localhost:8765/status.html:

```bash
python -m dotasim.ui.serve runs/demo --port 8765
```

Tests (game tests need the install; `slow` is the 12-instance run):

```bash
python -m pytest -m "not slow"
python -m pytest -m slow
```

Replays: every instance records SourceTV to `~/dota2/game/dota/replays/`. Copy a `.dem` into the
Windows client's `game/dota/replays/` and launch the client with `+playdemo replays/<file>.dem`.

## How it works

- `dotasim/env/instance.py` launches one dedicated server with unique ports and a per-instance
  console log, and reads the Lua handshake and stats from that log.
- The game streams `CMsgBotWorldState` over TCP; `dotasim/env/worldstate.py` types it.
- `dotasim/env/lua/core.lua` runs inside the game. The controlled bot blocks each decision tick
  until Python has written `bots/actions_<token>_<idx>.lua`, which makes the game wait for the
  policy (the virtual clock). Bots in the `valve` role keep Valve's AI; fillers idle.
- `dotasim/env/env.py` exposes `DotaEnv.reset(matchup)` and `step(actions)`.
- `dotasim/train/rollout.py` runs N envs and writes `status.json` for `dotasim/ui/status.html`.

Not affiliated with Valve. The repository contains no game files.
