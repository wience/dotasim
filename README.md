# dotasim

Self-play reinforcement learning for Dota 2 1v1 (Solo Mid) on one desktop, in the spirit of
[WarcraftSim](https://warlock.ai/blog/warcraftsim-1/). The real game runs headless and faster than
real time; a Python harness drives it in lockstep through Valve's bot scripting API.

Design: `docs/superpowers/specs/2026-10-22-dotasim-design.md`. Current phase plan: `docs/plans/`.
Experiment log: `docs/experiments.md`. Spike notes: `docs/2026-10-22-spike-headless-harness.md`.

Phase 1 (done): harness and fleet page. Phase 2 (built; imitation exit not met) and phase 3 (self-play loop built and tested; training barely started) are on `main` too; see `docs/experiments.md` for the honest numbers. 12 headless servers in WSL2 at 10x real time, 0 lockstep
timeouts, about 62x real time aggregate with replay recording on (72x without), 1 GB and half a
core per server.

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

Replays: every instance records SourceTV to `~/dota2/game/dota/replays/` (newest 30 kept). Copy a `.dem`
into the Windows client's `game/dota/replays/` and launch the client with `+playdemo replays/<file>.dem`.

## Phase 2: imitation

Record Valve-vs-Valve games (both views, observer stats), train the cloning model, evaluate vs passive bots:

```bash
python -m dotasim.data.record --games 24 --instances 8 --frames 5 --difficulty 4 --out data/sf_mirror_v1
```

```bash
python -m dotasim.train.bc --data data/sf_mirror_v1 --epochs 8 --batch 16 --out runs/bc_v2
```

```bash
python -m dotasim.eval.eval --checkpoint runs/bc_v2/policy.pt --games 3 --instances 3 --difficulty 0 --cap 300
```

TensorBoard: `tensorboard --logdir runs` then http://localhost:6006.

## Dashboard

`web/index.html` is one static page: KPI cards, learning curves (training last hits, both evaluations, outcomes per
wave, KL and entropy, throughput), the training configuration, a card per server and a detail drawer with both heroes'
stats and a live minimap.

- Local, live at 1 Hz: `python -m dotasim.ui.serve runs/<run>` then http://localhost:8765/ (the plain fleet table is
  at /status.html).
- Hosted: https://dotasimstatus.wience.tech (Vercel, deploys from `web/` on every push to `main`). It reads
  `live.json` from the `live` branch of this repo, which `python scripts/publish_status.py runs/<run>` rewrites in
  place every minute from the training machine (fleet status, config, minimaps, the iteration log).

## Phase 3: self-play

```bash
python -m dotasim.train.selfplay --init runs/bc_v2/policy.pt --instances 12 --iterations 300 --cap 300 --out runs/sp_v9 --eval-every 5 --device cuda --learner-python ~/dotasim-rocm-venv/bin/python
```

The two flags run the PPO update on the GPU in a separate process (`scripts/setup_rocm_wsl.sh` installs ROCm 7.2 and
AMD's PyTorch wheels into `~/dotasim-rocm-venv`); drop them for CPU only. Rollouts always run on the CPU.

Each iteration plays one episode per server, runs one PPO update, and logs a JSON line to `runs/<run>/log.jsonl`.
The opponent mix is `--p-pool` (past checkpoints, default 0.2) and `--p-valve` (Valve bots at `--difficulty`, default
0); the rest is self-play. The curriculum runs stage 1 as `--p-valve 1 --difficulty 0` (farm vs passive bots, gate:
20 last hits per side), stage 2 vs easy to unfair, stage 3 self-play with a Valve share. `--overlap` runs the GPU
update while the next wave plays (one update of policy lag). Both heroes see the map in the Radiant frame: Dire's
world is rotated 180 degrees about the mid towers (`canon_sign`), so one network plays both sides.

## How it works

- `dotasim/env/instance.py` launches one dedicated server with unique ports and a per-instance
  console log, and reads the Lua handshake and stats from that log.
- The game streams `CMsgBotWorldState` over TCP; `dotasim/env/worldstate.py` types it.
- `dotasim/env/lua/core.lua` runs inside the game. The controlled bot blocks each decision tick
  until Python has written `bots/actions_<token>_<idx>.lua`, which makes the game wait for the
  policy (the virtual clock). Bots in the `valve` role keep Valve's AI; fillers idle.
- `dotasim/env/env.py` exposes `DotaEnv.reset(matchup)` and `step(actions)`.
- `dotasim/train/rollout.py` runs N envs and writes `status.json` for `dotasim/ui/status.html`.

## License

MIT. Not affiliated with or endorsed by Valve. The repository contains no game files; you need
your own copy of Dota 2. Dota 2 is a trademark of Valve Corporation.
