# Experiment log

One entry per run. Newest at the bottom. Numbers come from the run's `status.json` or the CLI's printed totals.

## 2026-10-22 Phase 1 fleet baseline

Setup: 12 headless Linux servers in WSL2 (Ubuntu 26.04, 16 GB allotted), Dota 2 build 6965, mode 21, Shadow Fiend mirror, both heroes Valve default AI (difficulty cvar 4), host_timescale 10, 10 frames per worldstate, 600 game-second cap, no policy inference.

Command: `python -m dotasim.train.rollout --instances 12 --episodes 1 --policy valve --run runs/phase1-baseline`

| metric | value |
|---|---|
| episodes | 12 (all reached the 600 s cap) |
| worldstate steps | 24,421 |
| wall time | 112.8 s (includes staggered launch, 3 s apart) |
| games per hour | 383 |
| aggregate speed | about 72x real time (24,421 steps x 0.333 s / 112.8 s) |
| per instance | about 6 to 8x real time while all 12 run; 10x alone |
| Lua timeouts | 0 |
| errors | 0 |

Observations:
- A Valve-vs-Valve SF mirror rarely ends inside 10 game minutes; in a 30-minute run it ended 1-1 at exactly 30:00, which looks like a mode 21 time limit.
- At POST_GAME the server stops streaming worldstates and keeps the socket open for a few seconds before closing. The env detects the end from the console log.
- Memory: 12 instances fit in the default 16 GB WSL2 allotment with about 2 GB spare after map loads. No `.wslconfig` change was needed.
- Random-policy inference (two instances, 200 steps each): p95 under 0.01 ms. The per-decision budget at 10x and 10 frames is 33 ms, so a 3M-parameter CPU forward pass has room; batching across instances stays deferred.

Verdict: harness exit criteria met. Decision interval, batching and a real model are phase 2 questions.

![Fleet status page during a two-instance random-policy run](img/phase1-status-page.jpg)

## 2026-10-22 Live view: spectating a WSL2 server from the Windows client

Goal: watch a running training game with the real client instead of the canvas minimap.

| attempt | server flags | client | result |
|---|---|---|---|
| direct connect | `+sv_lan 1`, no lobby | `+connect 172.21.211.69:27015` | handshake completes, server rejects with `NETWORK_DISCONNECT_REJECT_NOLOBBY` |
| SourceTV | `+tv_enable 1 +tv_port 28000 +tv_delay 0` | `+connect 172.21.211.69:28000` | connects, reaches level load, dropped with `NETWORK_DISCONNECT_BADRELAYPASSWORD` |
| SourceTV + password | above plus `+tv_password dotasim +tv_relaypassword dotasim` | `+password dotasim +connect ...:28000` | connects, dropped with `NETWORK_DISCONNECT_INVALID` |

Networking between the Windows client and the WSL2 server works (challenge and connect exchanged on UDP 27015 and 28000, no firewall change needed). The rejection is the game's lobby and TV policy, not the network. Time-boxed to three attempts; live spectating is deferred. The canvas minimap on the status page stays the live view, and a Windows instance launched without `-dedicated` is the rendered debugging option.

SourceTV auto-recording works on the dedicated server: `~/dota2/game/dota/replays/auto-<date>-start-Dota_2.dem` (about 7 MB for 2 minutes at 10x). The server keeps `+tv_enable 1 +tv_autorecord 1 +tv_delay 0 +tv_port 28000+i`; the TV password flags were dropped.

Replay playback verified: copying the server's `.dem` into the Windows client's `game/dota/replays/` and launching with `+playdemo replays/<file>.dem` plays it with the real UI (reached PRE_GAME in the client log). That is the "watch what it learned" path for the writeup.
