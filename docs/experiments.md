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
