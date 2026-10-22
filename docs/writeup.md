# dotasim: teaching a Dota 2 1v1 bot on one desktop (draft)

*Status: draft written during phase 2/3 training; numbers marked TBD are filled from `docs/experiments.md` as runs finish.*

## Why

Robin Kahlow's WarcraftSim showed that a single desktop can train a competitive Warcraft III agent against
the real game if you remove enough waste. Dota 2 is a harder game but a friendlier target for the harness:
Valve ships a bot scripting API, a headless dedicated server, and a timescale cvar. This project is that
idea applied to Dota 2's 1v1 Solo Mid mode, starting with the Shadow Fiend mirror.

## The harness (phase 1)

- The game runs headless under WSL2 at 10x real time. One instance costs about half a core and 1 GB.
- The server streams a protobuf world state over TCP every N ticks. Python answers with one small Lua file
  per decision; the bot's Lua `Think` blocks until that file exists. That block is the virtual clock:
  the game cannot advance until the policy has answered, which is what WarcraftSim needed a DLL for.
- A bot whose Lua script defines no `Think` keeps Valve's AI, so the opponent for early training is free.
- Twelve servers at once: about 62x real time aggregate with replay recording on.
- Things that did not work: spectating a WSL2 server from the Windows client (rejected for lacking a
  lobby, then on the SourceTV relay password). Replays recorded by the server play fine in the client.

## Imitation (phase 2)

- 23 recorded Valve-vs-Valve games, both views, 180k decision steps at 0.167 s.
- Valve's worldstate has no last hits, gold or net worth. A filler bot on each team acts as an observer
  and prints them for its allies, so the data carries what the hero token needs.
- Labels come from each unit's current order type, its attack target handle, the active ability, and the
  displacement to the next step. A finished deny shows up as an attack order on a dead creep, which the
  label check caught on the first run.
- Model: 48 unit tokens through a 4-layer transformer (width 128), a GRU, and masked heads for action
  type, move direction, move distance and attack target. About 1M parameters, 0.6 ms per instance batched.
- Result: TBD (type accuracy, target accuracy, last hits vs passive bots at 5 minutes, with the random
  baseline and the Valve bot's own number as the ceiling).

## Self-play (phase 3)

- Potential-based shaping on last hits, denies, level, net worth and hp difference; +1/-1 on a win or loss;
  gamma 0.9995 so the terminal signal survives a 3,600-step episode.
- PPO over 32-step GRU windows with stored initial states, a KL penalty to the cloning checkpoint, a
  league of past checkpoints and Valve's unfair bots.
- Result: TBD.

## What I would tell someone starting this

1. Verify the mechanism on the real build before designing around docs. The order enum, the file-loading
   functions available in the bot VM, the worldstate cadence and the socket's drop-on-stall behavior all
   differed from what the 2018 projects described.
2. Budget memory, not CPU, under WSL2. Twelve servers fit in 16 GB with little to spare.
3. Keep the harness boring: files and sockets, no gRPC, no containers. The whole env layer is under 700 lines.
