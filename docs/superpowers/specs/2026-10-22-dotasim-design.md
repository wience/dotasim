# dotasim: self-play RL for Dota 2 1v1 on one desktop

Status: approved design, 2026-10-22. Supersedes nothing. Spike evidence in `docs/2026-10-22-spike-headless-harness.md`.

## Goal

Train an agent that plays Dota 2 1v1 Solo Mid (game mode 21) from the real game, on one desktop, through self-play reinforcement learning, and publish the result as an open repo plus a writeup with replays and an experiment log. Start with the Shadow Fiend mirror, extend to a Shadow Fiend / Puck / Queen of Pain pool. Beating humans is the north star, not a milestone.

Measurable target for the first result: win rate against Valve's built-in bots at each difficulty in the SF mirror, with 50% against "unfair" as the phase 3 exit.

## Constraints and assumptions

- Hardware: Ryzen 7 7800X3D (8 cores), 32 GB RAM, RX 9070 XT. The RTX 4080 machine is unavailable.
- Platform: Linux Dota 2 build inside WSL2 Ubuntu is the training platform (`/home/wince/dota2`, downloaded with SteamCMD). Windows is for debugging and spectating. The Steam install under Program Files is never modified again.
- Training on CPU first. PyTorch ROCm 7.2 on gfx1201 is an optimisation attempted after phase 1; the 10.1 wheels hang under WSL2.
- Python 3.12 via uv (torch has no 3.14 wheels). Lua 5.1 bot scripts. protobuf with the native backend. TensorBoard. Standard library HTTP server. No gRPC, no Docker, no web framework.
- Dota 2 patches may change the worldstate proto. The proto files are pinned by commit of SteamDatabase/GameTracking-Dota2 and re-pinned on each patch.
- Open Hyper AI as an opponent is deferred: its scripts share the same `bots/` folder as ours and would collide.

## Verified facts the design relies on

From the spike, Dota 2 build 6965, Windows and Linux:

- `-dedicated` runs headless. Game mode 21 applies with `+dota_force_gamemode 21` together with `+map start gamemode 21` (separate tokens).
- `-fill_with_bots` seats 10 bots even in 1v1.
- The game hosts one TCP port per team and streams `CMsgBotWorldState` with a 4-byte little-endian length prefix, one message every `-botworldstatetosocket_frames` server ticks (30 ticks/s). 10 frames gives 0.333 s between decisions; 5 frames gives 0.175 s.
- `bot_<heroname>.lua` defining a global `Think()` takes full control of that bot and runs every tick. A bot whose script defines no `Think` falls back to Valve's default AI. A no-op `Think` idles the bot.
- Lua loads files only through `require("bots/<name>")` with `package.loaded` cleared first; `io` and `os` are absent. Rewriting a file the game is reading fails on Windows, so action files are one per index.
- A Lua `Think` that busy-waits blocks the whole server. With `-nowatchdog` this is the lockstep clock: the game advances only when Python has answered.
- `-con_logfile <path>` sends all console output, including Lua prints, to that file instead of the shared `console.log`.
- `host_timescale` caps at 10x. One instance at 10x: about 0.5 core, about 1 GB of private memory, with a 1 GB spike at map load.
- Worldstate units carry `action_type`, `attack_target_handle`, `attack_target_name`, `ability_target_handle`, `ability_target_name`, `is_using_ability`. Bot-API unit objects expose no handle, so targets are resolved in Lua by name and nearest position.
- On Linux, launch the binary directly from `~/dota2/game` with `LD_LIBRARY_PATH=$PWD/bin/linuxsteamrt64 ENABLE_PATHMATCH=1`; `dota.sh` refuses to run outside Valve's container. `~/.steam/sdk64/steamclient.so` must link to SteamCMD's copy. No Steam client is needed.
- Native protobuf parses a 46 KB worldstate in 0.1 ms.

## Architecture

```
dotasim/
  env/
    instance.py     launch, monitor, kill one server; port allocation; console log tail; token handshake
    worldstate.py   proto -> typed WorldState (units, players, events) with the reductions the UI needs
    channel.py      action file writer (one file per index), calibration file, config file
    env.py          DotaEnv: reset(matchup) / step(actions_by_team) -> obs_by_team, done, info
    lua/
      hero_selection.lua   picks heroes per config
      bot_generic.lua      role dispatch (controlled / valve / filler), token handshake, lockstep loop
      macro.lua            item build, ability leveling, courier (scripted)
      heroes/              per-hero tables: build order, skill order, ability slot metadata
  model/
    features.py     WorldState -> tensors + masks (unit tokens, hero token, events)
    net.py          unit-token transformer + GRU + heads
  train/
    rollout.py      fleet of N DotaEnv, policy inference, trajectory buffer, status.json writer
    bc.py           behavioral cloning from recorded Valve-AI games
    ppo.py          PPO + GAE + KL anchor
    league.py       opponent sampling and checkpoint pool
    reward.py       terminal + potential-based shaping
  eval/
    eval.py         fixed matchup suite, win rates, replay collection
  ui/
    status.html     fleet table + minimap canvas, reads runs/<run>/status.json and ws_<i>.json
docs/
  superpowers/specs/   this file
  plans/               implementation plans per phase
  experiments.md       one entry per run
runs/<run>/            tensorboard events, checkpoints, status.json, ws_<i>.json, replays
```

### Instance and identity

One instance is one dedicated server process with server port `27015+i`, worldstate ports `12120+2i` and `12121+2i`, and console log `bots/console_<i>.log` via `-con_logfile`. All instances share the single `bots/` folder inside `~/dota2`.

Because Lua cannot read ports or convars, each bot script mints a token at load time (`RealTime()` plus the address of a fresh table) and prints `LUARDY {token, team, player_id, role}`. The instance's Python owner tails that instance's console log, learns the tokens for its two teams, and names the files for that instance `actions_<token>_<idx>.lua` and `calib_<token>.lua`. Stale action files older than five indices are deleted by the writer.

### Roles

Lua cannot know which instance it belongs to before the token handshake, so per-instance config files are impossible. Instead a single `bots/config.lua` is written per launch wave and read once at hero selection and once at bot load. It holds, per team, the player id that is controlled, the player id that plays as Valve's AI (or none), the Valve difficulty, and the hero picks. Everything else is a filler. Roles at script load:

- controlled: defines `Think` with the lockstep loop.
- valve: returns before defining `Think`, so Valve's default AI drives it.
- filler: defines a no-op `Think`, so it idles at the fountain.

Because the config file is shared, all instances in a wave run the same matchup kind. Mixed waves (some self-play, some vs Valve) are a later extension and are not needed for phases 1 to 3.

### Lockstep

Python writes `calib_<token>.lua` with the `dota_time` of the first PRE_GAME worldstate (index 0). Lua computes `m = round((DotaTime() - calib_t) / (1/30))` each tick and on `m % N == 1` blocks until `actions_<token>_<(m-1)/N>.lua` exists, with a 1 s real-time timeout that is counted and reported. N is the frames flag and is passed in the calibration file. Python answers each worldstate with one action file per controlled team. Every action carries the index it answers.

### Episode

launch -> hero selection (from config) -> PRE_GAME calibration -> step loop -> POST_GAME, or a cap of 10 game minutes -> quit -> relaunch. Relaunch costs about 20 s of wall time and is accepted for phases 1 to 3. The rcon path (`-usercon`, `restart`) is the optimisation if relaunch exceeds 25% of wall time.

### DotaEnv

```
env = DotaEnv(instance_id, config)       # owns one server
obs = env.reset(matchup)                 # matchup: heroes per team, roles, difficulty, frames
obs, done, info = env.step(actions)      # actions: {team: Action}; obs: {team: WorldState}
```

`Action` is one of none, move(dx, dy), attack(target name, target x, y), cast(slot, target or point). `info` carries kills, last hits, denies, xp, gold, hero hp, tower hp per team, game time, speed, and the Lua timeout counter. Both teams are controlled in self-play; one in vs-Valve.

### Macro (scripted, Lua)

Fixed item build per hero, fixed skill order per hero, courier delivery automated. All three live in `macro.lua` and `heroes/<hero>.lua`. They are not learned in phases 1 to 4.

## Observation and model

Features per decision, for the controlled hero's team:

- Up to 48 nearest visible units: type class (hero, lane creep, tower, other), team, position relative to the hero, distance, facing, hp fraction, hp, mana fraction, level, attack damage, is alive, last-hittable flag (hp below the hero's attack damage), attack target name equals our hero.
- Hero token: hero id embedding, level, hp, mana, gold, xp, game time, each ability's level, cooldown, mana cost, castable flag, item slots by id.
- Events since last step: damage dealt, damage taken, last hits, denies, kills.

Network, about 3M parameters: unit MLP embedding to width 128; 4 transformer encoder layers, 4 heads; mean-pool plus hero token into a GRU with hidden 256 carried across steps; heads:

- action type: none, move, attack, ability 1 to 4 (masked by castable and alive).
- move target: 16 directions x 4 distances relative to the hero.
- target unit: attention over the 48 unit tokens, masked to valid targets for the chosen type.
- ability point: the same polar grid.
- value: fed both teams' features during training (hidden information), unused at play time.

Per-hero ability slot metadata (targets unit, point, or none) lives in `heroes/<hero>.lua` and in `features.py` so masks agree on both sides. Lua resolves a target unit by name and nearest position from `GetNearbyCreeps`, `GetNearbyHeroes`, `GetNearbyTowers`; a mismatch counter is logged and reported in `info`.

## Training

### Behavioral cloning (phase 2)

Record Valve-AI-vs-Valve-AI 1v1 games with both heroes in the valve role. Labels come from the worldstate: `action_type`, attack and ability targets, and position deltas for moves. Cross-entropy per head, masked. Exit: the agent lanes and takes last hits against passive bots.

### Self-play PPO (phase 3)

12 instances in WSL2 (memory bound: raise the WSL2 cap to 24 GB in `.wslconfig`). PPO with GAE, clipped objective, entropy bonus, and a KL penalty to a frozen anchor refreshed every K updates. League sampling per episode: 70% latest policy, 20% uniformly from past checkpoints, 10% scripted (Valve unfair). Reward: +1 win, -1 loss; potential-based shaping on last hits, denies, xp, gold, hero hp difference and tower hp difference, so loops earn nothing; a capped episode receives shaping only. Discount tuned for 10-minute episodes at the chosen decision interval.

Inference: measured in phase 1 whether one forward pass per instance per decision fits in the wall budget (33 ms at 10 frames and 10x) or whether instances must be batched. The design allows both: rollout owns all envs and can gather decisions before one forward pass.

### Curriculum

SF mirror vs Valve passive, easy, medium, hard, unfair (phase 2 to 3), then self-play with the league (phase 3), then the Puck and QoP pool with mixed matchups (phase 4).

## Evaluation

Fixed suite: 100 games against each Valve difficulty with held-out seeds, SF mirror first. Metrics: win rate, last hits at 5 minutes, kills, first tower time. SourceTV replay per eval game (`+tv_enable 1 +tv_autorecord 1`); notable replays kept for the writeup. `docs/experiments.md` gets one entry per run: config, numbers, what changed, verdict.

## Interface

- TensorBoard (phase 1 onward): win rate per opponent, reward components, PPO losses, KL, steps per second, games per hour, instances alive, inference ms, Lua timeouts, target mismatches; hparams per run.
- Fleet page (phase 1): `rollout.py` writes `runs/<run>/status.json` once a second with, per instance: id, pid, phase, game time, speed, heroes, kills, last hits, episode number, last error. `ui/status.html` is one static file served by `python -m http.server` from `runs/<run>/` and renders the table. No buttons; the terminal kills and restarts.
- Live view (phase 3): first try the Windows Dota client connecting to the WSL2 LAN server as a spectator; that gives the real UI for free. If it fails, `rollout.py` also writes `ws_<i>.json` for the selected instance (units reduced to x, y, team, type, hp fraction, plus hero stats) and `status.html` draws a canvas minimap polled every 300 ms. A Windows instance launched without `-dedicated` gives a rendered game for debugging at any time.

## Phases

Only phase 1 gets an implementation plan now. Each later phase gets its own plan when the previous exit test passes.

1. Harness and fleet page. Exit test: 12 instances in WSL2 at 10x, Valve-vs-Valve 1v1 games run end to end, status page live, zero lockstep timeouts over one episode per instance, inference budget measured with a random policy.
2. Features, model, behavioral cloning. Exit: lanes and takes last hits against passive bots.
3. Self-play PPO and live view. Exit: 50% win rate vs unfair in the SF mirror; curves in TensorBoard; experiment log started.
4. Hero pool: Puck and QoP, mixed matchups.
5. Writeup: repo README, replays, experiment log, blog post.

## Testing

Three kinds only:

- End to end: launch one instance with a random policy, run 300 steps, assert worldstates arrive for each controlled team, the Lua exec counter advances, timeouts are zero, and the episode reaches POST_GAME or the cap and the server exits cleanly.
- Integration: parse a recorded worldstate stream (golden file from the spike) through `worldstate.py` and `features.py`; shapes, masks and reductions match stored values.
- Golden: `reward.py` on a recorded episode trace equals stored values.

## Risks

- Token handshake and shared config are new. Fallback: one game directory per instance via symlinks, or containers.
- Relaunch overhead of about 20 s per episode. Fallback: rcon `restart`.
- WSL2 memory cap. Fix: `.wslconfig` memory=24GB.
- ROCm on gfx1201 under WSL2 is fragile; CPU remains the primary path.
- Valve patches can change the proto and the bot API. Pin the proto commit; keep the end-to-end test as the canary.
- Dota 2 is not deterministic; experiments compare distributions, not single games.
