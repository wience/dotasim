# Phase 2: Features, Model and Behavioral Cloning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record Valve-vs-Valve Shadow Fiend mirror games, turn them into (features, labels) sequences, train the spec's unit-token transformer + GRU by behavioral cloning, and show the cloned agent laning and taking last hits against passive bots inside `DotaEnv`.

**Architecture:** An observer role (one filler per team) prints the stats the proto lacks for every configured hero, so recordings carry last hits, denies, gold, net worth and level. `record.py` saves raw worldstates per episode. `features.py` maps a `WorldState` to unit tokens, a hero token and masks; `labels.py` derives order labels from the next worldstate. `net.py` is the policy; `bc.py` trains it on episode windows with TensorBoard logging; `policy.py` gains `NetPolicy` so the checkpoint plays in the existing fleet. Macro (skill order, starting items) is scripted in Lua for the controlled bot.

**Tech Stack:** Python 3.12, PyTorch 2.14 CPU, numpy, tensorboard, protobuf; Lua 5.1 bot scripts; the phase 1 harness.

**Spec:** `docs/superpowers/specs/2026-10-22-dotasim-design.md` (sections Observation and model, Training/Behavioral cloning, Evaluation, Interface). Phase 1 plan and ledger: `docs/plans/2026-10-22-phase1-*.md`.

## Global Constraints

- Everything from the phase 1 plan's Global Constraints still holds (WSL2 venv, `~/dota2`, launch flags, file channel, team ids 2/3, test rules: end-to-end, integration, golden only; no mocks).
- Hero pool for this phase: Shadow Fiend mirror only (`npc_dota_hero_nevermore`). Hero-conditioning stays in the tensors (hero id field) but only one id occurs.
- Unit order codes (verified on the golden message and pinned by Task 3's golden check): 0 none, 1 move to position, 4 attack target, 6 cast target; 5 cast position, 8 cast no target are expected for razes and requiem and are pinned the same way.
- Shadow Fiend ability slots in the worldstate: slot 0, 1, 2 are the three razes (ids 5059, 5060, 5061), slot 4 is Requiem of Souls (id 5063 on this build, verify in Task 3); the rest are passives and talents and are never targets of a cast label.
- Decision interval: recordings at `frames=5` (0.167 s, 5 ticks); training and eval at the frames the recording used. Subsampling to 10 is a later experiment.
- Episode cap for recording and eval: 600 game seconds. Difficulty for the imitation source: 4 (unfair). Difficulty for the eval opponent: 0 (passive), verified to make the enemy hero non-aggressive.
- Model: unit MLP to width 128, 4 transformer layers (4 heads, FF 256), mean-pool + hero token into a `GRUCell(128+128 -> 256)`, heads: action type (7 = none, move, attack, cast slot 0, 1, 2, 4), move direction (16 bins), move distance (4 bins: 150, 300, 600, 1200 units), target unit (pointer over K=48 unit tokens). ~1M parameters.
- Training: Adam, lr 3e-4, batch of 32 windows of 32 steps (truncated BPTT), masked cross-entropy per head, 20 epochs over the recording, TensorBoard scalars under `runs/<run>/tb`.
- Dependencies added to `pyproject.toml`: `numpy>=2`, `tensorboard>=2.17`; torch comes from the CPU index via `scripts/setup_wsl.sh`.
- New tests that need the game are marked `game`; the recording job and the eval game are marked `slow`.

## Review Focus

1. A recorded episode where the Valve hero dies and respawns: labels during death must be "none" and the pointer label must never index a dead or absent unit. Pinned in Task 3 (golden check over the recording: no label points at a unit with `alive == False`).
2. `attack_target_handle` naming a unit outside the K=48 nearest tokens: the label must become "none" for the pointer head, not an index error. Pinned in Task 3.
3. Observer stats arriving one step late or missing for a hero that is dead: `Hero.last_hits` etc. must keep the last known value, never reset to -1 mid-episode. Pinned in Task 1.
4. The cloned agent issuing an attack on a target name that no longer exists by the time Lua resolves it: `mismatch` increments and the step still completes with no timeout. Pinned in Task 7 (eval asserts `lua_timeouts == 0` while `lua_mismatch` may be > 0 and is reported).
5. A checkpoint trained at `frames=5` run at `frames=10`: the env must refuse or warn, since the GRU dynamics assume the interval. Pinned in Task 6 (`NetPolicy` records its frames in the checkpoint and `Fleet` asserts the matchup matches).

---

### Task 1: Observer role and per-hero stats in the env

**Files:**
- Modify: `dotasim/env/lua/core.lua` (replace the temporary PROBE block in the filler branch), `dotasim/env/instance.py`, `dotasim/env/env.py`, `dotasim/env/worldstate.py`
- Test: `tests/test_env_game.py` (add), `tests/test_instance_log.py` (add)

**Interfaces:**
- Lua: the filler with the lowest player id on each team becomes the observer. Every 15 ticks it prints, for every player id in `config.heroes` on its team: `LUAALLY token=<tok> team=<t> pid=<n> lh=<n> dn=<n> gold=<n> nw=<n> lvl=<n>` (from `GetTeamMember(i)` unit objects; `GetXP` does not exist and is dropped).
- `Instance.lua.ally: dict[int, dict]` keyed by player id with keys `last_hits, denies, gold, net_worth, level`, updated by `_poll_log` (regex `LUAALLY`).
- `Hero` gains `net_worth: int = -1`; `xp` is removed from `Hero` (not observable). `summary()` keeps `last_hits` per team.
- `DotaEnv._merge_lua_obs` merges `instance.lua.ally` into every hero in the worldstate (controlled heroes keep LUAOBS, which is per step). Values persist: once a hero has stats, a missing line leaves the previous values.
- `Matchup` gains `watch_teams: tuple[int, ...] = ()`: extra teams whose worldstate stream is consumed and returned in `obs` even when not controlled (recording both perspectives). `reset` and `step` return obs for controlled teams plus `watch_teams`.

- [ ] **Step 1: Write the failing tests.** In `tests/test_instance_log.py`: append a `LUAALLY token=abc team=2 pid=0 lh=3 dn=1 gold=640 nw=900 lvl=2` line, poll, assert `inst.lua.ally[0] == {"last_hits": 3, "denies": 1, "gold": 640, "net_worth": 900, "level": 2}`. In `tests/test_env_game.py` (game): `test_valve_heroes_carry_observer_stats`: matchup roles `{0: "valve", 5: "valve"}`, `watch_teams=(2, 3)`, step until `dota_time >= 120`; assert `obs[2].hero_of(2).last_hits >= 0`, `obs[3].hero_of(3).gold > 0`, and that `obs[2].hero_of(3).level >= 1` (enemy hero stats also visible through team 3's observer merged by player id). Then kill the observer's lines by stepping 10 more times without new log lines (not possible to force; instead assert the values never go back to -1 across 200 steps).

- [ ] **Step 2: Run to verify they fail**, then **Step 3: Implement** per Interfaces, then **Step 4: Run to verify they pass** (`python -m pytest tests/test_instance_log.py tests/test_env_game.py::test_valve_heroes_carry_observer_stats -q -m "game or not game"`), then run `python -m pytest -q -m "not slow"`.

- [ ] **Step 5: Commit** `git commit -am "feat: observer role reports ally hero stats; watch_teams"`

---

### Task 2: Recording

**Files:**
- Create: `dotasim/data/__init__.py`, `dotasim/data/record.py`
- Test: `tests/test_record_game.py` (game, slow)

**Interfaces:**
- `record.py` CLI: `python -m dotasim.data.record --games 20 --instances 12 --frames 5 --difficulty 4 --out data/sf_mirror_v1` runs Valve-vs-Valve episodes with `watch_teams=(2, 3)` and writes one file per episode per team: `data/<name>/ep<NNNN>_t<2|3>.npz` with arrays: `raw` (object array of zlib-compressed worldstate bytes, one per step), `dota_time` (float32), `ally_stats` (int32 [T, 10, 5] by player id: last_hits, denies, gold, net_worth, level, -1 when unknown), plus `meta.json` per dataset (frames, difficulty, heroes, build 6965, proto PIN, date).
- `def load_episode(path: Path) -> list[WorldState]` decompresses and parses, re-applying `ally_stats` to heroes.
- Recording uses `Fleet` with a `RecordingPolicy` that returns `{}` and appends each step's obs; add a `Fleet.on_step(callback)` hook: `callback(env_index, obs_by_team, info)`.

- [ ] **Step 1: Write the failing test** (`game`, `slow`): record 2 games on 2 instances with `cap_s=120`, assert four `.npz` files exist, `load_episode` returns > 300 steps each (120 s / 0.175 s minus startup), consecutive `dota_time` differences are within 0.17 to 0.19 for 99% of steps, and `hero_of(team).last_hits >= 0` on the last step.

- [ ] **Step 2: Run to verify it fails**, **Step 3: Implement**, **Step 4: Run to verify it passes** (`python -m pytest tests/test_record_game.py -q -m slow`).

- [ ] **Step 5: Record the phase 2 dataset:** `python -m dotasim.data.record --games 24 --instances 12 --frames 5 --difficulty 4 --out data/sf_mirror_v1`. Expected: 48 episode files, about 24 x 2 x 3400 steps. Add `data/` to `.gitignore`. Note the wall time and disk size in `docs/experiments.md`.

- [ ] **Step 6: Commit** `git commit -am "feat: episode recording to npz with observer stats"`

---

### Task 3: Features and labels

**Files:**
- Create: `dotasim/model/__init__.py`, `dotasim/model/features.py`, `dotasim/model/labels.py`, `dotasim/model/heroes.py`
- Test: `tests/test_features_golden.py`, `tests/golden/features_6965.npz`

**Interfaces:**
- `heroes.py`: `HERO_IDS = {"npc_dota_hero_nevermore": 11}`, `ACTIVE_SLOTS = {"npc_dota_hero_nevermore": (0, 1, 2, 4)}`, `CAST_KIND = {"npc_dota_hero_nevermore": {0: "none", 1: "none", 2: "none", 4: "none"}}` (no-target casts), `SKILL_ORDER` and `START_ITEMS` for Task 5.
- `features.py`:
  ```python
  K = 48
  UNIT_DIM, HERO_DIM = 16, 24
  @dataclass class Features: units: np.ndarray  # [K, UNIT_DIM] float32, zero-padded
      unit_mask: np.ndarray    # [K] bool, True where a real unit
      unit_handles: np.ndarray # [K] int64, 0 where padded
      hero: np.ndarray         # [HERO_DIM] float32
      type_mask: np.ndarray    # [7] bool: none, move, attack, cast s0, s1, s2, s4 (castable now)
  def featurize(ws: WorldState, team: int) -> Features
  ```
  Unit features (per nearest unit to our hero, sorted by distance): one-hot kind (hero, creep, tower, other) [4], is enemy [1], dx/1500, dy/1500 [2], dist/1500 [1], cos/sin facing [2], hp frac [1], hp/1000 [1], level/25 [1], attack_damage/200 [1], alive [1], last-hittable (hp <= our attack_damage) [1]. Hero features: level/25, hp frac, mana frac, gold/2000, last_hits/100, denies/50, net_worth/5000, dota_time/600, x/8000, y/8000, cos/sin facing, per active slot (4 slots x 3: level/4, cooldown_remaining/20, castable) = 12 -> 24.
- `labels.py`:
  ```python
  @dataclass class Labels: action_type: int; move_dir: int; move_dist: int; target_idx: int   # -1 where not applicable
  def label(ws: WorldState, next_ws: WorldState, feats: Features, team: int) -> Labels
  ```
  Rules: hero dead or absent -> type none. `action_type == 4` and `attack_target_handle` among `feats.unit_handles` -> attack with that index; else (handle outside K) -> none. `is_using_ability` with `active_ability_handle` resolving to an active slot -> cast slot. Else displacement to `next_ws` hero position > 100 units -> move with direction bin `atan2` into 16 and distance bin by thresholds (150, 300, 600). Else none.
- Golden check script `tests/golden/make_features_golden.py` writes `features_6965.npz` from the golden worldstate (features only, no next step).

- [ ] **Step 1: Write the failing golden test:** `featurize(parse(golden), 2)` has `units.shape == (48, 16)`, `unit_mask.sum() == min(48, len(ws.units))`, the first unit is our hero with dx == dy == 0, `type_mask[0] and type_mask[2]`, and the arrays equal `features_6965.npz` (generated on the first run, then pinned). Plus a dataset check (`game`-free, uses `data/sf_mirror_v1` when present, else skipped): over the first recorded episode, label every step; assert type 4 rows with a resolvable handle have `attack_target_name != ""` in > 95% of cases, type 1 rows have displacement > 50 in > 80% of cases, no label points at a unit with `alive == False`, and no pointer index exceeds `unit_mask.sum()`.

- [ ] **Step 2: Run to verify it fails**, **Step 3: Implement**, **Step 4: Run to verify it passes** (`python -m pytest tests/test_features_golden.py -q`).

- [ ] **Step 5: Commit** `git commit -am "feat: unit-token features and order labels"`

---

### Task 4: Network

**Files:**
- Create: `dotasim/model/net.py`
- Test: `tests/test_net_golden.py`

**Interfaces:**
  ```python
  class Policy(nn.Module):
      def __init__(self, unit_dim=16, hero_dim=24, d=128, layers=4, hidden=256, k=48): ...
      def forward(self, units, unit_mask, hero, h) -> tuple[dict[str, Tensor], Tensor]
          # units [B,K,unit_dim], unit_mask [B,K] bool, hero [B,hero_dim], h [B,hidden]
          # returns logits {"type":[B,7], "dir":[B,16], "dist":[B,4], "target":[B,K] (masked to -inf where not a unit)}, new h
      def initial_state(self, batch:int) -> Tensor
  def masked_log_probs(logits, type_mask) -> Tensor     # applies the castable mask to "type"
  def sample(logits, type_mask, greedy: bool) -> dict[str, int]
  ```

- [ ] **Step 1: Write the failing test:** forward on the golden features (batch 1) returns the four logit shapes, `target` logits are `-inf` exactly where `unit_mask` is False, parameter count is between 0.7M and 1.5M, and a 12-batch forward takes under 15 ms on 4 threads (`torch.set_num_threads(4)`).

- [ ] **Step 2: Run to verify it fails**, **Step 3: Implement**, **Step 4: Run to verify it passes** (`python -m pytest tests/test_net_golden.py -q`).

- [ ] **Step 5: Commit** `git commit -am "feat: unit-token transformer + GRU policy"`

---

### Task 5: Scripted macro for the controlled bot

**Files:**
- Create: `dotasim/env/lua/macro.lua`
- Modify: `dotasim/env/lua/core.lua` (call `macro.tick(bot, hero_name)` once per 30 ticks inside the controlled Think, before the lockstep wait), `dotasim/model/heroes.py` (SKILL_ORDER, START_ITEMS as the Python mirror of the Lua tables)
- Test: `tests/test_env_game.py` (add)

**Interfaces:**
- `macro.lua`: `M.tick(bot, name)`: if `bot:GetAbilityPoints() > 0`, level the next ability from `SKILL_ORDER[name]` (SF: raze 2, necromastery, raze 2, necromastery, raze 2, ult, ... as a list of ability names); at the first tick, `bot:ActionImmediate_PurchaseItem` for each of `START_ITEMS[name]` (SF: `item_tango`, `item_branches`, `item_branches`, `item_circlet`, `item_faerie_fire`) while gold allows.
- Macro runs for controlled bots only; valve and filler untouched.

- [ ] **Step 1: Write the failing test** (`game`): `test_controlled_bot_levels_and_buys`: controlled SF vs passive, random policy, run until `dota_time >= 150`; assert `obs[2].hero_of(2).level >= 2` and that at least one ability in slots (0, 1, 2) has `level >= 1` (expose ability levels on `Hero` as `ability_levels: tuple[int, ...]` for slots 0..5 in `worldstate.py`), and `len(items) >= 1` (`Hero.item_ids: tuple[int, ...]`).

- [ ] **Step 2: Run to verify it fails**, **Step 3: Implement**, **Step 4: Run to verify it passes** (`python -m pytest tests/test_env_game.py::test_controlled_bot_levels_and_buys -q -m game`).

- [ ] **Step 5: Commit** `git commit -am "feat: scripted skill order and starting items for controlled bots"`

---

### Task 6: Behavioral cloning trainer and NetPolicy

**Files:**
- Create: `dotasim/train/bc.py`, `dotasim/train/dataset.py`
- Modify: `dotasim/train/policy.py` (add `NetPolicy`), `dotasim/train/rollout.py` (assert `policy.frames` matches the matchup when present)
- Test: `tests/test_bc_integration.py`

**Interfaces:**
- `dataset.py`: `class EpisodeWindows(torch.utils.data.Dataset)`: built from `data/<name>` files, featurizes and labels every step once (cached to `data/<name>/cache_<frames>.npz`), yields windows of 32 consecutive steps from one episode: tensors `units, unit_mask, hero, type_mask, y_type, y_dir, y_dist, y_target`.
- `bc.py` CLI: `python -m dotasim.train.bc --data data/sf_mirror_v1 --epochs 20 --out runs/bc_v1` trains with the Global Constraints hyperparameters, logs `loss/type, loss/dir, loss/dist, loss/target, acc/type, acc/target, steps_per_s` to TensorBoard each 20 batches, saves `runs/bc_v1/policy.pt` = `{"state_dict", "frames", "hero_pool", "config"}` every epoch, prints final accuracies as JSON.
- `NetPolicy(checkpoint_path, greedy=True)`: `act(obs)` featurizes each controlled team's obs, keeps a GRU state per team, returns `Action` built from the sampled heads: move -> `Action("move", hero.x + dist*cos, hero.y + dist*sin)`; attack -> `Action("attack", unit.x, unit.y, unit.name)` for the pointed unit; cast slot -> `Action("cast", name=str(slot))` (Lua: `bot:Action_UseAbility(bot:GetAbilityByName(...))` by slot via `GetAbilityInSlot(slot)`); none -> `Action()`. Add `cast` handling to `core.lua`'s `act`. Exposes `.frames`.

- [ ] **Step 1: Write the failing integration test:** on `data/sf_mirror_v1` (skip if absent) take 4 episodes, train 3 epochs with batch 8, assert `loss/type` at the last log is below the first by at least 30% and `acc/type` on the same episodes is above the majority-class rate. Then `NetPolicy(ckpt).act({2: ws})` on the golden worldstate returns an `Action` whose kind is one of none/move/attack/cast.

- [ ] **Step 2: Run to verify it fails**, **Step 3: Implement**, **Step 4: Run to verify it passes** (`python -m pytest tests/test_bc_integration.py -q`).

- [ ] **Step 5: Train the real model:** `python -m dotasim.train.bc --data data/sf_mirror_v1 --epochs 20 --out runs/bc_v1`; record final accuracies and wall time in `docs/experiments.md`. Expected: type accuracy well above the none-majority rate; target accuracy above 50% on attack rows.

- [ ] **Step 6: Commit** `git commit -am "feat: behavioral cloning trainer and NetPolicy"`

---

### Task 7: Evaluation against passive bots

**Files:**
- Create: `dotasim/eval/__init__.py`, `dotasim/eval/eval.py`
- Test: `tests/test_eval_game.py` (game, slow)

**Interfaces:**
- `eval.py` CLI: `python -m dotasim.eval.eval --checkpoint runs/bc_v1/policy.pt --games 3 --instances 3 --difficulty 0 --cap 300 --out runs/bc_v1/eval` runs the checkpoint as the controlled SF vs Valve SF at the given difficulty, prints JSON: per game `last_hits, denies, kills, deaths, level, lua_timeouts, lua_mismatch` at the end, plus medians. `--policy random` runs the baseline the same way. The Valve bot's own last hits (observer stats for the enemy hero) are reported as the ceiling.
- Writes `runs/<out>/eval.json` and TensorBoard scalars `eval/last_hits_median` etc.

- [ ] **Step 1: Write the failing test** (`game`, `slow`): run eval with the Task 6 checkpoint for 3 games, cap 300 s, 3 instances; assert `lua_timeouts == 0` for every game, every game reached the cap or post-game without error, and `median last_hits >= 5`. Also run the random baseline once and record it.

- [ ] **Step 2: Run to verify it fails**, **Step 3: Implement**, **Step 4: Run to verify it passes** (`python -m pytest tests/test_eval_game.py -q -m slow`).

- [ ] **Step 5: Record the phase 2 exit numbers** in `docs/experiments.md`: random baseline, BC agent, Valve bot ceiling (last hits at 5 game-minutes, median of 3), dataset size, training time, accuracies. Add the TensorBoard screenshot.

- [ ] **Step 6: Commit** `git commit -am "feat: eval suite vs Valve bots; phase 2 exit numbers"`

---

### Task 8: Phase 2 exit

- [ ] **Step 1:** Full suite: `python -m pytest -q -m "not slow"` then `-m slow`. Expected: all PASS.
- [ ] **Step 2:** README: add the data/train/eval commands and the phase 2 numbers. Spec: record the observer role and the no-target-cast simplification under Observation and model.
- [ ] **Step 3:** Commit, tag `phase2`, push main and the tag.
