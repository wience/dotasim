# Phase 3: Self-Play PPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Starting from the phase 2 cloning checkpoint, improve the Shadow Fiend mirror agent by self-play PPO with a league of past checkpoints and Valve bots, measured by win rate against Valve's unfair bots.

**Architecture:** A synchronized fleet steps every playing server in lockstep and runs one batched forward pass per step for all controlled heroes (both teams of every server). Rollouts collect features, actions, log-probs, values, GRU states and shaped rewards per hero; GAE at episode end; PPO updates on 32-step windows with stored initial GRU states, a KL penalty to a frozen anchor and an entropy bonus. A league samples each server's opponent per episode. Evaluation reuses phase 2's eval suite. Replays are pruned.

**Tech Stack:** phase 1 and 2 code; PyTorch CPU; TensorBoard.

**Spec:** `docs/superpowers/specs/2026-10-22-dotasim-design.md` (Training: Self-play PPO; Evaluation; Risks).

## Global Constraints

- Phase 1 and 2 Global Constraints hold. Shadow Fiend mirror only; `frames=5`; cap 600 game seconds; 12 instances in WSL2.
- Inference budget: at 10x and 5 frames a decision is due every 16.7 ms wall per server. One batched forward for all controlled heroes (up to 24) must stay under 12 ms on 4 threads. Measured phase 2 cost: 0.64 ms per instance batched.
- PPO: Adam lr 1e-4, clip 0.2, 3 epochs per iteration (2 if the update exceeds its budget), minibatch 64 windows of 32 steps, GAE lambda 0.95, gamma 0.9995 (horizon about 2,000 steps = 5.5 game-minutes at 5 frames; a 600 s episode is 3,600 steps), value loss coefficient 0.5, entropy coefficient 0.003, KL penalty 0.02 to the anchor over the masked type distribution plus the move and target heads, gradient clip 1.0, advantages normalized per iteration. Update budget: under 3 minutes per iteration (about 86k steps x 3 epochs), measured in Task 3 before any training.
- Reward: terminal +1 win / -1 loss / 0 on cap, plus potential-based shaping `gamma * phi(s') - phi(s)` with `phi = 0.02 * (lh - lh_enemy) + 0.01 * (denies - denies_enemy) + 0.1 * (level - level_enemy) + 0.0005 * (net_worth - net_worth_enemy) + 0.5 * (hp_frac - hp_frac_enemy)`; enemy stats come from the observer merge (both teams' observers write to the same log). Loops earn nothing because every term is a difference of state potentials.
- Winner: read from the worldstate `players` kills (present regardless of visibility or death) and the mid T1 towers (`npc_dota_goodguys_tower1_mid`, `npc_dota_badguys_tower1_mid`): at POST_GAME the team with 2 kills, else the team whose enemy T1 mid is dead, else draw. The server stops streaming at POST_GAME, so the last worldstate may predate the ending kill: Task 2 verifies the rule on real game ends from the phase 2 eval vs unfair and falls back to the console log if needed.
- League per episode: 70% latest vs latest (self-play, both teams controlled), 20% latest vs a uniformly sampled past checkpoint (the past one runs with greedy=False and no gradient), 10% latest vs Valve unfair (role `valve`, difficulty 4). Checkpoint pool: every 5th iteration, max 20 kept.
- Anchor: the phase 2 BC checkpoint at start; refreshed to the current policy every 20 iterations.
- Iteration: 12 servers, one episode each (about 90 s wall including relaunch), then the update. Evaluation every 10 iterations: 6 games vs Valve unfair and 6 vs passive with `eval.py`, logged to TensorBoard under `eval/*`.
- Replays: keep the newest 30 `.dem` files in `~/dota2/game/dota/replays/`; prune at every launch.

## Review Focus

1. A server that dies mid-iteration: its partial trajectory must be dropped, not trained on with a fake terminal reward. Pinned in Task 3 (integration test with a truncated trajectory flagged `error`).
2. GRU state at window boundaries: the stored initial state must come from the same policy version that produced the rollout, and be detached; otherwise PPO ratios are computed against wrong log-probs. Pinned in Task 3 (ratio at epoch 0 equals 1 within 1e-4 on every step).
3. A past-checkpoint opponent running the same network class with a different `frames`: refused at load. Pinned in Task 4.
4. Reward shaping when the enemy hero is not visible: potentials must use the observer stats (always available) rather than the visible-units copy, so phi does not jump when the enemy leaves vision. Pinned in Task 2 (golden trace where the enemy disappears has no reward spike).
5. Episode cap reached with no kills: terminal reward 0, done flagged, value bootstrapped from the last state (not zero). Pinned in Task 3.

---

### Task 1: Synchronized fleet with batched inference, replay pruning

**Files:**
- Create: `dotasim/train/syncfleet.py`
- Modify: `dotasim/train/policy.py` (`NetPolicy.act_batch`), `dotasim/env/instance.py` (prune replays at launch)
- Test: `tests/test_syncfleet_game.py` (game, slow)

**Interfaces:**
```python
class BatchPolicy(Protocol):
    def act_batch(self, items: list[tuple[int, int, WorldState]]) -> list[Action]   # (env_index, team, obs) -> action per item
    def reset_env(self, env_index: int) -> None
class SyncFleet:
    def __init__(self, n, matchup_for_env: Callable[[int], Matchup], policy_for: Callable[[int, int], BatchPolicy | None],
                 run_dir, dota_dir=None, timescale=10)
        # policy_for(env_index, team) routes each controlled team to a policy (the live net, a frozen league opponent);
        # items are batched per policy; teams routed to None are Valve bots. Only the live policy's teams enter the buffer.
    async def run(self, episodes_total: int, on_step=None, on_episode=None) -> dict
        # resets each env (its own matchup), steps all "playing" envs together, one act_batch per step; an env that finishes
        # is reset as a background task and rejoins when ready; stops after episodes_total episodes; same status.json as Fleet
```
- `NetPolicy.act_batch` keeps GRU states keyed by `(env_index, team)` and runs one forward for all items.
- `Instance.launch` deletes all but the newest 30 `.dem` in the replays folder before starting.

- [ ] **Step 1: Write the failing test** (game, slow): `SyncFleet(12, lambda i: Matchup(heroes SF/SF, roles {0:"controlled",5:"controlled"}, frames=5, cap_s=120), NetPolicy(bc_v1), ...)`, `run(12)`; assert 12 episodes, `lua_timeouts == 0`, mean `speed_x >= 5`, `act_batch` mean latency under 12 ms (recorded in totals), and the replays folder has at most 30 files.
- [ ] **Step 2: Run red**, **Step 3: Implement**, **Step 4: Run green**, **Step 5: Commit** `feat: synchronized fleet with batched inference`.

---

### Task 2: Reward and winner

**Files:**
- Create: `dotasim/train/reward.py`
- Test: `tests/test_reward_golden.py`, `tests/golden/reward_trace.json`

**Interfaces:**
```python
def potential(ws: WorldState, team: int, ally: dict[int, dict]) -> float        # phi from Global Constraints; enemy from ally stats
def step_reward(prev_ws, ws, team, ally, gamma) -> float                      # gamma*phi(ws) - phi(prev)
def winner(ws: WorldState, info: dict) -> int | None                          # 2, 3 or None(draw)
def terminal_reward(team, winner_team) -> float
```
- [ ] **Step 1: Golden test:** on the first recorded episode (`data/sf_mirror_v1/ep0000_t2.npz`) compute the shaped reward sequence; assert the discounted return `sum(gamma^t * r_t)` equals `gamma^T * phi(last) - phi(first)` within 1e-6 (telescoping, so loops earn nothing), that `abs(step_reward) < 1.0` everywhere, that a stretch where the enemy hero leaves vision produces no reward spike (enemy terms come from observer stats), and that the sequence equals the stored `reward_trace.json` (generated on the first run). Winner on a synthetic final state (players kills 2 vs 0) is team 2; verified on a real post-game end in Task 4's test.
- [ ] **Step 2: Run red**, **Step 3: Implement**, **Step 4: Run green**, **Step 5: Commit** `feat: potential-based reward and winner detection`.

---

### Task 3: Rollout buffer, GAE, PPO update

**Files:**
- Create: `dotasim/train/ppo.py`, `dotasim/train/buffer.py`
- Test: `tests/test_ppo_integration.py`

**Interfaces:**
```python
@dataclass class Step: units, unit_mask, hero, type_mask, action: dict[str,int], logp: float, value: float, h0: Tensor, policy_version: int, reward: float
class Trajectory: steps: list[Step]; done: bool; error: bool; last_value: float
class Buffer: add(env, team, step); finish(env, team, terminal_reward, error, last_value); windows(gamma, lam) -> list[Window]   # GAE per trajectory, windows of 32 with h0 of the first step
def ppo_update(net, anchor, windows, cfg) -> dict   # returns losses, kl, entropy, clipfrac; applies 3 epochs of minibatch updates
```
- [ ] **Step 1: Integration test:** build a buffer from 4 recorded episodes by running the BC net on them (actions sampled, log-probs and values recorded, rewards from Task 2, one trajectory flagged `error`); assert windows exclude the errored trajectory, advantages are finite, every window's `policy_version` matches the net that produced it (a mismatched window is refused), PPO ratio at the first epoch equals 1 within 1e-4 on every step, an update runs and returns finite losses, a capped trajectory's last advantage uses `last_value` (not zero), and a full-size update (86k steps, 3 epochs, minibatch 64) completes under 3 minutes on 8 threads (else the plan drops to 2 epochs).
- [ ] **Step 2: Run red**, **Step 3: Implement**, **Step 4: Run green**, **Step 5: Commit** `feat: PPO with GAE over GRU windows and KL anchor`.

---

### Task 4: League, training loop, evaluation hook

**Files:**
- Create: `dotasim/train/league.py`, `dotasim/train/selfplay.py`
- Test: `tests/test_selfplay_game.py` (game, slow)

**Interfaces:**
- `League(pool_dir)`: `sample(iteration) -> Matchup + opponent policy or None`, `add(checkpoint_path)`, refuses checkpoints with a different `frames`.
- `selfplay.py` CLI: `python -m dotasim.train.selfplay --init runs/bc_v1/policy.pt --instances 12 --iterations 200 --out runs/sp_v1 --eval-every 10` runs the loop: sample league matchups, SyncFleet iteration, PPO update, log `train/*` scalars, save `policy.pt` each iteration and pool checkpoints, run eval every N iterations and log `eval/*`, append a line per iteration to `runs/sp_v1/log.jsonl`.
- [ ] **Step 1: Test** (game, slow): 2 instances, 2 iterations, cap 120 s, eval-every 1 with 2 games; assert two `log.jsonl` lines with finite losses, `policy.pt` updated twice, pool has one checkpoint, eval scalars present.
- [ ] **Step 2: Run red**, **Step 3: Implement**, **Step 4: Run green**, **Step 5: Commit** `feat: self-play league training loop`.

---

### Task 5: Train

- [ ] **Step 1:** `python -m dotasim.train.selfplay --init runs/bc_v1/policy.pt --instances 12 --iterations 150 --out runs/sp_v1 --eval-every 10` in a terminal tab. Expected wall: about 150 x 2 min = 5 hours.
- [ ] **Step 2:** Every 10 iterations read `log.jsonl` and `eval/*`; if the KL to the anchor explodes or win rate collapses, halve the learning rate and note it. Record the curve and the final eval (vs passive, vs unfair) in `docs/experiments.md`.
- [ ] **Step 3:** Exit: median win rate vs unfair over 12 games at the best checkpoint. Spec target 50%; report the achieved number either way.

---

### Task 6: Phase 3 exit

- [ ] Full suite (`not slow`, then `slow`), README commands, spec note on batched stepping, commit, tag `phase3`, push.
