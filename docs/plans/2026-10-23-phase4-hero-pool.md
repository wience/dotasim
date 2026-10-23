# Phase 4: hero pool (Shadow Fiend, Puck, Queen of Pain) — notes before the plan

Not started. Gated on the SF pipeline producing an agent that lanes (bc_v2 evaluation, then sp_v5).

## Ability ids (from Valve's `scripts/npc/npc_ability_ids.txt`; SF ids match the recorded worldstates)

| hero | ability | id | target |
|---|---|---|---|
| nevermore | shadowraze1 / 2 / 3 | 5059 / 5060 / 5061 | none |
| nevermore | requiem | 5064 | none |
| puck | illusory_orb | 5069 | point |
| puck | ethereal_jaunt | 5070 | none (needs an orb in flight) |
| puck | waning_rift | 5071 | none |
| puck | phase_shift | 5072 | none |
| puck | dream_coil | 5073 | point |
| queenofpain | shadow_strike | 5173 | unit (enemy) |
| queenofpain | blink | 5174 | point |
| queenofpain | scream_of_pain | 5175 | none |
| queenofpain | sonic_wave | 5176 | point |

Slot order on this build: SF raze1, raze2, raze3, frenzy, dark_lord, requiem; Puck orb, rift, phase_shift, jaunt,
puckish (passive), dream_coil; QoP shadow_strike, blink, scream, succubus (passive), hidden, sonic_wave.

## What changes

- `heroes.py`: ACTIVE_IDS / ACTIVE_NAMES / CAST_KIND / SKILL_ORDER / START_ITEMS for both heroes; Puck has five
  actives, so N_TYPES becomes 3 + 5 and HERO_DIM 12 + 3 * 5; both read from the checkpoint config so old SF
  checkpoints keep loading (pad the type head, or retrain: the mixed dataset needs a retrain anyway).
- Point casts reuse the dir/dist heads, unit casts the target head (`Policy` already has them); `core.lua` act()
  gets `Action_UseAbilityOnLocation` / `Action_UseAbilityOnEntity`.
- Labels: a point cast's point is not in the worldstate; label the direction toward the nearest visible enemy hero
  (or none when none is visible) and let RL fix it.
- `hero_selection.lua` already picks from config; `Matchup` already takes a hero pair. Recording: a mixed-pool
  dataset (9 matchups, 3 games each) at 8 instances is about 10 minutes.
