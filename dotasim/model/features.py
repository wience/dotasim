"""WorldState -> fixed-size tensors for the policy: K nearest unit tokens, a hero token, action masks."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dotasim.env.worldstate import Hero, WorldState
from dotasim.model.heroes import ACTIVE_IDS

K = 48
UNIT_DIM = 16
HERO_DIM = 24
N_TYPES = 7   # none, move, attack, cast x4
KINDS = ("hero", "creep", "tower", "other")
POS_SCALE, MAP_SCALE = 1500.0, 8000.0


@dataclass
class Features:
    units: np.ndarray         # [K, UNIT_DIM] float32, zero padded
    unit_mask: np.ndarray     # [K] bool
    unit_handles: np.ndarray  # [K] int64, 0 where padded
    hero: np.ndarray          # [HERO_DIM] float32
    type_mask: np.ndarray     # [N_TYPES] bool


def _facing(deg: float) -> tuple[float, float]:
    r = math.radians(deg)
    return math.cos(r), math.sin(r)


def featurize(ws: WorldState, team: int) -> Features:
    me = ws.hero_of(team)
    units = np.zeros((K, UNIT_DIM), np.float32)
    mask = np.zeros(K, bool)
    handles = np.zeros(K, np.int64)
    hero = np.zeros(HERO_DIM, np.float32)
    type_mask = np.zeros(N_TYPES, bool)
    type_mask[0] = True
    if me is None:
        return Features(units, mask, handles, hero, type_mask)

    ordered = sorted(ws.units, key=lambda u: (u is not me, (u.x - me.x) ** 2 + (u.y - me.y) ** 2))[:K]
    for i, u in enumerate(ordered):
        dx, dy = (u.x - me.x) / POS_SCALE, (u.y - me.y) / POS_SCALE
        c, s = _facing(u.facing)
        units[i, KINDS.index(u.kind)] = 1.0
        units[i, 4] = float(u.team != team)
        units[i, 5], units[i, 6], units[i, 7] = dx, dy, math.hypot(dx, dy)
        units[i, 8], units[i, 9] = c, s
        units[i, 10] = u.hp / u.hp_max if u.hp_max else 0.0
        units[i, 11] = u.hp / 1000.0
        units[i, 12] = u.level / 25.0
        units[i, 13] = u.attack_damage / 200.0
        units[i, 14] = float(u.alive)
        units[i, 15] = float(u.alive and u.team != team and 0 < u.hp <= me.attack_damage)
        mask[i] = True
        handles[i] = u.handle

    c, s = _facing(me.facing)
    hero[:12] = [me.level / 25.0, me.hp / me.hp_max if me.hp_max else 0.0, me.mana / me.mana_max if me.mana_max else 0.0,
                 max(me.gold, 0) / 2000.0, max(me.last_hits, 0) / 100.0, max(me.denies, 0) / 50.0,
                 max(me.net_worth, 0) / 5000.0, ws.dota_time / 600.0, me.x / MAP_SCALE, me.y / MAP_SCALE, c, s]
    by_id = {a[0]: a for a in me.abilities}
    for j, aid in enumerate(ACTIVE_IDS.get(me.name, ())):
        a = by_id.get(aid)
        if a is not None:
            hero[12 + 3 * j: 15 + 3 * j] = [a[1] / 4.0, min(a[2], 20.0) / 20.0, float(a[3])]
            type_mask[3 + j] = bool(a[3]) and me.alive
    type_mask[1] = type_mask[2] = me.alive
    return Features(units, mask, handles, hero, type_mask)
