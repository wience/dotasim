"""Behavioral-cloning labels from consecutive worldstates of the imitated hero."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dotasim.env.worldstate import WorldState
from dotasim.model.features import Features
from dotasim.model.heroes import ACTIVE_IDS

NONE, MOVE, ATTACK, CAST_BASE = 0, 1, 2, 3
ORDER_ATTACK_TARGET = 4
N_DIRS = 16
DIST_EDGES = (150.0, 300.0, 600.0)    # bins: <150, <300, <600, else
DIST_VALUES = (150.0, 300.0, 600.0, 1200.0)  # distance each bin executes at
MOVE_MIN = 25.0                        # one-step displacement below this (about 150 u/s at 5 frames) is "none"
LOOKAHEAD = 6                          # steps (about 1 s at 5 frames) used for the distance bin


@dataclass
class Labels:
    action_type: int
    move_dir: int = -1
    move_dist: int = -1
    target_idx: int = -1


def direction_bin(dx: float, dy: float) -> int:
    return int(((math.atan2(dy, dx) + 2 * math.pi) % (2 * math.pi)) / (2 * math.pi) * N_DIRS) % N_DIRS


def distance_bin(d: float) -> int:
    return int(np.searchsorted(DIST_EDGES, d, side="right"))


def label(ws: WorldState, next_ws: WorldState, feats: Features, team: int, far_ws: WorldState | None = None) -> Labels:
    """far_ws is the state LOOKAHEAD steps ahead (distance bin); direction comes from the next step."""
    me = ws.hero_of(team)
    if me is None or not me.alive:
        return Labels(NONE)
    actives = ACTIVE_IDS.get(me.name, ())
    if me.is_using_ability and me.active_ability_id in actives:
        return Labels(CAST_BASE + actives.index(me.active_ability_id))
    if me.action_type == ORDER_ATTACK_TARGET and me.attack_target_handle:
        hits = np.nonzero(feats.unit_mask & (feats.unit_handles == me.attack_target_handle))[0]
        if len(hits) and feats.units[hits[0], 14] > 0:   # target present among the K tokens and alive
            return Labels(ATTACK, target_idx=int(hits[0]))
        return Labels(NONE)   # target outside the K nearest tokens, or already dead (e.g. a finished deny)
    nxt = next_ws.hero_of(team)
    if nxt is not None:
        dx, dy = nxt.x - me.x, nxt.y - me.y
        if math.hypot(dx, dy) >= MOVE_MIN:
            far = far_ws.hero_of(team) if far_ws is not None else None
            d = math.hypot(far.x - me.x, far.y - me.y) if far is not None else math.hypot(dx, dy) * LOOKAHEAD
            return Labels(MOVE, move_dir=direction_bin(dx, dy), move_dist=distance_bin(d))
    return Labels(NONE)
