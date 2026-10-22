"""Policies for phase 1: random micro, or nothing (Valve bots play themselves)."""
from __future__ import annotations

import math
import random
from typing import Protocol

from dotasim.env.channel import Action
from dotasim.env.worldstate import WorldState


class Policy(Protocol):
    def act(self, obs: dict[int, WorldState]) -> dict[int, Action]: ...


class ValveOnlyPolicy:
    def act(self, obs: dict[int, WorldState]) -> dict[int, Action]:
        return {}


class RandomPolicy:
    """Half the time walk somewhere nearby, half the time attack the nearest enemy lane creep."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, obs: dict[int, WorldState]) -> dict[int, Action]:
        out = {}
        for team, ws in obs.items():
            hero = ws.hero_of(team)
            if hero is None:
                out[team] = Action()
                continue
            enemy = 5 - team
            creeps = [u for u in ws.units if u.kind == "creep" and u.team == enemy and u.alive]
            if creeps and self.rng.random() < 0.5:
                c = min(creeps, key=lambda u: math.hypot(u.x - hero.x, u.y - hero.y))
                out[team] = Action("attack", c.x, c.y, c.name)
            else:
                out[team] = Action("move", hero.x + self.rng.uniform(-400, 400), hero.y + self.rng.uniform(-400, 400))
        return out
