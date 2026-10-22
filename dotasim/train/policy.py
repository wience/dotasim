"""Policies for phase 1: random micro, or nothing (Valve bots play themselves)."""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Protocol

from dotasim.env.channel import Action
from dotasim.env.worldstate import WorldState


class Policy(Protocol):
    def act(self, obs: dict[int, WorldState]) -> dict[int, Action]: ...


class ValveOnlyPolicy:
    def act(self, obs: dict[int, WorldState]) -> dict[int, Action]:
        return {}


class NetPolicy:
    """Runs a behavioral-cloning / RL checkpoint; one GRU state per controlled team."""

    def __init__(self, checkpoint: str | Path, greedy: bool = True, threads: int = 4):
        import torch
        from dotasim.model.net import Policy

        torch.set_num_threads(threads)
        ck = torch.load(Path(checkpoint), map_location="cpu")
        self.net = Policy()
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        self.frames = int(ck["frames"])
        self.greedy = greedy
        self.states: dict[int, "torch.Tensor"] = {}
        self._torch = torch

    def reset(self) -> None:
        self.states.clear()

    def for_env(self, index: int) -> "NetPolicy":
        """A per-instance view sharing the network but with its own GRU states."""
        other = object.__new__(NetPolicy)
        other.__dict__.update(self.__dict__)
        other.states = {}
        return other

    def act(self, obs: dict[int, WorldState]) -> dict[int, Action]:
        import torch
        from dotasim.model.features import featurize
        from dotasim.model.heroes import ACTIVE_NAMES
        from dotasim.model.labels import ATTACK, CAST_BASE, DIST_VALUES, MOVE, N_DIRS
        from dotasim.model.net import sample

        out = {}
        for team, ws in obs.items():
            hero = ws.hero_of(team)
            f = featurize(ws, team)
            h = self.states.get(team)
            if h is None:
                h = self.net.initial_state(1)
            with torch.inference_mode():
                logits, h = self.net(torch.from_numpy(f.units[None]), torch.from_numpy(f.unit_mask[None]),
                                     torch.from_numpy(f.hero[None]), h)
                a = sample(logits, torch.from_numpy(f.type_mask[None]), greedy=self.greedy)
            self.states[team] = h
            if hero is None or a["type"] == 0:
                out[team] = Action()
            elif a["type"] == MOVE:
                ang = (a["dir"] + 0.5) * 2 * math.pi / N_DIRS
                d = DIST_VALUES[a["dist"]]
                out[team] = Action("move", hero.x + d * math.cos(ang), hero.y + d * math.sin(ang))
            elif a["type"] == ATTACK:
                handle = int(f.unit_handles[a["target"]])
                u = next((u for u in ws.units if u.handle == handle), None)
                out[team] = Action("attack", u.x, u.y, u.name) if u else Action()
            else:
                names = ACTIVE_NAMES.get(hero.name, ())
                j = a["type"] - CAST_BASE
                out[team] = Action("cast", hero.x, hero.y, names[j]) if j < len(names) else Action()
        return out


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
