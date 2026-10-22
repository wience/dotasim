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

    # ---- batched interface (SyncFleet) ----------------------------------
    def reset_env(self, env_index: int) -> None:
        for key in [k for k in self.states if isinstance(k, tuple) and k[0] == env_index]:
            del self.states[key]

    def act_batch_full(self, items: list[tuple[int, int, WorldState]]) -> list[dict]:
        """One forward pass for all (env, team, obs) items. Returns per item:
        {"action": Action, "a": {type,dir,dist,target}, "logp", "value", "h0": np[hidden], "feats": Features}."""
        import numpy as np
        import torch
        from dotasim.model.features import featurize
        from dotasim.model.net import masked_log_probs
        from dotasim.train.ppo import joint_log_prob

        if not items:
            return []
        feats = [featurize(ws, team) for _, team, ws in items]
        h0 = torch.stack([self.states.get((e, t), self.net.initial_state(1)[0]) for e, t, _ in items])
        units = torch.from_numpy(np.stack([f.units for f in feats]))
        umask = torch.from_numpy(np.stack([f.unit_mask for f in feats]))
        hero = torch.from_numpy(np.stack([f.hero for f in feats]))
        tmask = torch.from_numpy(np.stack([f.type_mask for f in feats]))
        with torch.inference_mode():
            logits, h = self.net(units, umask, hero, h0)
            lp = masked_log_probs(logits, tmask)
            if self.greedy:
                a = {k: lp[k].argmax(1) for k in ("type", "dir", "dist", "target")}
            else:
                a = {k: torch.multinomial(lp[k].exp(), 1).squeeze(1) for k in ("type", "dir", "dist", "target")}
            logp = joint_log_prob(lp, a)
            value = logits["value"]
        out = []
        for i, (e, t, ws) in enumerate(items):
            self.states[(e, t)] = h[i].clone()
            ai = {k: int(v[i]) for k, v in a.items()}
            out.append({"action": self._to_action(ws, t, feats[i], ai), "a": ai, "logp": float(logp[i]),
                        "value": float(value[i]), "h0": h0[i].numpy().copy(), "feats": feats[i]})
        return out

    def act_batch(self, items: list[tuple[int, int, WorldState]]) -> list[Action]:
        return [d["action"] for d in self.act_batch_full(items)]

    @staticmethod
    def _to_action(ws: WorldState, team: int, f, a: dict[str, int]) -> Action:
        from dotasim.model.heroes import ACTIVE_NAMES
        from dotasim.model.labels import ATTACK, CAST_BASE, DIST_VALUES, MOVE, N_DIRS

        hero = ws.hero_of(team)
        if hero is None or a["type"] == 0:
            return Action()
        if a["type"] == MOVE:
            ang = (a["dir"] + 0.5) * 2 * math.pi / N_DIRS
            d = DIST_VALUES[a["dist"]]
            return Action("move", hero.x + d * math.cos(ang), hero.y + d * math.sin(ang))
        if a["type"] == ATTACK:
            handle = int(f.unit_handles[a["target"]])
            u = next((u for u in ws.units if u.handle == handle), None)
            return Action("attack", u.x, u.y, u.name) if u else Action()
        names = ACTIVE_NAMES.get(hero.name, ())
        j = a["type"] - CAST_BASE
        return Action("cast", hero.x, hero.y, names[j]) if j < len(names) else Action()

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
