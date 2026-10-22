"""Rollout storage: per-(env, team) trajectories, GAE, fixed-length windows with stored initial GRU states."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

ACTION_KEYS = ("type", "dir", "dist", "target")


@dataclass
class Step:
    units: np.ndarray
    unit_mask: np.ndarray
    hero: np.ndarray
    type_mask: np.ndarray
    action: dict[str, int]
    logp: float
    value: float
    h0: np.ndarray            # GRU state before this step
    policy_version: int
    reward: float


@dataclass
class Trajectory:
    env: int
    team: int
    steps: list[Step] = field(default_factory=list)
    done: bool = False
    error: bool = False
    last_value: float = 0.0
    terminal: float = 0.0
    capped: bool = False
    adv: np.ndarray | None = None
    ret: np.ndarray | None = None


@dataclass
class Window:
    env: int
    team: int
    version: int
    start: int
    tensors: dict[str, torch.Tensor]   # units, unit_mask, hero, type_mask, a_type, a_dir, a_dist, a_target, logp, adv, ret, valid
    h0: torch.Tensor                   # [hidden]

    @property
    def valid(self) -> torch.Tensor:
        return self.tensors["valid"]

    @property
    def adv(self) -> torch.Tensor:
        return self.tensors["adv"]

    @property
    def ret(self) -> torch.Tensor:
        return self.tensors["ret"]


class Buffer:
    def __init__(self, window: int = 32):
        self.window = window
        self.trajs: dict[tuple[int, int], Trajectory] = {}
        self.finished: list[Trajectory] = []

    def add(self, env: int, team: int, step: Step) -> None:
        self.trajs.setdefault((env, team), Trajectory(env, team)).steps.append(step)

    def finish(self, env: int, team: int, terminal_reward: float, error: bool, last_value: float,
               capped: bool | None = None) -> None:
        """capped=True bootstraps the last step from last_value (episode cut at cap_s); a finished game bootstraps 0."""
        t = self.trajs.pop((env, team), Trajectory(env, team))
        if capped is None:
            capped = terminal_reward == 0.0 and not error
        t.done, t.error, t.last_value, t.terminal = True, error, (last_value if capped else 0.0), terminal_reward
        t.capped = capped
        self.finished.append(t)

    def trajectory(self, env: int, team: int) -> Trajectory:
        return next(t for t in self.finished if t.env == env and t.team == team)

    def steps(self) -> int:
        return sum(len(t.steps) for t in self.finished if not t.error)

    def _gae(self, t: Trajectory, gamma: float, lam: float) -> None:
        n = len(t.steps)
        r = np.array([s.reward for s in t.steps], np.float64)
        v = np.array([s.value for s in t.steps], np.float64)
        r[-1] += t.terminal
        next_v = np.append(v[1:], t.last_value if t.capped else 0.0)   # bootstrap V(s_T+1) only when cut at the cap
        adv = np.zeros(n)
        last = 0.0
        for i in range(n - 1, -1, -1):
            delta = r[i] + gamma * next_v[i] - v[i]
            last = delta + gamma * lam * last
            adv[i] = last
        t.adv, t.ret = adv, adv + v

    def windows(self, gamma: float, lam: float) -> list[Window]:
        out = []
        for t in self.finished:
            if t.error or not t.steps:
                continue
            self._gae(t, gamma, lam)
            for s in range(0, len(t.steps), self.window):
                chunk = t.steps[s:s + self.window]
                out.append(self._window(t, s, chunk))
        return out

    def _window(self, t: Trajectory, start: int, chunk: list[Step]) -> Window:
        W, n = self.window, len(chunk)
        pad = W - n
        def stack(arrs, dtype):
            a = np.stack(arrs).astype(dtype)
            if pad:
                a = np.concatenate([a, np.zeros((pad,) + a.shape[1:], dtype)], 0)
            return torch.from_numpy(a)
        ten = {
            "units": stack([c.units for c in chunk], np.float32),
            "unit_mask": stack([c.unit_mask for c in chunk], bool),
            "hero": stack([c.hero for c in chunk], np.float32),
            "type_mask": stack([c.type_mask for c in chunk], bool),
            "logp": stack([c.logp for c in chunk], np.float32),
            "adv": torch.from_numpy(np.concatenate([t.adv[start:start + n], np.zeros(pad)]).astype(np.float32)),
            "ret": torch.from_numpy(np.concatenate([t.ret[start:start + n], np.zeros(pad)]).astype(np.float32)),
            "valid": torch.from_numpy(np.concatenate([np.ones(n, bool), np.zeros(pad, bool)])),
        }
        for k in ACTION_KEYS:
            ten[f"a_{k}"] = stack([max(c.action[k], 0) for c in chunk], np.int64)
        if pad:  # padded rows need a legal mask so log-softmax stays finite
            ten["type_mask"][n:, 0] = True
            ten["unit_mask"][n:, 0] = True
        return Window(t.env, t.team, chunk[0].policy_version, start, ten, torch.from_numpy(chunk[0].h0.astype(np.float32)))

    @staticmethod
    def collate(wins: list[Window], expect_version: int | None = None) -> dict[str, torch.Tensor]:
        if expect_version is not None and any(w.version != expect_version for w in wins):
            raise ValueError(f"window produced by policy version {[w.version for w in wins]} but expected {expect_version}")
        batch = {k: torch.stack([w.tensors[k] for w in wins]) for k in wins[0].tensors}
        batch["h0"] = torch.stack([w.h0 for w in wins])
        return batch
