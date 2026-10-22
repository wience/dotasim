"""Opponent pool for self-play: latest vs latest, a past checkpoint, or Valve's unfair bots."""
from __future__ import annotations

import random
import shutil
from pathlib import Path

import torch

from dotasim.train.policy import NetPolicy

P_SELF, P_POOL, P_VALVE = 0.7, 0.2, 0.1


class League:
    def __init__(self, pool_dir: Path, frames: int, max_pool: int = 20, seed: int = 0):
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.frames = frames
        self.max_pool = max_pool
        self.rng = random.Random(seed)
        self.pool: list[Path] = sorted(self.pool_dir.glob("*.pt"))
        self._cache: dict[Path, NetPolicy] = {}

    def add(self, checkpoint: Path, name: str) -> Path:
        ck = torch.load(checkpoint, map_location="cpu")
        if int(ck["frames"]) != self.frames:
            raise ValueError(f"checkpoint {checkpoint} has frames={ck['frames']}, league runs at {self.frames}")
        dst = self.pool_dir / f"{name}.pt"
        shutil.copy(checkpoint, dst)
        self.pool.append(dst)
        while len(self.pool) > self.max_pool:
            old = self.pool.pop(0)
            self._cache.pop(old, None)
            old.unlink(missing_ok=True)
        return dst

    def sample(self) -> tuple[str, NetPolicy | None]:
        """Returns ('self', None) | ('pool', opponent policy) | ('valve', None)."""
        r = self.rng.random()
        if r < P_VALVE:
            return "valve", None
        if r < P_VALVE + P_POOL and self.pool:
            path = self.rng.choice(self.pool)
            if path not in self._cache:
                self._cache[path] = NetPolicy(path, greedy=False)
            return "pool", self._cache[path]
        return "self", None
