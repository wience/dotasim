"""Opponent pool for self-play: latest vs latest, a past checkpoint, or Valve's unfair bots."""
from __future__ import annotations

import random
import shutil
from pathlib import Path

import torch

from dotasim.train.policy import NetPolicy

P_SELF, P_POOL, P_VALVE = 0.8, 0.2, 0.0   # Valve waves (12 games all lost or drawn vs unfair, advantages normalised within the wave) taught hiding: sp_v9, sp_v10


class League:
    def __init__(self, pool_dir: Path, frames: int, max_pool: int = 20, seed: int | None = None):
        # seed 0 gave S x12, P, S, S, P ... and no Valve wave until iteration 36; default to fresh entropy
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.frames = frames
        self.max_pool = max_pool
        self.rng = random.Random(seed)
        self._cache: dict[Path, NetPolicy] = {}
        self.pool: list[Path] = []
        for p in sorted(self.pool_dir.glob("*.pt")):   # pre-existing pool files get the same frames check as add()
            if int(torch.load(p, map_location="cpu")["frames"]) == frames:
                self.pool.append(p)

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
                self._cache[path].name = f"pool {Path(path).stem}"
            return "pool", self._cache[path]
        return "self", None
