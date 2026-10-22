"""Recorded episodes -> cached (features, labels) arrays -> windows of consecutive steps for BPTT."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dotasim.data.record import load_episode
from dotasim.model.features import featurize
from dotasim.model.labels import LOOKAHEAD, label

WINDOW = 32


def build_cache(ep_path: Path, cache_dir: Path) -> Path:
    """Featurize and label one episode file (team from the file name) into <cache_dir>/<stem>.npz."""
    out = cache_dir / (ep_path.stem + ".npz")
    if out.exists():
        return out
    team = int(ep_path.stem[-1])
    steps = load_episode(ep_path)
    n = len(steps) - 1
    feats = [featurize(ws, team) for ws in steps]
    cols = {k: [] for k in ("units", "unit_mask", "hero", "type_mask", "y_type", "y_dir", "y_dist", "y_target")}
    for i in range(n):
        f = feats[i]
        y = label(steps[i], steps[i + 1], f, team, steps[min(i + LOOKAHEAD, n)])
        cols["units"].append(f.units); cols["unit_mask"].append(f.unit_mask); cols["hero"].append(f.hero)
        cols["type_mask"].append(f.type_mask)
        cols["y_type"].append(y.action_type); cols["y_dir"].append(y.move_dir)
        cols["y_dist"].append(y.move_dist); cols["y_target"].append(y.target_idx)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out, units=np.stack(cols["units"]), unit_mask=np.stack(cols["unit_mask"]), hero=np.stack(cols["hero"]),
             type_mask=np.stack(cols["type_mask"]), y_type=np.array(cols["y_type"], np.int64),
             y_dir=np.array(cols["y_dir"], np.int64), y_dist=np.array(cols["y_dist"], np.int64),
             y_target=np.array(cols["y_target"], np.int64))
    return out


class EpisodeWindows(Dataset):
    def __init__(self, data_dir: Path, max_episodes: int | None = None, window: int = WINDOW):
        self.data_dir = Path(data_dir)
        self.meta = json.loads((self.data_dir / "meta.json").read_text())
        self.frames = int(self.meta["frames"])
        files = sorted(self.data_dir.glob("ep*_t*.npz"))
        if max_episodes:
            files = files[:max_episodes]
        cache_dir = self.data_dir / f"cache_f{self.frames}"
        self.episodes = [np.load(build_cache(f, cache_dir)) for f in files]
        self.window = window
        self.index = [(e, s) for e, ep in enumerate(self.episodes) for s in range(0, len(ep["y_type"]) - window + 1, window)]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        e, s = self.index[i]
        ep = self.episodes[e]
        sl = slice(s, s + self.window)
        return {k: torch.from_numpy(np.ascontiguousarray(ep[k][sl])) for k in
                ("units", "unit_mask", "hero", "type_mask", "y_type", "y_dir", "y_dist", "y_target")}

    def label_stats(self) -> dict:
        y = np.concatenate([ep["y_type"] for ep in self.episodes])
        counts = np.bincount(y, minlength=7)
        return {"steps": int(len(y)), "type_counts": counts.tolist(), "majority_rate": float(counts.max() / len(y))}
