"""Compare bc_v1 single-step move labels with the lookahead labels on one recorded episode."""
import math, sys
from pathlib import Path
import numpy as np
from dotasim.data.record import load_episode
from dotasim.model.features import featurize
from dotasim.model.labels import LOOKAHEAD, MOVE, direction_bin, label

steps = load_episode(sorted(Path(sys.argv[1] if len(sys.argv) > 1 else "data/sf_mirror_v1").glob("ep*_t2.npz"))[0])
n = len(steps) - 1
old, new = [], []
for i in range(n):
    f = featurize(steps[i], 2)
    y = label(steps[i], steps[i + 1], f, 2, steps[min(i + LOOKAHEAD, n)])
    new.append((y.action_type, y.move_dir))
    me, nx = steps[i].hero_of(2), steps[i + 1].hero_of(2)
    if y.action_type == MOVE or (me and nx and math.hypot(nx.x - me.x, nx.y - me.y) >= 25):
        old.append((MOVE, direction_bin(nx.x - me.x, nx.y - me.y)) if me and nx else (0, -1))
    else:
        old.append((y.action_type, -1))
def same_dir(seq):
    pairs = [(a, b) for a, b in zip(seq[:-1], seq[1:]) if a[0] == MOVE and b[0] == MOVE]
    return sum(a[1] == b[1] for a, b in pairs) / max(len(pairs), 1), len(pairs)
for name, seq in (("old", old), ("new", new)):
    types = np.bincount([t for t, _ in seq], minlength=7)
    print(name, "type mix", (types / n).round(3).tolist(), "consecutive moves same dir", same_dir(seq))
