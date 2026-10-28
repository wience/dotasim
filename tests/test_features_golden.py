from pathlib import Path

import numpy as np
import pytest

from dotasim.env.worldstate import parse
from dotasim.model.features import HERO_DIM, K, UNIT_DIM, featurize
from dotasim.model.labels import ATTACK, CAST_BASE, MOVE, NONE, label

GOLDEN = Path(__file__).parent / "golden" / "features_6965.npz"
DATA = Path(__file__).resolve().parents[1] / "data" / "sf_mirror_v1"


def test_golden_features(golden_ws_bytes):
    ws = parse(golden_ws_bytes)
    f = featurize(ws, 2)
    assert f.units.shape == (K, UNIT_DIM) and f.units.dtype == np.float32
    assert f.hero.shape == (HERO_DIM,)
    assert f.unit_mask.sum() == min(K, len(ws.units))
    assert f.units[0, 5] == 0 and f.units[0, 6] == 0 and f.unit_handles[0] == ws.hero_of(2).handle  # self first, dx=dy=0
    assert f.type_mask[NONE] and f.type_mask[MOVE] and f.type_mask[ATTACK]
    g = np.load(GOLDEN)
    for k in ("units", "unit_mask", "unit_handles", "hero", "type_mask"):
        np.testing.assert_allclose(getattr(f, k), g[k], rtol=1e-5, atol=1e-6, err_msg=k)


@pytest.mark.skipif(not DATA.exists(), reason="dataset not recorded")
def test_labels_on_recorded_episode():
    from dotasim.data.record import load_episode

    steps = load_episode(sorted(DATA.glob("ep*_t2.npz"))[0])
    n = {"attack_named": 0, "attack": 0, "move_disp": 0, "move": 0, "cast": 0, "none": 0}
    dirs = []
    for i, (ws, nxt) in enumerate(zip(steps[:-1], steps[1:])):
        f = featurize(ws, 2)
        far = steps[min(i + 6, len(steps) - 1)]
        y = label(ws, nxt, f, 2, far)
        me, me_next = ws.hero_of(2), far.hero_of(2)
        dirs.append(y.move_dir if y.action_type == MOVE else None)
        if y.action_type == ATTACK:
            n["attack"] += 1
            tgt = f.unit_handles[y.target_idx]
            assert y.target_idx < f.unit_mask.sum()
            unit = next(u for u in ws.units if u.handle == tgt)
            assert unit.alive
            n["attack_named"] += me.attack_target != ""
        elif y.action_type == MOVE:
            n["move"] += 1
            assert 0 <= y.move_dir < 16 and 0 <= y.move_dist < 4
            n["move_disp"] += 100 <= ((me_next.x - me.x) ** 2 + (me_next.y - me.y) ** 2) ** 0.5 <= 1500
        elif y.action_type >= CAST_BASE:
            n["cast"] += 1
        else:
            n["none"] += 1
            assert y.target_idx == -1 and y.move_dir == -1
    assert n["attack"] > 50 and n["attack_named"] / n["attack"] > 0.95, n
    assert n["move"] > 200 and n["move_disp"] == n["move"], n
    assert n["cast"] >= 5, n
    pairs = [(a, b) for a, b in zip(dirs[:-1], dirs[1:]) if a is not None and b is not None]
    assert sum(a == b for a, b in pairs) / len(pairs) > 0.7, "move direction labels should mostly persist step to step"


def test_dire_sees_the_canonical_frame(golden_ws_bytes):
    """Both heroes must see the enemy's mid tier-2 tower down the lane axis (canonical north-east, dx + dy > 0):
    Dire's world is rotated 180 degrees so one policy plays both sides. Before this, Dire farmed 0 while
    Radiant farmed 7 (sp_v14). Without the rotation Dire's value here is about -3.5."""
    ws = parse(golden_ws_bytes)
    for team, enemy in ((2, "badguys"), (3, "goodguys")):
        f = featurize(ws, team)
        t = next(u for u in ws.units if u.name == f"npc_dota_{enemy}_tower2_mid")
        i = int(np.nonzero(f.unit_handles == t.handle)[0][0])
        assert f.units[i, 5] + f.units[i, 6] > 1.0, (team, f.units[i, 5:7])
