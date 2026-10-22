import json
from pathlib import Path

import numpy as np
import pytest

from dotasim.env.worldstate import WorldState
from dotasim.train.reward import GAMMA, potential, step_reward, terminal_reward, winner

DATA = Path(__file__).resolve().parents[1] / "data" / "sf_mirror_v1"
TRACE = Path(__file__).parent / "golden" / "reward_trace.json"


@pytest.mark.skipif(not DATA.exists(), reason="dataset not recorded")
def test_shaping_telescopes_and_matches_trace():
    from dotasim.data.record import load_episode

    steps = load_episode(sorted(DATA.glob("ep*_t2.npz"))[0])
    r = np.array([step_reward(prev, cur, 2, GAMMA) for prev, cur in zip(steps[:-1], steps[1:])])
    assert np.all(np.abs(r) < 1.5), r[np.abs(r) >= 1.5][:5]   # a kill swings hp and stats terms at once
    T = len(r)
    disc = float(np.sum(r * GAMMA ** np.arange(T)))
    expected = GAMMA ** T * potential(steps[-1], 2) - potential(steps[0], 2)
    assert abs(disc - expected) < 1e-4, (disc, expected)
    # enemy leaving or entering vision must not spike the reward: enemy terms come from the observer snapshot
    vis = np.array([ws.hero_of(3) is not None for ws in steps[1:]])
    deaths = np.array([ws.player_deaths.get(5, 0) + ws.player_deaths.get(0, 0) for ws in steps[1:]])
    dead_before = np.array([ws.stats.get(5, {}).get("hp_frac", 1.0) == 0.0 for ws in steps[1:]])
    # vision changes that are neither a death nor a respawn
    flips = np.nonzero((vis[1:] != vis[:-1]) & (deaths[1:] == deaths[:-1]) & ~dead_before[:-1])[0]
    assert len(flips) > 0
    assert np.all(np.abs(r[flips + 1]) < 0.3), sorted(np.abs(r[flips + 1]))[-5:]
    if not TRACE.exists():
        TRACE.write_text(json.dumps([round(float(x), 6) for x in r]))
    np.testing.assert_allclose(r, np.array(json.loads(TRACE.read_text())), atol=1e-5)


def test_winner_from_player_kills():
    ws = WorldState(team=2, dota_time=300, game_time=400, game_state=6, player_kills={0: 2, 5: 0})
    assert winner(ws) == 2
    ws.player_kills = {0: 1, 5: 1}
    assert winner(ws) is None
    assert terminal_reward(2, 2) == 1.0 and terminal_reward(3, 2) == -1.0 and terminal_reward(2, None) == 0.0
