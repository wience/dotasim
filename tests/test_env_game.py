import asyncio
import math
import os
import random
import signal
import time

import pytest

from dotasim.env.channel import Action, install_scripts
from dotasim.env.env import DotaEnv, Matchup
from dotasim.env.instance import InstanceConfig
from tests.conftest import dota_dir

SF = "npc_dota_hero_nevermore"
pytestmark = pytest.mark.game


def _matchup():
    return Matchup(heroes={0: SF, 5: SF}, roles={0: "controlled", 5: "valve"})


def _env(index=0):
    bots = install_scripts(dota_dir())
    return DotaEnv(InstanceConfig(index=index, dota_dir=dota_dir()), bots)


def _random_action(ws, rng):
    hero = ws.hero_of(2)
    creeps = [u for u in ws.units if u.kind == "creep" and u.team == 3 and u.alive]
    if creeps and rng.random() < 0.5:
        c = min(creeps, key=lambda u: math.hypot(u.x - hero.x, u.y - hero.y))
        return Action("attack", c.x, c.y, c.name)
    return Action("move", hero.x + rng.uniform(-400, 400), hero.y + rng.uniform(-400, 400))


def test_random_policy_episode():
    env = _env()
    rng = random.Random(0)

    async def run():
        obs = await env.reset(_matchup())
        assert set(obs) == {2}
        spawn = (obs[2].hero_of(2).x, obs[2].hero_of(2).y)
        last_t, info, steps = obs[2].dota_time, {}, 0
        try:
            for _ in range(300):
                obs, done, info = await env.step({2: _random_action(obs[2], rng)})
                steps += 1
                if done:
                    break
                assert 2 in obs and obs[2].dota_time > last_t
                last_t = obs[2].dota_time
            assert steps >= 200, info
            assert info["lua_timeouts"] == 0, info
            assert "error" not in info, info
            hero = obs[2].hero_of(2)
            assert math.hypot(hero.x - spawn[0], hero.y - spawn[1]) > 1000
        finally:
            await env.close()

    asyncio.run(asyncio.wait_for(run(), 400))


def test_valve_game_ends_clean():
    """A Valve-vs-Valve game ends with POST_GAME and a closed stream: that is a clean finish, not an error."""
    env = _env()

    async def run():
        await env.reset(Matchup(heroes={0: SF, 5: SF}, roles={0: "valve", 5: "valve"}, cap_s=3000))
        try:
            for _ in range(20000):
                obs, done, info = await env.step({})
                if done:
                    break
            assert done, info
            assert "error" not in info, info
            assert env.phase == "postgame"
            assert info["game_time"] < 3000
        finally:
            await env.close()

    asyncio.run(asyncio.wait_for(run(), 900))


def test_valve_heroes_carry_observer_stats():
    """Valve-driven heroes never print LUAOBS; the observer filler reports their stats instead, for both teams."""
    env = _env()

    async def run():
        obs = await env.reset(Matchup(heroes={0: SF, 5: SF}, roles={0: "valve", 5: "valve"}, watch_teams=(2, 3), cap_s=400))
        assert set(obs) == {2, 3}
        try:
            seen = False
            for _ in range(2000):
                obs, done, info = await env.step({})
                assert not done, info
                h2, h3 = obs[2].hero_of(2), obs[3].hero_of(3)
                if seen:
                    # once reported, stats never fall back to unknown
                    assert h2.last_hits >= 0 and h2.gold >= 0 and h2.net_worth >= 0 and h3.level >= 1
                    foe = obs[2].hero_of(3)  # only present in team 2's view when visible (fog of war)
                    if foe is not None:
                        assert foe.level >= 1 and foe.gold >= 0  # enemy stats arrive via team 3's observer, merged by player id
                elif h2.last_hits >= 0 and h3.gold > 0:
                    seen = True
                if obs[2].dota_time >= 120 and seen:
                    break
            assert seen, "observer stats never arrived"
        finally:
            await env.close()

    asyncio.run(asyncio.wait_for(run(), 400))


def test_controlled_bot_levels_and_buys():
    """The scripted macro levels abilities and buys starting items for the controlled bot."""
    env = _env()

    def lane_farm(ws):  # walk to our side of the river, then hit the nearest enemy creep
        hero = ws.hero_of(2)
        if math.hypot(hero.x + 700, hero.y + 650) > 400:
            return Action("move", -700, -650)
        creeps = [u for u in ws.units if u.kind == "creep" and u.team == 3 and u.alive]
        if not creeps:
            return Action()
        c = min(creeps, key=lambda u: math.hypot(u.x - hero.x, u.y - hero.y))
        return Action("attack", c.x, c.y, c.name)

    async def run():
        obs = await env.reset(Matchup(heroes={0: SF, 5: SF}, roles={0: "controlled", 5: "valve"}, difficulty=0, cap_s=200))
        try:
            done = False
            while not done and obs[2].dota_time < 150:
                obs, done, info = await env.step({2: lane_farm(obs[2])})
            me = obs[2].hero_of(2)
            assert me.level >= 2, me.level
            levels = {aid: lvl for aid, lvl, _, _ in me.abilities}
            assert max(levels.get(a, 0) for a in (5059, 5060, 5061)) >= 1, levels
            assert len(me.item_ids) >= 1, me.item_ids
            assert info["lua_timeouts"] == 0
        finally:
            await env.close()

    asyncio.run(asyncio.wait_for(run(), 400))


def test_process_death_is_done():
    env = _env()

    async def run():
        obs = await env.reset(_matchup())
        try:
            os.kill(env.instance.proc.pid, signal.SIGKILL)
            t0 = time.monotonic()
            obs, done, info = await env.step({2: Action()})
            assert done is True
            assert "error" in info
            assert time.monotonic() - t0 < 5
        finally:
            await env.close()

    asyncio.run(asyncio.wait_for(run(), 300))


def test_stalled_server_is_done_within_five_seconds():
    env = _env()

    async def run():
        await env.reset(_matchup())
        pid = env.instance.proc.pid
        try:
            os.kill(pid, signal.SIGSTOP)
            t0 = time.monotonic()
            obs, done, info = await env.step({2: Action()})
            elapsed = time.monotonic() - t0
            assert done is True and "error" in info, info
            assert elapsed < 5.5, elapsed
        finally:
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
            await env.close()

    asyncio.run(asyncio.wait_for(run(), 300))


def test_two_controlled_teams_close_cleanly_after_death():
    env = _env()

    async def run():
        obs = await env.reset(Matchup(heroes={0: SF, 5: SF}, roles={0: "controlled", 5: "controlled"}))
        assert set(obs) == {2, 3}
        pid = env.instance.proc.pid
        os.kill(pid, signal.SIGKILL)
        obs, done, info = await env.step({2: Action(), 3: Action()})
        assert done is True and "error" in info
        await env.close()  # must not raise even though both streams failed together
        assert env.instance.alive() is False

    asyncio.run(asyncio.wait_for(run(), 300))


def test_worldstate_index_matches_tick_index():
    """Python's worldstate index must be the tick-derived index Lua uses, not a message counter."""
    env = _env()

    async def run():
        obs = await env.reset(_matchup())
        calib = obs[2].dota_time
        try:
            for _ in range(60):
                obs, done, info = await env.step({2: Action()})
                if done:
                    break
            tm = env._teams[2]
            expected = round((tm.last.dota_time - calib) * 30 / env.matchup.frames)
            assert tm.ws_idx == expected, (tm.ws_idx, expected)
            assert env.ws_index(2) == expected
        finally:
            await env.close()

    asyncio.run(asyncio.wait_for(run(), 300))


def test_slow_policy_is_counted():
    env = _env()

    async def run():
        obs = await env.reset(_matchup())
        try:
            await asyncio.sleep(1.5)
            obs, done, info = await env.step({2: Action()})
            for _ in range(20):
                obs, done, info = await env.step({2: Action()})
                assert not done, info
                assert 2 in obs
            assert info["lua_timeouts"] >= 1, info
        finally:
            await env.close()

    asyncio.run(asyncio.wait_for(run(), 300))
