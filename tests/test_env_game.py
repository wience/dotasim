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
