import asyncio
import socket

import pytest

from dotasim.env.channel import Channel, install_scripts
from dotasim.env.instance import Instance, InstanceConfig, InstanceError
from dotasim.env.worldstate import parse
from tests.conftest import dota_dir

SF = "npc_dota_hero_nevermore"

pytestmark = pytest.mark.game


def test_instance_handshake_and_worldstates():
    bots = install_scripts(dota_dir())
    Channel(bots).write_config(10, {0: SF, 5: SF}, {0: "valve", 5: "valve"})
    inst = Instance(InstanceConfig(index=0, dota_dir=dota_dir()))

    async def run():
        await inst.launch()
        try:
            tokens = await inst.handshake()
            assert set(tokens) == {2, 3}
            n = 0
            async for raw in inst.worldstates(2):
                ws = parse(raw)
                assert ws.team == 2
                n += 1
                if n == 30:
                    break
            assert n == 30
            assert inst.stats()["rss_mb"] > 100
        finally:
            await inst.kill()
        assert inst.alive() is False

    asyncio.run(asyncio.wait_for(run(), 150))


def test_handshake_times_out_with_console_tail_when_lua_never_reports():
    bots = install_scripts(dota_dir())
    # No bot_axe.lua stub exists, so the radiant bot never prints LUARDY.
    Channel(bots).write_config(10, {0: "npc_dota_hero_axe", 5: SF}, {0: "controlled", 5: "valve"})
    inst = Instance(InstanceConfig(index=0, dota_dir=dota_dir()))

    async def run():
        await inst.launch()
        try:
            with pytest.raises(InstanceError) as e:
                await inst.handshake(timeout=25)
            msg = str(e.value)
            assert "no LUARDY" in msg and "tokens seen {3:" in msg  # dire reported, radiant never did
            assert "console tail:" in msg and len(msg.splitlines()) > 5
        finally:
            await inst.kill()

    asyncio.run(asyncio.wait_for(run(), 120))


def test_launch_fails_fast_when_udp_server_port_is_bound():
    cfg = InstanceConfig(index=0, dota_dir=dota_dir())
    blocker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    blocker.bind(("0.0.0.0", cfg.server_port))
    inst = Instance(cfg)
    try:
        with pytest.raises(InstanceError, match=str(cfg.server_port)):
            asyncio.run(asyncio.wait_for(inst.launch(), 2))
    finally:
        blocker.close()
        asyncio.run(inst.kill())


def test_launch_fails_fast_when_port_is_bound():
    cfg = InstanceConfig(index=0, dota_dir=dota_dir())
    blocker = socket.socket()
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("0.0.0.0", cfg.ws_port(2)))
    blocker.listen(1)
    inst = Instance(cfg)
    try:
        with pytest.raises(InstanceError, match=str(cfg.ws_port(2))):
            asyncio.run(asyncio.wait_for(inst.launch(), 2))
    finally:
        blocker.close()
        asyncio.run(inst.kill())
