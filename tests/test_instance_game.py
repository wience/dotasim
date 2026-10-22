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
