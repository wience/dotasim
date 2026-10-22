"""Console-log parsing against a real file the way the game writes it: appended, sometimes mid-line."""
from pathlib import Path

from dotasim.env.instance import Instance, InstanceConfig


def _instance(tmp_path: Path) -> Instance:
    cfg = InstanceConfig(index=0, dota_dir=tmp_path)
    cfg.bots_dir.mkdir(parents=True)
    cfg.console_log.write_text("")
    return Instance(cfg)


def test_split_luardy_line_is_parsed_once_complete(tmp_path):
    inst = _instance(tmp_path)
    log = inst.cfg.console_log
    with open(log, "a") as f:
        f.write("10/22 21:00:00 [VScript] LUARDY token=abc123 team=2 pid=0 role=cont")
    inst.console_tail()
    assert inst.lua.tokens == {}
    with open(log, "a") as f:
        f.write("rolled\n10/22 21:00:00 [VScript] LUARDY token=def456 team=3 pid=5 role=valve\n")
    inst.console_tail()
    assert inst.lua.tokens == {2: "abc123", 3: "def456"}
    assert inst.lua.roles == {"abc123": "controlled", "def456": "valve"}


def test_ally_stats_line_is_parsed(tmp_path):
    inst = _instance(tmp_path)
    with open(inst.cfg.console_log, "a") as f:
        f.write("10/22 21:00:00 [VScript] LUAALLY token=abc team=2 pid=0 lh=3 dn=1 gold=640 nw=900 lvl=2\n")
    inst.console_tail()
    assert inst.lua.ally[0] == {"last_hits": 3, "denies": 1, "gold": 640, "net_worth": 900, "level": 2}


def test_split_post_game_line_is_seen(tmp_path):
    inst = _instance(tmp_path)
    log = inst.cfg.console_log
    with open(log, "a") as f:
        # a POST_GAME line before our bots report in belongs to the previous server exiting into the same file
        f.write("S:Gamerules: entering state 'DOTA_GAMERULES_STATE_POST_GAME'\n")
    inst.console_tail()
    assert inst.post_game_seen is False
    with open(log, "a") as f:
        f.write("10/22 21:00:00 [VScript] LUARDY token=abc team=2 pid=0 role=controlled\n")
        f.write("S:Gamerules: entering state 'DOTA_GAMERULES_STATE_PO")
    inst.console_tail()
    assert inst.post_game_seen is False
    with open(log, "a") as f:
        f.write("ST_GAME'\n")
    inst.console_tail()
    assert inst.post_game_seen is True
