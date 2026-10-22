import json
from pathlib import Path

from dotasim.env.worldstate import parse, summary

GOLDEN_JSON = Path(__file__).parent / "golden" / "worldstate_6965.json"


def test_golden_worldstate_typed(golden_ws_bytes):
    ws = parse(golden_ws_bytes)
    assert ws.team == 2
    assert len(ws.heroes) == 10
    assert ws.hero_of(2).player_id == 0
    assert ws.hero_of(3).player_id == 5
    assert all(u.kind in ("hero", "creep", "tower", "other") for u in ws.units)
    red = ws.reduced()
    assert red["t"] == ws.dota_time
    assert len(red["units"]) == len(ws.units)
    assert len(red["units"][0]) == 5


def test_golden_worldstate_summary(golden_ws_bytes):
    ws = parse(golden_ws_bytes)
    got = summary(ws)
    expected = json.loads(GOLDEN_JSON.read_text())
    assert got == expected
