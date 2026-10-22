from dotasim.env.proto import bot_script_pb2 as pb


def test_golden_worldstate_parses(golden_ws_bytes):
    ws = pb.CMsgBotWorldState()
    ws.ParseFromString(golden_ws_bytes)
    assert ws.team_id == 2 and ws.game_state == 5
    hero = pb.CMsgBotWorldState.UnitType.Value("HERO")
    heroes = [u for u in ws.units if u.unit_type == hero]
    assert len(heroes) == 10
    assert all(u.name == "npc_dota_hero_nevermore" for u in heroes)
    assert ws.units[0].HasField("action_type")
