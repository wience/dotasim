"""Print one worldstate (the golden frame) in human terms."""
import collections
from pathlib import Path
from dotasim.env.worldstate import parse

raw = next(Path("tests/golden").glob("*.bin")).read_bytes() if list(Path("tests/golden").glob("*.bin")) else None
if raw is None:
    import glob
    raw = Path(sorted(glob.glob("tests/golden/*ws*"))[0]).read_bytes()
ws = parse(raw)
print("bytes", len(raw), "team", ws.team, "dota_time", round(ws.dota_time, 1), "game_state", ws.game_state, "units", len(ws.units))
print("kinds", dict(collections.Counter(u.kind for u in ws.units)))
me = ws.hero_of(ws.team)
print("hero", {k: getattr(me, k) for k in ("name", "player_id", "x", "y", "hp", "hp_max", "mana", "level", "attack_damage", "facing", "action_type", "attack_target_handle", "is_using_ability", "alive")})
print("abilities (id, level, cooldown, castable)", me.abilities[:6])
print("items", me.item_ids)
for u in sorted(ws.units, key=lambda u: (u.x - me.x) ** 2 + (u.y - me.y) ** 2)[1:4]:
    print("nearby", {k: getattr(u, k) for k in ("name", "kind", "team", "x", "y", "hp", "hp_max", "level", "attack_damage", "alive")})
print("player_kills", ws.player_kills, "player_deaths", ws.player_deaths)
