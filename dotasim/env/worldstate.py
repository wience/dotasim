"""Typed view of CMsgBotWorldState: what the policy, the reward and the UI need."""
from __future__ import annotations

from dataclasses import dataclass, field

from dotasim.env.proto import bot_script_pb2 as pb

KIND = {1: "hero", 3: "creep", 6: "tower"}  # HERO, LANE_CREEP, TOWER; everything else is "other"
TEAMS = (2, 3)


@dataclass(slots=True)
class Unit:
    handle: int
    name: str
    kind: str
    team: int
    x: float
    y: float
    facing: float
    hp: int
    hp_max: int
    mana: float
    mana_max: float
    level: int
    alive: bool
    attack_damage: int
    attack_range: int
    action_type: int
    attack_target: str


@dataclass(slots=True)
class Hero(Unit):
    player_id: int = -1
    kills: int = 0
    deaths: int = 0
    attack_target_handle: int = 0
    is_using_ability: bool = False
    active_ability_id: int = -1
    abilities: tuple = ()      # (ability_id, level, cooldown_remaining, is_fully_castable) per ability
    item_ids: tuple = ()
    # Not in the proto: filled by the env from the Lua LUAOBS (controlled) or LUAALLY (observer) lines; -1 until then.
    last_hits: int = -1
    denies: int = -1
    gold: int = -1
    net_worth: int = -1


@dataclass(slots=True)
class WorldState:
    team: int
    dota_time: float
    game_time: float
    game_state: int
    units: list[Unit] = field(default_factory=list)
    heroes: dict[int, Hero] = field(default_factory=dict)
    player_kills: dict[int, int] = field(default_factory=dict)   # from players: present even when a hero is dead or hidden
    player_deaths: dict[int, int] = field(default_factory=dict)
    stats: dict[int, dict] = field(default_factory=dict)         # observer snapshot per player id at this step (env/loader)

    def hero_of(self, team: int) -> Hero | None:
        mine = [h for h in self.heroes.values() if h.team == team]
        return min(mine, key=lambda h: h.player_id) if mine else None

    def reduced(self) -> dict:
        return {
            "t": self.dota_time,
            "units": [[round(u.x), round(u.y), u.team, u.kind, round(u.hp / u.hp_max, 2) if u.hp_max else 0.0] for u in self.units],
            "heroes": {h.player_id: {"name": h.name, "team": h.team, "hp": h.hp, "hp_max": h.hp_max, "level": h.level,
                                     "kills": h.kills, "deaths": h.deaths, "last_hits": h.last_hits, "gold": h.gold}
                       for h in self.heroes.values()},
        }


def _unit(u, cls=Unit, **extra):
    return cls(
        handle=u.handle, name=u.name, kind=KIND.get(u.unit_type, "other"), team=u.team_id,
        x=u.location.x, y=u.location.y, facing=u.facing, hp=u.health, hp_max=u.health_max,
        mana=u.mana, mana_max=u.mana_max, level=u.level, alive=u.is_alive,
        attack_damage=u.attack_damage, attack_range=u.attack_range,
        action_type=u.action_type, attack_target=u.attack_target_name, **extra,
    )


def parse(buf: bytes) -> WorldState:
    msg = pb.CMsgBotWorldState()
    msg.ParseFromString(buf)
    players = {p.player_id: p for p in msg.players}
    ws = WorldState(team=msg.team_id, dota_time=msg.dota_time, game_time=msg.game_time, game_state=msg.game_state,
                    player_kills={p.player_id: p.kills for p in msg.players},
                    player_deaths={p.player_id: p.deaths for p in msg.players})
    for u in msg.units:
        if u.unit_type == pb.CMsgBotWorldState.UnitType.Value("HERO") and u.player_id >= 0:
            p = players.get(u.player_id)
            by_handle = {a.handle: a.ability_id for a in u.abilities}
            hero = _unit(u, Hero, player_id=u.player_id, kills=p.kills if p else 0, deaths=p.deaths if p else 0,
                         attack_target_handle=u.attack_target_handle, is_using_ability=u.is_using_ability,
                         active_ability_id=by_handle.get(u.active_ability_handle, -1),
                         abilities=tuple((a.ability_id, a.level, a.cooldown_remaining, a.is_fully_castable) for a in u.abilities),
                         item_ids=tuple(a.ability_id for a in u.items))
            ws.units.append(hero)
            ws.heroes[u.player_id] = hero
        else:
            ws.units.append(_unit(u))
    return ws


def summary(ws: WorldState) -> dict:
    out = {"dota_time": round(ws.dota_time, 2), "game_state": ws.game_state, "kills": {}, "last_hits": {}}
    for t in TEAMS:
        h = ws.hero_of(t)
        out["kills"][str(t)] = h.kills if h else 0
        out["last_hits"][str(t)] = h.last_hits if h else -1
    return out
