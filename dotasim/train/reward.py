"""Potential-based shaping plus terminal win/loss for the 1v1 mirror.

phi(s) reads the observer snapshot stored on the worldstate (ws.stats, set by the env or the episode loader),
so enemy terms do not jump when the enemy hero leaves vision.
"""
from __future__ import annotations

from dotasim.env.worldstate import WorldState

GAMMA = 0.9995
W_LH, W_DN, W_LVL, W_NW, W_HP = 0.02, 0.01, 0.1, 0.0005, 0.5
TOWER_MID = {2: "npc_dota_goodguys_tower1_mid", 3: "npc_dota_badguys_tower1_mid"}
PID = {2: 0, 3: 5}   # player ids of the 1v1 heroes


def _stat(ws: WorldState, pid: int, key: str) -> float:
    return float(max(ws.stats.get(pid, {}).get(key, -1), 0))


def _hp_frac(ws: WorldState, team: int) -> float:
    cached = ws.stats.get(PID[team], {}).get("hp_frac")
    if cached is not None:
        return float(cached)   # last-known value, carried while out of vision
    h = ws.hero_of(team)
    if h is None or not h.alive or not h.hp_max:
        return 0.0
    return h.hp / h.hp_max


def potential(ws: WorldState, team: int) -> float:
    me, foe = PID[team], PID[5 - team]
    d = lambda key: _stat(ws, me, key) - _stat(ws, foe, key)
    return (W_LH * d("last_hits") + W_DN * d("denies") + W_LVL * d("level") + W_NW * d("net_worth")
            + W_HP * (_hp_frac(ws, team) - _hp_frac(ws, 5 - team)))


def step_reward(prev: WorldState, cur: WorldState, team: int, gamma: float = GAMMA) -> float:
    return gamma * potential(cur, team) - potential(prev, team)


def winner(ws: WorldState) -> int | None:
    """2, 3 or None (draw), for a game that has ended. On this build the 1v1 ends at the first hero death (measured:
    every finished eval game had exactly one death), so the team with fewer deaths wins; kills alone miss deaths to
    the tower. Fallback: the team whose enemy mid T1 is gone."""
    deaths = {2: 0, 3: 0}
    for pid, d in ws.player_deaths.items():
        t = 2 if pid < 5 else 3
        deaths[t] = max(deaths[t], d)
    if deaths[2] != deaths[3]:
        return 2 if deaths[2] < deaths[3] else 3
    alive = {t: any(u.name == name and u.alive for u in ws.units) for t, name in TOWER_MID.items()}
    present = {t: any(u.name == name for u in ws.units) for t, name in TOWER_MID.items()}
    if present[3] and not alive[3]:
        return 2
    if present[2] and not alive[2]:
        return 3
    return None


def terminal_reward(team: int, winner_team: int | None) -> float:
    if winner_team is None:
        return 0.0
    return 1.0 if winner_team == team else -1.0
