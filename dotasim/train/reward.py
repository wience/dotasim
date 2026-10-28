"""Potential-based shaping plus terminal win/loss for the 1v1 mirror.

phi(s) reads the observer snapshot stored on the worldstate (ws.stats, set by the env or the episode loader),
so enemy terms do not jump when the enemy hero leaves vision.
"""
from __future__ import annotations

from dotasim.env.worldstate import WorldState

GAMMA = 0.9995
W_LH, W_DN, W_LVL, W_NW, W_HP = 0.15, 0.05, 0.1, 0.0005, 0.1   # W_LH 0.02 -> 0.15, W_DN 0.01 -> 0.05 (OpenAI Five's per-last-hit scale). No explicit death penalty, but until st1_v1 the collector scored rule-ended games (kills, tier-1 tower) by winner()'s deaths fallback, which was a -1 for dying once; terminal is now the fort line only
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
    """Own progress in absolute terms plus the hp difference. The first version used mine-minus-enemy for every
    term; in self-play that is zero-sum, so two copies idling in their fountains score the same as two copies farming,
    and with death costing 1.5 the policy converged to standing still (sp_v7, iteration 20: level 1 after 5 min)."""
    me = PID[team]
    mine = lambda key: _stat(ws, me, key)
    return (W_LH * mine("last_hits") + W_DN * mine("denies") + W_LVL * mine("level") + W_NW * mine("net_worth")
            + W_HP * (_hp_frac(ws, team) - _hp_frac(ws, 5 - team)))


def step_reward(prev: WorldState, cur: WorldState, team: int, gamma: float = GAMMA) -> float:
    return gamma * potential(cur, team) - potential(prev, team)


def winner(ws: WorldState) -> int | None:
    """Fallback winner from the last worldstate, 2, 3 or None (draw). The engine ends a 1v1 by destroying the
    loser's fort and the stream stops before that state arrives, so the console's fort line (Instance.winner) is the
    primary source; this reads deaths (fewer wins) and then the mid T1 towers."""
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
