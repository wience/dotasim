"""Probe: which cvar controls the Valve opponent's difficulty on a dedicated server?
Park our hero on our side of the river; count steps where the enemy hero's attack target is our hero,
how close it comes, our min hp and deaths. Passive bots should never target the hero."""
import asyncio
import os
import sys
from pathlib import Path

from dotasim.env.channel import Action, install_scripts
from dotasim.env.env import DotaEnv, Matchup
from dotasim.env.instance import InstanceConfig

SF = "npc_dota_hero_nevermore"
DOTA = Path(os.environ.get("DOTASIM_DOTA_DIR", "~/dota2")).expanduser()
PARK = (-1000, -900)

CONFIGS = {
    "set_difficulty=0": (0, []),
    "set=4+practice=0": (4, ["+dota_bot_practice_difficulty", "0"]),
    "set_difficulty=4": (4, []),
}


async def one(name: str, difficulty: int, extra: list[str], cap: float) -> dict:
    bots = install_scripts(DOTA)
    env = DotaEnv(InstanceConfig(index=0, dota_dir=DOTA, extra_args=extra), bots)
    obs = await env.reset(Matchup(heroes={0: SF, 5: SF}, roles={0: "controlled", 5: "valve"}, difficulty=difficulty, cap_s=cap))
    hp_min, deaths, near, targets_me, steps, info = 10**9, 0, 0, 0, 0, {}
    try:
        done = False
        while not done:
            obs, done, info = await env.step({2: Action("move", *PARK)})
            steps += 1
            ws = obs.get(2)
            if ws is None:
                break
            me, foe = ws.hero_of(2), ws.hero_of(3)
            if me and me.alive:
                hp_min = min(hp_min, me.hp)
            if me:
                deaths = max(deaths, me.deaths)
            if foe:
                if ((foe.x - PARK[0]) ** 2 + (foe.y - PARK[1]) ** 2) ** 0.5 < 900:
                    near += 1
                if foe.attack_target == SF:
                    targets_me += 1
        return {"config": name, "steps": steps, "game_time": round(info.get("game_time", 0)), "hp_min": hp_min,
                "deaths": deaths, "foe_near_steps": near, "foe_targets_hero_steps": targets_me,
                "ended": "error" if "error" in info else ("post_game" if info.get("post_game") else "cap")}
    finally:
        await env.close()


async def main():
    cap = float(sys.argv[1]) if len(sys.argv) > 1 else 180
    for name, (d, extra) in CONFIGS.items():
        print(await one(name, d, extra, cap), flush=True)


asyncio.run(main())
