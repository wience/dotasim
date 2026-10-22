"""Per-hero tables shared by features, labels and the Lua macro (keep in sync with lua/macro.lua)."""

SF = "npc_dota_hero_nevermore"

HERO_IDS = {SF: 11}

# Active abilities keyed by ability id (slot numbers are not stable across a game on this build).
# Order defines the cast action types: CAST_BASE + index.
ACTIVE_IDS = {SF: (5059, 5060, 5061, 5064)}  # shadowraze 1, 2, 3, requiem of souls
ACTIVE_NAMES = {SF: ("nevermore_shadowraze1", "nevermore_shadowraze2", "nevermore_shadowraze3", "nevermore_requiem")}

# All current actives are no-target casts; point/unit-target casts arrive with Puck and QoP.
CAST_KIND = {SF: ("none", "none", "none", "none")}

SKILL_ORDER = {SF: ("nevermore_shadowraze1", "nevermore_necromastery", "nevermore_shadowraze1", "nevermore_necromastery",
                    "nevermore_shadowraze1", "nevermore_requiem", "nevermore_shadowraze1", "nevermore_necromastery",
                    "nevermore_necromastery", "nevermore_dark_lord", "nevermore_requiem", "nevermore_dark_lord",
                    "nevermore_dark_lord", "nevermore_dark_lord", "nevermore_requiem")}
START_ITEMS = {SF: ("item_tango", "item_branches", "item_branches", "item_circlet", "item_faerie_fire")}
