-- Picks heroes from bots/config.lua (heroes = { [player_id] = "npc_dota_hero_..." }); fillers get Shadow Fiend.
local core = require("bots/core")
local DEFAULT = "npc_dota_hero_nevermore"

function Think()
    local cfg = core.config()
    for _, pid in pairs(GetTeamPlayers(GetTeam())) do
        if IsPlayerBot(pid) and GetSelectedHeroName(pid) == "" then
            SelectHero(pid, (cfg.heroes or {})[pid] or DEFAULT)
        end
    end
end

function UpdateLaneAssignments()
    local t = {}
    for _, pid in pairs(GetTeamPlayers(GetTeam())) do t[pid] = LANE_MID end
    return t
end
