-- SPIKE (throwaway): both sides pick Shadow Fiend; print diagnostics.
local printed = false
function Think()
    local players = GetTeamPlayers(GetTeam())
    if not printed then
        printed = true
        print(string.format("[SPIKE-SEL] team=%d gamemode=%s players=%d", GetTeam(), tostring(GetGameMode()), #players))
    end
    for _, pid in pairs(players) do
        if IsPlayerBot(pid) and GetSelectedHeroName(pid) == "" then
            SelectHero(pid, "npc_dota_hero_nevermore")
        end
    end
end
function UpdateLaneAssignments()
    local t = {}
    for _, pid in pairs(GetTeamPlayers(GetTeam())) do t[pid] = LANE_MID end
    return t
end
