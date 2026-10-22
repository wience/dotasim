-- Scripted macro for controlled bots: fixed skill order and starting items.
-- Mirror of dotasim/model/heroes.py SKILL_ORDER / START_ITEMS; keep both in sync.
local M = {}

local SKILL_ORDER = {
    npc_dota_hero_nevermore = {
        "nevermore_shadowraze1", "nevermore_necromastery", "nevermore_shadowraze1", "nevermore_necromastery",
        "nevermore_shadowraze1", "nevermore_requiem", "nevermore_shadowraze1", "nevermore_necromastery",
        "nevermore_necromastery", "nevermore_dark_lord", "nevermore_requiem", "nevermore_dark_lord",
        "nevermore_dark_lord", "nevermore_dark_lord", "nevermore_requiem",
    },
}
local START_ITEMS = {
    npc_dota_hero_nevermore = { "item_tango", "item_branches", "item_branches", "item_circlet", "item_faerie_fire" },
}

local bought = {}

local function level_up(bot, name)
    local order = SKILL_ORDER[name]
    if order == nil or bot:GetAbilityPoints() <= 0 then return end
    local want = {}
    for _, an in ipairs(order) do
        want[an] = (want[an] or 0) + 1
        local ab = bot:GetAbilityByName(an)
        if ab ~= nil and ab:GetLevel() < want[an] then
            if ab:CanAbilityBeUpgraded() then bot:ActionImmediate_LevelAbility(an) end
            return
        end
    end
end

local function buy_start(bot, name)
    if bought[bot:GetPlayerID()] then return end
    local items = START_ITEMS[name]
    if items == nil then bought[bot:GetPlayerID()] = true; return end
    for _, it in ipairs(items) do
        if bot:GetGold() >= GetItemCost(it) then bot:ActionImmediate_PurchaseItem(it) end
    end
    bought[bot:GetPlayerID()] = true
end

-- Called every ~30 ticks by core.lua for controlled bots only.
function M.tick(bot)
    local name = bot:GetUnitName()
    local ok, err = pcall(function() level_up(bot, name); buy_start(bot, name) end)
    if not ok then print("LUAERR macro " .. tostring(err)) end
end

return M
