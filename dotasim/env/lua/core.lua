-- dotasim bot core: role dispatch, token handshake, lockstep loop, action execution.
-- Loaded by the per-hero stubs (bot_<hero>.lua) with: Think = require("bots/core").make_think(GetBot())
local M = {}

local TICK = 1 / 30
local WAIT_TIMEOUT_S = 1.0
local FIND_RADIUS = 1600
local OBS_EVERY_TICKS = 15

-- Re-reads a Lua file under bots/ every call (require caches, so the cache entry is cleared first).
function M.load_table(name)
    local mod = "bots/" .. name
    package.loaded[mod] = nil
    local ok, v = pcall(require, mod)
    if ok and type(v) == "table" then return v end
    return nil
end

function M.config()
    return M.load_table("config") or { frames = 10, heroes = {}, roles = {} }
end

local function mint_token()
    local addr = tostring({}):match("0x(%x+)") or tostring({}):match("(%x+)$") or "0"
    return string.format("%x%s", math.floor(RealTime() * 1000) % 0xFFFFFF, addr)
end

local function safe(bot, method)
    local f = bot[method]
    if f == nil then return -1 end
    local ok, v = pcall(f, bot)
    if ok and type(v) == "number" then return v end
    return -1
end

local function find_unit(bot, name, x, y)
    local best, bestd = nil, math.huge
    local lists = {
        bot:GetNearbyCreeps(FIND_RADIUS, true),
        bot:GetNearbyHeroes(FIND_RADIUS, true, BOT_MODE_NONE),
        bot:GetNearbyTowers(FIND_RADIUS, true),
    }
    for _, list in ipairs(lists) do
        for _, u in ipairs(list) do
            if u:GetUnitName() == name then
                local p = u:GetLocation()
                local d = (p.x - x) ^ 2 + (p.y - y) ^ 2
                if d < bestd then best, bestd = u, d end
            end
        end
    end
    return best
end

local function act(st, a)
    local bot = st.bot
    if a.kind == "move" then
        bot:Action_MoveToLocation(Vector(a.x, a.y, 0))
        st.exec = st.exec + 1
    elseif a.kind == "attack" then
        local u = find_unit(bot, a.name or "", a.x or 0, a.y or 0)
        if u then
            bot:Action_AttackUnit(u, false)
            st.exec = st.exec + 1
        else
            st.mismatch = st.mismatch + 1
        end
    end
end

local function step(st)
    local bot = st.bot
    local now = DotaTime()
    if st.calib == nil then
        local c = M.load_table("calib_" .. st.token)
        if c and c.t then st.calib = c.t; st.n = c.n or st.n end
        return
    end
    -- m = ticks since the calibration worldstate; one worldstate every n ticks; block on the tick after each
    local m = math.floor((now - st.calib) / TICK + 0.5)
    if m < 1 or m % st.n ~= 1 then return end
    local want = (m - 1) / st.n
    st.waits = st.waits + 1
    local t0, data = RealTime(), nil
    while true do
        data = M.load_table("actions_" .. st.token .. "_" .. want)
        if data and data.idx == want then break end
        if RealTime() - t0 > WAIT_TIMEOUT_S then st.timeouts = st.timeouts + 1; data = nil; break end
    end
    if data then act(st, data) end
    print(string.format("LUAOBS token=%s idx=%d lh=%d dn=%d gold=%d xp=%d lvl=%d to=%d mm=%d",
        st.token, want, safe(bot, "GetLastHits"), safe(bot, "GetDenies"), safe(bot, "GetGold"),
        safe(bot, "GetXP"), safe(bot, "GetLevel"), st.timeouts, st.mismatch))
    if st.waits % 50 == 0 then
        print(string.format("LUASTAT token=%s waits=%d exec=%d timeouts=%d mismatch=%d",
            st.token, st.waits, st.exec, st.timeouts, st.mismatch))
    end
end

-- Returns the Think function for this bot's role, or nil for the Valve-AI role (no Think = default AI).
function M.make_think(bot)
    local team, pid = bot:GetTeam(), bot:GetPlayerID()
    local cfg = M.config()
    local role = (cfg.roles or {})[pid] or "filler"
    local token = mint_token()
    print(string.format("LUARDY token=%s team=%d pid=%d role=%s", token, team, pid, role))
    if role == "valve" then return nil end
    if role == "filler" then
        -- The lowest-id filler on each team is the observer: it reports stats the worldstate
        -- lacks (last hits, denies, gold, net worth, level) for every configured hero on its team.
        local observer = true
        for _, p in pairs(GetTeamPlayers(team)) do
            if (cfg.roles or {})[p] == nil and p < pid then observer = false end
        end
        if not observer then return function() end end
        local n = 0
        return function()
            n = n + 1
            if n % OBS_EVERY_TICKS ~= 0 then return end
            for i = 1, 5 do
                local u = GetTeamMember(i)
                if u ~= nil and (cfg.heroes or {})[u:GetPlayerID()] ~= nil then
                    print(string.format("LUAALLY token=%s team=%d pid=%d lh=%d dn=%d gold=%d nw=%d lvl=%d",
                        token, team, u:GetPlayerID(), safe(u, "GetLastHits"), safe(u, "GetDenies"),
                        safe(u, "GetGold"), safe(u, "GetNetWorth"), safe(u, "GetLevel")))
                end
            end
        end
    end
    local st = { token = token, team = team, bot = bot, n = cfg.frames or 10, calib = nil,
                 waits = 0, exec = 0, timeouts = 0, mismatch = 0, errs = 0 }
    return function()
        local ok, err = pcall(step, st)
        if not ok then
            st.errs = st.errs + 1
            if st.errs <= 5 then print("LUAERR token=" .. token .. " " .. tostring(err)) end
        end
    end
end

return M
