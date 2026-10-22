-- SPIKE (throwaway): full takeover (bot_<hero>.lua), lockstep with Python via files.
local team = GetBot():GetTeam()
-- only the lowest player id on each team is controlled; the rest idle at the fountain
local minpid = 1e9
for _, pid in pairs(GetTeamPlayers(team)) do if pid < minpid then minpid = pid end end
local controlled = (GetBot():GetPlayerID() == minpid)
print(string.format("[SPIKE-LOAD] team=%d pid=%d controlled=%s base=%s", team, GetBot():GetPlayerID(), tostring(controlled), "require"))

local N, TICK, WAIT_TIMEOUT_S = nil, 1/30, 1.0
local probed = false
local calib_t, n_exec, n_timeout, n_waits, n_diag, n_ahead = nil, 0, 0, 0, 0, 0
local last_idx = -1

local function load_table(name)   -- name without .lua; require re-reads the file once its cache entry is cleared
    local mod = "bots/" .. name
    package.loaded[mod] = nil
    local ok, v = pcall(require, mod)
    if ok and (type(v) == "table" or type(v) == "number") then return v end
    return nil
end

local function think_impl()
    local bot = GetBot()
    local now = DotaTime()
    if calib_t == nil then
        local c = load_table("calib_" .. team)
        if type(c) == "table" then calib_t = c.t; N = c.n; print(string.format("[SPIKE] team=%d calibrated calib_t=%.3f n=%d now=%.3f", team, calib_t, N, now)) end
        return
    end
    -- m = ticks since the calibration worldstate; a worldstate is emitted every N ticks, so block on m % N == 1
    local m = math.floor((now - calib_t) / TICK + 0.5)
    if m < 1 or m % N ~= 1 then return end
    local want = (m - 1) / N
    n_waits = n_waits + 1
    local t0, data = RealTime(), nil
    while true do
        data = load_table("actions_" .. team .. "_" .. want)
        if data and data.idx == want then break end
        if RealTime() - t0 > WAIT_TIMEOUT_S then n_timeout = n_timeout + 1; data = nil; break end
    end
    if data then
        if data.idx > want then n_ahead = n_ahead + 1 end
        last_idx = data.idx
        if data.kind == "move" then bot:Action_MoveToLocation(Vector(data.x, data.y, 0)); n_exec = n_exec + 1
        elseif data.kind == "attack" then
            local c = bot:GetNearbyCreeps(1200, true)
            if #c == 0 then c = bot:GetNearbyHeroes(1200, true, BOT_MODE_NONE) end
            if #c > 0 then
                bot:Action_AttackUnit(c[1], false); n_exec = n_exec + 1
                if not probed then
                    probed = true
                    local u = c[1]
                    for _, m in ipairs({"GetHandle", "GetUnitHandle", "GetEntityIndex", "GetIndex", "GetUnitName", "GetPlayerID"}) do
                        local ok, v = pcall(function() return u[m](u) end)
                        print(string.format("[SPIKE-HANDLE] team=%d %s ok=%s v=%s", team, m, tostring(ok), tostring(v)))
                    end
                end
            end
        end
    end
    if n_waits % 50 == 0 or n_waits <= 3 then
        print(string.format("[SPIKE] team=%d m=%d want=%d got=%d dotatime=%.2f waits=%d exec=%d timeouts=%d ahead=%d pos=(%.0f,%.0f) hp=%d lasthits=%d",
            team, m, want, last_idx, now, n_waits, n_exec, n_timeout, n_ahead, bot:GetLocation().x, bot:GetLocation().y, bot:GetHealth(), bot:GetLastHits()))
    end
end

local n_err = 0
function Think()
    if not controlled then return end
    local ok, err = pcall(think_impl)
    if not ok then n_err = n_err + 1; if n_err <= 5 then print("[SPIKE-ERR] team=" .. team .. " " .. tostring(err)) end end
end
