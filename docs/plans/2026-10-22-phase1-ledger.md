# SDD ledger — plan: docs/plans/2026-10-22-phase1-harness-fleet.md
Spec: docs/superpowers/specs/2026-10-22-dotasim-design.md (read). Branch: phase1. Python runs in WSL2 (~/dotasim-venv).
Pre-flight (shared interfaces):
- T2 -> T6: parse/summary/WorldState/hero_of — consistent.
- T4 -> T5,T6,T7: Channel.write_config/write_calib/write_action, Action, install_scripts — consistent.
- T5 -> T6: Instance.launch/handshake/worldstates/alive/stats/kill/lua_stats — consistent.
- T6 -> T7: DotaEnv.reset/step/status, Matchup — consistent; status schema reused by T8.
- T7 -> T8: status.json / ws_0.json schema — consistent.
No conflicts found.
Task 1: complete (commits cd425c4..2512dff, tests: wsl.exe -d Ubuntu -- bash /mnt/c/Users/ADMIN/Documents/GitHub/dotasim/scripts/wsl.sh python -m pytest tests/test_proto_golden.py -q → 1 passed in 0.11s)
Task 2: Ruling: CMsgBotWorldState carries no last_hits/denies/gold/xp (neither Unit nor Player). Hero keeps those fields defaulting to -1; parse fills kills/deaths from players; Lua will report lh/dn/gold/xp per step via a LUAOBS console line that env.py merges (Task 3/6). Cost if wrong: one extra parse of the console tail per step; if the log lags, those four stats lag one step.
Task 2: complete (commits 2512dff..6ae0303, tests: wsl.exe -d Ubuntu -- bash /mnt/c/Users/ADMIN/Documents/GitHub/dotasim/scripts/wsl.sh python -m pytest tests -q → 3 passed in 0.15s)
Task 3: Ruling: hero stubs assign Think = core.make_think(GetBot()) in the hero file's own scope instead of core setting a global, so the valve role is Think=nil; added a per-step LUAOBS line (lh/dn/gold/xp/lvl) per the Task 2 ruling. Lua is exercised by Task 6's end-to-end test. Cost if wrong: Task 6 fails and the stubs change.
Task 3: complete (commits 6ae0303..c5646a7, tests: wsl.exe -d Ubuntu -- bash /mnt/c/Users/ADMIN/Documents/GitHub/dotasim/scripts/wsl.sh python -m pytest tests/test_proto_golden.py tests/test_worldstate_golden.py -q → 3 passed in 0.17s)
Task 4: complete (commits c5646a7..0fecef6, tests: wsl.exe -d Ubuntu -- bash /mnt/c/Users/ADMIN/Documents/GitHub/dotasim/scripts/wsl.sh python -m pytest tests -q → 8 passed in 0.20s)
Task 5: complete (commits 0fecef6..5b298ae, tests: wsl.exe -d Ubuntu -- bash /mnt/c/Users/ADMIN/Documents/GitHub/dotasim/scripts/wsl.sh python -m pytest tests/test_instance_game.py -q -m game → 2 passed in 13.82s)
Task 6: Ruling: action index = index of the worldstate it answers (calib ws = 0), so Python resynchronises with Lua's tick-derived index after a stall instead of drifting; LUAOBS now also carries to=<timeouts> mm=<mismatch> so per-step info reflects timeouts without waiting for a LUASTAT. Cost if wrong: one extra integer pair per console line.
Task 6: complete (commits 5b298ae..4c1ba29, tests: wsl.exe -d Ubuntu -- bash /mnt/c/Users/ADMIN/Documents/GitHub/dotasim/scripts/wsl.sh python -m pytest tests/test_env_game.py -q -m game → 3 passed in 52.17s)
Task 7: Ruling: Fleet takes dota_dir and run(max_steps) beyond the plan's signature so tests can point at the install and bound a random-policy episode. Cost if wrong: two optional kwargs.
Task 7: Ruling: at POST_GAME the server stops streaming worldstates (socket stays open, later closes); DotaEnv treats a stream failure as a clean finish when the console log shows DOTA_GAMERULES_STATE_POST_GAME within 3 s, else an error. Found by the 12-instance test (2 of 12 marked dead at 2 kills). Cost if wrong: a crash right after POST_GAME would be reported as a clean finish.
Task 7: Ruling: inference runs per instance (one policy.act per env step), not batched; batching is deferred until a real model measures over budget. Cost if wrong: phase 3 adds a gather-then-batch loop.
Task 7: complete (commits 4c1ba29..d83efe1, tests: wsl.exe -d Ubuntu -- bash /mnt/c/Users/ADMIN/Documents/GitHub/dotasim/scripts/wsl.sh python -m pytest tests/test_fleet_game.py -q -m game → 2 passed in 134.59s (0:02:14))
Task 8: complete (commits d83efe1..cfd9a0a, tests: wsl.exe -d Ubuntu -- bash /mnt/c/Users/ADMIN/Documents/GitHub/dotasim/scripts/wsl.sh python -m pytest tests -q -m 'not game' → 8 passed, 8 deselected in 0.30s)
Task 8: verified in the browser: 2 live instances, table + minimap rendering, no console errors (docs/img/phase1-status-page.jpg).
Task 9: Ruling: live LAN spectating is deferred after three rejected connects (NOLOBBY, BADRELAYPASSWORD, INVALID); every instance now runs SourceTV with auto-record, which the Windows client plays back. Cost if wrong: SourceTV costs a little CPU per instance; measured fleet numbers predate it.
Task 9: complete (commits cfd9a0a..f8c6c01, tests: wsl.exe -d Ubuntu -- bash /mnt/c/Users/ADMIN/Documents/GitHub/dotasim/scripts/wsl.sh python -m pytest tests/test_instance_game.py tests/test_channel_golden.py -q → 7 passed in 10.32s)
Task 10: full suite run in WSL2: -m 'not slow' -> 15 passed in 277.5s; -m slow -> 1 passed in 130.1s; tag phase1.
Task 10: complete (commits f8c6c01..fbc2b89, tests: wsl.exe -d Ubuntu -- bash /mnt/c/Users/ADMIN/Documents/GitHub/dotasim/scripts/wsl.sh python -m pytest -q -m 'not game' → 8 passed, 8 deselected in 0.29s)
Final review: fresh reviewer (fable) on cd425c4..fbc2b89: 0 critical, 8 important, 14 minor. Re-graded by effect: all 8 important enter the fix pass (each a small change with real effect on long runs).
Final: fixed handshake timeout untested — test_handshake_times_out_with_console_tail_when_lua_never_reports RED→GREEN; error now names tokens seen and carries the console tail.
Final: fixed _poll_log splitting partial lines — test_split_luardy_line_is_parsed_once_complete, test_split_post_game_line_is_seen RED→GREEN (consume only to the last newline).
Final: fixed server/TV port check (UDP, not TCP) — test_launch_fails_fast_when_udp_server_port_is_bound RED→GREEN.
Final: fixed stalled server taking 8 s — test_stalled_server_is_done_within_five_seconds RED→GREEN (no post-game wait after a read timeout; dead process short-circuits).
Final: fixed worldstate index as message counter — test_worldstate_index_matches_tick_index RED→GREEN (tick-derived index, same formula as Lua). Ruling: a dropped-message injection test is not possible without mocking the socket, which the test rules forbid; the formula equality is the pin. Cost if wrong: a drop desyncs one step instead of permanently.
Final: fixed two-controlled-team close path — test_two_controlled_teams_close_cleanly_after_death kept as regression (gather return_exceptions, aclose guarded, kill unconditional).
Final: fixed dead 120 s connect budget — first worldstate in reset now uses CONNECT_TIMEOUT_S; covered by the existing reset path in every game test (no RED test possible without a slow host).
Final: fixed one exception killing the fleet — test_one_policy_exception_does_not_kill_the_fleet RED→GREEN.
Final: fixed committed egg-info (rode along): untracked and ignored.
Final: minor (deferred): calib/actions files for finished episodes linger in bots/ until the next install_scripts.
Final: minor (deferred): lua_literal emits inf/nan for non-finite floats (Lua syntax error -> shows as a timeout).
Final: minor (deferred): status.html uses innerHTML for last_error and hero names (local page).
Final: minor (deferred): handshake requires tokens for both teams; an all-filler team times out with no hint.
Final: minor (deferred): info["post_game"] is also set on cap_s end; misleading name.
Final: minor (deferred): test_lua_literal_types is a unit-style test under the user's test rule; LUASTAT exec counter is parsed but never asserted.
Final: minor (deferred): frames=1 never blocks (m % n ~= 1); token entropy comment; inspect_run.sh duplicates inspect_run.py; launch.json hardcodes paths; run_dir default on /mnt/c; repeated last-ws lookup in env.py; _out opened before exec (now closed on failure).
Suite after fix pass: -m "not slow" 23 passed (272.6 s); -m slow 1 passed (126.9 s).
