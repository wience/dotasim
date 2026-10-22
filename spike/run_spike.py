"""SPIKE (throwaway). Launch a headless Dota 2 1v1 server, receive worldstate over TCP,
hand actions to the Lua bot via file, measure throughput and resource use."""
import asyncio, os, sys, time, subprocess, struct
import psutil
import bot_script_pb2 as pb

DOTA = r"C:\Program Files (x86)\Steam\steamapps\common\dota 2 beta"
BIN = os.path.join(DOTA, "game", "bin", "win64", "dota2.exe")
BOTS = os.path.join(DOTA, "game", "dota", "scripts", "vscripts", "bots")
CONSOLE_LOG = os.path.join(DOTA, "game", "dota", "console.log")
TIMESCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 10
DURATION = float(sys.argv[3]) if len(sys.argv) > 3 else 180.0
PORTS = {2: 12120, 3: 12121}  # radiant, dire

ARGS = [BIN, "-dedicated", "-insecure", "-console", "-condebug", "-novid", "-nosound", "-nowatchdog", "-dev",
        "+dota_force_gamemode", "21", "+dota_1v1_skip_strategy", "1", "+host_timescale", str(TIMESCALE),
        "+sv_lan", "1", "+sv_cheats", "1", "+sv_hibernate_when_empty", "0",
        "+map", "start", "gamemode", "21", "-fill_with_bots",
        "-botworldstatetosocket_radiant", str(PORTS[2]), "-botworldstatetosocket_dire", str(PORTS[3]),
        "-botworldstatetosocket_frames", str(FRAMES), "-botworldstatesocket_threaded",
        "+dota_auto_surrender_all_disconnected_timeout", "180"]

stats = {t: dict(n=0, first=None, last=None, first_wall=None, last_wall=None, bad=0, states=set(), units_max=0)
         for t in PORTS}
step = {t: 0 for t in PORTS}

def write_action(team, ws):
    if "idx" not in stats[team]: return          # not calibrated yet
    stats[team]["idx"] += 1
    idx = stats[team]["idx"]
    step[team] += 1
    me = next((u for u in ws.units if u.unit_type == pb.CMsgBotWorldState.UnitType.Value("HERO") and u.team_id == team), None)
    if me is None:
        body = f"return {{idx={idx}, kind='none'}}"
    else:
        tx, ty = (-600, -600) if team == 2 else (600, 600)   # river, between the mid T1 towers
        far = ((me.location.x - tx) ** 2 + (me.location.y - ty) ** 2) ** 0.5 > 400
        body = f"return {{idx={idx}, kind='move', x={tx}, y={ty}}}" if far else f"return {{idx={idx}, kind='attack'}}"
    # one file per index: never rewrite a file the game may be reading (Windows locks it)
    tmp = os.path.join(BOTS, f"actions_{team}_{idx}.tmp")
    with open(tmp, "w") as f: f.write(body)
    os.replace(tmp, os.path.join(BOTS, f"actions_{team}_{idx}.lua"))
    old = os.path.join(BOTS, f"actions_{team}_{idx-5}.lua")
    try: os.remove(old)
    except OSError: pass

async def handle(team, reader, writer):
    s = stats[team]
    print(f"[py] team {team}: game connected on port {PORTS[team]}", flush=True)
    while True:
        hdr = await reader.readexactly(4)
        n = struct.unpack("<I", hdr)[0]
        if n > 50_000_000: n = struct.unpack(">I", hdr)[0]
        buf = await reader.readexactly(n)
        ws = pb.CMsgBotWorldState()
        try: ws.ParseFromString(buf)
        except Exception as e:
            s["bad"] += 1; continue
        now = time.time()
        if s["first"] is None:
            s["first"], s["first_wall"] = ws.dota_time, now
        if s["n"] < 25: print(f"[py-ws] team {team} #{s['n']+1} dota_time={ws.dota_time:.3f} state={ws.game_state}", flush=True)
        if ws.game_state == 4 and not s.get("calib") and os.path.isdir(BOTS):
            s["calib"] = True; s["idx"] = -1
            with open(os.path.join(BOTS, f"calib_{team}.lua"), "w") as f: f.write(f"return {{t={ws.dota_time:.3f}, n={FRAMES}}}")
            print(f"[py] team {team} wrote calib {ws.dota_time:.3f}", flush=True)
        s["last"], s["last_wall"], s["n"] = ws.dota_time, now, s["n"] + 1
        s["states"].add(ws.game_state); s["units_max"] = max(s["units_max"], len(ws.units))
        if s["n"] == 500 and team == 2:
            with open("ws_dump.txt", "w") as f:
                f.write(f"dota_time={ws.dota_time} state={ws.game_state}\nPLAYERS:\n")
                for p in ws.players: f.write("  " + str(p).replace("\n", " ") + "\n")
                f.write("HEROES:\n")
                for u in ws.units:
                    if u.unit_type == pb.CMsgBotWorldState.UnitType.Value("HERO"):
                        f.write(f"  name={u.name} team={u.team_id} pid={u.player_id} alive={u.is_alive} hp={u.health}/{u.health_max} lvl={u.level} loc=({u.location.x:.0f},{u.location.y:.0f}) lasthits={getattr(u,'last_hits',-1)}\n")
            print("[py] wrote ws_dump.txt", flush=True)
        if s["n"] % 50 == 1:
            print(f"[py] team {team} msg#{s['n']} dota_time={ws.dota_time:.2f} game_state={ws.game_state} units={len(ws.units)} bytes={n}", flush=True)
        write_action(team, ws)

async def connect(team):
    """The game HOSTS the worldstate port; we connect to it (with retries while it boots)."""
    for _ in range(600):
        try:
            r, w = await asyncio.open_connection("127.0.0.1", PORTS[team])
            await handle(team, r, w); return
        except ConnectionRefusedError:
            await asyncio.sleep(0.5)
        except OSError as e:
            print(f"[py] team {team}: socket error {e!r}, reconnecting", flush=True); await asyncio.sleep(0.5)
        except asyncio.IncompleteReadError:
            print(f"[py] team {team}: stream closed", flush=True); return
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[py] team {team}: handler died: {e!r}", flush=True); return
    print(f"[py] team {team}: never connected", flush=True)

async def main():
    for t in PORTS:
        if not os.path.isdir(BOTS): break
        for name in os.listdir(BOTS):
            if name.startswith(("actions_", "calib_")): os.remove(os.path.join(BOTS, name))
    if os.path.exists(CONSOLE_LOG): os.remove(CONSOLE_LOG)
    print("[py] launching:", " ".join(ARGS[1:]), flush=True)
    proc = subprocess.Popen(ARGS, cwd=os.path.dirname(BIN))
    tasks = [asyncio.create_task(connect(t)) for t in PORTS]
    ps = psutil.Process(proc.pid)
    cpu, rss, priv, t0 = [], [], [], time.time()
    try:
        while time.time() - t0 < DURATION and proc.poll() is None:
            await asyncio.sleep(5)
            try:
                procs = [ps] + ps.children(recursive=True)
                for p in procs: p.cpu_percent(None)
                await asyncio.sleep(1.0)
                cpu.append(sum(p.cpu_percent(None) for p in procs)); rss.append(sum(p.memory_info().rss for p in procs) / 2**20)
                priv.append(sum(p.memory_info().private for p in procs) / 2**20)
                if len(cpu) == 2: print("[py] process tree:", [(p.pid, p.name()) for p in procs], flush=True)
            except psutil.Error: break
            tot = sum(s["n"] for s in stats.values())
            print(f"[py] t+{time.time()-t0:5.0f}s msgs={tot} cpu%={cpu[-1]:.0f} rss={rss[-1]:.0f}MB", flush=True)
            if len(cpu) == 4:
                out = subprocess.run("netstat -an | findstr 1212", shell=True, capture_output=True, text=True).stdout
                print("[py] netstat:", " | ".join(out.split()) or "(no 1212x ports)", flush=True)
    finally:
        try:
            for p in ps.children(recursive=True): p.kill()
        except psutil.Error: pass
        if proc.poll() is None: proc.kill()
        for tk in tasks: tk.cancel()
    print("\n=== RESULTS ===")
    print(f"timescale={TIMESCALE} frames_per_ws={FRAMES} exit_code={proc.poll()}")
    for t, s in stats.items():
        if s["n"]:
            dt, dw = s["last"] - s["first"], s["last_wall"] - s["first_wall"]
            print(f"team {t}: msgs={s['n']} bad={s['bad']} dota_time {s['first']:.1f}->{s['last']:.1f} ({dt:.1f}s) over wall {dw:.1f}s => {dt/max(dw,1e-6):.2f}x realtime; "
                  f"{s['n']/max(dw,1e-6):.1f} msg/s; states={sorted(s['states'])}; max_units={s['units_max']}")
        else: print(f"team {t}: NO MESSAGES")
    if cpu:
        h = len(cpu) // 2
        print(f"dota2.exe cpu% avg={sum(cpu)/len(cpu):.0f} max={max(cpu):.0f} (100 = one core); rss avg={sum(rss)/len(rss):.0f}MB max={max(rss):.0f}MB; "
              f"private bytes: 2nd-half avg={sum(priv[h:])/max(len(priv[h:]),1):.0f}MB max={max(priv):.0f}MB; samples={priv}")
    if os.path.exists(CONSOLE_LOG):
        lines = [l.rstrip() for l in open(CONSOLE_LOG, errors="replace") if "[SPIKE]" in l or "hero_selection" in l.lower() or "error" in l.lower() or "lua" in l.lower()]
        print(f"console.log: {len(lines)} relevant lines, last 12:"); print("\n".join(lines[-12:]))
    else: print("console.log not found")

asyncio.run(main())
