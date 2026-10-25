"""Side bias in the recorded Valve-vs-Valve games (no lockstep involved): final last hits, net worth, level and deaths
per side. If Dire is ahead here too, the asymmetry is the map, not our protocol."""
import glob, numpy as np
from dotasim.data.record import load_episode
rows = []
for f in sorted(glob.glob("data/sf_mirror_v1/ep*_t2.npz")):
    ws = load_episode(f)[-1]
    r, d = ws.stats.get(0, {}), ws.stats.get(5, {})
    rows.append((r.get("last_hits", -1), d.get("last_hits", -1), r.get("net_worth", -1), d.get("net_worth", -1),
                 r.get("level", -1), d.get("level", -1), ws.player_deaths.get(0, 0), ws.player_deaths.get(5, 0), ws.dota_time))
a = np.array(rows, float)
print("games", len(a), "end dota_time mean", a[:, 8].mean().round(0))
for name, i in (("last_hits", 0), ("net_worth", 2), ("level", 4), ("deaths", 6)):
    print(f"{name:10s} Radiant {a[:, i].mean():7.1f}  Dire {a[:, i+1].mean():7.1f}  Dire ahead in {(a[:, i+1] > a[:, i]).sum()} of {len(a)}")
