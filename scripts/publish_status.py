"""Push a compact live snapshot of a run to a GitHub gist every N seconds, for the hosted status page (web/).
Usage (Windows or WSL, needs `gh` logged in): python scripts/publish_status.py runs/sp_v12 --gist <id> [--every 30]
Create the gist once:  gh gist create --desc "dotasim live status" live.json   (a secret gist is fine)."""
import argparse
import json
import subprocess
import time
from pathlib import Path

KEEP = ("iteration", "skipped", "steps", "ends", "outcomes", "mean_last_hits", "mean_return", "kl_anchor", "entropy",
        "act_ms", "speed_x", "update_wall_s", "wall_s", "eval", "error")


def snapshot(run: Path) -> dict:
    out = {"run": run.name, "published": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "iterations": []}
    st = run / "run" / "status.json"
    if st.exists():
        try:
            out["status"] = json.loads(st.read_text())
        except json.JSONDecodeError:
            pass
    log = run / "log.jsonl"
    if log.exists():
        for line in log.read_text().splitlines()[-200:]:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            out["iterations"].append({k: d.get(k) for k in KEEP if k in d})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--gist", required=True)
    p.add_argument("--every", type=float, default=30)
    a = p.parse_args()
    run = Path(a.run)
    tmp = Path("live.json")
    while True:
        snap = snapshot(run)
        tmp.write_text(json.dumps(snap))
        r = subprocess.run(["gh", "gist", "edit", a.gist, "-f", "live.json", str(tmp)], capture_output=True, text=True)
        n = len(snap["iterations"])
        print(time.strftime("%H:%M:%S"), "pushed" if r.returncode == 0 else f"gh failed: {r.stderr.strip()[:120]}",
              f"iterations={n}", f"last={snap['iterations'][-1].get('iteration') if n else None}", flush=True)
        time.sleep(a.every)


if __name__ == "__main__":
    main()
