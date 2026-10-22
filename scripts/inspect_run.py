"""Print per-instance outcome of a fleet run. usage: python scripts/inspect_run.py <run_dir>"""
import json
import sys
from pathlib import Path

d = json.load(open(Path(sys.argv[1]) / "status.json"))
print("totals:", d["totals"])
for i in d["instances"]:
    print(i["index"], i["phase"], round(i["game_time"]), i["kills"], (i["last_error"] or "")[:200])
