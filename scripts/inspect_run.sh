#!/bin/bash
# Print per-instance outcome of a fleet run plus the last game states from each console log.
# usage: inspect_run.sh <run_dir>
D="${1:?run dir}"
BOTS=~/dota2/game/dota/scripts/vscripts/bots
python3 - "$D" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1] + "/status.json"))
print("totals:", d["totals"])
for i in d["instances"]:
    print(i["index"], i["phase"], round(i["game_time"]), i["kills"], (i["last_error"] or "")[:150])
EOF
for f in "$BOTS"/console_*.log; do
  echo "$(basename "$f"): $(grep -o 'DOTA_GAMERULES_STATE_[A-Z_]*' "$f" | tail -1)  post_game_lines=$(grep -c POST_GAME "$f")"
done
echo "leftover dota2 processes: $(pgrep -c dota2 || echo 0)"
