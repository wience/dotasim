#!/bin/bash
# Copy each server console log the first time it contains a POST_GAME line (one copy per launch), for diagnosis.
out=${1:-/tmp/pg}; mkdir -p "$out"
end=$((SECONDS + ${2:-480}))
while [ $SECONDS -lt $end ]; do
  for f in ~/dota2/game/dota/scripts/vscripts/bots/console_*.log; do
    if grep -q "STATE_POST_GAME" "$f" 2>/dev/null; then
      key=$(head -c 400 "$f" | md5sum | cut -c1-8); n=$(basename "$f" .log)
      [ -f "$out/${n}_$key.log" ] || { cp "$f" "$out/${n}_$key.log"; echo "caught $n $key"; }
    fi
  done
  sleep 2
done
ls "$out" | wc -l
