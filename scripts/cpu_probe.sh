#!/bin/bash
# Mean CPU% of the dota2 server during one probe episode, with the given env settings. Usage: cpu_probe.sh [VAR=1 ...]
cd "$(dirname "$0")/.."
export PATH="$HOME/dotasim-venv/bin:$PATH"
env "$@" python scripts/probe_end.py runs/bc_v2/policy.pt 90 > /tmp/cpu_probe.out 2>&1 &
pp=$!
samples=()
while kill -0 $pp 2>/dev/null; do
  c=$(ps -o %cpu= -C dota2 | awk '{s+=$1} END {printf "%.0f", s}')
  [ -n "$c" ] && [ "$c" != "0" ] && samples+=("$c")
  sleep 3
done
grep -a -E "^\{|done True" /tmp/cpu_probe.out | tail -1 | cut -c1-120
n=${#samples[@]}; s=0; for v in "${samples[@]}"; do s=$((s+v)); done
echo "settings: $* | dota2 mean cpu% ${samples[@]:+$((s/n))} over $n samples"
