#!/bin/bash
# Grep ability ids out of the Linux install's pak directory (small text files are stored inline).
cd ~/dota2/game/dota || exit 1
for a in "$@"; do
  printf "%s " "$a"
  grep -a -m1 -A12 "\"$a\"" pak01_dir.vpk 2>/dev/null | grep -a -E '"ID"|AbilityBehavior' | tr -s ' \t' ' ' | tr '\n' ' ' | cut -c1-160
  echo
done
