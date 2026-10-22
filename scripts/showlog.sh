#!/bin/bash
# Summarise an instance's console log: states, Lua lines, errors. usage: showlog.sh <index> [pattern]
I="${1:-0}"
PAT="${2:-}"
L="${DOTASIM_DOTA_DIR:-$HOME/dota2}/game/dota/scripts/vscripts/bots/console_${I}.log"
echo "log: $L ($(stat -c %s "$L" 2>/dev/null || echo missing) bytes, $(wc -l < "$L" 2>/dev/null || echo 0) lines)"
echo "states: $(grep -o 'DOTA_GAMERULES_STATE_[A-Z_]*' "$L" 2>/dev/null | tr '\n' ' ')"
echo "LUARDY=$(grep -c LUARDY "$L" 2>/dev/null) LUAOBS=$(grep -c LUAOBS "$L" 2>/dev/null) LUAALLY=$(grep -c LUAALLY "$L" 2>/dev/null) LUAERR=$(grep -c LUAERR "$L" 2>/dev/null)"
grep "LUAERR\|LUAALLY" "$L" 2>/dev/null | head -6 | cut -c1-220
if [ -n "$PAT" ]; then echo "--- $PAT ---"; grep -i "$PAT" "$L" 2>/dev/null | grep -v CommandLine | head -12 | cut -c1-220; fi
