#!/bin/bash
# Why did a game end? State transitions and Lua lines from one instance's console log.
f=~/dota2/game/dota/scripts/vscripts/bots/console_${1:-0}.log
ls -la "$f" | cut -c25-; wc -l "$f"
grep -a -n -E "entering state|LUARDY|LUAERR|LUAOBS|Lua|SelectHero|hero|disconnect|Dropped|kicked|timed out|GAME_OVER|winner|Victory" "$f" | grep -v -i "LUAALLY\|LUASTAT" | head -60
echo "--- tail"; tail -5 "$f" | cut -c1-200
