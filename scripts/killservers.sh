#!/bin/bash
# Stop every dotasim training process and game server, and wait until the servers have exited and their ports are free.
pkill -f "dotasim.train.selfplay" 2>/dev/null
pkill -f "dotasim.train.rollout" 2>/dev/null
pkill -TERM -f "linuxsteamrt64/dota2" 2>/dev/null
for i in $(seq 1 40); do
  n=$(pgrep -c -f "linuxsteamrt64/dota2")
  if [ "$n" = "0" ]; then break; fi
  sleep 1
done
pkill -KILL -f "linuxsteamrt64/dota2" 2>/dev/null
sleep 1
echo "servers left: $(pgrep -c -f 'linuxsteamrt64/dota2')  listening worldstate ports: $(ss -ltn | grep -c ':121[2-4][0-9] ')"
