#!/bin/bash
# CPU share of python and dota2 processes, one line each: pid %cpu rss-MB cmd
top -bn1 -w 200 | awk 'NR>7 && ($12 ~ /python|dota2/) {printf "%s %s%% %dMB %s\n", $1, $9, $6/1024, $12}'
echo "cores $(nproc) load $(cut -d' ' -f1-3 /proc/loadavg)"
