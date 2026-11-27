#!/bin/bash
# 守护进程：服务器死了自动拉起，保证"她"还在
cd /root/workspace/digital-human || exit 1
while true; do
  if ! pgrep -f "python3 server.py --port 8100" >/dev/null 2>&1; then
    echo "$(date '+%F %T') restarting server" >> data/guardian.log
    nohup python3 server.py --port 8100 >> data/server-8100.log 2>&1 &
  fi
  sleep 5
done
