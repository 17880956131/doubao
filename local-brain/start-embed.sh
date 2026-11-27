#!/bin/sh
cd "$(dirname "$0")" || exit 1
if [ -f embed.pid ] && kill -0 "$(cat embed.pid)" 2>/dev/null; then
  echo "向量服务已在运行：http://127.0.0.1:8182"
  exit 0
fi
nohup node embed-server.mjs --port 8182 > embed.log 2>&1 &
echo $! > embed.pid
sleep 3
echo "向量服务已启动：http://127.0.0.1:8182"
