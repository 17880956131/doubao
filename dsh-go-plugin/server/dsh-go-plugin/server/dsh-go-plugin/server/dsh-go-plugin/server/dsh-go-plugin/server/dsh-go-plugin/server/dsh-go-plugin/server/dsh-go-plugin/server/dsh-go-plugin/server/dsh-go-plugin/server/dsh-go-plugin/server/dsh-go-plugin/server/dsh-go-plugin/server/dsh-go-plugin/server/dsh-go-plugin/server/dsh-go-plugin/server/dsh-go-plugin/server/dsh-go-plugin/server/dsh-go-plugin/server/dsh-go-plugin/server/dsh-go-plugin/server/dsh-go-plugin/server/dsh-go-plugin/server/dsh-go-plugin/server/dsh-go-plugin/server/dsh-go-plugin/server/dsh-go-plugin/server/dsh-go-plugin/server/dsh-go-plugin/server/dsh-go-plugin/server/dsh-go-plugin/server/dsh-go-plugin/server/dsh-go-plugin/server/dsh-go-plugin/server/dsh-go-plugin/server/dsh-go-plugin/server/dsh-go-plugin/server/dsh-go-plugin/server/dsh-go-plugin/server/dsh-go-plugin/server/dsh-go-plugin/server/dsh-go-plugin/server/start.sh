#!/bin/sh
# 启动豆宝数字人舞台（本地服务）
cd "$(dirname "$0")" || exit 1
mkdir -p data
if [ -f data/server.pid ] && kill -0 "$(cat data/server.pid)" 2>/dev/null; then
  echo "已经在跑了：http://127.0.0.1:8100"
  exit 0
fi
nohup python3 server.py --port 8100 > data/server.log 2>&1 &
echo $! > data/server.pid
sleep 1
echo "数字人舞台已启动：http://127.0.0.1:8100"
