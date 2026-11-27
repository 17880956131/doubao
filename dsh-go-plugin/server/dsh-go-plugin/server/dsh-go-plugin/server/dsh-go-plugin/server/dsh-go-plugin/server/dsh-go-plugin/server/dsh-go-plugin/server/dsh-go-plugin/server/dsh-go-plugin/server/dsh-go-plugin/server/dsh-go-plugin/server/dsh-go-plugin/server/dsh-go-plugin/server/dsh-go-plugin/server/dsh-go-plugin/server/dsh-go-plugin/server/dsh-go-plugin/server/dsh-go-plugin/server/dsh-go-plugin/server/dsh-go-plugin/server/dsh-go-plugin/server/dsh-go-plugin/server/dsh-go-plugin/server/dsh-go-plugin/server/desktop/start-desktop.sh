#!/usr/bin/env bash
# 在电脑上运行豆宝（macOS / Linux）
set -e
cd "$(dirname "$0")/.."

if ! curl -s -m 1 http://127.0.0.1:8100/api/health > /dev/null 2>&1; then
  echo "启动豆宝服务…"
  nohup python3 server.py --port 8100 > data/server-8100.log 2>&1 &
  for i in $(seq 1 30); do
    if curl -s -m 1 http://127.0.0.1:8100/api/health > /dev/null 2>&1; then break; fi
    sleep 0.5
  done
fi

echo "豆宝在 http://127.0.0.1:8100"
if command -v open > /dev/null 2>&1; then
  open http://127.0.0.1:8100
elif command -v xdg-open > /dev/null 2>&1; then
  xdg-open http://127.0.0.1:8100
else
  echo "请手动用浏览器打开 http://127.0.0.1:8100"
fi
