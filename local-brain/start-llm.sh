#!/bin/sh
cd "$(dirname "$0")" || exit 1
if [ -f llm.pid ] && kill -0 "$(cat llm.pid)" 2>/dev/null; then
  echo "本地模型服务已在运行：http://127.0.0.1:8181"
  exit 0
fi
nohup node llm-server.mjs --port 8181 --name r1-1.5b --model "/sdcard/模型文件/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf" > llm.log 2>&1 &
echo $! > llm.pid
sleep 3
echo "本地模型服务已启动：http://127.0.0.1:8181"
