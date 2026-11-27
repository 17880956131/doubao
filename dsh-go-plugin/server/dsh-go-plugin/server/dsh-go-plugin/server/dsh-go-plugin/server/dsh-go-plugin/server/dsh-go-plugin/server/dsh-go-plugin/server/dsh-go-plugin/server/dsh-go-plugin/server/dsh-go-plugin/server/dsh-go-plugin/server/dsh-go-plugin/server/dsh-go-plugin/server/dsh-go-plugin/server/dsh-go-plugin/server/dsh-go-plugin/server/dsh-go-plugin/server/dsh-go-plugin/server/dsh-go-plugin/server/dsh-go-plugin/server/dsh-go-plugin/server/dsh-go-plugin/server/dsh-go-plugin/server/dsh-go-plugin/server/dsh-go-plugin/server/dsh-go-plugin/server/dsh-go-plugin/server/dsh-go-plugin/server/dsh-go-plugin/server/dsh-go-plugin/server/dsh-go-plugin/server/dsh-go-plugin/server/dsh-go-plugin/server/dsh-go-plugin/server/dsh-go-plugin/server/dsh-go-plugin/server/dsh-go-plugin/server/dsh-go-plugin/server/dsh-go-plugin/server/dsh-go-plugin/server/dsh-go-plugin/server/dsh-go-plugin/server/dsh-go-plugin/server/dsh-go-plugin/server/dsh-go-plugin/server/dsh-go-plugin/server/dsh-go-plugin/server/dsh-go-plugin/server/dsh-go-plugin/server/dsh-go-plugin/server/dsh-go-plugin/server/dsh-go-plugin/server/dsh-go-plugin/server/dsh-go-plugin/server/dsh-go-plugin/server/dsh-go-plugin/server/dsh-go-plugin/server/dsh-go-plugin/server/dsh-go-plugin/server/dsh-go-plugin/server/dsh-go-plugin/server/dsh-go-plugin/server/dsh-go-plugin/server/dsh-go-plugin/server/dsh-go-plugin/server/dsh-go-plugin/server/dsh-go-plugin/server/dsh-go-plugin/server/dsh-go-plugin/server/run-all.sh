#!/bin/sh
# 一键启动：本地模型大脑 + 向量记忆 + 数字人舞台
cd "$(dirname "$0")" || exit 1
sh /root/workspace/local-brain/start-llm.sh
sh /root/workspace/local-brain/start-embed.sh
sh ./start.sh
