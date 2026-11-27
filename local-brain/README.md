# 本地大脑（手机推理）

- `llm-server.mjs`：OpenAI 兼容接口，默认 8181
- `embed-server.mjs`：向量服务，默认 8182
- `start-llm.sh` / `start-embed.sh`：启动脚本

## 模型

默认读 `/sdcard/模型文件/`：

- 对话：DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf
- 向量：embeddinggemma-300m-Q4_0.gguf

改模型路径：编辑 `start-llm.sh` 里的 `--model` / `--name`。

## 关键修复

`llm-server.mjs` 现在把 system 作为 system 消息、用户内容作为 user 消息，
让 llama.cpp 应用模型文件里的 `tokenizer.chat_template`，而不是手拼提示词。
手拼会导致回声、JSON 泄漏、答非所问。

## 依赖

```sh
npm install
# 需要 node-llama-cpp（ARM64 可跑）
```
