# 豆宝 · 桌面智能体版

同一套服务，换到电脑上跑。电脑版的意义：能接 GPU、能常驻、能当桌宠/桌面智能体。

## 快速开始

需要 Python 3.10+。

- macOS / Linux：`./desktop/start-desktop.sh`
- Windows：双击 `desktop/start-desktop.bat`
- 然后浏览器打开 http://127.0.0.1:8100

## 可选：本地大脑（不联网也能聊）

```bash
cd local-brain
npm i
./start-llm.sh     # Qwen3-0.6B，8181 端口
./start-embed.sh   # embeddinggemma，8182 端口
```

页面设置里把「大脑」切到 `local`。电脑 CPU/GPU 更强时可以换更大的 GGUF（gemma、Ministral 等）。

## 和手机版的区别

| | 手机（现在） | 电脑（桌面智能体） |
|---|---|---|
| 形象 | 网页 Live2D | 浏览器/PWA 窗口，可做桌宠悬浮窗 |
| 大脑 | 账号模型 / 0.6B 本地 | 账号模型 / 更大的本地 GGUF |
| 训练 | numpy 小模型，CPU | 可上 torch + GPU，真正做策略蒸馏 |
| 常驻 | guardian + App 插件 | 开机自启/托盘常驻 |
| 数据 | data/ 目录 | 同格式，可复制过去 |

## 接 DSH Harness（让她能干活）

手机端已经用 MCP 工具桥把数字人接口暴露给 Agent。电脑端同样可以把 `mcp-server.mjs` 注册到 DSH 的 `cordis.patch.yml`：

```yaml
- insert:
    - id: digital-human-mcp
      name: '@deepseek-ai/dsh-mcp-client'
      config:
        serverName: digital-human
        transport: stdio
        command: !!js process.execPath
        args: ['/path/to/digital-human/mcp-server.mjs']
        toolCallTimeoutMs: 30000
```

这样桌宠负责"形象和陪伴"，Harness 负责"执行任务"，和 dsh-pet-indesktop 的架构一致。

## RSI 在电脑上的落点

手机上只能训 2KB 小模型；电脑上可以：

1. 用 torch 实现策略网络/注意力路由，在 GPU 上训练
2. 对账号模型做语言风格蒸馏（LoRA 小模型），把"她怎么说话"变成本地权重
3. 保留手机端的冻结地层/巩固/自我状态格式，电脑训练、手机推理

数据不离开本地，API 模型不会吸收这些数据。
