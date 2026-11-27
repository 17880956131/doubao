// 数字人工具桥：把 8100 的自我/记忆/表现接口注册成 Agent 可调用的 MCP 工具
// 协议：MCP stdio（每行一个 JSON-RPC 消息），无第三方依赖
import readline from 'node:readline'

const BASE = process.env.DH_BASE || 'http://127.0.0.1:8100'

const TOOLS = [
  {
    name: 'digital_human_state',
    description: '读取数字人豆宝的自我状态：当前表情动作、心情、正在关注的时代层、自我一致性。',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
  },
  {
    name: 'digital_human_status',
    description: '读取数字人整体运行看板：层数、巩固次数、自评基准分、自我一致性、三观摘要。',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
  },
  {
    name: 'digital_human_presence',
    description: '读取数字人最后一次活动的持久化状态（表情、动作、心情、最近聊天），用于崩溃/重启后恢复。',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
  },
  {
    name: 'digital_human_chat',
    description: '和数字人豆宝说一句话，返回她的回复、表情和动作（会走她的记忆、情绪、仲裁与账号模型）。',
    inputSchema: {
      type: 'object',
      properties: {
        message: { type: 'string', description: '要对豆宝说的话' },
        lang: { type: 'string', description: '语言 zh/ja/en，默认 zh' },
      },
      required: ['message'],
      additionalProperties: false,
    },
  },
  {
    name: 'digital_human_narrate',
    description: '让豆宝基于她自己的身份、地层、心情和自评分，用第一人称讲一段自己是谁。',
    inputSchema: {
      type: 'object',
      properties: { lang: { type: 'string', description: '语言 zh/ja/en，默认 zh' } },
      additionalProperties: false,
    },
  },
  {
    name: 'digital_human_consolidate',
    description: '触发一次巩固：把积累的推理数据归一化训练成新的一层（只叠不改，旧层永久冻结）。',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
  },
  {
    name: 'digital_human_self_check',
    description: '让豆宝做一次自我评估：跑固定基准分，分数下滑就自己巩固/调权重。',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
  },
  {
    name: 'digital_human_landscape',
    description: '读取认知景观：各层损失/梯度/曲率、层间分歧、交叉点（各层意见最分裂的状态）。',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
  },
  {
    name: 'digital_human_values',
    description: '读取豆宝的三观原则与规则，以及有没有从行为里浮现出的候选三观。',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
  },
  {
    name: 'digital_human_perceive',
    description: '把外部感知（通知、电量、你在做什么、环境变化）推给数字人，让她"知道"当下发生了什么（写入事件总线，影响心情与注意力）。',
    inputSchema: {
      type: 'object',
      properties: {
        kind: { type: 'string', description: '事件类型，如 notification/battery/user_activity/environment' },
        text: { type: 'string', description: '感知内容（一句话）' },
        intensity: { type: 'number', description: '强度 0-1，可省略' },
      },
      required: ['text'],
      additionalProperties: false,
    },
  },
  {
    name: 'digital_human_attention',
    description: '查询给定语境下，豆宝会自主关注哪些时代层（注意力分布）。',
    inputSchema: {
      type: 'object',
      properties: { context: { type: 'string', description: '语境/状态描述' } },
      additionalProperties: false,
    },
  },
]

async function callApi(path, options = {}) {
  const response = await fetch(BASE + path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!response.ok) {
    throw new Error('HTTP ' + response.status + ' ' + path)
  }
  return response.json()
}

async function runTool(name, args) {
  switch (name) {
    case 'digital_human_state': {
      const state = await callApi('/api/self/state')
      return { context: state.context, focus: (state.attention || []).slice(0, 3), applied: state.applied, consistency: state.consistency }
    }
    case 'digital_human_status': {
      const dash = await callApi('/api/dashboard')
      return {
        layers: dash.layers,
        consolidations: dash.loop && dash.loop.consolidations,
        benchmark: dash.benchmark && dash.benchmark.score,
        consistency: dash.consistency && dash.consistency.score,
        values: (dash.values && dash.values.principles || []).slice(0, 5),
        inference: dash.inference,
      }
    }
    case 'digital_human_presence': {
      const p = await callApi('/api/presence')
      return { presence: p.presence, uptime: p.uptime_seconds, restarts: p.server_restarts, mood: p.mood }
    }
    case 'digital_human_chat': {
      const reply = await callApi('/api/orchestrate', {
        method: 'POST',
        body: JSON.stringify({ content: String(args.message || ''), lang: String(args.lang || 'zh'),
                               session: 'main' }),
      })
      return { text: reply.text, expression: reply.expression, gesture: reply.gesture, attention: reply.arbitration && reply.arbitration.attention }
    }
    case 'digital_human_narrate': {
      const narration = await callApi('/api/self/narrate', {
        method: 'POST',
        body: JSON.stringify({ lang: String(args.lang || 'zh') }),
      })
      return { text: narration.text, source: narration.source }
    }
    case 'digital_human_consolidate':
      return await callApi('/api/loop/consolidate', { method: 'POST', body: '{}' })
    case 'digital_human_self_check':
      return await callApi('/api/self-check', { method: 'POST', body: '{}' })
    case 'digital_human_landscape': {
      const landscape = await callApi('/api/landscape')
      return { layers: landscape.layers, disagreement: landscape.disagreement, contest_points: landscape.contest_points }
    }
    case 'digital_human_values':
      return await callApi('/api/values')
    case 'digital_human_perceive': {
      const payload = {
        type: 'external',
        kind: String(args.kind || 'external'),
        text: String(args.text || '').slice(0, 300),
        intensity: typeof args.intensity === 'number' ? args.intensity : 0.5,
      }
      const result = await callApi('/api/event', { method: 'POST', body: JSON.stringify(payload) })
      return { ok: true, pushed: payload, event: result }
    }
    case 'digital_human_attention': {
      const q = String(args.context || '')
      const attention = await callApi('/api/self/attention?q=' + encodeURIComponent(q))
      return { context: q, attention: attention.attention }
    }
    default:
      throw new Error('未知工具：' + name)
  }
}

function send(payload) {
  process.stdout.write(JSON.stringify(payload) + '\n')
}

const rl = readline.createInterface({ input: process.stdin, terminal: false })
rl.on('line', async (line) => {
  line = line.trim()
  if (!line) {
    return
  }
  let message
  try {
    message = JSON.parse(line)
  } catch (error) {
    send({ jsonrpc: '2.0', id: null, error: { code: -32700, message: 'JSON 解析失败' } })
    return
  }
  const id = message.id
  const method = message.method
  try {
    if (method === 'initialize') {
      send({
        jsonrpc: '2.0',
        id,
        result: {
          protocolVersion: '2024-11-05',
          capabilities: { tools: {} },
          serverInfo: { name: 'digital-human', version: '0.1.0' },
        },
      })
      return
    }
    if (method === 'notifications/initialized') {
      return
    }
    if (method === 'tools/list') {
      send({ jsonrpc: '2.0', id, result: { tools: TOOLS } })
      return
    }
    if (method === 'tools/call') {
      const name = message.params && message.params.name
      const args = (message.params && message.params.arguments) || {}
      const output = await runTool(String(name), args || {})
      send({
        jsonrpc: '2.0',
        id,
        result: { content: [{ type: 'text', text: JSON.stringify(output, null, 2) }] },
      })
      return
    }
    send({ jsonrpc: '2.0', id, error: { code: -32601, message: '方法不支持：' + method } })
  } catch (error) {
    send({ jsonrpc: '2.0', id, result: { content: [{ type: 'text', text: '调用失败：' + String(error && error.message || error) }], isError: true } })
  }
})
