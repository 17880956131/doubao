export const name = 'zhiming-bridge'
export const inject = ['tools']

const DEFAULT_BASE = 'http://127.0.0.1:8100'

async function call(path, body, timeout = 30000) {
  const response = await fetch(DEFAULT_BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
    signal: AbortSignal.timeout(timeout),
  })
  if (!response.ok) throw new Error('HTTP ' + response.status)
  return response.json()
}

export function apply(ctx) {
  ctx.effect(() => ctx.tools.register({
    name: 'zhiming_chat',
    description: '和智命（ZhiMing）里的数字人豆宝说话：她有自己的记忆、情绪和表情。返回她的回复、表情和动作。',
    parameters: {
      type: 'object',
      properties: {
        message: { type: 'string', description: '要对豆宝说的话' },
        lang: { type: 'string', description: '语言 zh/ja/en，默认 zh' },
      },
      required: ['message'],
      additionalProperties: false,
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          text: { type: 'string' },
          expression: { type: 'string' },
          gesture: { type: 'string' },
        },
      },
      render: (args, value) => [{
        type: 'text',
        text: `豆宝（${value.expression}）：${value.text}`,
      }],
    },
    async execute(args) {
      const reply = await call('/api/orchestrate', {
        content: String(args.message || ''),
        lang: String(args.lang || 'zh'),
        session: 'harness',
      })
      return { text: String(reply.text || ''), expression: String(reply.expression || 'neutral'), gesture: String(reply.gesture || '') }
    },
  }), 'zhiming-bridge tool')
}
