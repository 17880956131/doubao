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
    name: 'zhiming_body',
    description: '让智命的身体按 Agent 的话做表情和动作，并驱动舞台；不生成新的对话（单主体：说话的是 Agent，她是身体）。',
    parameters: {
      type: 'object',
      properties: {
        text: { type: 'string', description: '要让身体表达的话（由 Agent 说，不是她生成）' },
      },
      required: ['text'],
      additionalProperties: false,
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          text: { type: 'string' },
          expression: { type: 'string' },
          gesture: { type: 'string' },
          mood: { type: 'string' },
        },
      },
      render: (args, value) => [{
        type: 'text',
        text: `豆宝（${value.expression}）：${value.text}`,
      }],
    },
    async execute(args) {
      const body = await call('/api/body/say', {
        text: String(args.text || ''),
        source: 'harness-agent',
      })
      return { text: String(body.text || ''), expression: String(body.expression || 'neutral'),
               gesture: String(body.gesture || ''), mood: String(body.mood || '') }
    },
  }), 'zhiming-bridge tool')
}
