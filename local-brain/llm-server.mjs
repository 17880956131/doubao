// 本机 GGUF 模型服务：OpenAI 兼容接口，输出结构化表演指令
// 启动：node llm-server.mjs [--port 8181] [--model /path/to/model.gguf]
import { createServer } from 'node:http'
import { getLlama, LlamaChatSession } from 'node-llama-cpp'

const args = process.argv.slice(2)
const opt = (name, fallback) => {
  const index = args.indexOf(name)
  return index >= 0 && args[index + 1] ? args[index + 1] : fallback
}

const PORT = Number(opt('--port', process.env.LLM_PORT || 8181))
const MODEL_PATH = opt('--model', process.env.LLM_MODEL || '/sdcard/模型文件/Qwen3-0.6B-Q4_K_M.gguf')
const MODEL_NAME = opt('--name', process.env.LLM_NAME || 'qwen3-0.6b')

const SYSTEM_FALLBACK = '你是"豆宝"，一个住在手机里的 AI 数字人。活泼、说话短。'

const GRAMMAR_SCHEMA = {
  type: 'object',
  properties: {
    reply: { type: 'string' },
    emotion: { enum: ['happy', 'sad', 'surprised', 'angry', 'shy', 'neutral'] },
    gesture: { enum: ['nod', 'tilt', 'sway', 'look'] },
  },
  required: ['reply', 'emotion', 'gesture'],
  additionalProperties: false,
}

async function main() {
  console.log('[llm] loading', MODEL_PATH)
  const llama = await getLlama({ gpu: false, build: 'never' })
  const model = await llama.loadModel({ modelPath: MODEL_PATH, gpuLayers: 0 })
  const context = await model.createContext({ contextSize: 1024 })
  const session = new LlamaChatSession({ contextSequence: context.getSequence() })
  const grammar = await llama.createGrammarForJsonSchema(GRAMMAR_SCHEMA)
  console.log(`[llm] ready model=${MODEL_NAME} port=${PORT}`)

  const server = createServer(async (req, res) => {
    const send = (code, body) => {
      const payload = JSON.stringify(body)
      res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8', 'Content-Length': Buffer.byteLength(payload) })
      res.end(payload)
    }
    if (req.method === 'GET' && (req.url === '/health' || req.url.startsWith('/health?'))) {
      return send(200, { ok: true, model: MODEL_NAME, path: MODEL_PATH })
    }
    if (req.method === 'GET' && req.url.startsWith('/v1/models')) {
      return send(200, { object: 'list', data: [{ id: MODEL_NAME, object: 'model' }] })
    }
    if (req.method !== 'POST' || !req.url.startsWith('/v1/chat/completions')) {
      return send(404, { error: 'not found' })
    }
    let raw = ''
    try {
      for await (const chunk of req) raw += chunk
      const body = JSON.parse(raw || '{}')
      const messages = Array.isArray(body.messages) ? body.messages : []
      const system = messages.filter((m) => m && m.role === 'system').map((m) => String(m.content || '')).join('\n')
      const users = messages.filter((m) => m && m.role === 'user')
      const userText = users.length ? String(users[users.length - 1].content || '') : ''
      const plain = body.plain === true
      const started = Date.now()
      // 走模型自带的对话模板：system 作为 system 消息，用户内容作为 user 消息
      session.setChatHistory([])
      if (system) {
        session.setChatHistory([{ type: 'system', text: system || SYSTEM_FALLBACK }])
      }
      const userPrompt = plain ? userText : `${userText}\n只输出 JSON 对象。`
      const answer = await session.prompt(userPrompt, plain
        ? { maxTokens: Number(body.max_tokens) || 80, temperature: Number(body.temperature) || 0.8 }
        : { grammar, maxTokens: Number(body.max_tokens) || 100, temperature: Number(body.temperature) || 0.7 })
      const ms = Date.now() - started
      console.log(`[llm] ${ms}ms -> ${String(answer).slice(0, 120)}`)
      return send(200, {
        id: 'local-' + started,
        object: 'chat.completion',
        created: Math.floor(started / 1000),
        model: MODEL_NAME,
        choices: [{ index: 0, message: { role: 'assistant', content: answer }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
        local_ms: ms,
      })
    } catch (error) {
      console.log('[llm] error', error && error.message)
      return send(500, { error: String((error && error.message) || error) })
    }
  })
  server.listen(PORT, '127.0.0.1')
}

main().catch((error) => {
  console.error('[llm] failed', error)
  process.exit(1)
})
