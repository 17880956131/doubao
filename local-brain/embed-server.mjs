// 本地向量服务：给数字人的长期记忆做语义检索
// 启动：node embed-server.mjs [--port 8182]
import { createServer } from 'node:http'
import { getLlama } from 'node-llama-cpp'

const args = process.argv.slice(2)
const opt = (name, fallback) => {
  const i = args.indexOf(name)
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback
}
const PORT = Number(opt('--port', process.env.EMBED_PORT || 8182))
const MODEL_PATH = opt('--model', process.env.EMBED_MODEL || '/sdcard/模型文件/embeddinggemma-300m-Q4_0.gguf')

async function main() {
  console.log('[embed] loading', MODEL_PATH)
  const llama = await getLlama({ gpu: false, build: 'never' })
  const model = await llama.loadModel({ modelPath: MODEL_PATH, gpuLayers: 0 })
  const context = await model.createEmbeddingContext()
  console.log(`[embed] ready port=${PORT}`)

  const server = createServer(async (req, res) => {
    const send = (code, body) => {
      const payload = JSON.stringify(body)
      res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8', 'Content-Length': Buffer.byteLength(payload) })
      res.end(payload)
    }
    if (req.method === 'GET' && req.url.startsWith('/health')) {
      return send(200, { ok: true, model: 'embeddinggemma-300m', dims: 768 })
    }
    if (req.method !== 'POST' || !req.url.startsWith('/v1/embeddings')) {
      return send(404, { error: 'not found' })
    }
    let raw = ''
    try {
      for await (const chunk of req) raw += chunk
      const body = JSON.parse(raw || '{}')
      const inputs = Array.isArray(body.input) ? body.input : [body.input]
      const data = []
      for (let i = 0; i < inputs.length; i += 1) {
        const text = String(inputs[i] || '').slice(0, 2000)
        const embedding = await context.getEmbeddingFor(text)
        data.push({ object: 'embedding', index: i, embedding: Array.from(embedding.vector) })
      }
      return send(200, { object: 'list', model: 'embeddinggemma-300m', data })
    } catch (error) {
      return send(500, { error: String((error && error.message) || error) })
    }
  })
  server.listen(PORT, '127.0.0.1')
}

main().catch((error) => {
  console.error('[embed] failed', error)
  process.exit(1)
})
