// 豆宝数字人舞台：Live2D 形象 + 语音口型 + 表情 + 点按互动 + 语音对话
/* global PIXI */
const $ = (id) => document.getElementById(id)
const statusEl = $('status')
const subtitleEl = $('subtitle')
const chatLog = $('chatlog')

const state = {
  voice: localStorage.getItem('stts.voice') || 'zh-CN-XiaoxiaoNeural',
  rate: localStorage.getItem('stts.rate') || '+0%',
  key: localStorage.getItem('stts.key') || '',
  brain: localStorage.getItem('stts.brain') || 'deepseek',
  baseURL: localStorage.getItem('stts.baseURL') || '',
  model: localStorage.getItem('stts.model') || '',
  history: [],
  speaking: false,
}

const LANGS = {
  zh: {
    name: '中文', button: '中文',
    voices: [
      ['zh-CN-XiaoxiaoNeural', '晓晓 · 女声温柔'],
      ['zh-CN-YunxiNeural', '云希 · 男声阳光'],
      ['zh-CN-YunyangNeural', '云扬 · 男声播音'],
      ['zh-CN-XiaoyiNeural', '晓伊 · 女声活泼'],
    ],
    greeting: '你好呀，我是豆宝，一个住在你手机里的数字人。要不要听个冷笑话？',
    switchLine: '好，之后我只说中文。',
  },
  ja: {
    name: '日本語', button: '日本語',
    voices: [
      ['ja-JP-NanamiNeural', '七海 · 女声'],
      ['ja-JP-KeitaNeural', '圭太 · 男声'],
    ],
    greeting: 'こんにちは、豆宝だよ。スマホの中に住んでるデジタルヒューマン。寒いダジャレ、聞く？',
    switchLine: 'はい、これから日本語だけで話すね。',
  },
  en: {
    name: 'English', button: 'English',
    voices: [
      ['en-US-AriaNeural', 'Aria · female'],
      ['en-US-GuyNeural', 'Guy · male'],
      ['en-GB-SoniaNeural', 'Sonia · UK female'],
    ],
    greeting: "Hi, I'm Doubao, a digital human living in your phone. Want to hear a corny joke?",
    switchLine: "Okay, I'll speak English only from now on.",
  },
}
const VOICE_MEMORY = (() => {
  try { return JSON.parse(localStorage.getItem('stts.voices') || '{}') } catch (err) { return {} }
})()
state.lang = localStorage.getItem('stts.lang') || 'zh'
state.autoReact = localStorage.getItem('stts.autoReact') !== '0'
state.jokeSource = localStorage.getItem('stts.jokeSource') || 'auto'
state.gesture = ''
state.scenes = []
state.moodLabel = ''
let lastActivity = Date.now()

const EXPR_KINDS = ['neutral', 'happy', 'sad', 'surprised', 'angry', 'shy']

function log(event, detail) {
  fetch('/api/log', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ event, detail: detail || null }),
  }).catch(() => {})
}

window.addEventListener('error', (e) => log('js-error', { message: String(e.message || '').slice(0, 300), src: String(e.filename || '').slice(-40), line: e.lineno }))
window.addEventListener('unhandledrejection', (e) => log('js-reject', { message: String((e.reason && e.reason.message) || e.reason).slice(0, 300) }))

function setStatus(text) { statusEl.textContent = text || '' }

function showMood(mood) {
  const el = $('mood')
  if (el && mood && mood.label) {
    el.textContent = '· ' + mood.label
    state.moodLabel = mood.label
  }
}

let UI_PARAMS = { trajectory_interval_ms: 10000 }
state.mirror = localStorage.getItem('dsh.mirror') !== '0'
state.outbox = localStorage.getItem('dsh.outbox') === '1'
state.stageOnly = localStorage.getItem('dsh.stageOnly') !== '0'
state.visionLoop = localStorage.getItem('dsh.visionLoop') === '1'
let visionStream = null
let visionTimer = null

async function startVisionLoop() {
  try {
    visionStream = await navigator.mediaDevices.getUserMedia({ video: true })
    const video = document.createElement('video')
    video.srcObject = visionStream
    await video.play()
    const capture = async () => {
      if (!state.visionLoop || !visionStream) return
      try {
        const canvas = document.createElement('canvas')
        canvas.width = video.videoWidth || 640
        canvas.height = video.videoHeight || 480
        canvas.getContext('2d').drawImage(video, 0, 0)
        const dataUrl = canvas.toDataURL('image/jpeg', 0.6)
        await fetch('/api/multimodal/vision', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_base64: dataUrl, source: 'camera-loop' }),
        })
      } catch (err) { log('vision-loop-error', String(err)) }
      if (state.visionLoop) {
        visionTimer = setTimeout(capture, Number(UI_PARAMS.vision_interval_ms) || 15000)
      }
    }
    capture()
  } catch (err) {
    addBubble('系统', '打不开摄像头：' + String(err))
    state.visionLoop = false
  }
}

function stopVisionLoop() {
  state.visionLoop = false
  if (visionTimer) { clearTimeout(visionTimer); visionTimer = null }
  if (visionStream) { visionStream.getTracks().forEach((track) => track.stop()); visionStream = null }
}

$('vision-loop').addEventListener('change', () => {
  state.visionLoop = $('vision-loop').checked
  localStorage.setItem('dsh.visionLoop', state.visionLoop ? '1' : '0')
  if (state.visionLoop) startVisionLoop()
  else stopVisionLoop()
})

let mirrorSeq = 0
let mirrorStarted = false

async function mirrorLoop() {
  if (!state.mirror) return
  try {
    const data = await (await fetch('/api/mirror?since=' + mirrorSeq)).json()
    if (!mirrorStarted) {
      mirrorSeq = data.latest || 0
      mirrorStarted = true
      return
    }
    for (const msg of data.messages || []) {
      mirrorSeq = Math.max(mirrorSeq, msg.seq || 0)
      const text = String(msg.text || '').replace(/```[\s\S]*?```/g, '').trim()
      if (!text) continue
      addBubble(msg.role === 'user' ? '你' : '豆宝', text)
      if (msg.role === 'assistant' && text.length <= 160 && !state.speaking) {
        try {
          const body = await fetch('/api/body/say', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, source: 'mirror',
              voice: { name: state.voice || '', rate: state.rate || '+0%', volume: 1 } }),
          }).then((res) => res.json()).catch(() => null)
          await speak(text, (body && body.expression) || 'neutral')
        } catch (err) { /* 忽略 */ }
      }
    }
  } catch (err) { /* 忽略 */ }
}

async function refreshMood() {
  try {
    const data = await (await fetch('/api/mood')).json()
    showMood(data.mood)
    return data.mood
  } catch (err) {
    return null
  }
}

function addBubble(who, text) {
  const div = document.createElement('div')
  div.className = 'bubble' + (who === '你' ? ' me' : '')
  const label = document.createElement('b')
  label.textContent = who
  div.appendChild(label)
  div.appendChild(document.createTextNode(text))
  chatLog.appendChild(div)
  while (chatLog.children.length > 6) chatLog.removeChild(chatLog.firstChild)
  chatLog.scrollTop = chatLog.scrollHeight
}

function langConfig() { return LANGS[state.lang] || LANGS.zh }

function populateVoices() {
  const select = $('voice')
  const config = langConfig()
  select.innerHTML = ''
  for (const [id, label] of config.voices) {
    const option = document.createElement('option')
    option.value = id
    option.textContent = label
    select.appendChild(option)
  }
  if (!config.voices.some(([id]) => id === state.voice)) state.voice = config.voices[0][0]
  select.value = state.voice
  $('lang').textContent = config.button
  $('lang').title = config.name
}

function applyLanguage(lang, options = {}) {
  if (!LANGS[lang]) lang = 'zh'
  state.lang = lang
  const config = LANGS[lang]
  const remembered = VOICE_MEMORY[lang]
  state.voice = remembered && config.voices.some(([id]) => id === remembered) ? remembered : config.voices[0][0]
  localStorage.setItem('stts.lang', lang)
  populateVoices()
  log('lang', lang)
  if (options.announce) {
    stopSpeaking()
    speak(options.text || config.switchLine, 'happy')
  }
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]))
}

// ── 舞台 ──────────────────────────────────────────────
const canvas = $('stage')
const app = new PIXI.Application({
  view: canvas,
  resizeTo: window,
  backgroundAlpha: 0,
  antialias: true,
  resolution: Math.min(2, window.devicePixelRatio || 1),
  autoDensity: true,
  preserveDrawingBuffer: true,
})
window.__dshStage = app

let model = null
let modelBase = ''
let baseSize = { w: 1, h: 1 }
let exprList = []
let currentExpr = 'neutral'
let motionGroups = []
let mouth = 0
let mouthTarget = 0
let headX = 0
let headY = 0
let pointerX = 0
let pointerY = 0
let bodyPhase = 0
let idleTimer = null
let gesture = null
const SURFACE_PARAMS = ['PARAM_MOUTH_OPEN_Y', 'PARAM_MOUTH_FORM', 'PARAM_ANGLE_X', 'PARAM_ANGLE_Y', 'PARAM_ANGLE_Z', 'PARAM_BODY_ANGLE_Z', 'PARAM_EYE_L_OPEN', 'PARAM_EYE_R_OPEN', 'PARAM_BROW_L_Y', 'PARAM_BROW_R_Y', 'PARAM_EYE_BALL_X', 'PARAM_EYE_BALL_Y']

function fitModel() {
  if (!model) return
  const w = app.screen.width
  const h = app.screen.height
  const scale = Math.min((h * 0.92) / baseSize.h, (w * 0.96) / baseSize.w)
  model.scale.set(scale)
  model.position.set(w / 2, h * 1.02)
}

function loadExpressionList(settings) {
  const defs = (settings && settings.expressions) || []
  exprList = defs.map((def) => def.name)
  log('expr-ready', { names: exprList.join(',') })
}

async function setupModel() {
  const modelPath = new URLSearchParams(location.search).get('model')
    || localStorage.getItem('stage.model')
    || 'models/hibiki/hibiki.model.json'
  modelBase = new URL('.', new URL(modelPath, location.href)).href
  // 模型动作可能自带语音（这个示例模型带的是日语音效），会和我们的中文 TTS 叠在一起，关掉
  if (PIXI.live2d && PIXI.live2d.config) PIXI.live2d.config.sound = false
  model = await PIXI.live2d.Live2DModel.from(modelPath, { autoInteract: false })
  app.stage.addChild(model)
  model.anchor.set(0.5, 1)
  if ('eventMode' in model) model.eventMode = 'static'
  baseSize = { w: model.width, h: model.height }
  fitModel()
  window.addEventListener('resize', fitModel)
  window.addEventListener('orientationchange', () => setTimeout(fitModel, 120))

  const settings = model.internalModel.settings
  loadExpressionList(settings)
  motionGroups = Object.keys(settings.motions || {})
  log('motion-groups', motionGroups.join(','))

  // 口型 / 头部 / 身体微动 / 表情参数：在运动更新之后、形变之前叠加
  const core = model.internalModel.coreModel
  const origUpdate = core.update.bind(core)
  core.update = () => {
    try {
      headX += (pointerX * 14 - headX) * 0.07
      headY += (pointerY * 9 - headY) * 0.07
      mouth += (mouthTarget - mouth) * 0.38
      bodyPhase += 0.018
      let gx = 0
      let gy = 0
      let gz = 0
      let bx = 0
      if (gesture) {
        const t = (performance.now() - gesture.start) / gesture.duration
        if (t >= 1) {
          gesture = null
        } else {
          const s = Math.sin(Math.PI * t)
          if (gesture.type === 'nod') { gy = s * 26; gx = s * 6 }
          else if (gesture.type === 'tilt') { gz = s * 20; gx = s * 8 }
          else if (gesture.type === 'sway') { gx = s * 24; gz = s * 6 }
          else if (gesture.type === 'look') { gy = -s * 18; gx = -s * 10 }
          else if (gesture.type === 'bow') { gy = s * 34; bx = -s * 12 }
          else if (gesture.type === 'lean') { bx = s * 12; gz = s * 8 }
          else if (gesture.type === 'bounce') { gy = -s * 24; gx = s * 4 }
          else if (gesture.type === 'turn') { bx = s * 16; gz = -s * 10 }
          else if (gesture.type === 'wave') { gx = s * 22; gz = s * 12 }
        }
      }
      const ax = core.getParamFloat('PARAM_ANGLE_X')
      const ay = core.getParamFloat('PARAM_ANGLE_Y')
      const bz = core.getParamFloat('PARAM_BODY_ANGLE_Z')
      core.setParamFloat('PARAM_ANGLE_X', ax + headX + gx)
      core.setParamFloat('PARAM_ANGLE_Y', ay + headY + gy)
      core.setParamFloat('PARAM_ANGLE_Z', core.getParamFloat('PARAM_ANGLE_Z') + gz)
      core.setParamFloat('PARAM_BODY_ANGLE_Z', bz + gz * 1.4)
      core.setParamFloat('PARAM_BODY_ANGLE_X', core.getParamFloat('PARAM_BODY_ANGLE_X') + gx * 0.8 + bx)
      // 表情交给模型自己的表达式系统；这里只做口型（说话时开嘴）
      core.setParamFloat('PARAM_MOUTH_OPEN_Y', Math.max(0, Math.min(1, mouth)))
      if (!window.__surfaceParams) window.__surfaceParams = {}
      for (const key of SURFACE_PARAMS) {
        try { window.__surfaceParams[key] = Math.round(core.getParamFloat(key) * 100) / 100 } catch (err) { /* 该模型没有这个参数 */ }
      }
    } catch (err) {
      // 参数缺失的模型忽略
    }
    origUpdate()
  }

  try { model.motion('idle', 0, PIXI.live2d.MotionPriority.IDLE) } catch (err) { log('idle-error', String(err)) }

  canvas.addEventListener('pointerdown', (event) => {
    if (event.target === canvas) reactToTap()
  })
  model.on('pointerdown', () => reactToTap())

  window.addEventListener('pointermove', (event) => {
    pointerX = (event.clientX / window.innerWidth) * 2 - 1
    pointerY = (event.clientY / window.innerHeight) * 2 - 1
  })

  log('model-loaded', { w: Math.round(baseSize.w), h: Math.round(baseSize.h), expressions: exprList.length, motions: motionGroups.join('|') })
}

const GESTURES = ['nod', 'tilt', 'sway', 'look', 'bow', 'lean', 'bounce', 'turn', 'wave']
const EXPRESSION_GESTURE = { happy: 'bounce', sad: 'bow', surprised: 'lean', shy: 'turn', angry: 'turn', neutral: null }

const GESTURE_MOTION = { nod: 0, tilt: 1, sway: 2, look: 3 }

function playGestureMotion(type) {
  if (!model) return
  const index = GESTURE_MOTION[type]
  if (index == null) return
  try {
    const group = motionGroups.includes('null') ? 'null' : (motionGroups[0] || 'null')
    model.motion(group, index, PIXI.live2d.MotionPriority.FORCE)
  } catch (err) { log('gesture-motion-error', String(err)) }
}

function startGesture(type) {
  gesture = { type: type || GESTURES[Math.floor(Math.random() * GESTURES.length)], start: performance.now(), duration: 900 }
  state.gesture = gesture.type
  playGestureMotion(gesture.type)
  log('gesture', gesture.type)
  postState()
}

function makeThumb() {
  try {
    const source = app.renderer.view
    const small = document.createElement('canvas')
    small.width = 96
    small.height = 96
    const ctx = small.getContext('2d')
    ctx.drawImage(source, 0, 0, 96, 96)
    const gridCanvas = document.createElement('canvas')
    gridCanvas.width = 8
    gridCanvas.height = 8
    const gctx = gridCanvas.getContext('2d')
    // 先铺深色背景，避免透明区域被算成黑色
    gctx.fillStyle = '#141225'
    gctx.fillRect(0, 0, 8, 8)
    gctx.drawImage(source, 0, 0, 8, 8)
    const pixels = gctx.getImageData(0, 0, 8, 8).data
    const grid = []
    for (let i = 0; i < pixels.length; i += 4) {
      grid.push(Math.round(((pixels[i] * 0.299 + pixels[i + 1] * 0.587 + pixels[i + 2] * 0.114) / 255) * 100) / 100)
    }
    return { image: small.toDataURL('image/jpeg', 0.6), grid }
  } catch (err) {
    return { grid: [], image: null }
  }
}

let lastUserText = ''

function trajectorySnapshot() {
  const params = window.__surfaceParams || {}
  return {
    expression: currentExpr,
    gesture: state.gesture,
    speaking: state.speaking,
    mood: state.moodLabel,
    scene: state.scenes.length ? state.scenes[state.scenes.length - 1] : '',
    params: {
      PARAM_MOUTH_OPEN_Y: params.PARAM_MOUTH_OPEN_Y,
      PARAM_ANGLE_X: params.PARAM_ANGLE_X,
      PARAM_ANGLE_Y: params.PARAM_ANGLE_Y,
      PARAM_BODY_ANGLE_Z: params.PARAM_BODY_ANGLE_Z,
      PARAM_EYE_L_OPEN: params.PARAM_EYE_L_OPEN,
    },
  }
}

async function postTrajectory() {
  if (!model) return
  try {
    await fetch('/api/trajectory', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(trajectorySnapshot()),
    })
  } catch (err) { /* 忽略 */ }
}

async function captureSurface(force) {
  if (!model) return
  if (!force && document.hidden) return
  const thumb = makeThumb()
  const payload = {
    expression: currentExpr,
    gesture: state.gesture,
    speaking: state.speaking,
    params: window.__surfaceParams || {},
    grid: thumb.grid,
    image: thumb.image,
    at: Date.now(),
  }
  try {
    await fetch('/api/surface', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  } catch (err) { /* 忽略 */ }
}

async function selfCheckLoop() {
  setStatus('豆宝在看自己…')
  for (let round = 0; round < 2; round += 1) {
    await captureSurface(true)
    const question = round === 0
      ? '看看你现在的形象表面（状态里有表情、头、嘴、画面），你觉得自己看起来自然吗？需要换表情或动作的话，直接给出来。'
      : '再看一次你现在的样子，比刚才自然了吗？如果还僵，继续调整。'
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: question,
          lang: state.lang,
          state: loopState(),
          ...brainFields(),
        }),
      })
      const data = await res.json()
      addBubble('豆宝', data.text)
      await speak(data.text, data.expression || 'neutral')
      if (data.gesture) startGesture(data.gesture)
    } catch (err) {
      setStatus('自检失败')
    }
    await new Promise((resolve) => setTimeout(resolve, 1500))
  }
  await captureSurface(true)
  setStatus('')
}

function postState() {
  fetch('/api/state', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      expression: currentExpr,
      gesture: state.gesture,
      speaking: state.speaking,
      scenes: state.scenes,
      hour: new Date().getHours(),
    }),
  }).catch(() => {})
}

function brainFields() {
  if (state.brain === 'account') return { account: true }
  if (state.brain === 'local') return { baseURL: 'http://127.0.0.1:8181/v1', model: state.model || 'qwen3-0.6b' }
  if (state.brain === 'custom') return { baseURL: state.baseURL || undefined, model: state.model || undefined }
  return { key: state.key || undefined }
}

function loopState() {
  return {
    expression: currentExpr,
    gesture: state.gesture,
    speaking: state.speaking,
    scenes: state.scenes,
    hour: new Date().getHours(),
  }
}

async function playMotion(kind) {
  if (!model) { startGesture(); return }
  let group = null
  if (motionGroups.includes('null')) group = 'null'
  else if (motionGroups.includes('')) group = ''
  else if (motionGroups.includes('idle')) group = 'idle'
  if (group === null) { startGesture(); return }
  try {
    const priority = PIXI.live2d.MotionPriority.FORCE || PIXI.live2d.MotionPriority.NORMAL
    const ok = await model.motion(group, undefined, priority)
    log('motion', { group, ok })
    if (!ok) startGesture()
  } catch (err) {
    log('motion-error', { group, message: String(err && err.message || err) })
    startGesture()
  }
}

async function setExpression(kind) {
  if (!EXPR_KINDS.includes(kind)) kind = 'neutral'
  currentExpr = kind
  log('expr', { kind, by: 'model' })
  const index = EXPR_KINDS.indexOf(kind)
  if (!model || index < 0 || !exprList[index]) return
  try {
    let ok = await model.expression(exprList[index])
    if (!ok) {
      // 表达式管理器有时正忙（上一个表情还没淡出完），稍等重试一次
      await new Promise((resolve) => setTimeout(resolve, 200))
      ok = await model.expression(exprList[index])
    }
    log('expr-model', { name: exprList[index], ok })
    postState()
  } catch (err) {
    log('expr-api-error', String(err && err.message || err))
  }
  if (kind !== 'neutral') playMotion()
}

function reactToTap() {
  if (!model) return
  playMotion()
  if (!state.speaking && Math.random() < 0.5) {
    const lines = ['别戳啦～', '在呢在呢。', '你戳我干嘛呀。', '嘿嘿。']
    speak(lines[Math.floor(Math.random() * lines.length)], 'shy')
  }
}

// ── 声音与口型 ───────────────────────────────────────
let audio = null
let audioCtx = null
let analyser = null
let timeData = null

function ensureAudio() {
  if (audioCtx) return
  audio = new Audio()
  audio.crossOrigin = 'anonymous'
  audioCtx = new (window.AudioContext || window.webkitAudioContext)()
  const source = audioCtx.createMediaElementSource(audio)
  analyser = audioCtx.createAnalyser()
  analyser.fftSize = 512
  source.connect(analyser)
  analyser.connect(audioCtx.destination)
  timeData = new Uint8Array(analyser.fftSize)
  audio.addEventListener('ended', () => { mouthTarget = 0; state.speaking = false; clearSubtitle(); postState() })
  audio.addEventListener('error', () => { mouthTarget = 0; state.speaking = false; log('audio-error') })
}

app.ticker.add(() => {
  if (!state.speaking || !analyser) { mouthTarget = Math.max(0, mouthTarget - 0.08); return }
  analyser.getByteTimeDomainData(timeData)
  let sum = 0
  for (let i = 0; i < timeData.length; i += 1) {
    const v = (timeData[i] - 128) / 128
    sum += v * v
  }
  const rms = Math.sqrt(sum / timeData.length)
  mouthTarget = Math.min(1, rms * 5.2)
})

// ── 字幕 ─────────────────────────────────────────────
let subtitleTimer = null

function splitSentences(text) {
  return text.split(/(?<=[。！？!?；;])|\n+/).map((s) => s.trim()).filter(Boolean)
}

function clearSubtitle() {
  if (subtitleTimer) { clearInterval(subtitleTimer); subtitleTimer = null }
  subtitleEl.innerHTML = ''
}

function scheduleSubtitle(text) {
  clearSubtitle()
  const parts = splitSentences(text)
  if (parts.length === 0) return
  const total = parts.reduce((n, p) => n + p.length, 0) || 1
  const start = performance.now()
  const durationMs = (audio && isFinite(audio.duration) && audio.duration > 0 ? audio.duration : text.length / 5.2) * 1000
  subtitleTimer = setInterval(() => {
    const elapsed = performance.now() - start
    const progress = Math.min(1, elapsed / Math.max(1000, durationMs))
    let acc = 0
    let current = parts[parts.length - 1]
    for (const part of parts) {
      acc += part.length / total
      if (progress < acc) { current = part; break }
    }
    subtitleEl.innerHTML = '<b>豆宝</b> ' + escapeHtml(current)
  }, 120)
}

// ── 说话 ─────────────────────────────────────────────
async function speak(text, expression) {
  if (!text) return
  stopSpeaking()
  setExpression(expression || 'happy')
  setStatus('豆宝在想怎么念…')
  try {
    const res = await fetch('/api/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, voice: state.voice, rate: state.rate }),
    })
    if (!res.ok) {
      const detail = await res.text().catch(() => '')
      throw new Error('TTS ' + res.status + ' ' + detail.slice(0, 80))
    }
    const buffer = await res.arrayBuffer()
    ensureAudio()
    await audioCtx.resume()
    const url = URL.createObjectURL(new Blob([buffer], { type: 'audio/mpeg' }))
    audio.src = url
    state.speaking = true
    postState()
    await audio.play()
    setStatus('')
    scheduleSubtitle(text)
    log('speak-start', { chars: text.length })
  } catch (err) {
    state.speaking = false
    mouthTarget = 0
    setStatus('念不出来：' + (err && err.message ? err.message : String(err)))
    log('speak-error', String(err && err.message || err))
  }
}

function stopSpeaking() {
  if (audio) { try { audio.pause(); audio.currentTime = 0 } catch (err) { /* 忽略 */ } }
  state.speaking = false
  mouthTarget = 0
  clearSubtitle()
  setStatus('')
}

// ── 语音识别 ─────────────────────────────────────────
let recording = false
let mediaStream = null
let recordCtx = null
let processor = null
let chunks = []

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2)
  const view = new DataView(buffer)
  const writeString = (offset, text) => { for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i)) }
  writeString(0, 'RIFF')
  view.setUint32(4, 36 + samples.length * 2, true)
  writeString(8, 'WAVE')
  writeString(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  writeString(36, 'data')
  view.setUint32(40, samples.length * 2, true)
  let offset = 44
  for (let i = 0; i < samples.length; i += 1) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true)
    offset += 2
  }
  return new Uint8Array(buffer)
}

async function startRecording() {
  const talk = $('talk')
  if (recording) return
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } })
    const Ctx = window.AudioContext || window.webkitAudioContext
    recordCtx = new Ctx({ sampleRate: 16000 })
    const source = recordCtx.createMediaStreamSource(mediaStream)
    processor = recordCtx.createScriptProcessor(4096, 1, 1)
    chunks = []
    processor.onaudioprocess = (event) => {
      chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)))
    }
    source.connect(processor)
    processor.connect(recordCtx.destination)
    recording = true
    talk.textContent = '说完了'
    talk.classList.add('primary')
    setStatus('在听…说完点「说完了」')
  } catch (err) {
    setStatus('麦克风打不开：' + (err && err.message ? err.message : String(err)))
    log('mic-error', String(err && err.message || err))
  }
}

async function stopRecordingAndSend() {
  const talk = $('talk')
  recording = false
  talk.textContent = '说话'
  talk.classList.remove('primary')
  try {
    if (processor) { processor.disconnect(); processor.onaudioprocess = null }
    if (recordCtx) await recordCtx.close()
    if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop())
    processor = null
    recordCtx = null
    mediaStream = null
    const total = chunks.reduce((n, c) => n + c.length, 0)
    if (total < 4000) { setStatus('太短了，再说一次？'); return }
    const merged = new Float32Array(total)
    let offset = 0
    for (const chunk of chunks) { merged.set(chunk, offset); offset += chunk.length }
    chunks = []
    setStatus('听懂中…')
    const wav = encodeWav(merged, 16000)
    const res = await fetch('/api/asr', { method: 'POST', headers: { 'Content-Type': 'audio/wav' }, body: wav })
    const data = await res.json().catch(() => ({}))
    if (!res.ok || !data.text) {
      setStatus(data.error ? '没听清：' + data.error : '没听清，再说一次？')
      log('asr-fail', '')
      return
    }
    setStatus('')
    log('asr-ok', data.text.slice(0, 40))
    $('say').value = data.text
    addBubble('你（语音）', data.text)
    try {
      await fetch('/api/voice/transcript', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: data.text, speaker: 'owner' }),
      })
    } catch (err) { log('voice-transcript-error', String(err)) }
    await sendChatText(data.text)
  } catch (err) {
    setStatus('识别失败：' + (err && err.message ? err.message : String(err)))
    log('asr-error', String(err && err.message || err))
  }
}

// ── 大脑 ─────────────────────────────────────────────
async function tellJoke() {
  stopSpeaking()
  setStatus('豆宝在翻笑话本…')
  try {
    const res = await fetch('/api/joke?lang=' + state.lang)
    const data = await res.json()
    addBubble('豆宝', data.text)
    await speak(data.text, 'happy')
  } catch (err) {
    setStatus('笑话本掉地上了')
  }
}

async function postImage(dataUrl, caption) {
  try {
    const res = await fetch('/api/multimodal/image', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_base64: dataUrl, caption: caption || '', source: 'stage' }),
    })
    const data = await res.json()
    addBubble('系统', data.ok ? '看到了图片：' + (data.caption || '') : '图片没接上：' + (data.reason || ''))
  } catch (err) { log('mm-image-error', String(err)) }
}

async function postAudio(dataUrl) {
  try {
    const res = await fetch('/api/multimodal/audio', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ audio_base64: dataUrl, lang: state.lang || 'zh', source: 'stage' }),
    })
    const data = await res.json()
    addBubble('系统', data.ok ? '听到了：' + data.text : '语音没接上：' + (data.reason || ''))
  } catch (err) { log('mm-audio-error', String(err)) }
}

$('mm-image').addEventListener('change', (event) => {
  const file = event.target.files && event.target.files[0]
  if (!file) return
  const reader = new FileReader()
  reader.onload = () => postImage(reader.result, file.name)
  reader.readAsDataURL(file)
})

$('mm-audio-file').addEventListener('change', (event) => {
  const file = event.target.files && event.target.files[0]
  if (!file) return
  const reader = new FileReader()
  reader.onload = () => postAudio(reader.result)
  reader.readAsDataURL(file)
})

$('mm-camera').addEventListener('click', async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true })
    const video = document.createElement('video')
    video.srcObject = stream
    await video.play()
    await new Promise((resolve) => setTimeout(resolve, 600))
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth || 640
    canvas.height = video.videoHeight || 480
    canvas.getContext('2d').drawImage(video, 0, 0)
    stream.getTracks().forEach((track) => track.stop())
    postImage(canvas.toDataURL('image/jpeg', 0.8), '摄像头拍照')
  } catch (err) {
    addBubble('系统', '打不开摄像头：' + String(err))
  }
})

async function sendChatText(text) {
  lastUserText = text
  if (true) {
    addBubble('系统', '单主体模式：说话只在 DSH 会话里进行，舞台只负责演。')
    return
  }
  if (state.outbox) {
    try {
      const res = await fetch('/api/mirror/send', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })
      const data = await res.json()
      addBubble('系统', data.ok ? '已放进舞台信箱，等待 DSH 会话处理（几分钟内）' : '信箱写入失败')
    } catch (err) { log('outbox-error', String(err)) }
    return
  }
  const message = String(text || '').trim()
  if (!message) { setStatus('先打点字，或者点「说话」'); return }
  addBubble('你', message)
  stopSpeaking()
  state.history.push({ role: 'user', content: message })
  setStatus('豆宝在想要怎么回…')
  try {
    const res = await fetch('/api/orchestrate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        content: message,
        lang: state.lang,
        session: 'main',
        state: loopState(),
      }),
    })
    const data = await res.json()
    setStatus('')
    addBubble('豆宝', data.text)
    showMood(data.mood)
    state.history.push({ role: 'assistant', content: data.text })
    await speak(data.text, data.expression || 'happy')
    if (data.gesture) startGesture(data.gesture)
    if (data.arbitration) log('arbitration', JSON.stringify(data.arbitration.chosen) + '|' + JSON.stringify(data.arbitration.scores))
  } catch (err) {
    setStatus('大脑没连上')
    log('chat-error', String(err && err.message || err))
  }
}

async function sendChat() {
  const input = $('say')
  const text = input.value.trim()
  if (!text) { setStatus('先打点字，或者点「说话」'); return }
  input.value = ''
  await sendChatText(text)
}


// ── 环境感知与场景 ───────────────────────────────────
const SCENES = {
  'low-battery': {
    expr: 'sad', gesture: 'tilt', cooldown: 5 * 60000, speak: true,
    lines: { zh: '电量快没啦，记得充电哦。', ja: 'バッテリーが少ないよ、充電してね。', en: 'Battery is running low — should I get plugged in?' },
  },
  charging: {
    expr: 'happy', gesture: 'nod', cooldown: 2 * 60000, speak: true,
    lines: { zh: '充上电啦，安心～', ja: '充電きた、安心だね。', en: "Charging — that's better." },
  },
  offline: {
    expr: 'sad', gesture: 'tilt', cooldown: 3 * 60000, speak: false,
    lines: { zh: '网断啦，我先安静一会儿。', ja: 'ネットが切れたみたい。少し静かにしてるね。', en: "Network's gone — I'll stay quiet for now." },
  },
  online: {
    expr: 'happy', gesture: 'nod', cooldown: 60000, speak: true,
    lines: { zh: '网回来啦！', ja: 'ネット復活！', en: 'Network is back!' },
  },
  back: {
    expr: 'happy', gesture: 'sway', cooldown: 5 * 60000, speak: true,
    lines: { zh: '你回来啦～', ja: 'おかえり〜', en: 'Welcome back!' },
  },
  night: {
    expr: 'shy', gesture: 'tilt', cooldown: 90 * 60000, speak: true,
    lines: { zh: '这么晚了还不睡吗？', ja: 'こんな時間まで起きてるの？', en: "It's late — still awake?" },
  },
  idle: {
    expr: 'neutral', gesture: 'sway', cooldown: 10 * 60000, speak: false,
    lines: { zh: '有人吗…我有点无聊。', ja: 'ねえ、誰かいる？ちょっと退屈。', en: "Anyone there? I'm a bit bored." },
  },
  external: { expr: 'surprised', gesture: 'look', cooldown: 8000, speak: false, lines: { zh: '', ja: '', en: '' } },
}

function reactScene(type, extra = {}) {
  if (!model) return
  const scene = SCENES[type] || SCENES.external
  if (!scene) return
  if (!state.autoReact && type !== 'external') return
  const now = Date.now()
  if (scene.lastAt && now - scene.lastAt < scene.cooldown) return
  scene.lastAt = now
  if (state.speaking) return
  const expression = extra.expression || scene.expr
  setExpression(expression)
  startGesture(scene.gesture)
  const line = extra.text || scene.lines[state.lang] || scene.lines.zh
  state.scenes.push(type)
  if (state.scenes.length > 5) state.scenes.shift()
  log('scene', { type, line: String(line || '').slice(0, 30) })
  if (!line) return
  addBubble('豆宝', line)
  if (scene.speak !== false && audioCtx) speak(line, expression)
}

function setupPerception() {
  log('perception-ready', { autoReact: state.autoReact })
  // 电量
  try {
    if (navigator.getBattery) {
      navigator.getBattery().then((battery) => {
        const onLevel = () => {
          if (!battery.charging && battery.level <= 0.2) reactScene('low-battery')
        }
        battery.addEventListener('levelchange', onLevel)
        battery.addEventListener('chargingchange', () => {
          if (battery.charging) reactScene('charging')
        })
        onLevel()
      }).catch(() => {})
    }
  } catch (err) { /* 老浏览器没有 */ }
  // 网络
  window.addEventListener('online', () => reactScene('online'))
  window.addEventListener('offline', () => reactScene('offline'))
  // 回到页面
  document.addEventListener('visibilitychange', () => { if (!document.hidden) reactScene('back') })
  // 外部事件（DSH agent / 系统桥 / 定时任务都可以往 /api/event 推）
  // 用轮询而不是 SSE：手机 Chrome 切后台会切断长连接，轮询回来后能补上错过的
  let eventSeq = -1
  async function pollEvents() {
    try {
      const response = await fetch('/api/poll?since=' + eventSeq)
      if (!response.ok) return
      const data = await response.json()
      eventSeq = typeof data.seq === 'number' ? data.seq : eventSeq
      for (const event of data.events || []) {
        const silentTypes = new Set(['system', 'shadow', 'body', 'perception', 'multimodal'])
        if (silentTypes.has(event.type)) continue
        if (event.type === 'visual' && event.url) {
          const img = $('visual-overlay')
          if (img) {
            img.src = event.url
            img.style.display = 'block'
            setTimeout(() => { img.style.display = 'none' }, 9000)
          }
          addBubble('豆宝（画面）', event.text || '（生成了一张画面）')
        }
        if (event.type === 'vision' && event.text) {
          addBubble('豆宝（看到）', event.text)
          if (event.expression && model) {
            setExpression(event.expression)
            if (event.gesture) startGesture(event.gesture)
          }
          continue
        }
        if (event.type === 'imagination' && event.text) {
          addBubble('豆宝（想象）', event.text)
          if (event.expression) setExpression(event.expression)
          if (event.gesture) startGesture(event.gesture)
          if (event.visual) {
            const img = $('visual-overlay')
            if (img) {
              img.src = event.visual
              img.style.display = 'block'
              setTimeout(() => { img.style.display = 'none' }, 11000)
            }
          }
        }
        if (SCENES[event.type]) reactScene(event.type, event)
      }
    } catch (err) { /* 下一轮再试 */ }
  }
  pollEvents()
  setInterval(pollEvents, 2000)
  document.addEventListener('visibilitychange', () => { if (!document.hidden) pollEvents() })
  // 发呆与深夜
  setInterval(captureSurface, 2000)
  // 轨迹上报间隔从参数层读，不再写死
  fetch('/api/ui-params').then((r) => r.json()).then((params) => {
    UI_PARAMS = { ...UI_PARAMS, ...(params || {}) }
  }).catch(() => {})
  ;(function loopTrajectory() {
    setTimeout(async () => {
      try { await postTrajectory() } catch (e) { /* 忽略 */ }
      loopTrajectory()
    }, Number(UI_PARAMS.trajectory_interval_ms) || 10000)
  })()
  // 闲置时由本地训练出的行为模型决定她做什么
  setInterval(async () => {
    if (state.speaking || !state.autoReact || document.hidden) return
    const idle = Date.now() - lastActivity
    if (idle < 120000) return
    try {
      const decision = await (await fetch('/api/behavior/decide?idle=' + Math.floor(idle / 1000))).json()
      log('behavior', decision.action + ':' + (decision.reason || ''))
      if (!decision || decision.action === 'none' || decision.action === 'quiet') return
      const startedAt = Date.now()
      const outcome = (action) => {
        setTimeout(async () => {
          const engaged = lastActivity > startedAt
          try {
            await fetch('/api/behavior/outcome', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ action, engaged, idle: Math.floor(idle / 1000) }),
            })
          } catch (err) { /* 忽略 */ }
        }, 90000)
      }
      if (decision.action === 'gesture') {
        setExpression(decision.expression || currentExpr)
        if (decision.gesture) startGesture(decision.gesture)
        outcome('gesture')
      } else if (decision.action === 'joke') {
        await tellJoke()
        outcome('joke')
      } else if ((decision.action === 'care' || decision.action === 'greet') && decision.text) {
        addBubble('豆宝', decision.text)
        await speak(decision.text, decision.expression || 'neutral')
        outcome(decision.action)
      }
    } catch (err) { /* 忽略 */ }
  }, 60000)
  refreshMood()
  setInterval(refreshMood, 60000)
  // 长时间没人理时，她自己找个话题（由心情和记忆决定）
  setInterval(async () => {
    if (state.speaking || !state.autoReact || document.hidden) return
    if (Date.now() - lastActivity < 180000) return
    try {
      const res = await fetch('/api/initiative')
      const data = await res.json()
      if (!data || data.skip || !data.text) return
      addBubble('豆宝', data.text)
      showMood(data.mood)
      await speak(data.text, data.expression || 'neutral')
    } catch (err) { /* 忽略 */ }
  }, 90000)
  setInterval(checkIdle, 60000)
  setInterval(checkNight, 5 * 60000)
  setTimeout(checkNight, 2000)
}

function checkIdle() {
  if (Date.now() - lastActivity > 8 * 60000) {
    lastActivity = Date.now()
    reactScene('idle')
  }
}

function checkNight() {
  const hour = new Date().getHours()
  if (hour >= 23 || hour < 5) reactScene('night')
}

['pointerdown', 'keydown', 'input', 'click'].forEach((eventName) => {
  document.addEventListener(eventName, () => { lastActivity = Date.now() }, true)
})

// ── 交互绑定 ─────────────────────────────────────────
$('joke').addEventListener('click', () => { log('btn', 'joke'); tellJoke() })
$('stop').addEventListener('click', () => {
  log('btn', 'stop')
  if (!state.speaking) { setStatus('现在没在说话'); setTimeout(() => setStatus(''), 1200); return }
  stopSpeaking()
})
$('send').addEventListener('click', () => { log('btn', 'send'); sendChat() })
$('say').addEventListener('keydown', (e) => { if (e.key === 'Enter') sendChat() })
$('talk').addEventListener('click', () => { log('btn', 'talk'); if (recording) stopRecordingAndSend(); else startRecording() })
$('motion').addEventListener('click', () => { log('btn', 'motion'); playMotion() })
$('selfcheck').addEventListener('click', () => { log('btn', 'selfcheck'); selfCheckLoop() })
$('lang').addEventListener('click', () => {
  const order = ['zh', 'ja', 'en']
  const next = order[(order.indexOf(state.lang) + 1) % order.length]
  log('btn', 'lang:' + next)
  applyLanguage(next, { announce: true })
})
document.querySelectorAll('[data-expr]').forEach((btn) => {
  btn.addEventListener('click', () => { log('btn', 'expr:' + btn.dataset.expr); setExpression(btn.dataset.expr) })
})

// 设置
const settingsEl = $('settings')
async function renderValuesReview() {
  try {
    const emergent = await (await fetch('/api/values/emergent')).json()
    const values = await (await fetch('/api/values')).json()
    const children = (emergent.candidates || []).map((candidate, index) =>
      '<div style="display:flex;align-items:flex-start;gap:6px;margin:4px 0">' +
      '<span style="flex:1">候选：' + candidate.principle + '（支持 ' + candidate.support + '，占比 ' + candidate.share + '）</span>' +
      '<button class="ghost" data-index="' + index + '" style="flex-shrink:0">确认</button></div>').join('')
    $('valuesReview').innerHTML =
      '<div style="opacity:.75;margin-bottom:4px">我的三观：' + (values.principles || []).join('；') + '</div>' +
      (children || '<div style="opacity:.6">暂无候选三观</div>')
    $('valuesReview').querySelectorAll('button[data-index]').forEach((button) => {
      button.addEventListener('click', async () => {
        await fetch('/api/values/emergent/confirm', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ index: Number(button.dataset.index), confirm: true }),
        })
        renderValuesReview()
      })
    })
  } catch (err) { /* 忽略 */ }
}

async function sendFeedback(verdict) {
  const text = lastUserText || (document.querySelector('.bubble.me') ? document.querySelector('.bubble.me').textContent : '')
  if (!text) { addBubble('系统', '先说一句话，再反馈她的反应'); return }
  try {
    const res = await fetch('/api/affect/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, verdict }),
    })
    const data = await res.json()
    addBubble('系统', data.ok ? '已记录反馈：' + verdict + '（词：' + (data.word || '-') + '）' : '反馈没记上：' + (data.reason || ''))
  } catch (err) { log('feedback-error', String(err)) }
}

$('fb-over').addEventListener('click', () => sendFeedback('overreact'))
$('fb-miss').addEventListener('click', () => sendFeedback('missed'))
$('fb-ok').addEventListener('click', () => sendFeedback('correct'))

$('stage-only').addEventListener('change', () => {
  state.stageOnly = $('stage-only').checked
  localStorage.setItem('dsh.stageOnly', state.stageOnly ? '1' : '0')
})

$('outbox-toggle').addEventListener('change', () => {
  state.outbox = $('outbox-toggle').checked
  localStorage.setItem('dsh.outbox', state.outbox ? '1' : '0')
})

$('mirror-toggle').addEventListener('change', () => {
  state.mirror = $('mirror-toggle').checked
  localStorage.setItem('dsh.mirror', state.mirror ? '1' : '0')
  if (state.mirror) mirrorStarted = false
})

$('source').addEventListener('change', () => {
  localStorage.setItem('dsh.source', $('source').value)
})

$('correct-btn').addEventListener('click', async () => {
  const snapshot = trajectorySnapshot()
  const expression = (prompt('她应该是什么表情？happy / sad / shy / surprised / neutral') || '').trim()
  if (!expression) return
  const gesture = (prompt('什么动作？sway / nod / tilt / look / 留空') || '').trim()
  try {
    const res = await fetch('/api/layers/correct', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...snapshot,
        next_expression: expression,
        next_gesture: gesture,
        source: localStorage.getItem('dsh.source') || 'owner',
      }),
    })
    const data = await res.json()
    if (data.quarantined) {
      addBubble('系统', '这次纠正被三观闸门隔离：' + ((data.gate && data.gate.reason) || '') + '（来源 ' + (data.source || '') + '，可信度 ' + (data.credibility || '') + '）')
    } else {
      addBubble('系统', '已记下这次纠正：' + expression + (gesture ? ' / ' + gesture : '') + '（来源 ' + (data.source || '') + '）')
    }
    await speak('记下了。', 'neutral')
  } catch (err) {
    log('correct-error', String(err))
  }
})

$('settings-btn').addEventListener('click', async () => {
  settingsEl.classList.remove('hidden')
  try {
    $('source').value = localStorage.getItem('dsh.source') || 'owner'
    $('vision-loop').checked = state.visionLoop
    $('stage-only').checked = state.stageOnly
    $('mirror-toggle').checked = state.mirror
    $('outbox-toggle').checked = state.outbox
    renderValuesReview()
    const stats = await (await fetch('/api/behavior/stats')).json()
    try {
      const dash = await (await fetch('/api/dashboard')).json()
      const inf = dash.inference || {}
      const q = dash.quality || {}
      const st = dash.style || {}
      const loop = dash.loop || {}
      $('dashboard').textContent =
        '推理样本 ' + (inf.total || 0) + '（被回应 ' + (inf.engaged || 0) + '，均延迟 ' + (inf.avg_latency_ms || 0) + 'ms）' +
        '｜层 ' + (dash.layers || 0) + '（冻结 ' + (dash.frozen_layers || 0) + '，修正 ' + (dash.corrections || 0) + '）' +
        '｜边界 ' + (dash.boundaries || 0) + '｜巩固 ' + (loop.consolidations || 0) + ' 次，下轮阈值 ' + (loop.threshold || '-') +
        '｜层权重自适应 ' + ((dash.adaptation && dash.adaptation.adapted_layers) || 0) + ' 层' +
        '｜行为分 ' + ((dash.scores && dash.scores.behavior && dash.scores.behavior.score != null) ? dash.scores.behavior.score : '—') +
        '｜关系分 ' + ((dash.scores && dash.scores.relationship) ? dash.scores.relationship.score : '—') +
        '（语言来自 API，行为/关系来自本地）' +
        '｜自我一致性 ' + ((dash.consistency && dash.consistency.score != null) ? Math.round(dash.consistency.score * 100) + '%' : '待检查') +
        '｜自评基准 ' + ((dash.benchmark && dash.benchmark.score != null) ? dash.benchmark.score : '未跑') +
        '｜质量模型 ' + (q.loaded ? 'acc ' + q.accuracy : '待训练') +
        '｜风格：' + (st.hint || '还没学到')
    } catch (err) { /* 忽略 */ }
    const model = stats.model || {}
    const rate = (bucket) => bucket.count ? Math.round(bucket.engaged / bucket.count * 100) : 0
    const lines = Object.entries(stats.by_action || {}).map(([action, bucket]) => action + ' 回应率 ' + rate(bucket) + '%')
    document.getElementById('behaviorStats').textContent =
      '行为模型：' + (model.loaded ? '已训练（' + model.samples + ' 条反馈，准确率 ' + model.accuracy + '）' : '样本不足') +
      '｜' + lines.join(' · ')
  } catch (err) { /* 忽略 */ }
})
$('settings-close').addEventListener('click', () => settingsEl.classList.add('hidden'))

$('narrate').addEventListener('click', async () => {
  const button = $('narrate')
  button.disabled = true
  setStatus('她在想怎么介绍自己…')
  try {
    const res = await fetch('/api/self/narrate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lang: state.lang || 'zh' }),
    })
    const data = await res.json()
    if (data.text) {
      addBubble('豆宝', data.text)
      log('narrate', data.source || 'unknown')
      await speak(data.text, 'neutral')
    }
  } catch (err) {
    log('narrate-error', String(err))
  } finally {
    button.disabled = false
    setStatus('')
  }
})
applyLanguage(state.lang)
$('rate').value = state.rate
$('autoreact').checked = state.autoReact
$('jokesource').value = state.jokeSource
$('voice').addEventListener('change', () => {
  state.voice = $('voice').value
  VOICE_MEMORY[state.lang] = state.voice
  localStorage.setItem('stts.voices', JSON.stringify(VOICE_MEMORY))
})
$('key').value = state.key
$('brain').value = state.brain
$('baseurl').value = state.baseURL
$('model').value = state.model
$('settings-save').addEventListener('click', () => {
  state.voice = $('voice').value
  VOICE_MEMORY[state.lang] = state.voice
  localStorage.setItem('stts.voices', JSON.stringify(VOICE_MEMORY))
  state.rate = $('rate').value
  state.key = $('key').value.trim()
  state.brain = $('brain').value
  state.baseURL = $('baseurl').value.trim()
  state.model = $('model').value.trim()
  state.autoReact = $('autoreact').checked
  state.jokeSource = $('jokesource').value
  localStorage.setItem('stts.jokeSource', state.jokeSource)
  localStorage.setItem('stts.autoReact', state.autoReact ? '1' : '0')
  localStorage.setItem('stts.brain', state.brain)
  localStorage.setItem('stts.baseURL', state.baseURL)
  localStorage.setItem('stts.model', state.model)
  localStorage.setItem('stts.voice', state.voice)
  localStorage.setItem('stts.rate', state.rate)
  localStorage.setItem('stts.key', state.key)
  settingsEl.classList.add('hidden')
  setStatus('设置保存好了')
  setTimeout(() => setStatus(''), 1500)
})

// 唤醒（解锁声音 + 开场招呼）
$('wake-btn').addEventListener('click', async () => {
  $('wake').classList.add('gone')
  ensureAudio()
  try { await audioCtx.resume() } catch (err) { /* 忽略 */ }
  await setExpression('happy')
  await speak((LANGS[state.lang] || LANGS.zh).greeting, 'happy')
  idleTimer = setInterval(() => {
    if (state.speaking || !model) return
    try {
      model.motion('idle', Math.floor(Math.random() * 4), PIXI.live2d.MotionPriority.IDLE)
    } catch (err) { /* 忽略 */ }
  }, 24000)
})

async function autoPickBrain() {
  try {
    const saved = localStorage.getItem('stts.brain')
    if (saved === 'custom' || (saved === 'deepseek' && state.key)) return
    const health = await (await fetch('/api/health')).json()
    if (health && health.account) {
      state.brain = 'account'
      state.key = state.key || ''
    } else if (health && health.localBrain) {
      state.brain = 'local'
    } else {
      return
    }
    $('brain').value = state.brain
    localStorage.setItem('stts.brain', state.brain)
    log('brain', state.brain + '-auto')
  } catch (err) { /* 忽略 */ }
}

async function restorePresence() {
  try {
    const data = await (await fetch('/api/presence')).json()
    const presence = data.presence || {}
    const age = Date.now() / 1000 - (presence.last_seen || 0)
    if (presence.last_seen && age < 600) {
      if (presence.expression) await setExpression(presence.expression)
      if (presence.gesture) startGesture(presence.gesture)
      if (presence.mood) state.moodLabel = presence.mood
      log('presence-restored', { expression: presence.expression, gesture: presence.gesture, age: Math.round(age) })
    }
  } catch (err) { /* 忽略 */ }
}

// ── 启动 ─────────────────────────────────────────────
;(async () => {
  try {
    await setupModel()
    restorePresence()
    autoPickBrain()
    setupPerception()
    setInterval(mirrorLoop, 3000)
    setStatus('豆宝准备好了，点一下唤醒')
  } catch (err) {
    setStatus('模型加载失败：' + (err && err.message ? err.message : String(err)))
    log('model-error', String(err && err.stack || err))
  }
})()

// ── 手机系统数据上报（浏览器可见部分）─────────────────
async function reportSystem(kind, text) {
  try {
    await fetch('/api/system/ingest', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kind, text, source: 'stage-page' }),
    })
  } catch (err) { /* 忽略 */ }
}

const systemFlags = { network: null, charging: null, level: null, visibility: null, lastBeat: 0 }

async function systemTick() {
  try {
    const heartbeat = Date.now() - systemFlags.lastBeat > 10 * 60000
    const online = navigator.onLine
    if (heartbeat || systemFlags.network !== online) {
      systemFlags.network = online
      await reportSystem('network', online ? '网络在线' : '网络断开')
    }
    if (navigator.getBattery) {
      const battery = await navigator.getBattery()
      const level = Math.round(battery.level * 100)
      const charging = !!battery.charging
      if (heartbeat || systemFlags.charging !== charging || Math.abs((systemFlags.level || 0) - level) >= 5) {
        systemFlags.charging = charging
        systemFlags.level = level
        await reportSystem('battery', `电量 ${level}%，${charging ? '充电中' : '未充电'}`)
      }
    }
    const visibility = document.visibilityState === 'visible'
    if (heartbeat || systemFlags.visibility !== visibility) {
      systemFlags.visibility = visibility
      await reportSystem('visibility', visibility ? '页面在前台' : '页面在后台')
    }
    if (heartbeat) systemFlags.lastBeat = Date.now()
  } catch (err) { /* 忽略 */ }
}
setTimeout(systemTick, 8000)
setInterval(systemTick, 60000)
window.addEventListener('online', () => reportSystem('network', '网络恢复'))
window.addEventListener('offline', () => reportSystem('network', '网络断开'))

// ── 媒体测试：麦克风 / 扬声器 / 摄像头 ───────────────
let testPreviewStream = null

function testOut(text) {
  const el = $('test-result')
  if (el) el.textContent = text
}

async function testMic() {
  if (recording) return
  testOut('录音 3 秒…')
  await startRecording()
  setTimeout(async () => {
    try {
      if (processor) { processor.disconnect(); processor.onaudioprocess = null }
      if (recordCtx) await recordCtx.close()
      if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop())
      processor = null; recordCtx = null; mediaStream = null; recording = false
      const total = chunks.reduce((n, c) => n + c.length, 0)
      if (total < 4000) { testOut('太短，没录到声音'); return }
      const merged = new Float32Array(total)
      let offset = 0
      for (const chunk of chunks) { merged.set(chunk, offset); offset += chunk.length }
      chunks = []
      const wav = encodeWav(merged, 16000)
      const res = await fetch('/api/asr', { method: 'POST', headers: { 'Content-Type': 'audio/wav' }, body: wav })
      const data = await res.json().catch(() => ({}))
      testOut(data.text ? '识别：' + data.text : '没听清：' + (data.error || '未知'))
    } catch (err) { testOut('麦克风测试失败：' + String(err && err.message || err)) }
  }, 3000)
}

async function testSpeaker() {
  testOut('播放测试音…')
  try {
    ensureAudio(); await audioCtx.resume()
    const osc = audioCtx.createOscillator()
    const gain = audioCtx.createGain()
    osc.frequency.value = 660
    gain.gain.value = 0.08
    osc.connect(gain); gain.connect(audioCtx.destination)
    osc.start(); setTimeout(() => osc.stop(), 250)
  } catch (err) { testOut('测试音失败：' + String(err)) }
  try {
    testOut('测试音已播放，接着念一句…')
    await speak('这是扬声器测试，能听到吗？', 'happy')
    testOut('扬声器测试完成')
  } catch (err) { testOut('朗读失败（TTS 需要联网）：' + String(err && err.message || err)) }
}

async function toggleVideoTest() {
  const video = $('test-preview')
  if (testPreviewStream) {
    testPreviewStream.getTracks().forEach((t) => t.stop())
    testPreviewStream = null
    video.style.display = 'none'
    testOut('摄像头已关闭')
    return
  }
  try {
    testPreviewStream = await navigator.mediaDevices.getUserMedia({ video: true })
    video.srcObject = testPreviewStream
    video.style.display = 'block'
    testOut('摄像头预览中，点「拍一张」上传')
  } catch (err) { testOut('摄像头打不开：' + String(err && err.message || err)) }
}

async function testShot() {
  const video = $('test-preview')
  if (!video || !testPreviewStream) { testOut('先点「摄像头」打开预览'); return }
  try {
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth || 640
    canvas.height = video.videoHeight || 480
    canvas.getContext('2d').drawImage(video, 0, 0)
    const dataUrl = canvas.toDataURL('image/jpeg', 0.75)
    const res = await fetch('/api/multimodal/vision', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_base64: dataUrl, source: 'media-test' }),
    })
    const data = await res.json()
    testOut(data.ok ? '已上传图片：' + (data.caption || '未识别') : '上传失败：' + (data.reason || ''))
  } catch (err) { testOut('拍照失败：' + String(err && err.message || err)) }
}

if ($('model-select')) {
  const current = localStorage.getItem('stage.model') || ''
  $('model-select').value = current.includes('senko') ? 'senko' : 'hibiki'
  $('model-select').addEventListener('change', () => {
    const choice = $('model-select').value
    if (choice === 'senko') {
      localStorage.setItem('stage.model', 'models/senko/senko.model3.json')
      localStorage.setItem('stage.runtime', 'cubism4')
    } else {
      localStorage.removeItem('stage.model')
      localStorage.setItem('stage.runtime', 'cubism2')
    }
    location.reload()
  })
}

if ($('test-mic')) $('test-mic').addEventListener('click', testMic)
if ($('test-speaker')) $('test-speaker').addEventListener('click', testSpeaker)
if ($('test-video')) $('test-video').addEventListener('click', toggleVideoTest)
if ($('test-shot')) $('test-shot').addEventListener('click', testShot)

// ── 环境感知：定时抓帧，变化才描述 ─────────────────
let envStream = null
let envTimer = null
let envLastBrightness = null
let envBusy = false
let envLastPost = 0

async function envConfig() {
  try {
    const res = await fetch('/api/perception/environment')
    return await res.json()
  } catch (err) { return { enabled: false, interval_sec: 60, describe_interval_sec: 600, change_threshold: 0.08 } }
}

async function setEnvStatus(text) {
  const el = $('env-status')
  if (el) el.textContent = text
}

async function startEnvPerception() {
  if (envStream) return
  try {
    envStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
    const video = document.createElement('video')
    video.srcObject = envStream
    await video.play()
    await setEnvStatus('环境感知中（摄像头）')
    const loop = async () => {
      if (!envStream) return
      const cfg = await envConfig()
      if (!cfg.enabled) { stopEnvPerception(); return }
      if (document.visibilityState !== 'visible') { envTimer = setTimeout(loop, 5000); return }
      try {
        const canvas = document.createElement('canvas')
        canvas.width = 320
        canvas.height = 240
        const ctx = canvas.getContext('2d')
        ctx.drawImage(video, 0, 0, 320, 240)
        const pixels = ctx.getImageData(0, 0, 320, 240).data
        let sum = 0
        for (let i = 0; i < pixels.length; i += 4 * 64) {
          sum += (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3
        }
        const brightness = sum / (pixels.length / (4 * 64)) / 255
        const changed = envLastBrightness == null || Math.abs(brightness - envLastBrightness) > Number(cfg.change_threshold)
        envLastBrightness = brightness
        const due = Date.now() - envLastPost > Number(cfg.describe_interval_sec) * 1000
        if (changed && due && !envBusy) {
          envBusy = true
          envLastPost = Date.now()
          await setEnvStatus('看到环境有变化，正在识别…')
          try {
            const dataUrl = canvas.toDataURL('image/jpeg', 0.7)
            const res = await fetch('/api/multimodal/vision', {
              method: 'POST', headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ image_base64: dataUrl, source: 'environment' }),
            })
            const data = await res.json()
            if (data.ok && data.caption) addBubble('看到环境', data.caption)
          } catch (err) { log('env-error', String(err)) }
          envBusy = false
          await setEnvStatus('环境感知中（摄像头）')
        }
      } catch (err) { log('env-frame-error', String(err)) }
      envTimer = setTimeout(loop, Number(cfg.interval_sec) * 1000)
    }
    loop()
  } catch (err) {
    await setEnvStatus('摄像头打不开：' + String(err))
  }
}

function stopEnvPerception() {
  if (envTimer) { clearTimeout(envTimer); envTimer = null }
  if (envStream) { envStream.getTracks().forEach((t) => t.stop()); envStream = null }
  envLastBrightness = null
  setEnvStatus('已关闭')
}

if ($('env-quick')) {
  $('env-quick').addEventListener('click', async () => {
    const box = $('env-perception')
    if (!box) return
    box.checked = !box.checked
    box.dispatchEvent(new Event('change'))
    addBubble('系统', box.checked ? '环境感知已开启（摄像头）' : '环境感知已关闭')
  })
}

if ($('env-perception')) {
  const savedEnv = localStorage.getItem('zhiming.envPerception') === '1'
  $('env-perception').checked = savedEnv
  if (savedEnv) startEnvPerception()
  $('env-perception').addEventListener('change', async () => {
    const enabled = $('env-perception').checked
    localStorage.setItem('zhiming.envPerception', enabled ? '1' : '0')
    try {
      await fetch('/api/perception/environment', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      })
    } catch (err) { log('env-config-error', String(err)) }
    if (enabled) startEnvPerception()
    else stopEnvPerception()
  })
}

// ── 主动标注：挑她没把握的状态，让人补标签 ───────────
const EXPR_ZH = { neutral: '平静', happy: '开心', sad: '难过', surprised: '惊讶', angry: '生气', shy: '害羞' }

async function renderLabelQueue() {
  const box = $('label-queue')
  if (!box) return
  try {
    const data = await (await fetch('/api/labeling/queue')).json()
    const items = data.items || []
    if (!items.length) { box.textContent = '暂时没有需要标注的'; return }
    box.innerHTML = ''
    for (const item of items) {
      const wrap = document.createElement('div')
      wrap.style.cssText = 'margin:6px 0;padding:8px;border:1px solid rgba(160,140,220,.25);border-radius:10px'
      const meta = document.createElement('div')
      meta.style.cssText = 'font-size:11px;color:#8d82b5'
      meta.textContent = `${item.when} · 预测 ${EXPR_ZH[item.predicted] || item.predicted} · 置信 ${item.confidence}`
      const text = document.createElement('div')
      text.style.cssText = 'font-size:12px;margin:4px 0;color:#d8d0f0'
      text.textContent = item.context
      const row = document.createElement('div')
      row.style.cssText = 'display:flex;gap:6px;align-items:center'
      const select = document.createElement('select')
      for (const [value, label] of Object.entries(EXPR_ZH)) {
        const option = document.createElement('option')
        option.value = value
        option.textContent = label
        if (value === item.predicted) option.selected = true
        select.appendChild(option)
      }
      const save = document.createElement('button')
      save.textContent = '保存'
      save.onclick = async () => {
        save.disabled = true
        try {
          const res = await fetch('/api/labeling/submit', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id: item.id, expression: select.value }),
          })
          const data = await res.json()
          if (data.ok) wrap.remove()
        } catch (err) { log('label-error', String(err)) }
        save.disabled = false
      }
      row.appendChild(select)
      row.appendChild(save)
      wrap.appendChild(meta)
      wrap.appendChild(text)
      wrap.appendChild(row)
      box.appendChild(wrap)
    }
    if (!box.children.length) box.textContent = '暂时没有需要标注的'
  } catch (err) { box.textContent = '加载失败' }
}
setTimeout(renderLabelQueue, 4000)

setInterval(() => {
  if (!model || gesture || state.speaking) return
  try {
    const group = motionGroups.includes('idle') ? 'idle' : motionGroups[0]
    if (group) model.motion(group, Math.floor(Math.random() * 4), PIXI.live2d.MotionPriority.IDLE)
  } catch (err) { /* 忽略 */ }
}, 25000)

// ── 语音引擎切换（edge / kokoro）─────────────
async function initVoiceEngine() {
  const select = $('voice-engine')
  if (!select) return
  try {
    const identity = await (await fetch('/api/voice/identity')).json()
    select.value = identity.engine === 'kokoro' ? 'kokoro' : 'edge'
  } catch (err) { /* 忽略 */ }
  select.addEventListener('change', async () => {
    try {
      await fetch('/api/voice/identity', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ engine: select.value }),
      })
    } catch (err) { log('voice-engine-error', String(err)) }
  })
}
setTimeout(initVoiceEngine, 3500)

// ── 日志查看器 ───────────────────────────
async function loadLogs() {
  const box = $('log-list')
  if (!box) return
  const kind = $('log-kind') ? $('log-kind').value : ''
  try {
    const data = await (await fetch('/api/log/recent?lines=25&kind=' + encodeURIComponent(kind))).json()
    const items = data.items || []
    if (!items.length) { box.textContent = '没有记录'; return }
    box.innerHTML = items.slice().reverse().map((item) => {
      const detail = item.detail && typeof item.detail === 'object' ? JSON.stringify(item.detail) : String(item.detail || '')
      return `<div style="padding:2px 0;border-bottom:1px solid rgba(160,140,220,.12)"><span style="color:#8d82b5">${item.t || ''}</span> ${item.event || ''} <span style="color:#b9aee0">${detail.slice(0, 90)}</span></div>`
    }).join('')
  } catch (err) { box.textContent = '加载失败' }
}
if ($('log-refresh')) $('log-refresh').addEventListener('click', loadLogs)
if ($('log-kind')) $('log-kind').addEventListener('change', loadLogs)
setTimeout(loadLogs, 5000)
