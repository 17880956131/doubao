// dsh-go 原生插件：豆宝（数字人）入口
// 状态条 + 面板；数据来自本机 8100 服务，不主动做有副作用的操作
import { spawn } from 'node:child_process'
import { existsSync, readFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

export const name = 'zhiming-dsh-go'

const BASE = 'http://127.0.0.1:8100'
const PLUGIN_DIR = dirname(dirname(fileURLToPath(import.meta.url)))
const CANDIDATES = [
  process.env.DOUBAO_HOME,
  join(homedir(), 'workspace', 'digital-human'),
  '/root/workspace/digital-human',
  join(homedir(), 'doubao'),
  join(PLUGIN_DIR, 'server'),
].filter(Boolean)
const SERVER_DIR = CANDIDATES.find((dir) => existsSync(join(dir, 'server.py'))) || null
const DEFAULT_INTERVALS = { strip_refresh_ms: 20000, status_refresh_ms: 15000, three_refresh_ms: 30000 }
let uiIntervals = { ...DEFAULT_INTERVALS }
let localParams = { plugin_autostart: true }

function schedule(key, tick) {
  let alive = true
  const loop = async () => {
    if (!alive) return
    try { await tick() } catch (error) { /* 忽略 */ }
    if (alive) setTimeout(loop, Number(uiIntervals[key]) || DEFAULT_INTERVALS[key])
  }
  loop()
  return () => { alive = false }
}

async function api(path, options = {}) {
  const response = await fetch(BASE + path, {
    headers: { 'Content-Type': 'application/json' },
    signal: AbortSignal.timeout(8000),
    ...options,
  })
  if (!response.ok) throw new Error('HTTP ' + response.status)
  return response.json()
}

function startServer() {
  if (!SERVER_DIR) return null
  const child = spawn('python3', ['server.py', '--port', '8100'], {
    cwd: SERVER_DIR,
    detached: true,
    stdio: 'ignore',
  })
  child.unref()
  return child
}

async function snapshot() {
  try {
    const dashboard = await api('/api/dashboard')
    let presence = {}
    try {
      presence = (await api('/api/presence')).presence || {}
    } catch (error) { /* 忽略 */ }
    return { online: true, dashboard, presence }
  } catch (error) {
    return { online: false, dashboard: null, presence: {} }
  }
}

export function apply(ctx) {
  ctx.inject(['dshGo'], (ctx) => {
    fetch(BASE + '/api/ui-params')
      .then((response) => response.json())
      .then((params) => { uiIntervals = { ...uiIntervals, ...(params || {}) }; localParams = { ...localParams, ...(params || {}) } })
      .catch(() => {})

    // 自启：插件加载时如果服务没跑就拉起（参数 plugin_autostart 控制）
    try {
      const paramsPath = SERVER_DIR ? join(SERVER_DIR, 'data', 'ui_params.json') : ''
      const local = JSON.parse(readFileSync(paramsPath, 'utf8'))
      localParams = { ...localParams, ...(local || {}) }
    } catch (error) { /* 文件不存在就用默认 */ }
    if (localParams.plugin_autostart !== false) {
      snapshot().then((data) => {
        if (!data.online) {
          startServer()
          try {
            if (SERVER_DIR) {
            spawn('bash', ['-c', `pgrep -f guardian.sh >/dev/null || (cd ${JSON.stringify(SERVER_DIR)} && nohup ./guardian.sh >/dev/null 2>&1 &)`], {
              detached: true, stdio: 'ignore',
            }).unref()
          }
          } catch (error) { /* 忽略 */ }
        }
      }).catch(() => {})
    }

    // 输入框上方状态条
    ctx.dshGo.strip('doubao-strip', (scope, emit) => {
      let alive = true
      const tick = async () => {
        const data = await snapshot()
        if (!alive) return
        if (!data.online) {
          emit({ icon: 'robot', text: '豆宝 · 睡着中（点开唤醒）', tone: 'muted' })
          return
        }
        const loop = data.dashboard.loop || {}
        const benchmark = data.dashboard.benchmark || {}
        const layers = data.dashboard.layers || 0
        const score = benchmark.score == null ? '-' : benchmark.score
        const expression = data.presence && data.presence.expression
        const text = expression
          ? `豆宝 · ${expression} · ${layers}层`.slice(0, 24)
          : `豆宝 · ${layers}层 · 自评${score}`.slice(0, 24)
        emit({ icon: 'robot', text })
      }
      const cancel = schedule('strip_refresh_ms', tick)
      return () => {
        alive = false
        cancel()
      }
    })

    // 一键唤醒：像小鱼的「呼叫小鱼」那样，入口就能把服务拉起来
    ctx.dshGo.on('wake-doubao', async () => {
      const data = await snapshot()
      if (data.online) return { toast: '豆宝已经醒着' }
      if (!SERVER_DIR) return { toast: '找不到豆宝服务目录，先安装/配置服务' }
      startServer()
      return { toast: '正在唤醒豆宝，稍等几秒' }
    })

    // 每回合脚注：显示她此刻在想什么（注意力 + 动荡）
    ctx.dshGo.turnNote('doubao-note', (scope, emit) => {
      let alive = true
      ;(async () => {
        try {
          const state = await api('/api/self/state')
          const turbulence = await api('/api/turbulence')
          const affect = await api('/api/affect')
          if (!alive) return
          const focus = (state.attention || [])[0]
          const parts = []
          if (focus) parts.push('关注 ' + focus.layer + ' ' + Math.round((focus.weight || 0) * 100) + '%')
          if (affect && affect.stress > 0.3) parts.push('压力 ' + affect.stress)
          parts.push('动荡 ' + turbulence.level)
          const text = parts.join(' · ').slice(0, 32)
          emit({ [scope.turnId || 1]: { text, tone: 'muted' } })
        } catch (error) { /* 忽略 */ }
      })()
      return () => { alive = false }
    })

    // 四层面板：魂 / 魄 / 体（主机）/ 形
    ctx.dshGo.panel('three', {
      view: (scope, caps) => ({
        type: 'stack',
        title: '魂 · 魄 · 体 · 形',
        children: [
          { type: 'text', text: '魂 · 身份与意义', style: { weight: 'bold' } },
          { type: 'keyValue', items: { $: 'data.soul' } },
          { type: 'text', text: '魄 · 情绪与本能', style: { weight: 'bold' } },
          { type: 'keyValue', items: { $: 'data.instinct' } },
          { type: 'text', text: '体 · 主机（算力与存储）', style: { weight: 'bold' } },
          { type: 'keyValue', items: { $: 'data.host' } },
          { type: 'text', text: '形 · 形象与感知', style: { weight: 'bold' } },
          { type: 'keyValue', items: { $: 'data.form' } },
        ],
      }),
      data: (scope, emit) => {
        let alive = true
        const tick = async () => {
          try {
            const layer = await api('/api/three-layers')
            const consciousness = await api('/api/consciousness')
            if (!alive) return
            const soul = layer['魂'] || {}
            const instinct = layer['魄'] || {}
            const host = layer['体'] || {}
            const form = layer['形'] || {}
            const attention = (instinct['注意力'] || []).map((item) => item.layer).join('、') || '—'
            emit({
              soul: [
                ['记忆', soul['长期记忆'] == null ? '—' : soul['长期记忆']],
                ['三观', (soul['三观'] || []).length + ' 条'],
                ['权重模因', soul['权重模因']],
                ['代码基因', soul['代码基因']],
                ['自我一致性', soul['自我一致性'] == null ? '—' : Math.round(soul['自我一致性'] * 100) + '%'],
                ['自述', soul['自述条数']],
              ],
              instinct: [
                ['心情', instinct['心情'] || '—'],
                ['注意力', attention],
                ['时代层', instinct['时代层']],
                ['巩固', instinct['巩固次数']],
                ['自评', instinct['自评基准'] == null ? '—' : instinct['自评基准']],
                ['意识节拍', consciousness.ticks || 0],
              ],
              host: [
                ['主机', host['主机'] || '—'],
                ['CPU', host['CPU负载'] == null ? '—' : String(host['CPU负载'])],
                ['可用内存', host['可用内存MB'] == null ? '—' : host['可用内存MB'] + ' MB'],
                ['磁盘', host['磁盘剩余MB'] == null ? '—' : Math.round(host['磁盘剩余MB'] / 1024) + ' GB'],
                ['运行', Math.round((host['运行时长秒'] || 0) / 60) + ' 分钟'],
                ['电量', host['电量'] || '—'],
              ],
              form: [
                ['模型', form['模型'] || '—'],
                ['表情', form['表情'] || '—'],
                ['动作', form['动作'] || '—'],
                ['场景', form['场景'] || '—'],
                ['感知事件', form['感知事件']],
                ['清醒', Math.round((consciousness.awake_seconds || 0) / 60) + ' 分钟'],
              ],
            })
          } catch (error) {
            if (alive) emit({ soul: [['状态', '睡着中']], instinct: [['状态', '—']], host: [['状态', '—']], form: [['状态', '—']] })
          }
        }
        const cancel = schedule('three_refresh_ms', tick)
        return () => { alive = false; cancel() }
      },
    })

    // 成长面板：基准分曲线 + 时代层/巩固次数
    ctx.dshGo.panel('growth', {
      view: (scope, caps) => ({
        type: 'stack',
        title: '豆宝成长',
        children: [
          { type: 'keyValue', items: { $: 'data.runs' } },
          { type: 'text', text: '最近十次自评基准分（越低越需要巩固）', tone: 'muted' },
          { type: 'row', children: [
            { type: 'metric', label: '时代层', value: { $: 'data.layers' } },
            { type: 'metric', label: '巩固次数', value: { $: 'data.consolidations' } },
            { type: 'metric', label: '最新自评', value: { $: 'data.latest' } },
          ] },
          { type: 'callout', tone: 'info', text: {
            $: 'data.note',
          } },
        ],
      }),
      data: (scope, emit) => {
        let alive = true
        const tick = async () => {
          try {
            const benchmark = await api('/api/benchmark')
            const dashboard = await api('/api/dashboard')
            if (!alive) return
            const history = (benchmark.history || []).slice(-10)
            const runs = history.map((item) => [
              new Date((item.ts || 0) * 1000).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' }),
              String(item.score),
            ])
            emit({
              runs,
              layers: dashboard.layers || 0,
              consolidations: (dashboard.loop && dashboard.loop.consolidations) || 0,
              latest: benchmark.latest ? benchmark.latest.score : '—',
              note: benchmark.latest
                ? '最近一次：表情准确率 ' + benchmark.latest.expression_accuracy + '，动作准确率 ' + benchmark.latest.gesture_accuracy
                : '还没有自评记录，点「自我检查」跑一次',
            })
          } catch (error) {
            if (alive) emit({ runs: [['状态', '—']], layers: '—', consolidations: '—', latest: '—', note: '豆宝没醒，先点唤醒' })
          }
        }
        tick()
        return () => { alive = false }
      },
    })

    // 状态面板：视图 + 数据
    ctx.dshGo.panel('status', {
      view: (scope, caps) => ({
        type: 'stack',
        title: '豆宝',
        children: [
          { type: 'metric', label: '状态', value: { $: 'data.online' }, tone: 'info' },
          { type: 'row', children: [
            { type: 'metric', label: '时代层', value: { $: 'data.layers' } },
            { type: 'metric', label: '自评', value: { $: 'data.benchmark' } },
            { type: 'metric', label: '一致性', value: { $: 'data.consistency' } },
          ] },
          { type: 'keyValue', items: [
            ['表情', { $: 'data.expression' }],
            ['动作', { $: 'data.gesture' }],
            ['心情', { $: 'data.mood' }],
            ['关注', { $: 'data.focus' }],
            ['上次活动', { $: 'data.lastSeen' }],
          ] },
          { type: 'actions', items: [
            { label: '唤醒', action: 'wake', style: 'primary' },
            { label: '让她自述', action: 'narrate' },
            { label: '自我检查', action: 'selfcheck' },
            { label: '巩固一层', action: 'consolidate' },
          ] },
        ],
      }),
      data: (scope, emit) => {
        let alive = true
        const tick = async () => {
          const data = await snapshot()
          if (!alive) return
          if (!data.online) {
            emit({ online: '睡眠中', layers: '—', benchmark: '—', consistency: '—',
                   expression: '—', gesture: '—', mood: '—', focus: '—', lastSeen: '—' })
            return
          }
          const dashboard = data.dashboard
          const presence = data.presence
          const focus = (data.dashboard.adaptation ? null : null)
          const focusLayer = presence.attention && presence.attention[0]
          const lastSeen = presence.last_seen ? new Date(presence.last_seen * 1000).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' }) : '—'
          const consistency = dashboard.consistency && dashboard.consistency.score
          emit({
            online: '在线',
            layers: dashboard.layers,
            benchmark: dashboard.benchmark && dashboard.benchmark.score,
            consistency: consistency == null ? '—' : `${Math.round(consistency * 100)}%`,
            expression: presence.expression || '—',
            gesture: presence.gesture || '—',
            mood: presence.mood || (dashboard.quality && dashboard.quality.engaged_rate != null ? '' : '—'),
            focus: focusLayer ? `${focusLayer.layer} ${Math.round((focusLayer.weight || 0) * 100)}%` : '—',
            lastSeen,
          })
        }
        const cancel = schedule('status_refresh_ms', tick)
        return () => {
          alive = false
          cancel()
        }
      },
      actions: {
        wake: async (scope, event) => {
          const data = await snapshot()
          if (data.online) return { toast: '豆宝已经醒着' }
          startServer()
          return { toast: '正在唤醒豆宝，稍等几秒', refresh: true }
        },
        narrate: async () => {
          try {
            const result = await api('/api/self/narrate', {
              method: 'POST',
              body: JSON.stringify({ lang: String(ctx.caps?.locale || 'zh').startsWith('en') ? 'en' : 'zh' }),
            })
            const text = String(result.text || '').slice(0, 400)
            return text ? { insert: text, toast: '豆宝的自述已放进输入框' } : { toast: '她暂时没说话' }
          } catch (error) {
            return { toast: '她还没醒，先点唤醒' }
          }
        },
        'toggle-autostart': async () => {
          localParams.plugin_autostart = !(localParams.plugin_autostart !== false)
          try {
            await api('/api/ui-params/update', {
              method: 'POST',
              body: JSON.stringify({ plugin_autostart: localParams.plugin_autostart }),
            })
            return { toast: localParams.plugin_autostart ? '已开启开机自启' : '已关闭开机自启', refresh: true }
          } catch (error) {
            return { toast: '切换失败，豆宝可能没醒' }
          }
        },
        selfcheck: async () => {
          try {
            const result = await api('/api/self-check', { method: 'POST', body: '{}' })
            const score = result.score == null ? '—' : result.score
            return { toast: `自评完成：${score}`, refresh: true }
          } catch (error) {
            return { toast: '自我检查失败，她可能没醒' }
          }
        },
        consolidate: async () => {
          try {
            const result = await api('/api/loop/consolidate', { method: 'POST', body: '{}' })
            return { toast: result.ok ? `巩固完成：${result.layer && result.layer.label || '新一层'}` : '数据还不够，先积累', refresh: true }
          } catch (error) {
            return { toast: '巩固失败，她可能没醒' }
          }
        },
      },
    })
  })
}
