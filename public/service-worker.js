const CACHE = 'zhiming-stage-v3'
const SHELL = [
  '/', '/index.html', '/app.js', '/manifest.webmanifest',
  '/icons/icon-192.png', '/icons/icon-512.png',
  '/lib/pixi.min.js', '/lib/live2d.min.js', '/lib/cubism4core.min.js',
  '/lib/cubism2.min.js', '/lib/cubism4.min.js',
  '/models/hibiki/hibiki.model.json', '/models/hibiki/hibiki.physics.json',
  '/models/hibiki/moc/hibiki.moc', '/models/hibiki/moc/hibiki.2048/texture_00.png',
  '/models/hibiki/mtn/idle_01.mtn', '/models/hibiki/mtn/idle_02.mtn',
  '/models/hibiki/mtn/idle_03.mtn', '/models/hibiki/mtn/idle_04.mtn',
  '/models/hibiki/mtn/hibiki_01.mtn', '/models/hibiki/mtn/hibiki_02.mtn',
  '/models/hibiki/mtn/hibiki_03.mtn', '/models/hibiki/mtn/hibiki_04.mtn',
  '/models/hibiki/mtn/hibiki_05.mtn',
  '/models/hibiki/exp/f01.exp.json', '/models/hibiki/exp/f02.exp.json',
  '/models/hibiki/exp/f03.exp.json', '/models/hibiki/exp/f04.exp.json',
  '/models/hibiki/exp/f05.exp.json', '/models/hibiki/exp/f06.exp.json'
]

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE)
    await Promise.all(SHELL.map(async (path) => {
      try { await cache.add(path) } catch (err) { /* 单个失败不影响 */ }
    }))
    self.skipWaiting()
  })())
})

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const names = await caches.keys()
    await Promise.all(names.filter((n) => n !== CACHE).map((n) => caches.delete(n)))
    self.clients.claim()
  })())
})

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url)
  if (event.request.method !== 'GET' || url.origin !== location.origin || url.pathname.startsWith('/api/')) return
  const networkFirst = url.pathname === '/' || url.pathname === '/index.html' || url.pathname === '/app.js'
  event.respondWith((async () => {
    const cache = await caches.open(CACHE)
    if (networkFirst) {
      try {
        const response = await fetch(event.request)
        if (response.ok) cache.put(event.request, response.clone())
        return response
      } catch (err) {
        const hit = await cache.match(event.request, { ignoreSearch: true })
        return hit || cache.match('/index.html')
      }
    }
    const hit = await cache.match(event.request, { ignoreSearch: true })
    if (hit) return hit
    try {
      const response = await fetch(event.request)
      if (response.ok) cache.put(event.request, response.clone())
      return response
    } catch (err) {
      return cache.match('/index.html')
    }
  })())
})
