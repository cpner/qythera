const CACHE='qythera-v1';
const ASSETS=['/','/manifest.json','/icon-192.png','/icon-512.png'];

self.addEventListener('install',e=>{
  e.waitUntil(caches.open(CACHE).then(c=>c.addAll(ASSETS)));
  self.skipWaiting();
});

self.addEventListener('activate',e=>{
  e.waitUntil(caches.keys().then(k=>Promise.all(k.filter(x=>x!==CACHE).map(x=>caches.delete(x)))));
  self.clients.claim();
});

self.addEventListener('fetch',e=>{
  if(e.request.url.includes('/v1/')||e.request.url.includes('/api/'))return;
  e.respondWith(
    fetch(e.request).then(r=>{
      const c=r.clone();
      caches.open(CACHE).then(cache=>cache.put(e.request,c));
      return r;
    }).catch(()=>caches.match(e.request))
  );
});

self.addEventListener('push',e=>{
  const d=e.data?e.data.json():{};
  self.registration.showNotification(d.title||'Qythera',{body:d.body||'New message',icon:'/icon-192.png'});
});

self.addEventListener('notificationclick',e=>{
  e.notification.close();
  e.waitUntil(clients.openWindow('/'));
});
