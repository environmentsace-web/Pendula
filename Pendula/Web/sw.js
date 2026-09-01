/* Pendula - minimalni servisni radnik.
   Dvije stvari: telefon aplikaciju smatra instalabilnom tek kad ovo postoji,
   i sam interfejs se otvara i bez mreze. Podaci o zonama se uvijek povlace
   svjezi - stari podaci bi bili gori od nikakvih. */

const KES = "pendula-v3";   // podici broj pri svakoj izmjeni interfejsa
const OKOSNICA = [
  "./",
  "./index.html",
  "./manifest.webmanifest",
  "./ikone/pendula-192.png",
  "./ikone/pendula-512.png",
];

self.addEventListener("install", e => {
  e.waitUntil(
    caches.open(KES)
      .then(k => k.addAll(OKOSNICA))
      .catch(() => {})      // ako neki fajl fali, instalacija svejedno prolazi
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", e => {
  e.waitUntil(
    caches.keys()
      .then(imena => Promise.all(
        imena.filter(i => i !== KES).map(i => caches.delete(i))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);

  // Podaci o zonama i karta uvijek idu sa mreze.
  const uvijek_svjeze = url.pathname.endsWith(".geojson") ||
                        url.pathname.endsWith("index.json") ||
                        url.hostname.includes("tile.openstreetmap") ||
                        url.hostname.includes("arcgisonline");
  if (uvijek_svjeze) return;

  // Okosnica: prvo mreza, pa kes kao rezerva kad nema veze.
  e.respondWith(
    fetch(e.request)
      .then(odgovor => {
        const kopija = odgovor.clone();
        caches.open(KES).then(k => k.put(e.request, kopija)).catch(() => {});
        return odgovor;
      })
      .catch(() => caches.match(e.request).then(r => r || caches.match("./index.html")))
  );
});
