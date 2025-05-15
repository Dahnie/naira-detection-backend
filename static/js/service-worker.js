// Service Worker for Naira Detection System
const CACHE_NAME = 'naira-detector-v1';
const ASSETS = [
    '/',
    '/index.html',
    '/static/css/styles.css',
    '/static/js/app.js',
    '/static/assets/icons/icon-192x192.png',
    '/static/assets/icons/icon-512x512.png',
    '/static/manifest.json'
];

// Install event - cache all static assets
self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => {
                console.log('Opened cache');
                return cache.addAll(ASSETS);
            })
    );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys().then(cacheNames => {
            return Promise.all(
                cacheNames.map(cacheName => {
                    if (cacheName !== CACHE_NAME) {
                        return caches.delete(cacheName);
                    }
                })
            );
        })
    );
});

// Fetch event - serve from cache if available, otherwise fetch from network
self.addEventListener('fetch', event => {
    // Skip cross-origin requests and API calls
    if (event.request.url.startsWith(self.location.origin) &&
        !event.request.url.includes('/api/')) {
        event.respondWith(
            caches.match(event.request)
                .then(response => {
                    if (response) {
                        return response;
                    }

                    return fetch(event.request).then(
                        response => {
                            // Don't cache if response is not valid
                            if (!response || response.status !== 200 || response.type !== 'basic') {
                                return response;
                            }

                            // Clone response and cache it
                            let responseToCache = response.clone();
                            caches.open(CACHE_NAME)
                                .then(cache => {
                                    cache.put(event.request, responseToCache);
                                });

                            return response;
                        }
                    );
                })
        );
    }
});

// Handle push notifications
self.addEventListener('push', event => {
    const data = event.data.json();

    const options = {
        body: data.body || 'New scan result available',
        icon: '/static/assets/icons/icon-192x192.png',
        badge: '/static/assets/icons/badge-icon.png',
        vibrate: [100, 50, 100],
        data: {
            url: data.url || '/'
        }
    };

    event.waitUntil(
        self.registration.showNotification(
            data.title || 'Naira Detector Update',
            options
        )
    );
});

// Handle notification clicks
self.addEventListener('notificationclick', event => {
    event.notification.close();

    event.waitUntil(
        clients.openWindow(event.notification.data.url)
    );
});