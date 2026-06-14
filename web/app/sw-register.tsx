'use client';

import { useEffect } from 'react';

export function SWRegister() {
  useEffect(() => {
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker
        .register('/sw.js')
        .then((reg) => {
          console.log('[Qythera] SW registered:', reg.scope);
          // Check for updates periodically
          setInterval(() => reg.update(), 60 * 60 * 1000);
        })
        .catch((err) => console.warn('[Qythera] SW registration failed:', err));
    }

    // Request notification permission
    if ('Notification' in window && Notification.permission === 'default') {
      // Don't auto-request, let user trigger it
    }

    // Handle install prompt
    let deferredPrompt: any = null;
    window.addEventListener('beforeinstallprompt', (e) => {
      e.preventDefault();
      deferredPrompt = e;
      // Store for later use
      (window as any).__installPrompt = deferredPrompt;
    });

    window.addEventListener('appinstalled', () => {
      deferredPrompt = null;
      console.log('[Qythera] App installed');
    });
  }, []);

  return null;
}
