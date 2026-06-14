'use client';
import { useEffect } from 'react';

export function useKeyboardShortcuts(shortcuts: Record<string, () => void>) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const key = [e.ctrlKey && 'ctrl', e.metaKey && 'meta', e.shiftKey && 'shift', e.key].filter(Boolean).join('+');
      if (shortcuts[key]) { e.preventDefault(); shortcuts[key](); }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [shortcuts]);
}
