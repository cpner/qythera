import type { Metadata, Viewport } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Qythera AI - Production Superintelligence',
  description: 'Ask anything. Code, analyze, create.',
  manifest: '/manifest.json',
  appleWebApp: { capable: true, statusBarStyle: 'black-translucent', title: 'Qythera' },
};

export const viewport: Viewport = {
  width: 'device-width', initialScale: 1, maximumScale: 5,
  themeColor: [{ media: '(prefers-color-scheme: dark)', color: '#09090b' }],
  viewportFit: 'cover',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (<html lang="en" className="dark"><body><div className="bg-anim" /><div className="relative z-10 h-full">{children}</div></body></html>);
}
