import type { Metadata, Viewport } from 'next';
import './globals.css';
import { SWRegister } from './sw-register';

export const metadata: Metadata = {
  title: 'Qythera AI — Production Superintelligence',
  description: 'Ask anything. Code, analyze, create — powered by Vaelon model architecture with custom autodiff engine. No external AI APIs.',
  keywords: ['AI', 'superintelligence', 'Vaelon', 'Qythera', 'machine learning', 'autodiff', 'transformer'],
  authors: [{ name: 'Qythera Team' }],
  creator: 'Qythera',
  publisher: 'Qythera',
  formatDetection: { telephone: false, email: false, address: false },
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: 'https://qythera.ai',
    siteName: 'Qythera AI',
    title: 'Qythera AI — Production Superintelligence',
    description: 'Ask anything. Code, analyze, create — powered by Vaelon model architecture.',
    images: [{ url: '/icons/icon-512.png', width: 512, height: 512, alt: 'Qythera AI' }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Qythera AI',
    description: 'Production Superintelligence with Vaelon Model',
    images: ['/icons/icon-512.png'],
  },
  manifest: '/manifest.json',
  appleWebApp: {
    capable: true,
    statusBarStyle: 'black-translucent',
    title: 'Qythera',
  },
  other: {
    'mobile-web-app-capable': 'yes',
    'msapplication-TileColor': '#7c3aed',
    'msapplication-TileImage': '/icons/icon-144.png',
    'theme-color': '#7c3aed',
  },
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 5,
  userScalable: true,
  themeColor: [
    { media: '(prefers-color-scheme: dark)', color: '#09090b' },
    { media: '(prefers-color-scheme: light)', color: '#fafafa' },
  ],
  viewportFit: 'cover',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="icon" href="/favicon.ico" sizes="any" />
        <link rel="icon" href="/icons/logo.svg" type="image/svg+xml" />
        <link rel="apple-touch-icon" href="/icons/icon-180.png" />
        <link rel="apple-touch-icon" sizes="180x180" href="/icons/icon-180.png" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
        <meta name="apple-mobile-web-app-title" content="Qythera" />
        <meta name="application-name" content="Qythera" />
        <meta name="msapplication-TileColor" content="#7c3aed" />
        <meta name="msapplication-tap-highlight" content="no" />
        <meta name="theme-color" content="#7c3aed" />
      </head>
      <body>
        <div className="bg-anim" />
        <div className="relative z-10 h-full">
          {children}
        </div>
        <SWRegister />
      </body>
    </html>
  );
}
