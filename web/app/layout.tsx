import type { Metadata } from 'next';
import './globals.css';
export const metadata: Metadata = { title: 'Qythera AI', description: 'Production Superintelligence' };
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (<html lang="en" className="dark"><body><div className="bg-anim" /><div className="relative z-10 h-full">{children}</div></body></html>);
}
