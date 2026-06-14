'use client';
import { useState } from 'react';

interface Tab { id: string; label: string; }
interface TabsProps { tabs: Tab[]; activeTab?: string; onChange: (id: string) => void; }

export default function Tabs({ tabs, activeTab, onChange }: TabsProps) {
  const [active, setActive] = useState(activeTab || tabs[0]?.id);
  const current = activeTab || active;
  return (
    <div className="flex gap-1 p-1 bg-white/5 rounded-lg">
      {tabs.map(tab => (
        <button key={tab.id} onClick={() => { setActive(tab.id); onChange(tab.id); }}
          className={`px-3 py-1.5 rounded-md text-sm transition-all ${current === tab.id ? 'bg-white/10 text-white' : 'text-white/40 hover:text-white/60'}`}>
          {tab.label}
        </button>
      ))}
    </div>
  );
}
