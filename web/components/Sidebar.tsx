'use client';
import { Plus, MessageSquare, Settings } from 'lucide-react';

export default function Sidebar({ conversations, currentId, onSelect, onNew, onSettings }) {
  return (
    <div className="sidebar-glass w-[280px] h-full flex flex-col">
      <div className="p-4 border-b border-white/5">
        <button onClick={onNew} className="btn-gradient w-full flex items-center justify-center gap-2">
          <Plus size={18} /> New Chat
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-2">
        {conversations.map(conv => (
          <button key={conv.id} onClick={() => onSelect(conv.id)}
            className={`w-full text-left p-3 rounded-lg mb-1 transition-all ${currentId === conv.id ? 'bg-white/10 text-white' : 'text-white/60 hover:bg-white/5'}`}>
            <div className="flex items-center gap-2"><MessageSquare size={14} /><span className="truncate text-sm">{conv.title}</span></div>
          </button>
        ))}
      </div>
      <div className="p-4 border-t border-white/5">
        <button onClick={onSettings} className="w-full flex items-center gap-2 text-white/50 hover:text-white/80 p-2 rounded-lg hover:bg-white/5">
          <Settings size={16} /><span className="text-sm">Settings</span>
        </button>
      </div>
    </div>
  );
}
