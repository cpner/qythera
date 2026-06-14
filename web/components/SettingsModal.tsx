'use client';
import { X } from 'lucide-react';

export default function SettingsModal({ isOpen, onClose }) {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center" onClick={onClose}>
      <div className="glass w-[400px] p-6" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-bold">Settings</h2>
          <button onClick={onClose} className="p-1 hover:bg-white/10 rounded"><X size={18} /></button>
        </div>
        <div className="space-y-4">
          <div>
            <label className="text-sm text-white/60 block mb-1">Model</label>
            <select className="w-full bg-white/5 border border-white/10 rounded-lg p-2 text-sm text-white outline-none">
              <option>Vaelon 7B</option><option>Vaelon 13B</option><option>Vaelon 70B MoE</option>
            </select>
          </div>
          <div>
            <label className="text-sm text-white/60 block mb-1">Temperature</label>
            <input type="range" min="0" max="2" step="0.1" defaultValue="0.7" className="w-full" />
          </div>
        </div>
      </div>
    </div>
  );
}
