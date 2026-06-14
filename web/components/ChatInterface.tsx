'use client';
import { useState, useRef } from 'react';
import { Send } from 'lucide-react';

export default function ChatInterface({ onSendMessage, isLoading }) {
  const [input, setInput] = useState('');
  const ref = useRef(null);

  const submit = () => {
    if (!input.trim() || isLoading) return;
    onSendMessage(input.trim());
    setInput('');
  };

  return (
    <div className="px-4 pb-4">
      <div className="input-glass flex items-end gap-2">
        <textarea ref={ref} value={input} onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); } }}
          placeholder="Message Qythera..." rows={1}
          className="flex-1 bg-transparent text-white placeholder-white/30 outline-none resize-none text-sm py-1" />
        <button onClick={submit} disabled={!input.trim() || isLoading}
          className="p-2 rounded-xl bg-gradient-to-br from-qythera-500 to-blue-500 text-white disabled:opacity-30 transition-all">
          <Send size={18} />
        </button>
      </div>
    </div>
  );
}
