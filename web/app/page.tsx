'use client';
import { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Msg { id: string; role: 'user' | 'assistant'; content: string; }

export default function Home() {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sidebar, setSidebar] = useState(true);
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [msgs]);

  const send = useCallback(async () => {
    if (!input.trim() || loading) return;
    const userMsg: Msg = { id: Date.now().toString(), role: 'user', content: input.trim() };
    setMsgs(p => [...p, userMsg]);
    setInput('');
    setLoading(true);
    if (inputRef.current) inputRef.current.style.height = 'auto';
    try {
      const r = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: [...msgs, userMsg].map(m => ({ role: m.role, content: m.content })) }) });
      const d = await r.json();
      setMsgs(p => [...p, { id: (Date.now()+1).toString(), role: 'assistant', content: d.choices?.[0]?.message?.content || 'No response' }]);
    } catch {
      setMsgs(p => [...p, { id: (Date.now()+1).toString(), role: 'assistant', content: 'Server offline. Run: python -m inference.server' }]);
    }
    setLoading(false);
  }, [input, loading, msgs]);

  return (
    <div className="flex h-full">
      <AnimatePresence>
        {sidebar && (
          <motion.div initial={{x:-280}} animate={{x:0}} exit={{x:-280}} className="sidebar w-[280px] h-full flex flex-col">
            <div className="p-4"><button className="btn w-full" onClick={() => setMsgs([])}>+ New Chat</button></div>
            <div className="flex-1 overflow-y-auto p-2 text-sm text-white/50">
              {msgs.length > 0 && <div className="px-3 py-2 rounded-lg bg-white/5 mb-1">Current conversation</div>}
            </div>
            <div className="p-4 border-t border-white/5 text-xs text-white/30">Qythera v0.1.0 - Vaelon Model</div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="flex-1 flex flex-col h-full">
        <div className="glass flex items-center gap-3 px-4 py-3 m-2 mb-0">
          <button onClick={() => setSidebar(!sidebar)} className="p-2 rounded-lg hover:bg-white/10">☰</button>
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-q-500 to-blue-500 flex items-center justify-center text-white text-xs font-bold">Q</div>
          <span className="font-bold bg-gradient-to-r from-q-400 to-blue-400 bg-clip-text text-transparent">Qythera</span>
          <span className="ml-auto text-xs text-white/30">Vaelon</span>
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
          {msgs.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <motion.div initial={{scale:.8,opacity:0}} animate={{scale:1,opacity:1}}>
                <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-q-500 to-blue-500 flex items-center justify-center text-2xl animate-float">✦</div>
                <h2 className="text-xl font-bold mb-2 bg-gradient-to-r from-q-300 to-blue-300 bg-clip-text text-transparent">Welcome to Qythera</h2>
                <p className="text-white/40 text-sm max-w-md">Ask me anything. I can help with coding, analysis, creative tasks, and more.</p>
                <div className="flex flex-wrap gap-2 mt-6 justify-center">
                  {['Explain quantum computing','Write a Python script','Help me debug code'].map(s => (
                    <button key={s} onClick={() => setInput(s)} className="glass px-4 py-2 text-xs text-white/50 hover:text-white transition">{s}</button>
                  ))}
                </div>
              </motion.div>
            </div>
          ) : msgs.map(m => (
            <motion.div key={m.id} initial={{opacity:0,y:10}} animate={{opacity:1,y:0}} className={`flex ${m.role==='user'?'justify-end':'justify-start'}`}>
              <div className={m.role==='user'?'msg-user':'msg-bot'}>
                {m.role==='assistant' && <div className="flex items-center gap-1.5 mb-1"><div className="w-4 h-4 rounded bg-gradient-to-br from-q-500 to-blue-500"/><span className="text-xs text-q-300 font-medium">Vaelon</span></div>}
                <p className="text-sm leading-relaxed whitespace-pre-wrap">{m.content}</p>
              </div>
            </motion.div>
          ))}
          {loading && <div className="flex justify-start"><div className="msg-bot"><div className="flex gap-1 py-2"><span className="dot"/><span className="dot"/><span className="dot"/></div></div></div>}
          <div ref={endRef} />
        </div>

        <div className="px-4 pb-4">
          <div className="input-glass flex items-end gap-2">
            <textarea ref={inputRef} value={input}
              onChange={e => { setInput(e.target.value); e.target.style.height='auto'; e.target.style.height=Math.min(e.target.scrollHeight,200)+'px'; }}
              onKeyDown={e => { if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();} }}
              placeholder="Message Qythera..." rows={1} className="flex-1 bg-transparent text-white placeholder-white/30 outline-none resize-none text-sm py-1" />
            <button onClick={send} disabled={!input.trim()||loading} className="p-2 rounded-xl bg-gradient-to-br from-q-500 to-blue-500 text-white disabled:opacity-30 transition">➤</button>
          </div>
          <p className="text-center text-xs text-white/15 mt-1.5">Qythera AI - Powered by Vaelon Model</p>
        </div>
      </div>
    </div>
  );
}
