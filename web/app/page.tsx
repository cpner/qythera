'use client';
import { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Msg { id: string; role: 'user' | 'assistant'; content: string; time?: string; }

const SUGGESTIONS = [
  { icon: '💻', text: 'Write a Python sorting algorithm' },
  { icon: '🧠', text: 'Explain how transformers work' },
  { icon: '✍️', text: 'Help me write an email' },
  { icon: '🔬', text: 'What is machine learning?' },
  { icon: '🎨', text: 'Design a REST API' },
  { icon: '📊', text: 'Explain photosynthesis' },
];

function fmtTime() { return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }); }

export default function Home() {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sidebar, setSidebar] = useState(false);
  const [serverOk, setServerOk] = useState<boolean | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [msgs]);
  useEffect(() => { fetch('/health').then(r => r.json()).then(() => setServerOk(true)).catch(() => setServerOk(false)); }, []);

  const send = useCallback(async () => {
    if (!input.trim() || loading) return;
    const userMsg: Msg = { id: Date.now().toString(), role: 'user', content: input.trim(), time: fmtTime() };
    setMsgs(p => [...p, userMsg]); setInput(''); setLoading(true);
    if (inputRef.current) inputRef.current.style.height = 'auto';
    try {
      const r = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: [...msgs, userMsg].map(m => ({ role: m.role, content: m.content })) }) });
      const d = await r.json();
      setMsgs(p => [...p, { id: (Date.now()+1).toString(), role: 'assistant', content: d.choices?.[0]?.message?.content || 'No response', time: fmtTime() }]);
    } catch { setMsgs(p => [...p, { id: (Date.now()+1).toString(), role: 'assistant', content: 'Server offline. Run: python -m core.inference.server', time: fmtTime() }]); }
    setLoading(false);
  }, [input, loading, msgs]);

  return (
    <div className="flex h-full" style={{height:'100dvh'}}>
      <AnimatePresence>
        {sidebar && (<>
          <motion.div initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}} className="fixed inset-0 bg-black/60 z-40 md:hidden" onClick={() => setSidebar(false)} />
          <motion.div initial={{x:-300}} animate={{x:0}} exit={{x:-300}} transition={{type:'spring',damping:25,stiffness:200}} className="sidebar w-[280px] h-full flex flex-col z-50 fixed md:relative shrink-0">
            <div className="p-4 flex items-center justify-between border-b border-white/5"><span className="font-semibold text-sm">Chats</span><button onClick={() => setSidebar(false)} className="p-1.5 rounded-lg hover:bg-white/5 md:hidden">✕</button></div>
            <div className="flex-1 overflow-y-auto p-3"><button className="btn w-full mb-3 text-sm" onClick={() => { setMsgs([]); setSidebar(false); }}>+ New Chat</button></div>
            <div className="p-4 border-t border-white/5"><div className="flex items-center gap-2 text-[10px] text-white/30"><span className="w-1.5 h-1.5 rounded-full" style={{background: serverOk ? '#22c55e' : serverOk === false ? '#ef4444' : '#eab308'}} />{serverOk ? 'Connected' : serverOk === false ? 'Offline' : 'Checking...'}</div></div>
          </motion.div>
        </>)}
      </AnimatePresence>

      <div className="flex-1 flex flex-col h-full min-w-0">
        <div className="glass-strong flex items-center gap-3 px-3 py-2.5 m-2 mb-0 shrink-0">
          <button onClick={() => setSidebar(!sidebar)} className="p-2 rounded-xl hover:bg-white/5"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg></button>
          <div className="flex items-center gap-2"><div className="w-8 h-8 rounded-xl bg-gradient-to-br from-violet-500 to-blue-500 flex items-center justify-center"><span className="text-white text-sm font-bold">Q</span></div><div><h1 className="text-sm font-semibold gradient-text">Qythera</h1><p className="text-[10px] text-white/25 -mt-0.5">Vaelon AI</p></div></div>
          <div className="ml-auto"><span className="text-[10px] text-white/20 px-2 py-1 rounded-full bg-white/[.03]">{msgs.length} msgs</span></div>
        </div>

        <div className="flex-1 overflow-y-auto px-3 py-4 space-y-3">
          {msgs.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center px-4">
              <motion.div initial={{scale:.9,opacity:0}} animate={{scale:1,opacity:1}} transition={{duration:.5}}>
                <div className="w-20 h-20 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-violet-500 to-blue-500 flex items-center justify-center shadow-2xl shadow-violet-500/25 animate-float"><span className="text-3xl">✦</span></div>
                <h2 className="text-2xl font-bold mb-2 gradient-text">Welcome to Qythera</h2>
                <p className="text-white/30 text-sm max-w-md leading-relaxed mb-8">Ask me anything. I can help with coding, analysis, creative tasks, and more.</p>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 max-w-lg mx-auto">
                  {SUGGESTIONS.map(s => (<button key={s.text} onClick={() => setInput(s.text)} className="glass p-3 text-left hover:border-violet-500/20 transition-all group"><span className="text-lg">{s.icon}</span><p className="text-xs text-white/40 mt-1.5 group-hover:text-white/60 transition">{s.text}</p></button>))}
                </div>
              </motion.div>
            </div>
          ) : msgs.map(m => (
            <motion.div key={m.id} initial={{opacity:0,y:12}} animate={{opacity:1,y:0}} transition={{duration:.3}} className={`flex ${m.role==='user'?'justify-end':'justify-start'}`}>
              <div className={`${m.role==='user'?'msg-user':'msg-bot'}`}>
                {m.role==='assistant' && <div className="flex items-center gap-2 mb-1.5"><div className="w-5 h-5 rounded-lg bg-gradient-to-br from-violet-500 to-blue-500 flex items-center justify-center"><span className="text-[8px] text-white font-bold">Q</span></div><span className="text-[11px] text-violet-300/70 font-medium">Vaelon</span>{m.time && <span className="text-[9px] text-white/15">{m.time}</span>}</div>}
                <div className="text-[13.5px] leading-[1.65] whitespace-pre-wrap break-words">{m.content}</div>
              </div>
            </motion.div>
          ))}
          {loading && <motion.div initial={{opacity:0}} animate={{opacity:1}} className="flex justify-start"><div className="msg-bot"><div className="flex gap-1 py-2 px-1"><span className="typing-dot"/><span className="typing-dot"/><span className="typing-dot"/></div></div></motion.div>}
          <div ref={endRef} />
        </div>

        <div className="px-3 pb-3 shrink-0">
          <div className="input-glass flex items-end gap-2">
            <textarea ref={inputRef} value={input} onChange={e => { setInput(e.target.value); e.target.style.height='auto'; e.target.style.height=Math.min(e.target.scrollHeight,160)+'px'; }} onKeyDown={e => { if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();} }} placeholder="Message Qythera..." rows={1} className="flex-1 bg-transparent text-white placeholder-[#52525b] outline-none resize-none text-[13.5px] py-1" />
            <button onClick={send} disabled={!input.trim()||loading} className="p-2.5 rounded-2xl bg-gradient-to-br from-violet-500 to-blue-500 text-white disabled:opacity-20 transition-all active:scale-95 shrink-0"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg></button>
          </div>
          <p className="text-center text-[10px] text-white/10 mt-2">Qythera AI — Custom autodiff engine, no external APIs</p>
        </div>
      </div>
    </div>
  );
}
