'use client';

import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Settings, Plus, MessageSquare, Menu, X, Zap, Brain, Sparkles } from 'lucide-react';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
}

export default function Home() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConvId, setCurrentConvId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const currentConv = conversations.find(c => c.id === currentConvId);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const newConversation = () => {
    const conv: Conversation = {
      id: Date.now().toString(),
      title: 'New Chat',
      messages: [],
      createdAt: new Date(),
    };
    setConversations(prev => [conv, ...prev]);
    setCurrentConvId(conv.id);
    setMessages([]);
  };

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInput('');
    setIsLoading(true);

    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: newMessages.map(m => ({ role: m.role, content: m.content })),
        }),
      });

      const data = await response.json();
      const assistantMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.choices?.[0]?.message?.content || 'No response received.',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMsg]);
    } catch (error) {
      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Connection error. Make sure the inference server is running on port 8000.',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex h-full">
      {/* Sidebar */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            initial={{ x: -280 }}
            animate={{ x: 0 }}
            exit={{ x: -280 }}
            className="sidebar-glass w-[280px] h-full flex flex-col"
          >
            <div className="p-4 border-b border-white/5">
              <button onClick={newConversation} className="btn-gradient w-full flex items-center justify-center gap-2">
                <Plus size={18} />
                New Chat
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-2">
              {conversations.map(conv => (
                <button
                  key={conv.id}
                  onClick={() => {
                    setCurrentConvId(conv.id);
                    setMessages(conv.messages);
                  }}
                  className={`w-full text-left p-3 rounded-lg mb-1 transition-all ${
                    currentConvId === conv.id
                      ? 'bg-white/10 text-white'
                      : 'text-white/60 hover:bg-white/5 hover:text-white/80'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <MessageSquare size={14} />
                    <span className="truncate text-sm">{conv.title}</span>
                  </div>
                </button>
              ))}
            </div>
            <div className="p-4 border-t border-white/5">
              <button
                onClick={() => setSettingsOpen(true)}
                className="w-full flex items-center gap-2 text-white/50 hover:text-white/80 transition-colors p-2 rounded-lg hover:bg-white/5"
              >
                <Settings size={16} />
                <span className="text-sm">Settings</span>
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Content */}
      <div className="flex-1 flex flex-col h-full">
        {/* Header */}
        <div className="glass flex items-center justify-between px-4 py-3 m-2 mb-0">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 rounded-lg hover:bg-white/10 transition-colors"
            >
              {sidebarOpen ? <X size={18} /> : <Menu size={18} />}
            </button>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-qythera-500 to-blue-500 flex items-center justify-center">
                <Sparkles size={16} className="text-white" />
              </div>
              <h1 className="text-lg font-bold bg-gradient-to-r from-qythera-400 to-blue-400 bg-clip-text text-transparent">
                Qythera
              </h1>
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs text-white/40">
            <Brain size={14} />
            <span>Vaelon 7B</span>
          </div>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full">
              <motion.div
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                className="text-center"
              >
                <div className="w-20 h-20 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-qythera-500 to-blue-500 flex items-center justify-center animate-float">
                  <Sparkles size={36} className="text-white" />
                </div>
                <h2 className="text-2xl font-bold mb-2 bg-gradient-to-r from-qythera-300 to-blue-300 bg-clip-text text-transparent">
                  Welcome to Qythera
                </h2>
                <p className="text-white/50 max-w-md">
                  Powered by Vaelon model. Ask me anything — I can help with coding, analysis, creative tasks, and more.
                </p>
                <div className="flex gap-3 mt-8 justify-center">
                  {['Explain quantum computing', 'Write a Python script', 'Analyze this dataset'].map((suggestion) => (
                    <button
                      key={suggestion}
                      onClick={() => setInput(suggestion)}
                      className="glass-card px-4 py-2 text-sm text-white/60 hover:text-white transition-all"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </motion.div>
            </div>
          ) : (
            messages.map((msg) => (
              <motion.div
                key={msg.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={msg.role === 'user' ? 'message-user' : 'message-assistant'}>
                  {msg.role === 'assistant' && (
                    <div className="flex items-center gap-2 mb-2">
                      <div className="w-5 h-5 rounded bg-gradient-to-br from-qythera-500 to-blue-500 flex items-center justify-center">
                        <Zap size={10} className="text-white" />
                      </div>
                      <span className="text-xs text-qythera-300 font-medium">Vaelon</span>
                    </div>
                  )}
                  <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                </div>
              </motion.div>
            ))
          )}
          {isLoading && (
            <div className="flex justify-start">
              <div className="message-assistant">
                <div className="flex items-center gap-2 mb-1">
                  <div className="w-5 h-5 rounded bg-gradient-to-br from-qythera-500 to-blue-500 flex items-center justify-center">
                    <Zap size={10} className="text-white" />
                  </div>
                  <span className="text-xs text-qythera-300 font-medium">Vaelon</span>
                </div>
                <div className="flex gap-1 py-2">
                  <span className="typing-dot" />
                  <span className="typing-dot" />
                  <span className="typing-dot" />
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="px-4 pb-4">
          <div className="input-glass flex items-end gap-2">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                e.target.style.height = 'auto';
                e.target.style.height = Math.min(e.target.scrollHeight, 200) + 'px';
              }}
              onKeyDown={handleKeyDown}
              placeholder="Message Qythera..."
              rows={1}
              className="flex-1 bg-transparent text-white placeholder-white/30 outline-none resize-none text-sm leading-relaxed py-1"
            />
            <button
              onClick={sendMessage}
              disabled={!input.trim() || isLoading}
              className="p-2 rounded-xl bg-gradient-to-br from-qythera-500 to-blue-500 text-white disabled:opacity-30 disabled:cursor-not-allowed transition-all hover:shadow-lg hover:shadow-qythera-500/25"
            >
              <Send size={18} />
            </button>
          </div>
          <p className="text-center text-xs text-white/20 mt-2">
            Qythera AI - Powered by Vaelon Model Architecture
          </p>
        </div>
      </div>

      {/* Settings Modal */}
      <AnimatePresence>
        {settingsOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center"
            onClick={() => setSettingsOpen(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="glass w-[400px] p-6"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-lg font-bold">Settings</h2>
                <button onClick={() => setSettingsOpen(false)} className="p-1 hover:bg-white/10 rounded">
                  <X size={18} />
                </button>
              </div>
              <div className="space-y-4">
                <div>
                  <label className="text-sm text-white/60 block mb-1">Model</label>
                  <select className="w-full bg-white/5 border border-white/10 rounded-lg p-2 text-sm text-white outline-none">
                    <option>Vaelon 7B</option>
                    <option>Vaelon 13B</option>
                    <option>Vaelon 70B MoE</option>
                  </select>
                </div>
                <div>
                  <label className="text-sm text-white/60 block mb-1">Temperature: 0.7</label>
                  <input type="range" min="0" max="2" step="0.1" defaultValue="0.7" className="w-full" />
                </div>
                <div>
                  <label className="text-sm text-white/60 block mb-1">API Endpoint</label>
                  <input
                    type="text"
                    defaultValue="http://localhost:8000"
                    className="w-full bg-white/5 border border-white/10 rounded-lg p-2 text-sm text-white outline-none"
                  />
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
