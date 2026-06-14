'use client';
import { motion } from 'framer-motion';
import { Zap } from 'lucide-react';

export default function MessageList({ messages }) {
  return (
    <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
      {messages.map(msg => (
        <motion.div key={msg.id} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
          className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
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
      ))}
    </div>
  );
}
