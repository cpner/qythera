import React, { useState, useRef } from 'react';
import { View, Text, TextInput, FlatList, TouchableOpacity, StyleSheet, SafeAreaView, KeyboardAvoidingView, Platform } from 'react-native';

const API_URL = 'http://localhost:8000';

interface Message { id: string; role: 'user' | 'assistant'; content: string; }

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const flatListRef = useRef<FlatList>(null);

  const send = async () => {
    if (!input.trim() || loading) return;
    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: input.trim() };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const resp = await fetch(`${API_URL}/v1/chat/completions`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: [...messages, userMsg].map(m => ({ role: m.role, content: m.content })) }),
      });
      const data = await resp.json();
      const assistantMsg: Message = { id: (Date.now()+1).toString(), role: 'assistant',
        content: data.choices?.[0]?.message?.content || 'No response' };
      setMessages(prev => [...prev, assistantMsg]);
    } catch { setMessages(prev => [...prev, { id: (Date.now()+1).toString(), role: 'assistant', content: 'Connection error' }]); }
    setLoading(false);
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}><Text style={styles.title}>Qythera</Text><Text style={styles.subtitle}>Vaelon AI</Text></View>
      <KeyboardAvoidingView style={styles.chat} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <FlatList ref={flatListRef} data={messages} keyExtractor={m => m.id} style={styles.messages}
          renderItem={({item}) => (
            <View style={[styles.bubble, item.role === 'user' ? styles.userBubble : styles.assistantBubble]}>
              <Text style={styles.bubbleText}>{item.content}</Text>
            </View>
          )}
          onContentSizeChange={() => flatListRef.current?.scrollToEnd()} />
        <View style={styles.inputRow}>
          <TextInput style={styles.input} value={input} onChangeText={setInput}
            placeholder="Message Qythera..." placeholderTextColor="#666" multiline />
          <TouchableOpacity style={styles.sendBtn} onPress={send} disabled={loading}>
            <Text style={styles.sendText}>{loading ? '...' : 'Send'}</Text>
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0a0a1a' },
  header: { padding: 16, alignItems: 'center', borderBottomWidth: 1, borderBottomColor: '#222' },
  title: { fontSize: 20, fontWeight: 'bold', color: '#a78bfa' },
  subtitle: { fontSize: 12, color: '#666' },
  chat: { flex: 1 },
  messages: { flex: 1, padding: 12 },
  bubble: { padding: 12, borderRadius: 16, marginBottom: 8, maxWidth: '85%' },
  userBubble: { alignSelf: 'flex-end', backgroundColor: '#7c3aed' },
  assistantBubble: { alignSelf: 'flex-start', backgroundColor: '#1a1a2e' },
  bubbleText: { color: '#fff', fontSize: 15, lineHeight: 20 },
  inputRow: { flexDirection: 'row', padding: 8, gap: 8, alignItems: 'flex-end' },
  input: { flex: 1, backgroundColor: '#1a1a2e', borderRadius: 20, padding: 12, color: '#fff', fontSize: 15, maxHeight: 100 },
  sendBtn: { backgroundColor: '#7c3aed', borderRadius: 20, padding: 12, paddingHorizontal: 20 },
  sendText: { color: '#fff', fontWeight: '600' },
});
