import axios, { AxiosInstance } from 'axios';

interface ChatMessage { role: 'user' | 'assistant' | 'system'; content: string; }
interface ChatOptions { model?: string; temperature?: number; max_tokens?: number; stream?: boolean; }

export class QytheraClient {
  private http: AxiosInstance;

  constructor(apiUrl: string = 'http://localhost:8000', apiKey?: string) {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (apiKey) headers['Authorization'] = `Bearer ${apiKey}`;
    this.http = axios.create({ baseURL: apiUrl, headers, timeout: 120000 });
  }

  async chat(messages: ChatMessage[], options: ChatOptions = {}) {
    const resp = await this.http.post('/v1/chat/completions', { messages, ...options });
    return resp.data;
  }

  async generate(prompt: string, options: ChatOptions = {}): Promise<string> {
    const result = await this.chat([{ role: 'user', content: prompt }], options);
    return result.choices[0].message.content;
  }

  async *chatStream(messages: ChatMessage[], options: ChatOptions = {}) {
    const resp = await this.http.post('/v1/chat/completions',
      { messages, stream: true, ...options }, { responseType: 'stream' });
    for await (const chunk of resp.data) {
      const lines = chunk.toString().split('\n');
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') return;
          const parsed = JSON.parse(data);
          yield parsed.choices?.[0]?.delta?.content || '';
        }
      }
    }
  }

  async health(): Promise<boolean> {
    try { await this.http.get('/health'); return true; } catch { return false; }
  }

  async models() {
    const resp = await this.http.get('/v1/models');
    return resp.data.data;
  }
}
