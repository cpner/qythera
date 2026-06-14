import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    
    // Try multiple possible server URLs
    const urls = [
      process.env.VAELEN_API_URL || 'http://localhost:8000',
      'http://127.0.0.1:8000',
    ];
    
    for (const baseUrl of urls) {
      try {
        const response = await fetch(`${baseUrl}/v1/chat/completions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
          signal: AbortSignal.timeout(30000),
        });
        
        if (response.ok) {
          const data = await response.json();
          return NextResponse.json(data);
        }
      } catch {
        continue;
      }
    }
    
    // If server not found, use knowledge base directly
    const messages = body.messages || [];
    const lastMsg = messages[messages.length - 1]?.content || '';
    
    return NextResponse.json({
      id: `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      model: 'vaelon',
      choices: [{
        index: 0,
        message: {
          role: 'assistant',
          content: `Server not running. Start it with:\n\npython -m core.inference.server\n\nYour question: "${lastMsg}"`
        },
        finish_reason: 'stop'
      }],
      usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
    });
  } catch (error) {
    return NextResponse.json({
      choices: [{
        message: { role: 'assistant', content: `Error: ${error.message}. Make sure the server is running.` }
      }]
    });
  }
}
