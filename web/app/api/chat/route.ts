import { NextRequest, NextResponse } from 'next/server';

const API_URL = process.env.VAELEN_API_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { messages } = body;

    const response = await fetch(`${API_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages,
        max_tokens: 2048,
        temperature: 0.7,
        top_p: 0.9,
      }),
    });

    if (!response.ok) {
      return NextResponse.json(
        { choices: [{ message: { role: 'assistant', content: 'Inference server error. Please check if Vaelon server is running.' } }] },
        { status: 200 }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { choices: [{ message: { role: 'assistant', content: 'Cannot connect to inference server. Start it with: qythera serve' } }] },
      { status: 200 }
    );
  }
}
