import { NextRequest, NextResponse } from 'next/server';
const API = process.env.VAELEN_API_URL || 'http://localhost:8000';
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const r = await fetch(`${API}/v1/chat/completions`, { method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(body) });
    return NextResponse.json(await r.json());
  } catch {
    return NextResponse.json({choices:[{message:{role:'assistant',content:'Server offline. Start: python -m inference.server'}}]});
  }
}
