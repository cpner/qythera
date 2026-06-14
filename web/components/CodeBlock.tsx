'use client';
export default function CodeBlock({ language, children }) {
  return (
    <pre className="bg-black/40 rounded-lg p-3 overflow-x-auto text-sm">
      {language && <div className="text-xs text-white/40 mb-2">{language}</div>}
      <code className="text-green-300">{children}</code>
    </pre>
  );
}
