'use client';
import { clsx } from 'clsx';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
}

export default function Input({ label, error, className, ...props }: InputProps) {
  return (
    <div className="space-y-1">
      {label && <label className="text-sm text-white/60">{label}</label>}
      <input className={clsx('w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white outline-none focus:border-qythera-500/50 transition-colors', error && 'border-red-500/50', className)} {...props} />
      {error && <p className="text-xs text-red-400">{error}</p>}
    </div>
  );
}
