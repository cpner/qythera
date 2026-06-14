'use client';
import { clsx } from 'clsx';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
}

export default function Button({ variant = 'primary', size = 'md', className, children, ...props }: ButtonProps) {
  const base = 'inline-flex items-center justify-center rounded-xl font-medium transition-all duration-200 disabled:opacity-50';
  const variants = {
    primary: 'bg-gradient-to-br from-qythera-500 to-blue-500 text-white hover:shadow-lg hover:shadow-qythera-500/25',
    secondary: 'bg-white/10 text-white border border-white/10 hover:bg-white/15',
    ghost: 'text-white/60 hover:text-white hover:bg-white/5',
    danger: 'bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30',
  };
  const sizes = { sm: 'px-3 py-1.5 text-xs', md: 'px-4 py-2 text-sm', lg: 'px-6 py-3 text-base' };
  return <button className={clsx(base, variants[variant], sizes[size], className)} {...props}>{children}</button>;
}
