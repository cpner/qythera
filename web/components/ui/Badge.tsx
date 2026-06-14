interface BadgeProps { children: React.ReactNode; variant?: 'default' | 'success' | 'warning' | 'error'; }

export default function Badge({ children, variant = 'default' }: BadgeProps) {
  const colors = { default: 'bg-white/10 text-white/60', success: 'bg-green-500/20 text-green-400',
    warning: 'bg-yellow-500/20 text-yellow-400', error: 'bg-red-500/20 text-red-400' };
  return <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs ${colors[variant]}`}>{children}</span>;
}
