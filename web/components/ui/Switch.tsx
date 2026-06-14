'use client';
interface SwitchProps { checked: boolean; onChange: (checked: boolean) => void; label?: string; }

export default function Switch({ checked, onChange, label }: SwitchProps) {
  return (
    <label className="flex items-center gap-2 cursor-pointer">
      <div className={`w-10 h-5 rounded-full transition-colors ${checked ? 'bg-qythera-500' : 'bg-white/10'}`}
        onClick={() => onChange(!checked)}>
        <div className={`w-4 h-4 rounded-full bg-white transition-transform mt-0.5 ${checked ? 'translate-x-5 ml-0.5' : 'translate-x-0.5'}`} />
      </div>
      {label && <span className="text-sm text-white/60">{label}</span>}
    </label>
  );
}
