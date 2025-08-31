import React, { useEffect, useRef, useState } from "react";
import { ChevronDown, Check } from "lucide-react";

export default function CategorySelect({ value, onChange, options }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  // close on outside click / Escape
  useEffect(() => {
    const onDoc = (e) => {
      if (!ref.current?.contains(e.target)) setOpen(false);
    };
    const onEsc = (e) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onEsc);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onEsc);
    };
  }, []);

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen((s) => !s)}
        className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-100 hover:bg-white/10 focus:outline-none focus:ring-2 focus:ring-indigo-500/40"
      >
        <span className="truncate max-w-[12rem]">{value}</span>
        <ChevronDown size={16} className="opacity-70" />
      </button>

      {open && (
        <ul
          role="listbox"
          className="absolute right-0 z-30 mt-1 w-56 overflow-hidden rounded-xl border border-white/10 bg-[#0f1528] p-1 shadow-[var(--shadow-soft)]"
        >
          {options.map((opt) => {
            const selected = opt === value;
            return (
              <li key={opt}>
                <button
                  role="option"
                  aria-selected={selected}
                  onClick={() => {
                    onChange(opt);
                    setOpen(false);
                  }}
                  className={`flex w-full items-center justify-between rounded-lg px-3 py-2 text-sm hover:bg-white/10 ${
                    selected
                      ? "bg-indigo-500/15 text-indigo-200 border border-indigo-500/20"
                      : "text-slate-100"
                  }`}
                >
                  <span className="truncate">{opt}</span>
                  {selected && <Check size={16} className="text-indigo-300" />}
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
