"use client";

import { type Lang, setLang, useT } from "@/lib/i18n";

// Flags as small inline SVGs: emoji flags don't render on Windows (they show "GB", "TR").

function FlagGB() {
  return (
    <svg viewBox="0 0 60 30" className="h-3.5 w-6 rounded-[2px]" aria-hidden>
      <clipPath id="gb-clip">
        <path d="M0,0 v30 h60 v-30 z" />
      </clipPath>
      <clipPath id="gb-diag">
        <path d="M30,15 h30 v15 z v15 h-30 z h-30 v-15 z v-15 h30 z" />
      </clipPath>
      <g clipPath="url(#gb-clip)">
        <path d="M0,0 v30 h60 v-30 z" fill="#012169" />
        <path d="M0,0 L60,30 M60,0 L0,30" stroke="#fff" strokeWidth="6" />
        <path d="M0,0 L60,30 M60,0 L0,30" clipPath="url(#gb-diag)" stroke="#C8102E" strokeWidth="4" />
        <path d="M30,0 v30 M0,15 h60" stroke="#fff" strokeWidth="10" />
        <path d="M30,0 v30 M0,15 h60" stroke="#C8102E" strokeWidth="6" />
      </g>
    </svg>
  );
}

function FlagTR() {
  return (
    <svg viewBox="0 0 30 20" className="h-3.5 w-6 rounded-[2px]" aria-hidden>
      <rect width="30" height="20" fill="#E30A17" />
      <circle cx="10.6" cy="10" r="5" fill="#fff" />
      <circle cx="11.85" cy="10" r="4" fill="#E30A17" />
      <polygon fill="#fff" points="14.70,10.00 19.40,11.53 16.50,7.53 16.50,12.47 19.40,8.47" />
    </svg>
  );
}

const OPTIONS: { lang: Lang; label: string; Flag: () => React.JSX.Element }[] = [
  { lang: "en", label: "EN", Flag: FlagGB },
  { lang: "tr", label: "TR", Flag: FlagTR },
];

export function LanguageSwitch() {
  const { lang, t } = useT();
  return (
    <div role="group" aria-label={t.language} className="inline-flex rounded-full bg-slate-100 p-1 text-xs font-semibold">
      {OPTIONS.map(({ lang: option, label, Flag }) => (
        <button
          key={option}
          onClick={() => setLang(option)}
          aria-pressed={lang === option}
          className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 transition ${
            lang === option ? "bg-white text-slate-900 shadow-sm" : "text-slate-500 hover:text-slate-800"
          }`}
        >
          <Flag />
          {label}
        </button>
      ))}
    </div>
  );
}
