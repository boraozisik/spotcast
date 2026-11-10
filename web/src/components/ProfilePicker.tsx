"use client";

import { useT } from "@/lib/i18n";
import { LEVELS, SPORTS, setProfile, splitId, useProfileId } from "@/lib/profile";

/** Sport and level. Remembered across pages and visits; every verdict depends on it. */
export function ProfilePicker() {
  const { t } = useT();
  const { sport, level } = splitId(useProfileId());

  const button = (active: boolean) =>
    `rounded-full px-3 py-1.5 text-sm transition ${
      active ? "bg-sky-900 text-white shadow-sm" : "bg-slate-100 text-slate-700 hover:bg-slate-200"
    }`;

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:gap-6">
        <div role="group" aria-label={t.picker.sport} className="flex flex-wrap items-center gap-1.5">
          <span className="mr-1 text-sm font-medium text-slate-700">{t.picker.sport}</span>
          {SPORTS.map((s) => (
            <button key={s} aria-pressed={s === sport} onClick={() => setProfile(s, level)} className={button(s === sport)}>
              {t.sports[s]}
            </button>
          ))}
        </div>
        <div role="group" aria-label={t.picker.level} className="flex flex-wrap items-center gap-1.5">
          <span className="mr-1 text-sm font-medium text-slate-700">{t.picker.level}</span>
          {LEVELS.map((l) => (
            <button key={l} aria-pressed={l === level} onClick={() => setProfile(sport, l)} className={button(l === level)}>
              {t.levels[l]}
            </button>
          ))}
        </div>
      </div>
      <p className="mt-2 text-xs text-slate-500">{t.picker.hint}</p>
    </div>
  );
}
