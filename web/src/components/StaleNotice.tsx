"use client";

import { useState } from "react";

import { useT } from "@/lib/i18n";

// The job is scheduled every 3 h, but GitHub may skip runs: 12 h means several were missed
// or failed, worth telling the visitor.
const STALE_AFTER_HOURS = 12;

/** Hours since the batch job wrote the data. generated_at is Turkish local time, written
 * without a zone; Turkey is UTC+3 all year, so the age is right in any visitor's timezone. */
export function hoursOld(generatedAt: string, now: number): number {
  return (now - Date.parse(`${generatedAt}:00+03:00`)) / 3_600_000;
}

/** Tells visitors when the forecast is old instead of showing it as if it were fresh. */
export function StaleNotice({ generatedAt }: { generatedAt: string }) {
  const { t } = useT();
  const [now] = useState(() => Date.now()); // read once, when the data is shown
  const age = Math.floor(hoursOld(generatedAt, now));
  if (age < STALE_AFTER_HOURS) return null;
  return (
    <p className="flex gap-2 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900">
      <span aria-hidden>🕒</span>
      <span>{t.stale(age)}</span>
    </p>
  );
}
