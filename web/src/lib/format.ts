import { type Dictionary, type Lang, formatNumber } from "./i18n";
import type { Label } from "./types";

// Tailwind classes per label: badge background/text and the bar colour in charts
export const LABEL_STYLE: Record<Label, { badge: string; bar: string }> = {
  good: { badge: "bg-emerald-100 text-emerald-800 ring-emerald-600/20", bar: "fill-emerald-500" },
  marginal: { badge: "bg-amber-100 text-amber-800 ring-amber-600/20", bar: "fill-amber-400" },
  bad: { badge: "bg-rose-100 text-rose-800 ring-rose-600/20", bar: "fill-rose-300" },
  unknown: { badge: "bg-slate-100 text-slate-600 ring-slate-500/20", bar: "fill-slate-300" },
};

/** 0° → "N" / "K". The direction the wind comes FROM. */
export function compass(degrees: number, t: Dictionary): string {
  return t.compass[Math.round(degrees / 45) % 8];
}

export function knots(value: number, lang: Lang, t: Dictionary): string {
  return `${formatNumber(value, lang)} ${t.knots}`;
}

export function metres(value: number | null, lang: Lang): string {
  return value === null ? "—" : `${formatNumber(value, lang, 2)} m`;
}
