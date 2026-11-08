"use client";

import type { ReactNode } from "react";

import { LABEL_STYLE } from "@/lib/format";
import { useT } from "@/lib/i18n";
import type { Label, Note } from "@/lib/types";

export function LabelBadge({ label, count }: { label: Label; count?: number }) {
  const { t } = useT();
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset ${LABEL_STYLE[label].badge}`}
    >
      {count !== undefined && <span className="font-semibold">{count}</span>}
      {t.labels[label]}
    </span>
  );
}

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div className={`rounded-2xl border border-slate-200 bg-white p-5 shadow-sm ${className}`}>
      {children}
    </div>
  );
}

export function PageTitle({ title, lead }: { title: string; lead?: ReactNode }) {
  return (
    <header className="mb-8">
      <h1 className="text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">{title}</h1>
      {lead && <p className="mt-3 max-w-3xl text-lg text-slate-600">{lead}</p>}
    </header>
  );
}

/** Warnings built by code from facts (never by an LLM), worded in the visitor's language. */
export function Notes({ notes }: { notes: Note[] }) {
  const { t } = useT();
  if (notes.length === 0) return null;
  return (
    <ul className="mt-3 space-y-1.5">
      {notes.map((note) => (
        <li
          key={note.code}
          className="flex gap-2 rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-900"
        >
          <span aria-hidden>⏳</span>
          <span>{t.note(note)}</span>
        </li>
      ))}
    </ul>
  );
}

export function Caution() {
  const { t } = useT();
  return (
    <p className="flex gap-2 rounded-xl border border-sky-200 bg-sky-50 px-4 py-3 text-sm text-sky-900">
      <span aria-hidden>⚠️</span>
      <span>{t.caution}</span>
    </p>
  );
}

export function Loading() {
  return (
    <div className="animate-pulse space-y-3" aria-busy="true">
      <div className="h-6 w-1/3 rounded bg-slate-200" />
      <div className="h-24 rounded-2xl bg-slate-100" />
      <div className="h-24 rounded-2xl bg-slate-100" />
    </div>
  );
}

export function LoadError({ message }: { message: string }) {
  const { t } = useT();
  return (
    <Card className="border-rose-200 bg-rose-50">
      <p className="font-medium text-rose-900">{t.loadError.title}</p>
      <p className="mt-1 text-sm text-rose-800">{t.loadError.body}</p>
      <p className="mt-2 font-mono text-xs text-rose-700">{message}</p>
    </Card>
  );
}
