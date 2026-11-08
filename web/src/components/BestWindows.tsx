"use client";

import { useState } from "react";

import { knots, metres } from "@/lib/format";
import { formatDay, useT } from "@/lib/i18n";
import type { Recommendations } from "@/lib/types";

import { Card, LabelBadge, Notes } from "./ui";

export function BestWindows({ recommendations }: { recommendations: Recommendations }) {
  const { lang, t } = useT();
  const [preset, setPreset] = useState<keyof Recommendations>("any");
  const { windows } = recommendations[preset];

  return (
    <section>
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-xl font-semibold text-slate-900">{t.home.best.title}</h2>
          <p className="text-sm text-slate-500">{t.home.best.subtitle}</p>
        </div>
        <div className="inline-flex rounded-full bg-slate-100 p-1 text-sm">
          {(["any", "weekend"] as const).map((key) => (
            <button
              key={key}
              onClick={() => setPreset(key)}
              aria-pressed={preset === key}
              className={`rounded-full px-3 py-1 ${preset === key ? "bg-white shadow-sm" : "text-slate-600"}`}
            >
              {t.home.best[key]}
            </button>
          ))}
        </div>
      </div>

      {windows.length === 0 ? (
        <Card>
          <p className="text-slate-700">{t.home.best.none}</p>
        </Card>
      ) : (
        <ol className="grid gap-4 md:grid-cols-2">
          {windows.map((w, i) => (
            <li key={`${w.spot}-${w.date}-${w.start}`}>
              <Card className={i === 0 ? "ring-2 ring-sky-500" : ""}>
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
                      {i === 0 ? t.home.best.first : t.home.best.nth(i + 1)}
                    </p>
                    <h3 className="mt-1 text-lg font-semibold text-slate-900">{w.spot}</h3>
                    <p className="text-slate-700">
                      {formatDay(w.date, lang)}, {w.start}-{w.end}
                    </p>
                  </div>
                  <div className="flex flex-col items-end gap-1">
                    {w.good_hours > 0 && <LabelBadge label="good" count={w.good_hours} />}
                    {w.marginal_hours > 0 && <LabelBadge label="marginal" count={w.marginal_hours} />}
                  </div>
                </div>
                <dl className="mt-4 grid grid-cols-2 gap-2 text-sm">
                  <div className="rounded-lg bg-slate-50 px-3 py-2">
                    <dt className="text-slate-500">{t.home.best.avgWind}</dt>
                    <dd className="font-semibold">{knots(w.avg_wind_knots, lang, t)}</dd>
                  </div>
                  <div className="rounded-lg bg-slate-50 px-3 py-2">
                    <dt className="text-slate-500">{t.home.best.maxWave}</dt>
                    <dd className="font-semibold">{metres(w.max_wave_metres, lang)}</dd>
                  </div>
                </dl>
                <Notes notes={w.notes} />
              </Card>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
