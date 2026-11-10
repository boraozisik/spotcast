"use client";

import { useMemo, useState } from "react";

import { compass, knots } from "@/lib/format";
import { formatDay, useT } from "@/lib/i18n";
import type { Label, Meta, Profile, Spot } from "@/lib/types";

import { HourlyBars } from "./HourlyBars";
import { Card, LabelBadge } from "./ui";

/** One guided spot: pick a day, see its daylight hours as bars. */
export function SpotForecast({
  spot,
  limits,
  info,
}: {
  spot: Spot;
  limits: Profile;
  info?: NonNullable<Meta["spots"]>[number];
}) {
  const { lang, t } = useT();
  const days = useMemo(() => {
    const byDay = new Map<string, Spot["hours"]>();
    for (const hour of spot.hours) {
      if (!hour.is_day) continue;
      const date = hour.time.slice(0, 10);
      byDay.set(date, [...(byDay.get(date) ?? []), hour]);
    }
    return [...byDay.entries()];
  }, [spot]);
  // Open the first day with a few daylight hours left (in the evening, today has almost none)
  const [selected, setSelected] = useState(() => Math.max(0, days.findIndex(([, h]) => h.length >= 3)));
  const [date, hours] = days[Math.min(selected, days.length - 1)] ?? ["", []];

  const counts = hours.reduce<Partial<Record<Label, number>>>((acc, h) => {
    acc[h.label] = (acc[h.label] ?? 0) + 1;
    return acc;
  }, {});
  const strongest = hours.reduce((a, b) => (b.wind_knots > a.wind_knots ? b : a), hours[0]);

  return (
    <Card className="min-w-0">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-lg font-semibold">{spot.name}</h3>
          {info?.sheltered && (
            <span className="text-xs font-medium text-sky-700">⚓ {t.home.spots.sheltered}</span>
          )}
        </div>
        <div className="flex flex-wrap gap-1">
          {(["good", "marginal", "bad"] as const).map(
            (label) => counts[label] && <LabelBadge key={label} label={label} count={counts[label]} />,
          )}
        </div>
      </div>
      <div className="mt-3 flex gap-1 overflow-x-auto pb-1">
        {days.map(([d], i) => (
          <button
            key={d}
            onClick={() => setSelected(i)}
            className={`whitespace-nowrap rounded-lg px-2.5 py-1 text-sm ${
              i === selected ? "bg-sky-900 text-white" : "bg-slate-100 text-slate-700 hover:bg-slate-200"
            }`}
          >
            {formatDay(d, lang, true)}
          </button>
        ))}
      </div>
      {date && (
        <>
          {info?.sports && !info.sports.includes(limits.sport) && (
            <p className="mb-2 rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-900">
              🚫 {t.home.spots.notForSport(t.sports[limits.sport])}
            </p>
          )}
          <HourlyBars
            bars={hours.map((h) => ({ ...h, hour: h.time.slice(11, 16) }))}
            idealMin={limits.good_wind_min_kn}
            idealMax={limits.good_wind_max_kn}
          />
          {strongest && (
            <p className="text-sm text-slate-600">
              {t.home.spots.strongest(
                strongest.time.slice(11, 16),
                knots(strongest.wind_knots, lang, t),
                compass(strongest.wind_direction, t),
              )}
            </p>
          )}
        </>
      )}
    </Card>
  );
}
