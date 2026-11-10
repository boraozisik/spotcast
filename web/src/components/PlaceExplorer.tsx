"use client";

import { useSearchParams } from "next/navigation";
import { useMemo, useState } from "react";

import { HourlyBars } from "@/components/HourlyBars";
import { NearbyPlaces } from "@/components/NearbyPlaces";
import { ProfilePicker } from "@/components/ProfilePicker";
import { StaleNotice } from "@/components/StaleNotice";
import { Caution, Card, LabelBadge, LoadError, Loading, Notes } from "@/components/ui";
import { useProfileData, useSiteData } from "@/lib/data";
import { knots, metres } from "@/lib/format";
import { type Dictionary, formatDay, useT } from "@/lib/i18n";
import { DEFAULT_PROFILE, profileLimits, useProfileId } from "@/lib/profile";
import type { Meta, Place, PlaceDay } from "@/lib/types";

function verdict(day: PlaceDay, t: Dictionary): { text: string; tone: string } {
  const c = day.label_counts;
  if (c.unknown) return { text: t.place.verdict.unknown, tone: "text-slate-700" };
  if (c.good) return { text: t.place.verdict.good(c.good), tone: "text-emerald-700" };
  if (c.marginal) return { text: t.place.verdict.marginal, tone: "text-amber-700" };
  return { text: t.place.verdict.bad, tone: "text-rose-700" };
}

export function PlaceExplorer() {
  const { lang, t } = useT();
  const profileId = useProfileId();
  const meta = useSiteData<Meta>("meta.json");
  const defaultId = meta.status === "ready" ? (meta.data.default_profile ?? DEFAULT_PROFILE) : DEFAULT_PROFILE;
  const places = useProfileData<Place[]>("places.json", profileId, defaultId);
  // Shareable links: /konum/?yer=bozcaada
  const searchParams = useSearchParams();
  const [slug, setSlug] = useState(searchParams.get("yer") ?? "mezitli-sahili");
  const [dayIndex, setDayIndex] = useState<number | null>(null);

  const regions = useMemo(() => {
    if (places.status !== "ready") return [];
    const groups = new Map<string, Place[]>();
    for (const p of places.data) {
      const group = p.region ?? t.place.guidedGroup;
      groups.set(group, [...(groups.get(group) ?? []), p]);
    }
    return [...groups.entries()];
  }, [places, t]);

  if (meta.status === "error") return <LoadError message={meta.message} />;
  if (places.status === "error") return <LoadError message={places.message} />;
  if (meta.status !== "ready" || places.status !== "ready")
    return (
      <div className="space-y-6">
        <ProfilePicker />
        <Loading />
      </div>
    );

  const limits = profileLimits(meta.data, profileId);
  const pick = (next: string) => {
    setSlug(next);
    setDayIndex(null);
    window.history.replaceState(null, "", `?yer=${next}`);
  };

  const place = places.data.find((p) => p.slug === slug) ?? places.data[0];
  // Unless a day was picked, open the first day with a few daylight hours left
  const firstUseful = Math.max(0, place.days.findIndex((d) => d.daylight_hours.length >= 3));
  const day = place.days[Math.min(dayIndex ?? firstUseful, place.days.length - 1)];
  const shownIndex = place.days.indexOf(day);
  const v = verdict(day, t);

  return (
    <div className="space-y-6">
      <StaleNotice generatedAt={meta.data.generated_at} />
      <ProfilePicker />
      <NearbyPlaces places={places.data} onPick={pick} />
      <Card className="grid gap-4 sm:grid-cols-[1fr_2fr]">
        <label className="text-sm">
          <span className="font-medium text-slate-700">{t.place.placeLabel}</span>
          <select
            value={place.slug}
            onChange={(e) => pick(e.target.value)}
            className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2"
          >
            {regions.map(([region, list]) => (
              <optgroup key={region} label={region}>
                {list.map((p) => (
                  <option key={p.slug} value={p.slug}>
                    {p.name}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        </label>
        <div className="text-sm">
          <span className="font-medium text-slate-700">{t.place.dayLabel}</span>
          <div className="mt-1 flex flex-wrap gap-1">
            {place.days.map((d, i) => (
              <button
                key={d.date}
                onClick={() => setDayIndex(i)}
                className={`rounded-lg px-2.5 py-2 ${
                  i === shownIndex ? "bg-sky-900 text-white" : "bg-slate-100 text-slate-700 hover:bg-slate-200"
                }`}
              >
                {formatDay(d.date, lang)}
              </button>
            ))}
          </div>
        </div>
      </Card>

      <Card>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-sm text-slate-500">
              {place.name}, {place.region ?? t.place.guidedSpot} · {formatDay(day.date, lang)}
            </p>
            <h2 className={`mt-1 text-2xl font-bold ${v.tone}`}>{v.text}</h2>
            {day.best_window && (
              <p className="mt-2 text-slate-700">
                {t.place.best(
                  day.best_window.hours,
                  knots(day.best_window.avg_wind_knots, lang, t),
                  day.best_window.good_hours,
                )}
              </p>
            )}
          </div>
          <div className="flex flex-wrap gap-1">
            {(["good", "marginal", "bad", "unknown"] as const).map(
              (label) =>
                day.label_counts[label] && (
                  <LabelBadge key={label} label={label} count={day.label_counts[label]} />
                ),
            )}
          </div>
        </div>

        <div className="mt-4">
          <HourlyBars
            bars={day.daylight_hours}
            idealMin={limits.good_wind_min_kn}
            idealMax={limits.good_wind_max_kn}
          />
        </div>

        <details className="mt-2 text-sm">
          <summary className="cursor-pointer text-slate-600">{t.place.table}</summary>
          <table className="mt-2 w-full text-left">
            <thead className="text-slate-500">
              <tr>
                {t.place.columns.map((column, i) => (
                  <th key={column} className={i === 0 ? "py-1" : ""}>
                    {column}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {day.daylight_hours.map((h) => (
                <tr key={h.hour} className="border-t border-slate-100">
                  <td className="py-1.5">{h.hour}</td>
                  <td>{knots(h.wind_knots, lang, t)}</td>
                  <td>{metres(h.wave_metres, lang)}</td>
                  <td className={h.shore_wind === "offshore" ? "text-rose-700" : ""}>
                    {t.shoreWind[h.shore_wind ?? "unknown"]}
                  </td>
                  <td>
                    <LabelBadge label={h.label} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>

        <Notes notes={day.notes} />
      </Card>

      <Caution />
    </div>
  );
}
