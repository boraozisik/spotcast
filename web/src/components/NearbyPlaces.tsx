"use client";

import { useState } from "react";

import { useT } from "@/lib/i18n";
import type { Place } from "@/lib/types";

import { Card } from "./ui";

/** Straight-line distance on the Earth's surface (haversine), the same formula as the
 * Python side (locations.distance_km). */
function distanceKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const rad = (d: number) => (d * Math.PI) / 180;
  const a =
    Math.sin(rad(lat2 - lat1) / 2) ** 2 +
    Math.cos(rad(lat1)) * Math.cos(rad(lat2)) * Math.sin(rad(lon2 - lon1) / 2) ** 2;
  return 2 * 6371 * Math.asin(Math.sqrt(a));
}

type State =
  | { status: "idle" }
  | { status: "locating" }
  | { status: "denied" }
  | { status: "ready"; latitude: number; longitude: number };

const SHOWN = 5;

/**
 * "Places near me": the browser asks for the visitor's location. It never leaves the
 * browser: the distances are computed here, against coordinates already in places.json.
 */
export function NearbyPlaces({ places, onPick }: { places: Place[]; onPick: (slug: string) => void }) {
  const { t } = useT();
  const [state, setState] = useState<State>({ status: "idle" });

  function locate() {
    if (!("geolocation" in navigator)) {
      setState({ status: "denied" });
      return;
    }
    setState({ status: "locating" });
    navigator.geolocation.getCurrentPosition(
      (position) =>
        setState({ status: "ready", latitude: position.coords.latitude, longitude: position.coords.longitude }),
      () => setState({ status: "denied" }),
      { maximumAge: 10 * 60_000, timeout: 15_000 },
    );
  }

  const nearest =
    state.status === "ready"
      ? places
          // Like the assistant: skip spots whose guide doesn't list this sport
          .filter((place) => !place.days.some((day) => day.notes.some((n) => n.code === "not_for_sport")))
          .map((place) => ({
            place,
            km: Math.round(distanceKm(state.latitude, state.longitude, place.latitude, place.longitude)),
            good: place.days.reduce((sum, day) => sum + (day.label_counts.good ?? 0), 0),
          }))
          .sort((a, b) => a.km - b.km)
          .slice(0, SHOWN)
      : [];

  return (
    <Card>
      <div className="flex flex-wrap items-center gap-3">
        <button
          onClick={locate}
          disabled={state.status === "locating"}
          className="rounded-full bg-sky-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-60"
        >
          📍 {state.status === "locating" ? t.place.nearby.locating : t.place.nearby.button}
        </button>
        <p className="text-xs text-slate-500">{t.place.nearby.privacy}</p>
      </div>
      {state.status === "denied" && <p className="mt-3 text-sm text-rose-800">{t.place.nearby.denied}</p>}
      {nearest.length > 0 && (
        <div className="mt-4">
          <p className="text-sm font-medium text-slate-700">{t.place.nearby.title}</p>
          <ul className="mt-2 divide-y divide-slate-100">
            {nearest.map(({ place, km, good }) => (
              <li key={place.slug} className="flex flex-wrap items-center justify-between gap-2 py-2 text-sm">
                <div>
                  <p className="font-medium text-slate-900">
                    {place.name}
                    {place.region && <span className="font-normal text-slate-500">, {place.region}</span>}
                  </p>
                  <p className="text-slate-500">
                    {t.place.nearby.distance(km)} ·{" "}
                    <span className={good ? "text-emerald-700" : "text-slate-500"}>
                      {good ? t.place.nearby.goodHours(good) : t.place.nearby.noGood}
                    </span>
                  </p>
                </div>
                <button
                  onClick={() => onPick(place.slug)}
                  className="rounded-lg bg-slate-100 px-3 py-1.5 text-slate-700 hover:bg-slate-200"
                >
                  {t.place.nearby.show}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </Card>
  );
}
