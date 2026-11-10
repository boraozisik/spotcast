"use client";

import { LABEL_STYLE, knots, metres } from "@/lib/format";
import { useT } from "@/lib/i18n";
import type { Label, ShoreWind } from "@/lib/types";

export interface Bar {
  hour: string; // "14:00"
  wind_knots: number;
  wave_metres: number | null;
  shore_wind?: ShoreWind;
  label: Label;
}

// The scale follows the profile: SUP's comfortable range is 0-6 knots, windsurf's 15-25
const MIN_SCALE_KNOTS = 10;
const MAX_SCALE_KNOTS = 40;
const TOP = 12; // room for the offshore marker above the tallest bar
const HEIGHT = 120 + TOP;
const BAR = 22;
const GAP = 6;

/**
 * Wind per hour as bars, coloured by SpotCast's verdict. Dashed lines mark the comfortable
 * wind range of the chosen profile (e.g. 15-25 knots), so the reason for a colour is
 * visible. ▼ above a bar: the wind blows from the land (offshore).
 */
export function HourlyBars({
  bars,
  idealMin,
  idealMax,
}: {
  bars: Bar[];
  idealMin: number;
  idealMax: number;
}) {
  const { lang, t } = useT();
  const width = bars.length * (BAR + GAP);
  const scale = Math.min(
    MAX_SCALE_KNOTS,
    Math.ceil(Math.max(MIN_SCALE_KNOTS, idealMax * 1.2, ...bars.map((b) => b.wind_knots)) / 5) * 5,
  );
  const y = (kn: number) => HEIGHT - (Math.min(kn, scale) / scale) * (HEIGHT - TOP);

  return (
    <div>
      {/* viewBox + w-full: the chart scales to the card instead of being cut off */}
      <svg
        viewBox={`0 0 ${width + 34} ${HEIGHT + 22}`}
        className="h-auto max-h-48 w-full"
        role="img"
        aria-label={t.home.spots.title}
      >
        {[idealMin, idealMax].map((kn) => (
          <g key={kn}>
            <line
              x1={30}
              x2={width + 34}
              y1={y(kn)}
              y2={y(kn)}
              className="stroke-sky-400"
              strokeDasharray="4 4"
            />
            <text x={0} y={y(kn) + 4} className="fill-slate-500 text-[10px]">
              {kn} kn
            </text>
          </g>
        ))}
        {bars.map((bar, i) => {
          const x = 34 + i * (BAR + GAP);
          const top = y(bar.wind_knots);
          return (
            <g key={bar.hour}>
              <title>
                {`${bar.hour}: ${knots(bar.wind_knots, lang, t)}, ${metres(bar.wave_metres, lang)}, ${t.labels[bar.label]}` +
                  (bar.shore_wind && bar.shore_wind !== "unknown" ? `, ${t.shoreWind[bar.shore_wind]}` : "")}
              </title>
              {bar.shore_wind === "offshore" && (
                <text x={x + BAR / 2} y={top - 3} textAnchor="middle" className="fill-rose-600 text-[10px]">
                  ▼
                </text>
              )}
              <rect
                x={x}
                y={top}
                width={BAR}
                height={Math.max(HEIGHT - top, 2)}
                rx={4}
                className={LABEL_STYLE[bar.label].bar}
              />
              <text
                x={x + BAR / 2}
                y={HEIGHT + 15}
                textAnchor="middle"
                className="fill-slate-500 text-[10px]"
              >
                {bar.hour.slice(0, 2)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
