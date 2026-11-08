"use client";

import { LABEL_STYLE, knots, metres } from "@/lib/format";
import { useT } from "@/lib/i18n";
import type { Label } from "@/lib/types";

export interface Bar {
  hour: string; // "14:00"
  wind_knots: number;
  wave_metres: number | null;
  label: Label;
}

const MAX_KNOTS = 30;
const HEIGHT = 120;
const BAR = 22;
const GAP = 6;

/**
 * Wind per hour as bars, coloured by SpotCast's verdict. Dashed lines mark the ideal
 * wind range of the profile (e.g. 15-25 knots), so the reason for a colour is visible.
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
  const y = (kn: number) => HEIGHT - (Math.min(kn, MAX_KNOTS) / MAX_KNOTS) * HEIGHT;

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
                {`${bar.hour}: ${knots(bar.wind_knots, lang, t)}, ${metres(bar.wave_metres, lang)}, ${t.labels[bar.label]}`}
              </title>
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
