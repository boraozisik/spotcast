"use client";

import Link from "next/link";

import { BestWindows } from "@/components/BestWindows";
import { SpotForecast } from "@/components/SpotForecast";
import { Caution, LoadError, Loading } from "@/components/ui";
import { useSiteData } from "@/lib/data";
import { formatDateTime, useT } from "@/lib/i18n";
import type { Meta, Recommendations, Spot } from "@/lib/types";

export default function Home() {
  const { lang, t } = useT();
  const meta = useSiteData<Meta>("meta.json");
  const recommendations = useSiteData<Recommendations>("recommendations.json");
  const spots = useSiteData<Spot[]>("spots.json");

  const failed = [meta, recommendations, spots].find((s) => s.status === "error");

  return (
    <div className="space-y-12">
      <section className="rounded-3xl bg-gradient-to-br from-sky-900 to-cyan-700 px-6 py-10 text-white sm:px-10">
        <p className="text-sm font-medium uppercase tracking-widest text-sky-200">{t.home.kicker}</p>
        <h1 className="mt-2 max-w-3xl text-3xl font-bold tracking-tight sm:text-5xl">{t.home.title}</h1>
        <p className="mt-4 max-w-2xl text-lg text-sky-100">{t.home.lead}</p>
        <div className="mt-6 flex flex-wrap gap-3 text-sm">
          <Link href="/konum/" className="rounded-full bg-white px-4 py-2 font-medium text-sky-900">
            {t.home.ctaPlace}
          </Link>
          <Link
            href="/nasil-calisir/"
            className="rounded-full bg-white/10 px-4 py-2 font-medium ring-1 ring-white/30"
          >
            {t.home.ctaHow}
          </Link>
        </div>
        {meta.status === "ready" && (
          <p className="mt-6 text-sm text-sky-200">
            {t.home.updated(formatDateTime(meta.data.generated_at, lang))}
          </p>
        )}
      </section>

      {failed?.status === "error" ? (
        <LoadError message={failed.message} />
      ) : meta.status !== "ready" || recommendations.status !== "ready" || spots.status !== "ready" ? (
        <Loading />
      ) : (
        <>
          <BestWindows recommendations={recommendations.data} />

          <section>
            <h2 className="text-xl font-semibold text-slate-900">{t.home.spots.title}</h2>
            <p className="mb-4 text-sm text-slate-500">
              {t.home.spots.text(meta.data.profile.good_wind_min_kn, meta.data.profile.good_wind_max_kn)}
            </p>
            <div className="grid gap-4 lg:grid-cols-3">
              {spots.data.map((spot) => (
                <SpotForecast key={spot.slug} spot={spot} meta={meta.data} />
              ))}
            </div>
          </section>

          <Caution />
        </>
      )}
    </div>
  );
}
