"use client";

import { Suspense } from "react";

import { PlaceExplorer } from "@/components/PlaceExplorer";
import { Loading, PageTitle } from "@/components/ui";
import { useT } from "@/lib/i18n";

export default function PlacePage() {
  const { t } = useT();
  return (
    <div className="space-y-6">
      <PageTitle title={t.place.title} lead={t.place.lead} />
      {/* Reads ?yer= from the URL, so it renders in the browser, inside Suspense */}
      <Suspense fallback={<Loading />}>
        <PlaceExplorer />
      </Suspense>
    </div>
  );
}
