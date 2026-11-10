"use client";

import { useEffect, useState } from "react";

// Where the batch job's JSON lives. In production: the `site-data` branch on GitHub,
// refreshed every 6 hours. Locally: web/public/data (run `uv run spotcast-publish --out web/public/data`).
export const DATA_URL = (process.env.NEXT_PUBLIC_DATA_URL ?? "/data").replace(/\/$/, "");

export type DataState<T> =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; data: T };

/** Load one JSON file in the browser. A static site has no server to do it at request time.
 * The result is kept with the file it belongs to: when the file changes (another profile),
 * the old data is never shown as if it were the new one. An empty name loads nothing. */
export function useSiteData<T>(file: string): DataState<T> {
  const [result, setResult] = useState<{ file: string; state: DataState<T> } | null>(null);

  useEffect(() => {
    if (!file) return;
    let cancelled = false;
    fetch(`${DATA_URL}/${file}`, { cache: "no-store" })
      .then((response) => {
        if (!response.ok) throw new Error(`${file}: HTTP ${response.status}`);
        return response.json() as Promise<T>;
      })
      .then((data) => !cancelled && setResult({ file, state: { status: "ready", data } }))
      .catch(
        (error: Error) =>
          !cancelled && setResult({ file, state: { status: "error", message: error.message } }),
      );
    return () => {
      cancelled = true;
    };
  }, [file]);

  return result && result.file === file ? result.state : { status: "loading" };
}

/** One of a profile's files (profiles/<id>/<file>). If it is missing for the default
 * profile, e.g. right after a deploy, before the batch job has written the new layout,
 * the files at the top are used: every data version has them. */
export function useProfileData<T>(file: string, profileId: string, defaultId: string): DataState<T> {
  const state = useSiteData<T>(`profiles/${profileId}/${file}`);
  const useFallback = state.status === "error" && profileId === defaultId;
  const fallback = useSiteData<T>(useFallback ? file : "");
  return useFallback ? fallback : state;
}
