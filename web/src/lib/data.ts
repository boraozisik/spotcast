"use client";

import { useEffect, useState } from "react";

// Where the batch job's JSON lives. In production: the `site-data` branch on GitHub,
// refreshed every 6 hours. Locally: web/public/data (run `uv run spotcast-publish --out web/public/data`).
export const DATA_URL = (process.env.NEXT_PUBLIC_DATA_URL ?? "/data").replace(/\/$/, "");

export type DataState<T> =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; data: T };

/** Load one JSON file in the browser. A static site has no server to do it at request time. */
export function useSiteData<T>(file: string): DataState<T> {
  const [state, setState] = useState<DataState<T>>({ status: "loading" });

  useEffect(() => {
    let cancelled = false;
    fetch(`${DATA_URL}/${file}`, { cache: "no-store" })
      .then((response) => {
        if (!response.ok) throw new Error(`${file}: HTTP ${response.status}`);
        return response.json() as Promise<T>;
      })
      .then((data) => !cancelled && setState({ status: "ready", data }))
      .catch((error: Error) => !cancelled && setState({ status: "error", message: error.message }));
    return () => {
      cancelled = true;
    };
  }, [file]);

  return state;
}
