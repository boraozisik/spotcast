"use client";

import { useSyncExternalStore } from "react";

import type { Level, Meta, Profile, Sport } from "./types";

// The visitor's sport and level, remembered like the language (see lib/i18n.tsx).
// Default: windsurfing, intermediate. The prerendered HTML uses the default too.

export const SPORTS: Sport[] = ["windsurf", "kitesurf", "sup", "kayak"];
export const LEVELS: Level[] = ["beginner", "intermediate", "advanced"];
export const DEFAULT_PROFILE = "windsurf_intermediate";

const KEY = "spotcast-profile";
const listeners = new Set<() => void>();
let memory: string | null = null; // when localStorage is blocked (private mode)

function valid(id: string | null): id is string {
  if (!id) return false;
  const [sport, level] = id.split("_");
  return SPORTS.includes(sport as Sport) && LEVELS.includes(level as Level);
}

function read(): string {
  try {
    const stored = localStorage.getItem(KEY);
    if (valid(stored)) return stored;
  } catch {
    // storage not available: use memory
  }
  return memory ?? DEFAULT_PROFILE;
}

function subscribe(onChange: () => void): () => void {
  listeners.add(onChange);
  window.addEventListener("storage", onChange);
  return () => {
    listeners.delete(onChange);
    window.removeEventListener("storage", onChange);
  };
}

export function setProfile(sport: Sport, level: Level): void {
  const id = `${sport}_${level}`;
  memory = id;
  try {
    localStorage.setItem(KEY, id);
  } catch {
    // ignore: memory keeps it for this visit
  }
  listeners.forEach((listener) => listener());
}

/** The chosen profile id, e.g. "sup_beginner". */
export function useProfileId(): string {
  return useSyncExternalStore(subscribe, read, () => DEFAULT_PROFILE);
}

export function splitId(id: string): { sport: Sport; level: Level } {
  const [sport, level] = id.split("_") as [Sport, Level];
  return { sport, level };
}

/** The chosen profile's limits from meta.json. Old data has only the default profile. */
export function profileLimits(meta: Meta, id: string): Profile {
  const found = meta.profiles?.find((p) => p.id === id);
  if (found) return found;
  return {
    id: DEFAULT_PROFILE,
    sport: "windsurf",
    level: "intermediate",
    ideal_wind_kn: null,
    max_offshore_wind_kn: null,
    ...meta.profile,
  };
}
