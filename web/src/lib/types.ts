// The shape of the JSON files written by `spotcast-publish` (src/spotcast/publish.py).
// If the Python side changes a field, change it here too: this is the contract between them.

export type Label = "good" | "marginal" | "bad" | "unknown";

// Warnings come as codes; the site words them in the visitor's language (lib/i18n.tsx)
export type Note =
  | { code: "lead_time"; days: number }
  | { code: "no_waves" }
  | { code: "no_guide" }
  | { code: "not_for_sport" } // the place's guide doesn't list this sport
  | { code: "no_shore" } // unknown shore direction: offshore could not be checked
  | { code: "offshore"; hours: number } // wind from the land, counted as a danger
  | { code: "sheltered_offshore"; hours: number }; // wind from the land, in a closed bay

export type Sport = "windsurf" | "kitesurf" | "sup" | "kayak";
export type Level = "beginner" | "intermediate" | "advanced";
export type ShoreWind = "onshore" | "side-shore" | "offshore" | "unknown";

export interface Profile {
  id: string; // "sup_beginner": the folder of its files under profiles/
  sport: Sport;
  level: Level;
  max_wind_kn: number;
  max_wave_m: number;
  min_wind_kn: number;
  good_wind_min_kn: number;
  good_wind_max_kn: number;
  good_wave_max_m: number;
  ideal_wind_kn: number | null;
  max_offshore_wind_kn: number | null;
}

export interface Meta {
  generated_at: string; // "2026-11-08T18:24", Turkish time
  timezone: string;
  sport: string;
  level: string;
  profile: {
    max_wind_kn: number;
    max_wave_m: number;
    min_wind_kn: number;
    good_wind_min_kn: number;
    good_wind_max_kn: number;
    good_wave_max_m: number;
  };
  model: { type: string; features: string[]; train_years: number[] };
  default_profile?: string; // missing in data written before the profiles existed
  profiles?: Profile[];
  spots?: { name: string; sports: Sport[] | null; sheltered: boolean }[];
  uncertain_from_days: number;
  code_version: { git_commit: string; git_dirty: string };
  quality?: {
    rows: number;
    places_total: number;
    places_skipped: string[];
    missing_wave_share: number;
    implausible_values: Record<string, number>;
    outside_training_range: Record<string, number>;
  };
}

export interface Window {
  spot: string;
  date: string; // "2026-11-14"
  start: string;
  end: string;
  hours: number;
  good_hours: number;
  marginal_hours: number;
  avg_wind_knots: number;
  max_wave_metres: number;
  days_ahead: number;
  notes: Note[];
}

export interface Preset {
  earliest_hour: number;
  latest_hour: number;
  windows: Window[];
}

export interface Recommendations {
  any: Preset;
  weekend: Preset;
}

export interface SpotHour {
  time: string; // "2026-11-09T14:00"
  wind_knots: number;
  wind_direction: number;
  wave_metres: number | null;
  is_day: boolean;
  shore_wind?: ShoreWind;
  label: Label;
}

export interface Spot {
  name: string;
  slug: string;
  latitude: number;
  longitude: number;
  hours: SpotHour[];
}

export interface DaylightHour {
  hour: string; // "14:00"
  wind_knots: number;
  wave_metres: number | null;
  shore_wind?: ShoreWind;
  label: Label;
}

export interface PlaceDay {
  date: string;
  days_ahead: number;
  daylight_hours: DaylightHour[];
  label_counts: Partial<Record<Label, number>>;
  best_window: { hours: string; good_hours: number; avg_wind_knots: number } | null;
  local_guide_spot: string | null;
  notes: Note[];
}

export interface Place {
  name: string;
  region: string | null; // null for our guided spots
  slug: string;
  latitude: number;
  longitude: number;
  guided: boolean;
  days: PlaceDay[];
}
