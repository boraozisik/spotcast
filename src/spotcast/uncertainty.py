"""How far a forecast can be trusted, as fixed warnings written by code (never by the LLM).

Two known reasons to doubt a result:
- Lead time: forecasts get worse the further ahead they look. In Stage 3 our own 48 h model
  was already much worse than at 1 h (MAE 0.69 → 2.43 knot), and weather forecasts often
  change a lot after about 5 days.
- No wave data (lake, land, very close to the shore): the model needs wave height, so there
  is no verdict at all, only wind.
"""

from datetime import date, datetime

UNCERTAIN_FROM_DAYS = 5  # a rule of thumb; Open-Meteo gives 7 days (today = 0 ... 6)

NO_WAVE_NOTE = (
    "Bu konum için dalga verisi yok (göl, kara ya da kıyıya çok yakın bir nokta). "
    "Bu yüzden uygunluk değerlendirmesi yapılamadı; sadece rüzgâr bilgisi var."
)


def days_ahead(day: date | datetime, now: datetime) -> int:
    """0 = today, 1 = tomorrow, ... (pandas Timestamps work too)."""
    if isinstance(day, datetime):
        day = day.date()
    return (day - now.date()).days


def lead_time_note(days: int) -> str | None:
    if days < UNCERTAIN_FROM_DAYS:
        return None
    return (
        f"{days} gün sonrası için tahmin: bu kadar uzak tahminler sık değişir. "
        f"Gitmeden önceki gün yeniden kontrol et."
    )


def notes_for_results(results: list[dict]) -> list[str]:
    """Warnings for an answer built from these check_conditions / recommend_spots results."""
    days: list[int] = []
    waves_missing = False
    for result in results:
        if "error" in result:
            continue
        if "days_ahead" in result:
            days.append(result["days_ahead"])
        days += [
            w["days_ahead"]
            for w in result.get("windows_best_first", [])
            if "days_ahead" in w
        ]
        if result.get("label_counts", {}).get("unknown"):
            waves_missing = True

    notes = [NO_WAVE_NOTE] if waves_missing else []
    farthest = lead_time_note(max(days)) if days else None
    return notes + ([farthest] if farthest else [])
