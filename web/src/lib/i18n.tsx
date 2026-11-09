"use client";

import { useEffect, useSyncExternalStore } from "react";

import type { Label, Note } from "./types";

export type Lang = "en" | "tr";

// ---------- Remembering the visitor's choice ----------
// English by default. The choice is kept in localStorage; useSyncExternalStore reads it
// without an extra render, and the prerendered HTML (English) matches the first paint.

const KEY = "spotcast-lang";
const listeners = new Set<() => void>();
let memory: Lang | null = null; // fallback when localStorage is blocked (private mode)

function read(): Lang {
  try {
    const stored = localStorage.getItem(KEY);
    if (stored === "en" || stored === "tr") return stored;
  } catch {
    // storage not available: use memory
  }
  return memory ?? "en";
}

function subscribe(onChange: () => void): () => void {
  listeners.add(onChange);
  window.addEventListener("storage", onChange); // other tabs
  return () => {
    listeners.delete(onChange);
    window.removeEventListener("storage", onChange);
  };
}

export function setLang(lang: Lang): void {
  memory = lang;
  try {
    localStorage.setItem(KEY, lang);
  } catch {
    // ignore: memory keeps it for this visit
  }
  listeners.forEach((listener) => listener());
}

export function useLang(): Lang {
  return useSyncExternalStore(subscribe, read, () => "en");
}

/** Keeps <html lang> in sync, for screen readers and search engines. */
export function HtmlLang() {
  const lang = useLang();
  useEffect(() => {
    document.documentElement.lang = lang;
  }, [lang]);
  return null;
}

// ---------- Formatting ----------

const LOCALE: Record<Lang, string> = { en: "en-GB", tr: "tr-TR" };

/** "2026-11-14" → "Saturday 14 November" / "14 Kasım Cumartesi" */
export function formatDay(date: string, lang: Lang, short = false): string {
  const options: Intl.DateTimeFormatOptions = short
    ? { weekday: "short", day: "numeric" }
    : { weekday: "long", day: "numeric", month: "long" };
  return new Intl.DateTimeFormat(LOCALE[lang], options).format(new Date(`${date}T12:00`));
}

/** "2026-11-08T18:41" → "Sunday 8 November, 18:41" (Turkish time, as written) */
export function formatDateTime(value: string, lang: Lang): string {
  const [date, time] = value.split("T");
  return `${formatDay(date, lang)}, ${time}`;
}

export function formatNumber(value: number, lang: Lang, digits = 1): string {
  return value.toLocaleString(LOCALE[lang], { maximumFractionDigits: digits });
}

// ---------- Texts ----------

const en = {
  brand: "windsurf forecasts",
  nav: {
    home: "Recommendations",
    place: "Check a location",
    assistant: "Assistant",
    evaluation: "Evaluation",
    how: "How it works",
  },
  language: "Language",
  labels: { good: "Good", marginal: "Marginal", bad: "Not suitable", unknown: "No data" } as Record<
    Label,
    string
  >,
  compass: ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
  knots: "knots",
  caution:
    "Forecasts can change. Check the wind and waves on the beach yourself before going in; if in doubt, don't go out.",
  note: (note: Note): string => {
    if (note.code === "lead_time")
      return `Forecast ${note.days} days ahead: forecasts this far out change often. Check again the day before you go.`;
    if (note.code === "no_waves")
      return "No wave data for this point (lake, land or too close to the shore), so there is no suitability verdict; wind only.";
    return "No local guide for this place: ask locals about currents, rocks and restricted areas.";
  },
  stale: (hours: number) =>
    `This forecast is ${hours} hours old (it is normally refreshed every 6 hours). Conditions may have changed: check a current forecast before you go.`,
  loadError: {
    title: "Forecast data could not be loaded.",
    body: "Data is refreshed every 6 hours. Please try again shortly.",
  },
  footer: {
    disclaimer: "Forecasts are guidance only. Always check the conditions on site before going out.",
    data: "Weather and wave data",
    source: "Source code",
  },
  home: {
    kicker: "AI-powered windsurf forecasts",
    title: "Is this spot right for my level, at this hour?",
    lead: "SpotCast evaluates wind and wave forecasts with a machine learning model and strict safety rules, then picks the best time windows along the Turkish coast.",
    ctaPlace: "Check a location",
    ctaHow: "How it works",
    updated: (time: string) =>
      `Updated ${time} · windsurfing, intermediate level · refreshed every 6 hours`,
    best: {
      title: "Best time windows",
      subtitle: "Guided spots, next 7 days, at least 2 hours between 09:00 and 18:00.",
      any: "Any day",
      weekend: "Weekend",
      none: "No window fits these preferences: the wind is either too light or too strong for an intermediate rider.",
      first: "Best option",
      nth: (n: number) => `Option ${n}`,
      avgWind: "Average wind",
      maxWave: "Highest wave",
    },
    spots: {
      title: "Hour by hour at guided spots",
      text: (min: number, max: number) =>
        `Bars show wind speed, colours show SpotCast's verdict. Dashed lines mark the ideal range for an intermediate rider (${min}-${max} knots). Daylight hours only.`,
      strongest: (time: string, wind: string, from: string) =>
        `Strongest wind at ${time}: ${wind}, from the ${from}.`,
    },
  },
  place: {
    title: "Check a location",
    lead: "Planning a session somewhere new? Pick a place on the Turkish coast and a day. Places outside our guided spots are assessed with the same model and safety rules, with an extra warning where no local guide is available.",
    placeLabel: "Place",
    dayLabel: "Day",
    guidedGroup: "Guided spots",
    guidedSpot: "guided spot",
    verdict: {
      unknown: "No verdict: no wave data",
      good: (n: number) => `Good hours available (${n} h)`,
      marginal: "Marginal hours only",
      bad: "Not suitable",
    },
    best: (hours: string, wind: string, good: number) =>
      `Best window ${hours}: average ${wind}, ${good} good hours.`,
    table: "Hour-by-hour table",
    columns: ["Hour", "Wind", "Waves", "Verdict"],
  },
  assistant: {
    title: "SpotCast Assistant",
    lead: (date: string, model: string) =>
      `A tool-calling AI agent that understands free-form questions and decides which tools to use: geocoding, condition checks, recommendations and a knowledge base of spot guides and safety rules. Conversations recorded on ${date} with the ${model} model.`,
    translated: "The assistant answers in Turkish. English translations are shown; switch to TR for the originals.",
    howTitle: "How to read",
    howTools:
      "Tool calls: functions the agent chose to run. Sport and level are fixed by the system; the model cannot change them.",
    howCode:
      "Blue boxes: fixed text added by the system after the model's answer: uncertainty notes, the safety caution and a verbatim quote of the source. Safety text never comes from the model.",
    tools: {
      find_location: "Find location",
      check_conditions: "Check conditions",
      recommend_spots: "Recommend spots",
      search_knowledge: "Search knowledge base",
    } as Record<string, string>,
    refusal:
      "Unsupported level: the agent did not run and the system answered with a fixed message, so a beginner is never given results meant for another level.",
    meta: (calls: number, seconds: number) => `${calls} tool calls · ${seconds} s`,
  },
  evaluation: {
    title: "Evaluation",
    lead: "Every part of SpotCast is measured on fixed test sets: retrieval quality, the agent's answers and the LLM judge that checks them.",
    retrieval: {
      title: "1. Retrieval quality (RAG)",
      text: "18 questions with known answer sections. hit@1: the right section ranks first; hit@3: it is in the top 3; MRR: mean of 1/rank.",
      worse: "↓ worse",
      finding:
        "Finding: restructuring the content had the largest impact. The first hybrid search version reduced quality, because PostgreSQL's ranking ignores how common a word is; filtering common words fixed it.",
    },
    judge: {
      title: "2. Measuring the LLM judge",
      text: "A second LLM checks whether answers contradict their sources. Since the judge is an LLM too, it was first measured on 10 hand-labelled examples.",
      contradictions: "Contradictions found",
      additions: "Unsupported additions found",
      perExample: (s: number) => `~${s} s per example`,
    },
    answers: {
      title: "3. Agent answers",
      text: "Tool results are frozen (fixtures), so the weather never changes during a test: the real LLM runs in a fixed world. 10 scenarios, each run 3 times, because an LLM can answer differently each time.",
      result:
        "First run: 8 of 10 scenarios passed. After the fixes: 30 of 30 runs passed. A known gap remains: the number check cannot catch a wrong sentence built from a real number.",
    },
    bugs: { title: "Issues found by evaluation", fix: "Fix" },
  },
  how: {
    title: "How it works",
    lead: "SpotCast combines a data pipeline, classical machine learning, deep learning, large language models and automated evaluation in one system.",
    pipelineTitle: "Data pipeline",
    pipeline: [
      "Forecasts are computed by a scheduled batch job every 6 hours and published as static data. The site has no backend: nothing to attack, nothing that can go down.",
      "Live on this site: recommendations, hourly forecasts and assessments for 29 coastal places, all from the same model and safety rules.",
      "The assistant runs on a locally hosted LLM; its conversations are shown on the Assistant page.",
    ],
    principlesTitle: "Design principles",
    stackTitle: "Technology",
    sourceLink: "Source code and documentation on GitHub →",
  },
};

export type Dictionary = typeof en;

const tr: Dictionary = {
  brand: "rüzgâr sörfü tahmini",
  nav: {
    home: "Öneriler",
    place: "Konum değerlendir",
    assistant: "Asistan",
    evaluation: "Değerlendirme",
    how: "Nasıl çalışır?",
  },
  language: "Dil",
  labels: { good: "Uygun", marginal: "Sınırda", bad: "Uygun değil", unknown: "Veri yok" },
  compass: ["K", "KD", "D", "GD", "G", "GB", "B", "KB"],
  knots: "knot",
  caution:
    "Tahminler değişebilir. Suya girmeden önce sahilde rüzgârı ve dalgayı kendin kontrol et; emin olmadığında girme.",
  note: (note: Note): string => {
    if (note.code === "lead_time")
      return `${note.days} gün sonrası için tahmin: bu kadar uzak tahminler sık değişir. Gitmeden önceki gün yeniden kontrol et.`;
    if (note.code === "no_waves")
      return "Bu nokta için dalga verisi yok (göl, kara ya da kıyıya çok yakın). Bu yüzden uygunluk değerlendirmesi yapılamadı; sadece rüzgâr bilgisi var.";
    return "Bu yer için yerel rehber yok: akıntı, kaya ve yasak alan gibi yerel tehlikeleri oradakilere sor.";
  },
  stale: (hours: number) =>
    `Bu tahmin ${hours} saat önce üretildi (normalde 6 saatte bir yenilenir). Koşullar değişmiş olabilir: gitmeden önce güncel bir tahmine bak.`,
  loadError: {
    title: "Tahmin verisi yüklenemedi.",
    body: "Veriler 6 saatte bir yenileniyor. Lütfen biraz sonra tekrar dene.",
  },
  footer: {
    disclaimer: "Tahminler yol göstericidir. Suya girmeden önce koşulları sahilde mutlaka kontrol et.",
    data: "Hava ve dalga verisi",
    source: "Kaynak kod",
  },
  home: {
    kicker: "Yapay zekâ destekli rüzgâr sörfü tahmini",
    title: "Bu spot, bu saatte, benim seviyem için uygun mu?",
    lead: "SpotCast rüzgâr ve dalga tahminlerini bir makine öğrenmesi modeli ve katı güvenlik kurallarıyla değerlendirir, Türkiye kıyılarındaki en iyi zaman aralıklarını seçer.",
    ctaPlace: "Bir konumu değerlendir",
    ctaHow: "Nasıl çalışır?",
    updated: (time: string) =>
      `Güncelleme: ${time} · rüzgâr sörfü, orta seviye · 6 saatte bir yenilenir`,
    best: {
      title: "En iyi zaman aralıkları",
      subtitle: "Rehberli spotlar, önümüzdeki 7 gün, 09:00-18:00 arası en az 2 saat.",
      any: "Her gün",
      weekend: "Hafta sonu",
      none: "Bu tercihlere uyan bir zaman aralığı yok: rüzgâr ya çok zayıf ya da orta seviye için fazla güçlü.",
      first: "En iyi seçenek",
      nth: (n: number) => `${n}. seçenek`,
      avgWind: "Ortalama rüzgâr",
      maxWave: "En yüksek dalga",
    },
    spots: {
      title: "Rehberli spotlarda saat saat",
      text: (min: number, max: number) =>
        `Çubuklar rüzgâr hızını, renkler SpotCast'in kararını gösterir. Kesikli çizgiler orta seviye için ideal aralık (${min}-${max} knot). Sadece gündüz saatleri.`,
      strongest: (time: string, wind: string, from: string) =>
        `En güçlü rüzgâr saat ${time}: ${wind}, ${from} yönünden.`,
    },
  },
  place: {
    title: "Konum değerlendir",
    lead: "Yeni bir yerde mi sörf yapacaksın? Türkiye kıyılarından bir yer ve gün seç. Rehberli spotlarımız dışındaki yerler de aynı model ve güvenlik kurallarıyla değerlendirilir; yerel rehber olmayan yerlerde ayrıca uyarı verilir.",
    placeLabel: "Yer",
    dayLabel: "Gün",
    guidedGroup: "Rehberli spotlar",
    guidedSpot: "rehberli spot",
    verdict: {
      unknown: "Karar verilemiyor: dalga verisi yok",
      good: (n: number) => `Uygun saatler var (${n} saat)`,
      marginal: "Sadece sınırda saatler var",
      bad: "Uygun değil",
    },
    best: (hours: string, wind: string, good: number) =>
      `En iyi aralık ${hours}: ortalama ${wind}, ${good} uygun saat.`,
    table: "Saat saat tablo",
    columns: ["Saat", "Rüzgâr", "Dalga", "Karar"],
  },
  assistant: {
    title: "SpotCast Asistanı",
    lead: (date: string, model: string) =>
      `Serbest soruları anlayıp hangi aracı kullanacağına kendisi karar veren bir yapay zekâ ajanı: yer bulma, koşul değerlendirme, öneri ve spot rehberleriyle güvenlik kurallarından oluşan bir bilgi bankası. Konuşmalar ${date} tarihinde ${model} modeliyle kaydedildi.`,
    translated: "",
    howTitle: "Nasıl okunur?",
    howTools:
      "Araç çağrıları: ajanın kendi kararıyla çalıştırdığı fonksiyonlar. Spor ve seviyeyi sistem belirler; model değiştiremez.",
    howCode:
      "Mavi kutular: modelin cevabından sonra sistemin eklediği sabit metinler: belirsizlik notu, güvenlik uyarısı ve kaynağın birebir alıntısı. Güvenlik metni asla modelden gelmez.",
    tools: {
      find_location: "Yer bul",
      check_conditions: "Koşulları değerlendir",
      recommend_spots: "Spot öner",
      search_knowledge: "Bilgi bankasında ara",
    },
    refusal:
      "Desteklenmeyen seviye: ajan çalışmadı ve sistem sabit bir mesajla cevap verdi. Böylece yeni başlayan birine başka bir seviyenin sonuçları asla verilmez.",
    meta: (calls: number, seconds: number) => `${calls} araç çağrısı · ${seconds} saniye`,
  },
  evaluation: {
    title: "Değerlendirme",
    lead: "SpotCast'in her parçası sabit test setleriyle ölçülür: arama kalitesi, ajanın cevapları ve bu cevapları denetleyen LLM hakemi.",
    retrieval: {
      title: "1. Arama (RAG) kalitesi",
      text: "Cevabının hangi bölümde olduğu bilinen 18 soru. hit@1: doğru bölüm 1. sırada; hit@3: ilk 3'te; MRR: 1/sıra ortalaması.",
      worse: "↓ düştü",
      finding:
        "Bulgu: en büyük etkiyi içeriğin yeniden düzenlenmesi yaptı. Hibrit aramanın ilk hali kaliteyi düşürdü, çünkü PostgreSQL'in sıralaması kelimelerin ne kadar yaygın olduğunu hesaba katmıyor; yaygın kelimeleri elemek sorunu çözdü.",
    },
    judge: {
      title: "2. LLM hakeminin ölçülmesi",
      text: "Cevapların kaynaklarla çelişip çelişmediğini ikinci bir LLM denetliyor. Hakem de bir LLM olduğu için önce elle etiketlenmiş 10 örnekte kendisi ölçüldü.",
      contradictions: "Çelişki tespiti",
      additions: "Kaynakta olmayan ekleme tespiti",
      perExample: (s: number) => `~${s} saniye / örnek`,
    },
    answers: {
      title: "3. Ajan cevapları",
      text: "Araç sonuçları sabitlenir (fixture), böylece test sırasında hava değişmez: gerçek LLM sabit bir dünyada çalışır. 10 senaryo, her biri 3 kez; çünkü bir LLM her seferinde farklı cevap verebilir.",
      result:
        "İlk ölçüm: 10 senaryonun 8'i geçti. Düzeltmelerden sonra: 30 çalıştırmanın 30'u geçti. Bilinen bir boşluk var: sayı kontrolü, gerçek bir sayıyla kurulmuş yanlış bir cümleyi yakalayamaz.",
    },
    bugs: { title: "Değerlendirmeyle bulunan sorunlar", fix: "Düzeltme" },
  },
  how: {
    title: "Nasıl çalışır?",
    lead: "SpotCast bir veri hattını, klasik makine öğrenmesini, derin öğrenmeyi, büyük dil modellerini ve otomatik değerlendirmeyi tek bir sistemde birleştirir.",
    pipelineTitle: "Veri hattı",
    pipeline: [
      "Tahminler 6 saatte bir zamanlanmış bir toplu iş (batch job) tarafından hesaplanıp statik veri olarak yayınlanır. Sitenin backend'i yoktur: saldırılacak ya da çökecek bir sunucu bulunmaz.",
      "Bu sitede canlı olanlar: öneriler, saatlik tahminler ve 29 kıyı noktası için değerlendirmeler; hepsi aynı model ve güvenlik kurallarıyla.",
      "Asistan yerel olarak barındırılan bir LLM üzerinde çalışır; konuşmaları Asistan sayfasında görülebilir.",
    ],
    principlesTitle: "Tasarım ilkeleri",
    stackTitle: "Teknolojiler",
    sourceLink: "Kaynak kod ve dokümantasyon GitHub'da →",
  },
};

export const DICTIONARY: Record<Lang, Dictionary> = { en, tr };

export function useT(): { lang: Lang; t: Dictionary } {
  const lang = useLang();
  return { lang, t: DICTIONARY[lang] };
}
