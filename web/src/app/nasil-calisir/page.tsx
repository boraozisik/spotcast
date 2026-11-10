"use client";

import { Card, PageTitle } from "@/components/ui";
import { useT } from "@/lib/i18n";
import { GITHUB_URL } from "@/lib/site";

type Text = { en: string; tr: string };

const PARTS: { title: Text; text: Text }[] = [
  {
    title: { en: "Data", tr: "Veri" },
    text: {
      en: "Wind and wave data from Open-Meteo: 7-day forecasts for 29 places in one batched request, and hourly history since 2022 from 9 places across 5 seas (380,000 hours), paced to the free API's limits. Every point was checked against the OpenStreetMap coastline: it must be on the water, and the shore direction decides when wind is offshore. pandas, Parquet, PostgreSQL.",
      tr: "Open-Meteo'dan rüzgâr ve dalga verisi: 29 yer için tek toplu istekte 7 günlük tahmin ve 5 denizden 9 yerin 2022'den bu yana saatlik geçmişi (380.000 saat); istekler ücretsiz API'nin sınırlarına göre ayarlanır. Her nokta OpenStreetMap kıyı çizgisiyle kontrol edildi: denizde olmalı, ve kıyının baktığı yön rüzgârın ne zaman karadan estiğini belirler. pandas, Parquet, PostgreSQL.",
    },
  },
  {
    title: { en: "Suitability model", tr: "Uygunluk modeli" },
    text: {
      en: "One scikit-learn model labels every hour as good, marginal or not suitable for 12 profiles (4 sports × 3 levels). It sees conditions relative to each profile's limits (\"3 knots above this level's maximum\"), so it works even for a profile it was not trained on. Models can fail on conditions they have never seen, so hard safety rules (guardrail) sit on top: wind, waves, darkness and offshore wind, with an exception only for sheltered bays confirmed by someone who knows the place.",
      tr: "Tek bir scikit-learn modeli 12 profil (4 spor × 3 seviye) için her saati uygun, sınırda veya uygun değil olarak etiketler. Koşulları her profilin sınırlarına göre görür (\"bu seviyenin üst sınırından 3 knot fazla\"); bu yüzden eğitimde görmediği bir profilde de çalışır. Modeller hiç görmedikleri koşullarda yanılabildiği için üstünde aşılamayan güvenlik kuralları (guardrail) bulunur: rüzgâr, dalga, karanlık ve karadan esen rüzgâr; tek istisna, yerini bilen birinin onayladığı korunaklı koylar.",
    },
  },
  {
    title: { en: "Forecasting models", tr: "Tahmin modelleri" },
    text: {
      en: "XGBoost and a PyTorch LSTM predict the wind 48 hours ahead and are benchmarked against persistence and climatology baselines.",
      tr: "XGBoost ve PyTorch LSTM modelleri 48 saat sonraki rüzgârı tahmin eder ve persistence ile climatology kıyas noktalarıyla karşılaştırılır.",
    },
  },
  {
    title: { en: "Recommendation engine", tr: "Öneri motoru" },
    text: {
      en: "An optimizer picks the best spot and time window under the user's constraints: hours, weekend, minimum session length, distance from where the user starts, and only spots where the sport is done.",
      tr: "Bir optimizasyon katmanı, kullanıcının kısıtlarına (saat aralığı, hafta sonu, en kısa seans süresi, kullanıcının bulunduğu yere uzaklık, sadece o sporun yapıldığı spotlar) uyan en iyi yeri ve zaman aralığını seçer.",
    },
  },
  {
    title: { en: "LLM and RAG", tr: "LLM ve RAG" },
    text: {
      en: "A locally hosted LLM explains recommendations in Turkish. Spot guides and safety documents are stored in pgvector and searched with hybrid (vector + full-text) retrieval; every tip cites its source.",
      tr: "Yerel olarak barındırılan bir LLM önerileri Türkçe açıklar. Spot rehberleri ve güvenlik belgeleri pgvector'de tutulur ve hibrit (vektör + tam metin) aramayla bulunur; her ipucu kaynağını gösterir.",
    },
  },
  {
    title: { en: "Agent and MCP", tr: "Ajan ve MCP" },
    text: {
      en: "A tool-calling agent (built both as a custom loop and with LangGraph) answers free-form questions. The same tools are exposed to other AI clients through an MCP server.",
      tr: "Araç çağıran bir ajan (hem özel bir döngü olarak hem LangGraph ile) serbest soruları cevaplar. Aynı araçlar bir MCP sunucusu üzerinden başka yapay zekâ istemcilerine açılır.",
    },
  },
  {
    title: { en: "Evaluation", tr: "Değerlendirme" },
    text: {
      en: "Retrieval quality (hit@k, MRR), agent answers (rule checks) and faithfulness to sources (LLM judge) are measured on fixed test sets. The judge itself is measured too.",
      tr: "Arama kalitesi (hit@k, MRR), ajan cevapları (kural kontrolleri) ve kaynağa sadakat (LLM hakemi) sabit test setleriyle ölçülür. Hakemin kendisi de ölçülür.",
    },
  },
  {
    title: { en: "MLOps", tr: "MLOps" },
    text: {
      en: "MLflow experiment tracking and model registry, a Docker image, GitHub Actions CI (lint, 300+ tests, Docker and website builds), the scheduled batch job behind this site with data-quality and drift checks, and weekly champion/challenger retraining with human approval.",
      tr: "MLflow ile deney takibi ve model kaydı, Docker imajı, GitHub Actions CI (lint, 300+ test, Docker ve site derlemesi), bu siteyi besleyen ve veri kalitesi ile veri kaymasını denetleyen zamanlanmış toplu iş, insan onaylı haftalık champion/challenger yeniden eğitim.",
    },
  },
];

const PRINCIPLES: { title: Text; text: Text }[] = [
  {
    title: { en: "Code computes, the LLM only words it", tr: "Kod hesaplar, LLM sadece sözlendirir" },
    text: {
      en: "Dates, comparisons and unit conversions are done in code; the LLM turns the results into sentences.",
      tr: "Tarihler, karşılaştırmalar ve birim dönüşümleri kodda yapılır; LLM sonuçları cümleye çevirir.",
    },
  },
  {
    title: { en: "Critical rules live in code, not in prompts", tr: "Kritik kurallar prompt'ta değil, kodda" },
    text: {
      en: "The user's level, the safety caution and the source quote are never left to the LLM.",
      tr: "Kullanıcının seviyesi, güvenlik uyarısı ve kaynak alıntısı asla LLM'e bırakılmaz.",
    },
  },
  {
    title: { en: "Safety text never comes from the LLM", tr: "Güvenlik metni asla LLM'den gelmez" },
    text: {
      en: "Cautions and uncertainty notes are fixed texts added by the system.",
      tr: "Uyarılar ve belirsizlik notları sistemin eklediği sabit metinlerdir.",
    },
  },
  {
    title: { en: "Measured, not assumed", tr: "Varsayım değil, ölçüm" },
    text: {
      en: "Every change is checked against a baseline on a fixed test set.",
      tr: "Her değişiklik sabit bir test setinde bir kıyas noktasına göre doğrulanır.",
    },
  },
  {
    title: { en: "Human review", tr: "İnsan onayı" },
    text: {
      en: "New knowledge documents enter as drafts and become searchable only after review.",
      tr: "Bilgi bankasına giren yeni belgeler önce taslaktır; ancak onaylandıktan sonra aramaya girer.",
    },
  },
];

const STACK =
  "Python 3.12, uv, pandas, scikit-learn, XGBoost, PyTorch, FastAPI, Pydantic, PostgreSQL + pgvector, Ollama (qwen3:8b, bge-m3, qwen2.5vl), LangGraph, MCP, MLflow, Docker, GitHub Actions, Next.js, TypeScript, Tailwind CSS, Netlify";

export default function HowItWorksPage() {
  const { lang, t } = useT();

  return (
    <div className="space-y-8">
      <PageTitle title={t.how.title} lead={t.how.lead} />

      <div className="grid gap-4 sm:grid-cols-2">
        {PARTS.map((part, i) => (
          <Card key={part.title.en}>
            <p className="text-xs font-medium text-sky-700">{i + 1}</p>
            <h2 className="font-semibold">{part.title[lang]}</h2>
            <p className="mt-1 text-sm text-slate-600">{part.text[lang]}</p>
          </Card>
        ))}
      </div>

      <Card>
        <h2 className="text-xl font-semibold">{t.how.pipelineTitle}</h2>
        <pre className="mt-3 overflow-x-auto rounded-xl bg-slate-900 p-4 text-xs leading-relaxed text-slate-100">
{`GitHub Actions (every 6 h):
   Open-Meteo forecast → ML model + guardrail → recommender → JSON
                                                               │
Next.js site (Netlify) ◀── browser reads the JSON ◀────────────┘`}
        </pre>
        <ul className="mt-4 list-disc space-y-1 pl-5 text-sm text-slate-700">
          {t.how.pipeline.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
      </Card>

      <Card>
        <h2 className="text-xl font-semibold">{t.how.principlesTitle}</h2>
        <dl className="mt-3 space-y-3">
          {PRINCIPLES.map((p) => (
            <div key={p.title.en}>
              <dt className="font-medium">{p.title[lang]}</dt>
              <dd className="text-sm text-slate-600">{p.text[lang]}</dd>
            </div>
          ))}
        </dl>
      </Card>

      <Card className="bg-slate-50">
        <h2 className="font-semibold">{t.how.stackTitle}</h2>
        <p className="mt-2 text-sm text-slate-700">{STACK}</p>
        <a href={GITHUB_URL} className="mt-3 inline-block text-sm font-medium text-sky-800 hover:underline">
          {t.how.sourceLink}
        </a>
      </Card>
    </div>
  );
}
