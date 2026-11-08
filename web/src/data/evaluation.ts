// Measured results from `uv run spotcast-eval`. Numbers are copied from real runs, not
// estimated. Re-run the evaluation and update here when the system changes.

type Text = { en: string; tr: string };

export const retrievalSteps: { step: Text; hit1: number; hit3: number; mrr: number; worse?: boolean }[] = [
  {
    step: { en: "Baseline (vector search only)", tr: "Başlangıç (sadece vektör araması)" },
    hit1: 72,
    hit3: 78,
    mrr: 0.74,
  },
  {
    step: {
      en: "1. Content restructured (mixed sections split by topic)",
      tr: "1. İçerik yeniden düzenlendi (karışık bölümler konuya göre ayrıldı)",
    },
    hit1: 83,
    hit3: 94,
    mrr: 0.87,
  },
  {
    step: { en: "2a. Hybrid search, first version", tr: "2a. Hibrit arama, ilk hali" },
    hit1: 78,
    hit3: 94,
    mrr: 0.85,
    worse: true,
  },
  {
    step: {
      en: "2b. Hybrid + common-word filter (IDF-like)",
      tr: "2b. Hibrit + yaygın kelime filtresi (IDF benzeri)",
    },
    hit1: 83,
    hit3: 94,
    mrr: 0.88,
  },
  {
    step: {
      en: "3. + Query expansion (knots → Beaufort, in code)",
      tr: "3. + Sorgu genişletme (knot → Beaufort, kodda)",
    },
    hit1: 89,
    hit3: 100,
    mrr: 0.94,
  },
];

export const judgeVersions: { version: Text; contradicts: number; adds: number; seconds: number }[] = [
  {
    version: { en: "v1: single verdict", tr: "v1: tek seferde karar" },
    contradicts: 80,
    adds: 70,
    seconds: 3,
  },
  {
    version: {
      en: "v2: split into claims, verdict by code",
      tr: "v2: iddialara böl, kararı kod versin",
    },
    contradicts: 100,
    adds: 100,
    seconds: 7,
  },
];

export const answerChecks: Text[] = [
  {
    en: "Were the right tools called, with the right date?",
    tr: "Doğru araçlar, doğru tarihle çağrıldı mı?",
  },
  {
    en: "Were the required warnings given (no local guide, no wave data)?",
    tr: "Gerekli uyarılar verildi mi (yerel rehber yok, dalga verisi yok)?",
  },
  { en: "Conditions are never called \"safe\".", tr: "Koşullara asla \"güvenli\" denmiyor." },
  {
    en: "\"Suitable\" is never used when no hour is good.",
    tr: "Hiç uygun saat yokken \"uygun\" denmiyor.",
  },
  {
    en: "Every knot and metre value in the answer comes from a tool result (no invented numbers).",
    tr: "Cevaptaki her knot ve metre değeri araç sonuçlarından geliyor (uydurma sayı yok).",
  },
  {
    en: "LLM judge: does the answer contradict its source or add advice the source doesn't contain?",
    tr: "LLM hakemi: cevap kaynağıyla çelişiyor mu, kaynakta olmayan tavsiye ekliyor mu?",
  },
];

export const issuesFound: { title: Text; fix: Text }[] = [
  {
    title: {
      en: "The agent turned \"yarın\" (tomorrow) into the wrong date, 3 times out of 3",
      tr: "Ajan \"yarın\"ı 3 denemenin 3'ünde yanlış güne çeviriyordu",
    },
    fix: {
      en: "The date table given to the model now includes the Turkish words. Result: 3/3 correct.",
      tr: "Modele verilen tarih tablosuna Türkçe karşılıklar eklendi. Sonuç: 3/3 doğru.",
    },
  },
  {
    title: {
      en: "The model changed a beginner's level to \"intermediate\" on its own",
      tr: "Model yeni başlayan bir kullanıcının seviyesini kendisi \"orta\" yaptı",
    },
    fix: {
      en: "The level is now fixed in code: removed from the schema the model sees and overwritten on every tool call.",
      tr: "Seviye artık kodda sabit: modelin gördüğü şemadan silindi ve her araç çağrısında kod tarafından üstüne yazılıyor.",
    },
  },
  {
    title: {
      en: "A rip current instruction was inverted (\"don't try to get out of the current\")",
      tr: "Dip akıntısı tavsiyesi tersine döndü (\"akıntının dışına çıkmaya çalışma\")",
    },
    fix: {
      en: "The system appends a verbatim quote of the source; the LLM judge catches this kind of contradiction.",
      tr: "Sistem kaynağın birebir alıntısını ekliyor; LLM hakemi bu tür çelişkileri yakalıyor.",
    },
  },
  {
    title: {
      en: "The first hybrid search version lowered retrieval quality",
      tr: "Hibrit aramanın ilk hali arama kalitesini düşürdü",
    },
    fix: {
      en: "PostgreSQL's ranking ignored how common a word is; common words are now filtered out.",
      tr: "PostgreSQL'in sıralaması kelime yaygınlığını hesaba katmıyordu; yaygın kelimeler artık eleniyor.",
    },
  },
  {
    title: {
      en: "The vision model consistently underestimated storms (force 10-12 judged as 6-8)",
      tr: "Görsel model fırtınaları sürekli küçümsüyordu (10-12 Beaufort'a 6-8 dedi)",
    },
    fix: {
      en: "Photo-based estimates are not used in decisions.",
      tr: "Fotoğraftan yapılan tahminler kararlarda kullanılmıyor.",
    },
  },
];
