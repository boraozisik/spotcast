// Faithful English translations of the recorded conversations in examples.json, in the
// same order. The assistant itself answers in Turkish; these are shown in English mode.
// Re-recording examples.json means updating these translations too.

export interface ExampleTranslation {
  topic: string;
  question: string;
  summaries: string[]; // one per tool call
  mainText: string; // the LLM's answer (or the system's fixed answer)
  codeParts: string[]; // text added by the system after the answer
}

export const examplesEn: ExampleTranslation[] = [
  {
    topic: "A place outside the guided spots",
    question:
      "I want to windsurf on the Mersin coast on Saturday, I'm intermediate. Is it suitable?",
    summaries: [
      "Mersin (Mersin); Mersin (Akçaabat, Trabzon); Mersin (Çatak, Van); Mersinli (Foça, İzmir); Mersinli (Salihli, Manisa)",
      "Saturday 14 November · hours: bad: 10 · best window: none",
    ],
    mainText:
      'Windsurfing on the Mersin coast on Saturday is not suitable. All hours were assessed as "not suitable" (bad). The wind blows offshore (pushing you away from the shore) and is light, which is not enough for your level. SpotCast has no local guide for this area; check local hazards.',
    codeParts: [
      "🌬️ Offshore wind (from the land out to sea) in 3 daylight hours: it pushes you away from the shore. Hours above your level's limit were counted as not suitable.",
      "⚠️ Forecasts can change. Check the wind and waves on the beach yourself before going in; if in doubt, don't go out.",
    ],
  },
  {
    topic: "Places near me",
    question: "I live in Izmir. Where should I go windsurfing on Saturday? I'm intermediate.",
    summaries: [
      "İzmir (İzmir); İzmirli (Düzce); Boz Dağ; Izmir Kaklıç Airport (Menemen, İzmir); Adnan Menderes Airport (Gaziemir, İzmir)",
      "Alaçatı (69 km) Saturday 14 November 08:00-18:00 (good 0, marginal 11) | Urla Gülbahçe (43 km) Saturday 14 November 08:00-18:00 (good 0, marginal 11) | Seferihisar Sığacık (39 km) Saturday 14 November 08:00-18:00 (good 0, marginal 11)",
    ],
    mainText:
      "Alaçatı, 69 km away, is the best option. On Saturday 14 November, 11 hours of marginal conditions are expected between 08:00 and 18:00. Average wind 15.3 knots, highest waves 1.46 metres. Urla Gülbahçe and Seferihisar Sığacık are alternatives, but they have no local guide.",
    codeParts: [
      "📏 Straight-line distance: Alaçatı 69 km, Urla Gülbahçe 43 km, Seferihisar Sığacık 39 km. The road distance may be longer.",
      "ℹ️ We have no local guide for Urla Gülbahçe, Seferihisar Sığacık: ask there about local hazards such as currents, rocks and restricted areas.",
      "⚠️ Forecasts can change. Check the wind and waves on the beach yourself before going in; if in doubt, don't go out.",
    ],
  },
  {
    topic: "Another sport and level",
    question: "I want to go stand-up paddling in Akyaka tomorrow, I'm a beginner. Is it suitable?",
    summaries: [
      "Error: These coordinates did not come from find_location. Call find_location with the place name first; never guess coordinates.",
      "Akyaka (Muğla); Akyaka (Kars); Akyaka (Ardahan); Akyaka (Demirözü, Bayburt); Akkaya (Burdur)",
      "Error: The user asked about 2026-11-11, but you sent 2026-11-14. Use the date table and call again.",
      "Wednesday 11 November · hours: good: 11 · best window: 08:00-18:00",
    ],
    mainText:
      "For stand-up paddling in Akyaka tomorrow, 11 hours are good (08:00-18:00) and 0 hours marginal. Average wind 1.2 knots. During the good hours the wind is light and the waves are around 0.1 metres. Something to watch out for: the wind blows from the shore out to sea and can carry you away from the shore.",
    codeParts: [
      "🌬️ Offshore wind (from the land out to sea) in 5 daylight hours: it pushes you away from the shore. Hours above your level's limit were counted as not suitable.",
      "⚠️ Forecasts can change. Check the wind and waves on the beach yourself before going in; if in doubt, don't go out.",
    ],
  },
  {
    topic: "Unsupported activity (answered by the system)",
    question: "I want to go diving in Kaş this weekend. What are the conditions like?",
    summaries: [],
    mainText:
      "SpotCast can currently assess conditions only for these sports: windsurfing, kitesurfing, SUP and sea kayaking. I can't give recommendations for this activity yet. Results for another sport or level may not suit you, so I don't recommend using them.",
    codeParts: [],
  },
  {
    topic: "Safety knowledge (RAG)",
    question: "What should I do if I get caught in a rip current?",
    summaries: [
      "Rip Currents → What to do if you are caught | Rip Currents → What is a rip current? | Rip Currents → General safety rules",
    ],
    mainText:
      "If you get caught in a rip current, don't fight the current. First get out of the current, then swim towards the shore. If you can't escape, float on your back or tread water where you are. If you need help, shout or wave for assistance. Follow safety rules such as not going into the sea if you can't swim and never swimming alone.",
    codeParts: [
      "📖 Source (Rip Currents → What to do if you are caught): «Don't fight the current. First get out of the current, then swim towards the shore. If you can't escape, float on your back or tread water where you are. If you need help, shout or wave for assistance.»",
    ],
  },
  {
    topic: "Safety knowledge (RAG)",
    question: "Is it risky to windsurf when the wind blows from the land towards the sea?",
    summaries: [
      "Windsurfing safety rules → Wind direction: offshore wind | Windsurfing area rules → Wind direction | Windsurfing safety rules → Know your level",
    ],
    mainText:
      "Wind blowing from the land towards the sea (offshore) is the most dangerous wind direction for windsurfing. Near the shore the wind looks light and the surface calm, but it gets stronger further out. If you fall or get tired, the wind drifts you out to sea and getting back to the shore becomes hard. So if the wind blows from the land towards the sea, the right decision is not to go in.",
    codeParts: [
      "📖 Source (Windsurfing safety rules → Wind direction: offshore wind): «Wind blowing from the land towards the sea (offshore) is the most dangerous wind direction. Near the shore the wind looks light and the surface calm, but it gets stronger further out. If you fall or get tired, the wind drifts you out to sea and getting back to the shore becomes hard. Wind parallel to the shore (side-shore) or blowing slightly from the sea (side-onshore) is the safest and best direction. With wind blowing straight from the sea (onshore), getting out is hard because of the waves, but the risk of drifting is low.»",
    ],
  },
];
