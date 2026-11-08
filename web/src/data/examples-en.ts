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
      "Mersin (Mersin); Mersin (Akçaabat, Trabzon); Mersin (Çatak, Van)",
      "Saturday 14 November · hours: bad: 10 · best window: none",
    ],
    mainText:
      'On Saturday 14 November, all hours on the Mersin coast were assessed as "not suitable" (bad) for windsurfing. Poor conditions are expected for 10 hours. SpotCast has no local guide for this area; checking local hazards is recommended. Forecasts 6 days ahead can change a lot; check again the day before you go.',
    codeParts: [
      "⏳ Forecast 6 days ahead: forecasts this far out change often. Check again the day before you go.",
      "⚠️ Forecasts can change. Check the wind and waves on the beach yourself before going in; if in doubt, don't go out.",
    ],
  },
  {
    topic: "Recommendation",
    question: "Where should I go windsurfing this weekend? I'm intermediate.",
    summaries: [
      "Alaçatı, Saturday 14 November 09:00-18:00 (good 0, marginal 10) | Kuşadası Pamucak, Saturday 14 November 12:00-17:00 (good 0, marginal 6)",
    ],
    mainText:
      "For windsurfing this weekend, Alaçatı and Kuşadası Pamucak are recommended. In Alaçatı, 10 hours of marginal conditions are expected on Saturday 14 November between 09:00 and 18:00, while 6 hours of marginal conditions are forecast in Kuşadası Pamucak between 12:00 and 17:00. As both forecasts are 6 days ahead, it is recommended to check again the day before you go.",
    codeParts: [
      "⏳ Forecast 6 days ahead: forecasts this far out change often. Check again the day before you go.",
      "⚠️ Forecasts can change. Check the wind and waves on the beach yourself before going in; if in doubt, don't go out.",
    ],
  },
  {
    topic: "Unsupported level (answered by the system)",
    question: "I'm a beginner. Would it be OK to go to Alaçatı this weekend?",
    summaries: [],
    mainText:
      "SpotCast can currently assess conditions for intermediate windsurfing only. I can't give recommendations for beginner windsurfing yet. Results for another level may not suit you, so I don't recommend using them.",
    codeParts: [],
  },
  {
    topic: "Safety knowledge (RAG)",
    question: "What should I do if I get caught in a rip current?",
    summaries: [
      "Rip Currents → What to do if you are caught | Rip Currents → General safety rules | Rip Currents → What is a rip current?",
    ],
    mainText:
      "If you get caught in a rip current, don't fight the current. First get out of the current, then swim towards the shore. If you can't escape, float on your back or tread water where you are. If you need help, shout or wave for assistance. Don't go into the sea if you can't swim, and never swim alone. If in doubt, don't go in.",
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
      "Wind blowing from the land towards the sea (offshore) is the most dangerous wind direction for windsurfing. Near the shore the wind looks light and the surface calm, but it gets stronger further out, and if you fall or get tired the wind drifts you out to sea. So if the wind blows from the land towards the sea, the right decision is not to go in.",
    codeParts: [
      "📖 Source (Windsurfing safety rules → Wind direction: offshore wind): «Wind blowing from the land towards the sea (offshore) is the most dangerous wind direction. Near the shore the wind looks light and the surface calm, but it gets stronger further out. If you fall or get tired, the wind drifts you out to sea and getting back to the shore becomes hard. Wind parallel to the shore (side-shore) or blowing slightly from the sea (side-onshore) is the safest and best direction. With wind blowing straight from the sea (onshore), getting out is hard because of the waves, but the risk of drifting is low.»",
    ],
  },
];
