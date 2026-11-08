"use client";

import { Card, PageTitle } from "@/components/ui";
import { examplesEn } from "@/data/examples-en";
import recorded from "@/data/examples.json";
import { formatDay, useT } from "@/lib/i18n";

/** The parts of an answer that the system adds after the LLM's text. */
function codeParts(answer: string, llmText: string): string[] {
  const rest = answer.startsWith(llmText) ? answer.slice(llmText.length) : "";
  return rest
    .split("\n\n")
    .map((part) => part.trim())
    .filter(Boolean);
}

export default function AssistantPage() {
  const { lang, t } = useT();
  const recordedOn = formatDay(recorded.recorded_at.slice(0, 10), lang);

  return (
    <div className="space-y-6">
      <PageTitle title={t.assistant.title} lead={t.assistant.lead(recordedOn, recorded.model)} />

      <Card className="bg-slate-50 text-sm text-slate-700">
        <p className="font-medium text-slate-900">{t.assistant.howTitle}</p>
        <ul className="mt-2 list-disc space-y-1 pl-5">
          <li>{t.assistant.howTools}</li>
          <li>{t.assistant.howCode}</li>
        </ul>
        {t.assistant.translated && <p className="mt-3 text-slate-500">{t.assistant.translated}</p>}
      </Card>

      {recorded.examples.map((example, index) => {
        const answeredBySystem = !example.llm_text; // the agent never ran
        const english = examplesEn[index];
        const view =
          lang === "en" && english
            ? english
            : {
                topic: example.topic,
                question: example.question,
                summaries: example.steps.map((step) => step.summary),
                mainText: answeredBySystem ? example.answer : example.llm_text,
                codeParts: answeredBySystem ? [] : codeParts(example.answer, example.llm_text),
              };

        return (
          <Card key={example.question}>
            <p className="text-xs font-medium uppercase tracking-wide text-sky-700">{view.topic}</p>
            <p className="mt-2 rounded-xl bg-sky-900 px-4 py-3 text-white">❓ {view.question}</p>

            {example.steps.length > 0 && (
              <ol className="mt-3 space-y-2">
                {example.steps.map((step, i) => (
                  <li key={i} className="rounded-lg border border-slate-200 px-3 py-2 text-sm">
                    <p className="font-medium text-slate-800">
                      🔧 {t.assistant.tools[step.tool] ?? step.tool}{" "}
                      <code className="text-xs text-slate-500">
                        {step.tool}({JSON.stringify(step.arguments)})
                      </code>
                    </p>
                    <p className="mt-1 text-slate-600">→ {view.summaries[i]}</p>
                  </li>
                ))}
              </ol>
            )}

            <p className="mt-3 whitespace-pre-line text-slate-800">💬 {view.mainText}</p>
            {answeredBySystem && <p className="mt-2 text-sm text-slate-500">{t.assistant.refusal}</p>}
            {view.codeParts.map((part) => (
              <p
                key={part}
                className="mt-2 whitespace-pre-line rounded-lg bg-sky-50 px-3 py-2 text-sm text-sky-900"
              >
                {part}
              </p>
            ))}
            <p className="mt-3 text-xs text-slate-400">
              {t.assistant.meta(example.steps.length, example.seconds)}
            </p>
          </Card>
        );
      })}
    </div>
  );
}
