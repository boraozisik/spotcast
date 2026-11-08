"use client";

import { Card, PageTitle } from "@/components/ui";
import { answerChecks, issuesFound, judgeVersions, retrievalSteps } from "@/data/evaluation";
import { useT } from "@/lib/i18n";

function Bar({ value, max = 100, tone = "bg-sky-600" }: { value: number; max?: number; tone?: string }) {
  return (
    <div className="h-2 w-full rounded-full bg-slate-100">
      <div className={`h-2 rounded-full ${tone}`} style={{ width: `${(value / max) * 100}%` }} />
    </div>
  );
}

export default function EvaluationPage() {
  const { lang, t } = useT();
  const e = t.evaluation;

  return (
    <div className="space-y-8">
      <PageTitle title={e.title} lead={e.lead} />

      <Card>
        <h2 className="text-xl font-semibold">{e.retrieval.title}</h2>
        <p className="mt-1 text-sm text-slate-600">{e.retrieval.text}</p>
        <div className="mt-4 space-y-4">
          {retrievalSteps.map((s) => (
            <div key={s.step.en}>
              <div className="flex flex-wrap justify-between gap-2 text-sm">
                <span className={s.worse ? "text-rose-700" : "text-slate-800"}>
                  {s.step[lang]} {s.worse && e.retrieval.worse}
                </span>
                <span className="font-mono text-slate-600">
                  hit@1 {s.hit1}% · hit@3 {s.hit3}% · MRR {s.mrr.toFixed(2)}
                </span>
              </div>
              <div className="mt-1">
                <Bar value={s.mrr} max={1} tone={s.worse ? "bg-rose-400" : "bg-sky-600"} />
              </div>
            </div>
          ))}
        </div>
        <p className="mt-4 rounded-lg bg-slate-50 px-3 py-2 text-sm text-slate-700">{e.retrieval.finding}</p>
      </Card>

      <Card>
        <h2 className="text-xl font-semibold">{e.judge.title}</h2>
        <p className="mt-1 text-sm text-slate-600">{e.judge.text}</p>
        <div className="mt-4 grid gap-4 md:grid-cols-2">
          {judgeVersions.map((j) => (
            <div key={j.version.en} className="rounded-xl border border-slate-200 p-4">
              <p className="font-medium">{j.version[lang]}</p>
              <p className="mt-3 text-sm text-slate-600">
                {e.judge.contradictions}: {j.contradicts}%
              </p>
              <Bar value={j.contradicts} />
              <p className="mt-2 text-sm text-slate-600">
                {e.judge.additions}: {j.adds}%
              </p>
              <Bar value={j.adds} />
              <p className="mt-2 text-xs text-slate-400">{e.judge.perExample(j.seconds)}</p>
            </div>
          ))}
        </div>
      </Card>

      <Card>
        <h2 className="text-xl font-semibold">{e.answers.title}</h2>
        <p className="mt-1 text-sm text-slate-600">{e.answers.text}</p>
        <ul className="mt-3 space-y-1.5 text-sm">
          {answerChecks.map((check) => (
            <li key={check.en} className="flex gap-2">
              <span className="text-emerald-600">✓</span>
              <span>{check[lang]}</span>
            </li>
          ))}
        </ul>
        <p className="mt-4 text-sm text-slate-700">{e.answers.result}</p>
      </Card>

      <Card>
        <h2 className="text-xl font-semibold">{e.bugs.title}</h2>
        <ul className="mt-3 space-y-3">
          {issuesFound.map((issue) => (
            <li key={issue.title.en} className="rounded-lg border border-slate-200 px-4 py-3">
              <p className="font-medium text-slate-900">{issue.title[lang]}</p>
              <p className="mt-1 text-sm text-slate-600">
                {e.bugs.fix}: {issue.fix[lang]}
              </p>
            </li>
          ))}
        </ul>
      </Card>
    </div>
  );
}
