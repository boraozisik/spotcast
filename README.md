# SpotcastAI

**Is this spot right for my level, at this hour?**
SpotcastAI is an AI-powered forecasting system for water sports on the Turkish coast: windsurf,
kitesurf, stand-up paddle (SUP) and sea kayak, each at beginner, intermediate and advanced level.
It turns raw wind and wave forecasts into clear, hour-by-hour verdicts and the best time windows
to go, and answers free-form questions through a tool-calling AI assistant.

[![CI](https://github.com/boraozisik/spotcast-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/boraozisik/spotcast-ai/actions/workflows/ci.yml)
[![Publish site data](https://github.com/boraozisik/spotcast-ai/actions/workflows/publish.yml/badge.svg)](https://github.com/boraozisik/spotcast-ai/actions/workflows/publish.yml)

**Live:** [spotcastai.netlify.app](https://spotcastai.netlify.app/) (English / Turkish)

## Features

- **Hour-by-hour verdicts for 12 profiles.** Every forecast hour is labelled _good_, _marginal_ or
  _not suitable_ for a sport and skill level. One model serves all profiles: it sees conditions
  relative to each profile's limits, so it also works for a profile it was not trained on.
- **Hard safety guardrails.** Safety rules sit on top of the model and can never be overridden:
  a storm is never "good", whatever the model predicts.
- **Offshore wind.** Every place knows which way its beach faces, so wind blowing from the land
  out to sea is recognised and limited per profile (never for kitesurf). Sheltered bays, where a
  drifting rider reaches another shore, are marked as such only when confirmed by local knowledge,
  and every answer explains it.
- **Best time windows.** An optimizer picks the best spot and session window under the user's
  constraints (hours, weekend, minimum session length), only at spots where the sport is done.
- **Places near me.** "I live in Izmir, where should I go on Saturday?": the assistant ranks every
  known place within a distance; the website does the same with the browser's location, which
  never leaves the page.
- **Any point on the coast.** 29 places on the Turkish coast, each checked against the OpenStreetMap
  coastline to be on the water. Places outside the curated spots are assessed with the same model
  and rules, with explicit warnings when there is no local guide.
- **Explanations with sources.** A locally hosted LLM explains each recommendation in Turkish,
  citing spot guides and safety documents retrieved from a vector database.
- **AI assistant and MCP server.** A tool-calling agent answers questions such as "Is Saturday in
  Mersin OK for windsurfing?". Code checks every tool call: made-up coordinates and a day other
  than the one the user named are sent back to the model. The same tools are available to other
  AI clients through MCP.
- **Uncertainty notes.** Forecasts five or more days ahead and points without wave data are flagged
  automatically.
- **Continuously evaluated.** Retrieval quality, agent answers and an LLM judge are measured on
  fixed test sets.

Sports, levels and places are data, not code (`profiles.toml`, `locations.toml`): adding one is a
new block in a file, checked when it is loaded.

## Architecture

```mermaid
flowchart LR
    OM[Open-Meteo<br/>batched, rate-limited] --> DATA[pandas<br/>Parquet / PostgreSQL]
    CAT[locations.toml<br/>profiles.toml] --> CLS
    DATA --> CLS[Suitability model<br/>scikit-learn + guardrail]
    DATA --> FC[Wind forecasting<br/>XGBoost / LSTM]
    CLS --> REC[Recommender<br/>optimization]
    REC --> API[FastAPI]
    KB[Knowledge base<br/>guides + OCR'd signs] --> RAG[pgvector +<br/>Turkish full-text<br/>hybrid search]
    RAG --> LLM[Local LLM<br/>Ollama qwen3:8b]
    REC --> LLM
    LLM --> AG[Agent<br/>tool calling, LangGraph]
    AG --> MCP[MCP server]
    REC --> PUB[Batch job<br/>GitHub Actions, every 6 h]
    PUB --> SITE[Next.js on Netlify<br/>static, no backend]
```

**The public website has no backend.** A scheduled GitHub Actions job runs the model, guardrail
and recommender every six hours and publishes the results as JSON to the `site-data` branch. The
static Next.js site reads them in the browser. With no server, there is nothing to attack, nothing
to go down and nothing to pay for (batch inference). The LLM components run on a locally hosted
model; recorded assistant conversations are shown on the site.

## Evaluation highlights

**Retrieval** (18 questions with known answer sections, top 3):

| Change                                               | hit@1   | hit@3    | MRR      |
| ---------------------------------------------------- | ------- | -------- | -------- |
| Baseline (vector search)                             | 72%     | 78%      | 0.74     |
| Content restructured (mixed sections split by topic) | 83%     | 94%      | 0.87     |
| Hybrid search, first version                         | 78%     | 94%      | 0.85 ↓   |
| Hybrid + common-word filter (IDF-like)               | 83%     | 94%      | 0.88     |
| + Query expansion (knots → Beaufort, in code)        | **89%** | **100%** | **0.94** |

Restructuring the content had the largest impact. The first hybrid version reduced quality
because PostgreSQL's ranking ignores how common a word is; filtering common words fixed it.

**LLM judge** (10 hand-labelled answer/source pairs): a single verdict agreed with the labels in
80% (contradictions) and 70% (unsupported additions) of cases. Splitting answers into claims and
deciding in code: 100% / 100%.

**Agent answers** (11 scenarios with frozen tool results, run several times each): 8/10 on the first
run. The evaluation caught the agent mapping "yarın" (tomorrow) to the wrong date in 3 of 3 runs;
after the fix, 30/30 runs pass. A "places near me" scenario added later caught the model claiming
that a place without a local guide had one: distances and the guide note are now written by code
(3/3).

**One model for every profile** (test year 2025, macro F1): one tree per profile 1.000; one tree
with raw wind and limit columns 0.742 (0.653 on an unseen profile, because a tree compares one
column at a time); one tree with relative features ("wind minus this profile's limit") 0.998, and
0.998 on a profile left out of training. With the offshore rule, tree depth was chosen by
measurement: depth 5 reached 0.933, depth 7 reaches 1.000 with 10 leaves.

**Data quality.** Checking the forecast points against the OpenStreetMap coastline showed that 15
of 29 were on land (one on a 76 m hill), unnoticed because the coarse wave model still returned
data. Training on one place per sea (Mediterranean, Aegean, Black Sea) instead of three Aegean
spots took the hours outside the training range from 8-10 per run to 0.

**48-hour wind forecasting** (MAE in knots, test year 2025): persistence 2.91, climatology 2.46,
XGBoost 2.43, LSTM 2.40. A horizon study (1 h: 0.69 → 48 h: 2.43) shows that the remaining error is
an information limit of local history, not a modelling limit.

## Design principles

- **Code computes, the LLM only words it.** Dates, comparisons and unit conversions are done in code.
- **Critical rules live in code, not in prompts.** The user's sport and level are fixed by the
  system and hidden from the model; tool calls with made-up coordinates or the wrong day are sent
  back; facts like "no local guide here" and what offshore means are written by code.
- **Safety text never comes from the LLM.** Cautions, uncertainty notes and source quotes are fixed
  texts added by the system.
- **Measured, not assumed.** Every change is checked against a baseline on a fixed test set.
- **Human review.** New knowledge documents enter as drafts and become searchable only after review;
  sheltered bays are marked only after local confirmation; a retrained model reaches the site only
  after approval.

## Tech stack

Python 3.12 · uv · pandas · scikit-learn · XGBoost · PyTorch · FastAPI · Pydantic ·
PostgreSQL + pgvector · Ollama (qwen3:8b, bge-m3, qwen2.5vl:7b) · LangGraph · MCP · MLflow ·
Docker · GitHub Actions · Next.js · TypeScript · Tailwind CSS · Netlify

## Repository layout

| Path                                                                   | Contents                                                |
| ---------------------------------------------------------------------- | ------------------------------------------------------- |
| `src/spotcast/data.py`, `config.py`                                    | Open-Meteo data: batched requests, rate limiting, cache |
| `src/spotcast/locations.toml`, `locations.py`, `shore.py`              | Places, shore direction, offshore rule                  |
| `src/spotcast/profiles.toml`, `profiles.py`, `labels.py`, `model.py`   | Activity profiles, labels, classifier, guardrail        |
| `src/spotcast/forecasting.py`, `wind_model.py`, `lstm_model.py`        | 48-hour wind forecasting                                |
| `src/spotcast/recommend.py`, `api.py`                                  | Optimizer and FastAPI service                           |
| `src/spotcast/llm.py`, `explain.py`, `rag.py`                          | LLM client, explanations, hybrid retrieval              |
| `src/spotcast/vision.py`, `ocr.py`                                     | Sea-state estimation from photos, OCR of signs          |
| `src/spotcast/tools.py`, `agent.py`, `agent_graph.py`, `mcp_server.py` | Tools, agents, MCP server                               |
| `src/spotcast/evals/`, `evals/`                                        | Evaluation code and test sets                           |
| `src/spotcast/tracking.py`                                             | MLflow experiment tracking                              |
| `src/spotcast/publish.py`, `monitoring.py`, `retrain.py`               | Batch job, data checks, champion/challenger retraining  |
| `scripts/shore_facing.py`                                              | Places checked against the OpenStreetMap coastline      |
| `knowledge/`                                                           | Spot guides, Beaufort scale, safety rules (RAG sources) |
| `web/`                                                                 | Next.js website (EN/TR)                                 |
| `tests/`                                                               | 300+ tests (pytest, no network or LLM needed)           |

## Running locally

```bash
uv sync                                   # Python dependencies
cp .env.example .env                      # then set your own database password
docker compose up -d db                   # PostgreSQL + pgvector
uv run spotcast                           # history, labels, training (tracked in MLflow)
uv run spotcast-api                       # http://127.0.0.1:8000/docs
```

LLM features need [Ollama](https://ollama.com) with `qwen3:8b` and `bge-m3`:

```bash
uv run spotcast-index                     # build the knowledge index
uv run spotcast-agent "Cumartesi Mersin'de rüzgâr sörfü uygun mu?"
uv run spotcast-eval answers              # agent evaluation
```

Website with local data:

```bash
uv run spotcast-publish --out web/public/data
cd web && npm install && npm run dev
```

Tests and checks:

```bash
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
```

## Data and licenses

Weather and wave data: [Open-Meteo.com](https://open-meteo.com/), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (historical data from ERA5 /
Copernicus). Requests stay within Open-Meteo's free-use limits. Coastline data used to check the
places and their shore direction: © [OpenStreetMap](https://www.openstreetmap.org/copyright)
contributors, ODbL. Photos and signs used in the vision and OCR experiments are public domain and are not
included in this repository.

SpotCast forecasts are guidance only. Always check the conditions on site before going out.
