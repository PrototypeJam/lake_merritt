# Live Agent Evaluation — interactive demo

**▶ https://prototypejam.github.io/lake_merritt/demo/**

A self-contained, offline walkthrough of what it looks like to run an AI agent loop, watch it emit
OpenTelemetry, and turn that telemetry into an evidence-backed evaluation.

Everything you see is **real captured data replayed** — 190 real events from a public agent
transcript, and real evaluation records. No model is called, no key is needed, nothing costs
anything. It is a faithful playback, not a live system and not a mock-up of invented numbers.

## Run it locally

Download the `demo/` folder and open `index.html`. That's it — no build step, no server, no
install. All data loads via `<script src>` rather than `fetch()`, so it works straight from
`file://`.

## What it shows

1. **Setup** — pick a mode (demo/live), a **loop model** and, separately, a **judge model**;
   choose how much to run (smoke / partial / full question / multi-question) and which question.
2. **Live run** — the actual loop output streaming by (assistant turns, tool calls, artifacts
   written), alongside live telemetry, deterministic checks, and evidence meters.
3. **Evaluation** — deliberately *not* a number that wiggles during the run. Evidence accumulates
   live; the rubric score resolves once, at completion, after a visible judging sequence.
4. **Results & Compare** — a citation-backed report, plus judge variance, a three-way producer
   comparison, and eval-pack regression drift.

## The idea it's demonstrating

A rubric score is a judgment about a *whole* session, so it can only be honest at the end. But
plenty of useful signal — completeness, budget, required outputs, policy checks, accumulating
evidence — is deterministic and can stream live. Separating those two layers is the whole design.

Because scope is a real control, you can see the point directly: the same run scored at smoke,
partial, and full scope produces **59.5 → 71.0 → 75.5**. A partial run is not comparable to a
full one, and the UI never pretends otherwise.

## Honest limits

- **Diagnostic only.** The composite here is derived from observed evidence, not a real
  LLM-as-judge call. In the integrated system this is replaced by the Lake Merritt rubric judge.
- **Not an official score** from any evaluation body, and not a benchmark result.
- **One question has demo data** (OQ-130). The others render with their real parameters but are
  visibly disabled.
- The scores shown are illustrative of the *mechanism*, and should not be cited as a measurement
  of any particular system's quality.

## Files

```
demo/
├── index.html          the whole UI
└── assets/
    ├── styles.css
    ├── data.js         questions, model lists, scopes, evaluation records
    ├── run.js          190 real captured events (loaded as a script, file:// safe)
    ├── run_oq130.json  the same event data as plain JSON, for reuse
    └── app.js          replay engine, meters, judging sequence, report, comparisons
```

Part of [Lake Merritt](https://github.com/PrototypeJam/lake_merritt). Apache-2.0.
