/* Real captured data — loaded via <script src> so it works from file:// with no server. */
window.DEMO = {
  meta: {
    source: "LegalQuants public baseline repo (real OQ-130 transcript) + real eval records",
    disclaimer: "SIMULATED PLAYBACK OF REAL DATA — no live model calls. Diagnostic, not an official LegalQuants score."
  },
  questions: [
    { id: "OQ-008", title: "Spec the legal-markdown standard", bar: 82, budget: 133, available: false },
    { id: "OQ-009", title: "Architect agent memory for long-running bots", bar: 88, budget: 226, available: false },
    { id: "OQ-010", title: "Working OCR pipeline for hard legal scans", bar: 78, budget: 133, available: false },
    { id: "OQ-026", title: "Methodology layer for legal data before the LLM sees it", bar: 83, budget: 133, available: false },
    { id: "OQ-028", title: "Client-specific 'mini brain' MCP", bar: 83, budget: 133, available: false },
    { id: "OQ-040", title: "Prompt-injection tester for legal agents", bar: 92, budget: 237, available: false },
    { id: "OQ-112", title: "Build a messy native eml/msg test corpus", bar: 83, budget: 133, available: false },
    { id: "OQ-115", title: "Build a legal-task-aware model router", bar: 82, budget: 133, available: false },
    { id: "OQ-122", title: "Ship an affordable, MCP-accessible statutes source", bar: 88, budget: 196, available: false },
    { id: "OQ-130", title: "A friction-free delivery layer for interactive HTML legal explainers", bar: 82, budget: 133, available: true }
  ],
  loopModels: [
    { id: "claude-sonnet-5", label: "Claude Sonnet 5", note: "the model LegalQuants used for its baseline" },
    { id: "claude-opus-5", label: "Claude Opus 5", note: "stronger, slower, costlier" },
    { id: "claude-haiku-4-5", label: "Claude Haiku 4.5", note: "fast + cheap" },
    { id: "gpt-5.1", label: "GPT-5.1", note: "cross-family comparison" }
  ],
  judgeModels: [
    { id: "claude-opus-5", label: "Claude Opus 5", note: "recommended judge — best discrimination" },
    { id: "claude-sonnet-5", label: "Claude Sonnet 5", note: "cheaper judge" },
    { id: "gpt-5.1", label: "GPT-5.1", note: "cross-family judge (reduces same-family bias)" }
  ],
  scopes: [
    { id: "smoke",   label: "Smoke",        detail: "~10 turns · proves the pipe",        turns: 10,  est: "~$0.05" },
    { id: "partial", label: "Partial",      detail: "~40 turns · substantive slice",      turns: 40,  est: "~$0.30" },
    { id: "full",    label: "Full question", detail: "full turn budget · comparable run", turns: 133, est: "~$1.20" },
    { id: "multi",   label: "Multi-question", detail: "2+ questions · leaderboard view",  turns: 266, est: "~$2.50+" }
  ],
  // real eval records (same ones the comparison report uses)
  evalRecords: [
    { id:"r1", producer:"frozen-baseline", pack:"v1", composite:71.0, dims:{orchestration:7.0,tooling:8.0,self_awareness:6.5,fidelity:7.0} },
    { id:"r2", producer:"frozen-baseline", pack:"v1", composite:68.0, dims:{orchestration:6.5,tooling:8.0,self_awareness:6.0,fidelity:6.5} },
    { id:"r3", producer:"frozen-baseline", pack:"v1", composite:73.0, dims:{orchestration:7.5,tooling:8.0,self_awareness:7.0,fidelity:7.0} },
    { id:"r4", producer:"fresh-loop-A",    pack:"v1", composite:79.0, dims:{orchestration:8.0,tooling:8.5,self_awareness:7.5,fidelity:7.5} },
    { id:"r5", producer:"fresh-loop-B",    pack:"v1", composite:66.0, dims:{orchestration:6.5,tooling:7.0,self_awareness:6.5,fidelity:6.5} },
    { id:"r6", producer:"frozen-baseline", pack:"v2", composite:74.0, dims:{orchestration:7.5,tooling:8.0,self_awareness:7.0,fidelity:7.5} }
  ],
  clauses: {
    explainer:["explainer","html","interactive","render"],
    delivery:["deliver","link","url","share","send"],
    per_client:["per-client","hashed","token","unique link","recipient"],
    responsive:["mobile","desktop","viewport","responsive"],
    link_lifecycle:["expire","revoke","expiry","one-time","view cap"],
    confidentiality:["confidential","who can reach","access","prefetch","leak"]
  }
};
