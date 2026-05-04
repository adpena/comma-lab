---
name: Ara paradigm partial pivot LANDED — best-writeup-prize differentiator
description: 2026-04-29 PM. arXiv 2604.24658 = "The Last Human-Written Paper: Agent-Native Research Artifacts" (Ara). 4-layer artifact (logic/src/trace/evidence). Subagent F + G landed PARTIAL PIVOT: docs/paper/ara/ skeleton + tools/ara_compile.py + 10 evidence files + 13/13 smoke tests. Seal Level 1 passes 0 errors. Next: install ARA skills locally + run rigor-reviewer.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Paper paradigm**: arXiv 2604.24658 "The Last Human-Written Paper: Agent-Native Research Artifacts" (Ara). Recasts research output from narrative document → machine-executable knowledge package with 4 layers:
- `logic/` — problem, claims, experiments, related-work (cognitive layer)
- `src/` — executable code with kernel/repo modes (physical layer)
- `trace/` — exploration DAG with decisions / dead-ends / pivots (provenance)
- `evidence/` — raw outputs only (ground truth)

Ara Compiler (4 stages) + Live Research Manager (typed events) + Ara Seal (3 verification levels). PaperBench gains: QA 72.4% → 93.7%, reproduction 57.4% → 64.4%.

**Companion repo**: github.com/Orchestra-Research/Agent-Native-Research-Artifact + AI-Research-SKILLs (3 skills: compiler, research-manager, rigor-reviewer). MIT-licensed, JS installer, ~3500 LOC.

**PARTIAL PIVOT landed in 4 commits (Subagents F + G)**:
- `e596a54a` — Subagent F: tools/ara_compile.py scaffold + auto-compiled evidence
- `47fd1ba2` — Subagent F: hand-curated Ara skeleton (12 layer files) + 13-test smoke suite
- `29df460c` — Subagent G: tools/ara_compile.py landed + 10 evidence files + Seal Level 1 passing 0 errors / 9 warns

**Files added (under docs/paper/ara/)**:
- `PAPER.md` — root manifest with disclosure-policy frontmatter
- `RECOMMENDATION_20260429.md` — pivot recommendation + 6-pass hand-off
- `logic/{problem,claims,experiments,related_work}.md` — 10 falsifiable claims (C1-C10) bound to 10 experiments (E1-E10)
- `src/index.md` — kernel-mode physical layer
- `trace/{exploration_tree.yaml,events.jsonl,seal_report.json}` — 3-era DAG + 408 classified events
- `evidence/{era1,era2,jacobian,preflight,cnn_residual}/` — 10 evidence files (2 real contest_auth_eval.json + 1 transcribed MPS-CUDA drift summary + 1 council findings markdown + 6 _pending_provenance placeholders)

**Next-up actions queued (task #220)**:
1. `npx @orchestra-research/ara-skills install --all --local` — 5 min, $0
2. `/rigor-reviewer docs/paper/ara/` — 1h, produces level2_report.json with 6-dimension scores
3. ARA citation in writeup (already done by Subagent F in RECOMMENDATION_20260429.md)

**Strategic value**: Most other challenge entrants will not name the paradigm they implicitly violate. Self-grading against an external rubric (rigor-reviewer's level2_report.json) is a credible signal to the writeup-prize judges.

**How to apply**:
- After Subagent H (lane hardening) + Subagent I (Lane AL) land, dedicate 1.5h to ARA skills install + rigor-reviewer pass + writeup citation polish.
- Strategic-secrecy: PRIVATE_TERMS redaction enforced via `tools/ara_compile.py`. Cloudflare site (CLAUDE.md exception) can stay specific; Lane W / Lane Ω / DARTS-S secret levers remain redacted.
