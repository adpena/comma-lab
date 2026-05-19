# comma-lab — Public Agent Operating Charter

> This is the public-facing summary of agent operating discipline for the
> comma-lab research environment. The internal `CLAUDE.md` contains additional
> operator-specific operational state (Tailscale fleet IPs, local-path
> conventions, session-history references) that is not appropriate for OSS
> consumption but is required by the in-repo agent harness.

## Mission

A dual-track research lab for the comma video compression challenge with
research-grade engineering rigor and a production-quality engineering surface.
Code, contracts, ledgers, and CI gates here are intended to be reusable beyond
this specific challenge by comma.ai / openpilot / external contributors.

## Architecture

- `src/tac/` — Task-Aware Compression library + runtime contracts. The reusable
  Python implementation: codec primitives, archive grammars, payload parsers,
  scorer/eval contracts, byte profilers, planning primitives, visualization
  primitives. Now published as standalone OSS at
  [`adpena/tac`](https://github.com/adpena/tac) (MIT licensed) for external
  reuse independent of the research environment.
- `src/comma_lab/` — research-state custody, public-frontier intake, hosted
  supplement builds, provider ledgers, recovery audits.
- `tools/` — operator-facing CLI surfaces (audit tools, dispatch wrappers,
  preflight gates).
- `experiments/` — experiment runners + dispatch harnesses.
- `submissions/` — frozen submission packets (one per submitted PR).
- `reverse_engineering/` — curated public-PR deconstruction artifacts.
- `upstream/` — pinned upstream snapshot (READ-ONLY by non-negotiable rule).

## Core engineering principles

1. **Apples-to-apples evidence discipline.** Every score claim is tagged with
   its measurement axis: `[contest-CUDA]` (NVIDIA GPU on Linux), `[contest-CPU]`
   (x86_64 Linux), `[macOS-CPU advisory]` (Apple Silicon — non-promotable),
   `[MPS-PROXY]` (Apple GPU — research signal only). Cross-axis inference is
   forbidden.

2. **eval_roundtrip = True everywhere.** Every training path simulates the
   contest's uint8 bottleneck (384 → 874 → uint8 → 384). Without it the
   proxy-auth gap is 2-11x on PoseNet.

3. **EMA shadows ship.** Inference checkpoints come from the EMA shadow, not
   the live final-epoch weights. Per-method engineering choice; default decay
   = 0.997 for weight EMAs.

4. **MPS auth eval is NOISE.** Local Apple Silicon GPU evaluation drifts 23×
   on PoseNet vs the contest's CUDA scorer. Never report MPS-derived scores as
   authoritative. Apple Silicon CPU is acceptable as advisory-only proxy when
   tagged `[macOS-CPU advisory]`; promotion requires Linux x86_64 verification.

5. **Strict-mode preflight catalog.** 295+ catalog gates structurally extinct
   reproducing bug classes (silent device fallbacks, dead CLI flags, archive
   non-determinism, scorer-at-inflate violations, custody-evidence corruption,
   stale state in committed artifacts). The catalog is documented inline in the
   internal `CLAUDE.md` "Meta-bug class catalog" section.

6. **Canonical 4-layer pattern for shared state.** Every persisted state file
   uses fcntl-locked JSONL append-only writes via a canonical helper module,
   with strict-load fail-closed loaders and corruption quarantine.

7. **Subagent commit serialization.** Multi-subagent commits go through
   `tools/subagent_commit_serializer.py` with `--expected-content-sha256`
   working-tree validation. This extincts the commit-swap bug class where
   concurrent subagents could absorb each other's edits.

8. **Council-grade design decisions.** Non-trivial tradeoffs require sextet
   council deliberation (Shannon + Dykstra + Yousfi + Fridrich + Contrarian +
   Assumption-Adversary) with verbatim dissent preserved. Council outputs land
   to a queryable continual-learning posterior.

9. **HISTORICAL_PROVENANCE append-only.** Forensic artifacts (recovery
   metadata, council deliberations, dispatch claim ledgers, contest auth-eval
   JSONs) are append-only. In-place mutation of historical state is forbidden.

10. **Canonical-vs-unique decision per layer.** Every substrate scaffold
    documents per-layer canonical-helper-adoption vs unique-implementation
    decisions explicitly. Default-canonical reflex is a documented anti-pattern
    that produces a structural plateau across substrates.

## Track separation

- **Track A: `submissions/exact_current`** — the currently-published contest
  workflow. Preserve transparency; demote immediately if upstream invalidates.
- **Track B: `submissions/robust_current`** — stricter rule-faithful
  interpretation. Add sparse residuals before heavier learned components.

## Mutation frontier

Allowed (without explicit human approval): `configs/`, `docs/`, `prompts/`,
`src/comma_lab/`, `submissions/robust_current/`, `runtime-rs/`, `cuda/`,
`jax/`, `mojo/`, `.omx/`, `.ralph/`, `.agents/`, `reports/`, `experiments/`.

Forbidden without explicit human approval: pinned upstream snapshot,
`submissions/exact_current/inflate.py` and `inflate.sh`, `start.sh`,
`LICENSE`, `THIRD_PARTY_NOTICES.md`.

## Public Disclosure Hygiene

- Credentials, private infrastructure URLs, local absolute paths, raw provider
  logs, unpublished operator state, and account metadata stay out of public
  surfaces.
- Operator state (`.omx/state/*`, `.omx/logs/*`, provider transcripts,
  per-machine tracker files) is gitignored and never committed.
- Detailed OSS / paper writeups are allowed when deliberately promoted, with
  sanitization into release manifests or dated research ledgers first.

## Related Open Source

The reusable codec, predictor, and search primitives are open-sourced as
standalone Python package [`adpena/tac`](https://github.com/adpena/tac) (MIT
licensed). The comma-lab repo here is the full research environment + state +
experimental scaffolding; `tac` is the curated production extract suitable for
OSS adoption.

---

For agents operating inside the local development environment: the internal
`CLAUDE.md` contains the full operating instructions and is the source of truth
for harness behavior. This `CLAUDE_PUBLIC.md` is a sanitized summary for public
review and OSS contributors.
