# ddm_sw1 — portable-paths + local-info scrub + secrets protection (public-repo hygiene)

## MANDATE

Operator directive 2026-08-20 verbatim: *"Let's also scrub those local paths and ensure we are
always using portable paths and protecting local secrets and info."* The repo origin
`github.com/adpena/comma-lab` is PUBLIC — every committed byte is published. The oc2 census
(2026-08-20, `.omx/state/operator_p0_ledger.jsonl` p0_jg5_custody row) measured **365 committed
files containing `/Volumes/` and 399 containing `/Users/adpena`**. The three packet documents are
clean (pq8 scanned them); the surrounding corpus is not. This arm executes the scrub, makes
portable paths STRUCTURAL, and runs a REAL secrets pass.

## PHASES (in order; each emits a receipt)

**P0 — fresh census + classification (measure before cutting).** Re-run the count on the live
tree (never carry oc2's numbers — they are 10+ hours old). Classify every hit into:
  (a) TOOLS/CODE (`src/`, `tools/`, `experiments/`, `scripts/`) — absolute paths here break
      reproduction on any other machine; portable-ize through the canonical resolver.
  (b) RESEARCH MEMOS (`.omx/research/`, docs) — hygiene class; textual scrub.
  (c) SHA-PINNED FILES — any file whose sha256 is recorded in a seal, ledger row, receipt, or
      MANIFEST. **DO NOT EDIT** (editing breaks the pin chain; Catalog #110/#113 append-only).
      List them with the pinning consumer named; supersede-don't-mutate where cure is needed.
  (d) GITIGNORED/LIVE_STATE — not public, out of scope (verify actually ignored).

**P1 — structural portable-path surface (RECALL FIRST).** `src/tac/checkpoint_retention.py` and
`src/tac/payload_retention.py` already carry the tier paths; CLAUDE.md's disk-rules section
defines the waterfall order. EXTEND the existing surface — do NOT build a twin. One canonical
resolver: env-var roots (`PACT_TIER1` → `/Volumes/VertigoDataTier/pact`, `PACT_TIER2` →
`/Volumes/APDataStore/pact`, `$HOME`) with the current values as defaults so NOTHING breaks on
this machine, and writers that emit placeholder forms for durable artifacts. Migrate class-(a)
callers to it.

**P2 — the sweep.** Mechanical rewrite of class (a) + (b): `/Users/adpena` → `~` or `$HOME`;
`/Volumes/VertigoDataTier/pact` → `$PACT_TIER1`; `/Volumes/APDataStore/pact` → `$PACT_TIER2`.
Emit a mapping receipt (old→new per file, counts per class) so the sweep is reversible and
auditable. CONSUMER CHECK before rewriting any memo a tool parses paths FROM (evidence-pointer
readers, recall tools) — if a reader resolves literal paths, either teach it the placeholders in
the same landing or leave that file in class (c) with rationale.

**P3 — the guard (two-landing).** Warn-only preflight gate refusing NEW/modified committed files
carrying `/Users/<name>` or `/Volumes/` absolute paths (same-line waiver
`# ABSOLUTE_PATH_OK:<rationale>`; placeholder rationales rejected). EXECUTED positive control —
plant a violation, watch it fire, remove it. Historical files pass via the P2 sweep, not a
blanket exemption.

**P4 — secrets + local info.** (1) gitleaks (or equivalent) over the WORKING TREE + report-only
history SCAN. **#1086 binding lesson: the prior positive control was VACUOUS** (planted gitleaks'
own allowlisted doc-example key). Plant a synthetic secret matching a LIVE rule, verify it FIRES,
then remove — the control must be able to fail. (2) Fleet-info check: verify no committed file
carries Tailscale IPs / fleet hostnames (fleet.local.toml must be gitignored — verify). (3) Any
REAL secret found in history → immediate P0 report to MAIN; the cure is ROTATION, never history
rewrite.

## HARD CONSTRAINTS

- **NO git history rewrite** — pushed shas are pinned by receipts everywhere.
- NO edits under `submissions/robust_current/jg5_sub015_runtime/` (seal-pinned),
  `generations/gen5*` (pq9 owns the packet), or `upstream/` (read-only).
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; receipts on APDataStore if bulky.

## OPTIMAL FORM

- Family reference: the canonical two-landing cure chain (fix + structural gate + executed
  positive control) at its landed form — the #1115 seal contract's 36 executed controls, commit
  `361608c875`. Provenance pins: oc2 census source = `.omx/state/operator_p0_ledger.jsonl`
  p0_jg5_custody row (365 //Volumes// + 399 //Users/adpena committed files); custody commit
  `2d61b51988` (the seal-pinned tree this arm must NOT touch); tier-path surfaces
  `src/tac/checkpoint_retention.py` + `src/tac/payload_retention.py` at HEAD.
- SCOPE reductions permitted, declared per row (e.g. sweep class (a) fully + class (b) top-N by
  sensitivity with the remainder counted). MECHANISM reductions FORBIDDEN: no regex-only "scan"
  without classification; no gate without an executed positive control; no secrets pass whose
  control cannot fail.
- **PRIOR-LAW PREDICTION (falsifiable):** `/Users/adpena` is already public via the repo URL, so
  most hits are LOW-sensitivity; the real value is (a)-class portability + the structural guard.
  Predict ≥80% of hits land in class (b) memos. FALSIFIER: any hit containing a credential,
  token, or fleet IP — that flips this from hygiene to P0 and MAIN must be told immediately.

## DELIVERABLE

`.omx/research/ddm_sw1_portable_paths_secrets_scrub_20260820.md` — per-phase rows {what EXECUTED
· scope · counts · receipt path}, the class-(c) do-not-touch list with pinning consumers, the
mapping receipt sha, gate + control evidence, secrets verdict. End with the own-vehicle frontier
line.
