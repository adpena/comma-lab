# P0 #1001 triage — the live population is verified, the CONFIRMED split is NOT

**Author:** MAIN · 2026-08-10 · `[static analysis, no scorer]` · `score_claim=false`

## What is MEASURED

1. **pr1's census reproduces independently.** Running the landed
   `audit_measure_and_discard_payload` from a clean invocation:
   repository **1,819 findings / 51,009 `.py` examined** (exact match) and, after applying the
   census's own exclusions (`results`, `tests`, `fixtures`), live scope **1,072 / 6,390** (exact
   match). Detector sha `7dc700c8d5513b23ccdc4db9303bfad302727d9b899590cf825ef138f1fba26f`.

2. **The 1,819 repo figure is inflated by ARCHIVED COPIES.** The 4 roots contain
   `experiments/results/**`, which holds thousands of run-artifact script copies. The honest
   live-code population is **1,072**. Quote that one.

3. **MAIN's earlier 427 / 1,289 figures were UNDERCOUNTS, not upper bounds.** I published them as
   upper bounds on the reasoning that `(root, file)` keying splits one payload across N roots. That
   mechanism is real, but a larger one ran the other way: the same keying also COLLAPSED distinct
   materializations (one name rebound to different payloads counted once). Net 4.3× (repo) and
   7.6× (SSD) low. The correction direction was opposite to my prediction.

## What is NOT established — and why I am not claiming it

I hand-read 37 sampled call sites (seeded stratified-random, `seed=20260810`, proportional over
13 producer×dir strata) and found **three** classes, not two:

| class | shape | is it the ANS defect? |
|---|---|---|
| CONSUMED | payload flows into a larger structure that IS written (`payload = header + body`) | **No** — safe |
| SIZING_HELPER | function exists only to return a size (`def lz(b): return len(lzma.compress(b))`) | **No** — owed a waiver |
| GENUINE | materialized, measured, dropped | **Yes** — retrofit target |

I then built an AST classifier for those classes and **validated it against the 37 hand labels
before quoting any population. It agreed on 18/37 = 49%.** Its full-population output
(CONSUMED 826 / GENUINE 175 / SIZING_HELPER 71) is therefore **NOT REPORTED AS A RESULT**. A
classifier that loses its own control does not get to publish a number, and tuning it to match
37 labels I wrote myself would be fitting the instrument to the answer.

## The precise diagnosis the failure produced (this is the usable output)

Both class boundaries are wrong in a *nameable* way:

1. **The gate asks "was it PERSISTED?" when the deciding question is "does it reach DISK?"**
   `compressed` passed to `_sha256_bytes(compressed)` or to `RangeDecoder(comp)` for a decode-verify
   is *used*, so a use-based rule says CONSUMED — but sha256 and round-trip checks are
   **measurements**, and the bytes still die. This is the ANS anchor exactly: it also recorded a
   verification, and the payload was still lost. **Consumption by another measurement is not
   retention.** The rule must be reachability-to-a-writer, not use-vs-unuse.

2. **`SIZING_HELPER` is not structurally detectable by a one-line-return test.** Real instances
   carry guards (`... if raw else 0`), multi-statement bodies, and no return annotation
   (`ddm_kl1:29`, `dykstra_legal_frame:370`, `lane_ground_factorization:359/521`). Detecting it
   needs the call-site question — *does any caller keep the bytes?* — not the callee's shape.

## Routing

Owner: **#1001 / the `ddm_pr1` successor.** The retrofit ordering pr1 queued (TZ1 → PK2 → LV1) is
unaffected — those were selected as coder races, which is the GENUINE shape by inspection. What
changes is the **census semantics**: the live 1,072 is a detector-finding count, and the confirmed
violation count remains **UNMEASURED**. `population_semantics` in the census already says
"bounded detector findings, not adjudicated confirmed runtime violations" — that label is correct
and must not be dropped when the number is quoted.

Persisted (P0 compliance — the findings list is itself a payload):
`/Volumes/VertigoDataTier/pact/ddm_main_triage_20260810/live_scope_findings.jsonl`
(213,971 B, sha256 `a47818d081d3bdc5…`), plus `live_source_findings.jsonl` (482,631 B, sha256
`cb41512b61636003bec0e54560de7b533e0e76c3b42b8e571dbbe1e5086395b3`), `sample_context.txt`,
`classified.jsonl` (retained WITH its failing-control label so nobody re-derives it as truth).
