# Codex findings — DDM E5 typed E4/WS1 exporter adapter — 2026-07-24

`research_only=true` · `score_claim=false` ·
`[macOS-CPU frozen-scorer advisory]` · pointer
`0.1910828242 [contest-CPU]` unchanged · MAIN landing review required.

## Verdict

**PASS the explicit typed WS1 admission route, source-byte reconstruction,
Brotli-Q11 packets, and both exact batch32 sealed endpoint reproductions.
BLOCK the Brotli-absent raw-LZMA1 fallback formulation because the
receiver-closed WS1 source grammar itself consumes Brotli.**

This is an instance-positive result for both materialized states and a
formulation-scoped fallback negative. It is not a contest score, promotion,
family negative, training authorization, paid dispatch, or pointer mutation.

## Typed admission, not a widened literal

The legacy `DDME1RuntimeExporterConfigV1` V15 path retains its sealed source,
state, name, run-id, and repository-relative-path literals. The new route is a
separate `DDME4WS1RuntimeExporterConfigV1` with:

- grammar version `ddm_ws1_receiver_closed_warm_start.v1`;
- an ordered, contiguous, gap-free stream partition;
- per-stream byte count, offset, SHA-256, and named receiver consumer;
- exact archive SHA and parser re-emit;
- strict packet member custody and source-state byte reconstruction.

No candidate SHA appears in exporter or receiver source. The two configs bind
the measured instances, while the reusable admission law binds grammar and
consumption rather than an allowlist.

## Exact primary results

| State | Source bytes / SHA | Brotli packet bytes / SHA | Rate delta vs unpacked | Exact batch32 d_seg | Exact batch32 d_pose |
|---|---|---|---:|---:|---:|
| `W_seg` | 138,031 / `264a09ab...81a9` | 130,870 / `1e45eb33...3368` | -7,161 B; -0.004768215963307869 | 0.024124510023328993 | 146.36493245487773 |
| `W_joint` | 138,801 / `5aa45850...433e` | 131,294 / `81ad65aa...164f` | -7,507 B; -0.00499860316108814 | 0.07051923116048177 | 36.618184751411334 |

Both packet parse-backs reconstruct the exact source bytes. Both standalone
receivers produce 3,662,409,600 raw bytes with the source receiver's measured
SHA:

- `W_seg`: `e4931dacb5e28494e480e6badcc3831ff3275093843d6988b411988f81f0d057`;
- `W_joint`: `9682164f0fc0b6aa0ea28d6f62cc4a712cb28fc0d4d8c9307eeac9a8815fb43b`.

The locked upstream `evaluate.sh` signature passed twice against final emitted
runtime source:

| State | Exit | Rounded harness d_seg | Rounded harness d_pose | Rounded advisory score | Wall clock |
|---|---:|---:|---:|---:|---:|
| `W_seg` | 0 | 0.02412450 | 146.36492920 | 40.76 | 1,582.157099 s |
| `W_joint` | 0 | 0.07051922 | 36.61817932 | 26.28 | 1,588.075822 s |

The separate frozen-scorer batch32 receipts—not rounded harness text—are the
authority for the exact endpoint equality above. They preserve 19 scorer rows
per state and now resume by validating row/raw bindings before reusing them.

## Raw-LZMA1 fallback disposition

The requested fallback row has no admissible packed byte count:

| State | Raw-LZMA1 fallback bytes | Disposition |
|---|---:|---|
| `W_seg` | — | `BLOCKED_FAIL_CLOSED` |
| `W_joint` | — | `BLOCKED_FAIL_CLOSED` |

This is not merely because the outer selected packet uses Brotli. In a
Brotli-absent environment, importing the WS1 receiver closure fails. After
guarding already-imported modules to isolate actual use, decoding the exact
source reaches:

`ddm_ws1_warm_start → carrier_compose → g1_worldsheet →
_decompress → brotli.decompress`.

Therefore an outer raw-LZMA1 packet would still have a hidden Brotli dependency
inside the reconstructed source grammar. Calling that dependency-closed would
be fake custody. The exporter now refuses this WS1 fallback before source work;
the legacy V15 E4 ImportError-only fallback remains unchanged.

Durable blocker logs:

- `source_import.stderr.txt` SHA
  `5e03ce496f2c2ddd6265413421e7bad54cb735e22ea76f662118c376e44410f2`;
- `inner_decode.stderr.txt` SHA
  `d9c5f67c7d5ec92066dcfa69be77633e6db5413caff91a209abc6bf1971efa25`;
- SSD root:
  `/Volumes/VertigoDataTier/pact/evidence/ddm_e5_e4_ws1_exporter_adapter_20260724/fallback_dependency_blocker_smoke`.

## Triality

- DSL: typed exporter config, grammar stream config, packet schema, and exact
  batch32 remeasure config.
- DAG: `FEED-ddm-e5-e4-ws1-exporter-adapter-20260724` records typed admission →
  exact source reconstruction → source receiver decode → preserved raw stages →
  frozen scorers, plus the fallback-refusal edge.
- Equations: `Admit_g`, receiver identity, the exact rate law
  `25*(packed-unpacked)/37,545,489`, and the fallback closure predicate are
  recorded in
  `.omx/research/ddm_e5_e4_ws1_canonical_equations_note_20260724.md`.

Directive disposition is machine-readable in
`.omx/research/ddm_e5_e4_ws1_directive_consumption_20260724.json`.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, and `docs/operating_manual_craft_handoff.md`, fully.
- Delegated authority file, SHA
  `1c5201fbde8619a2224e7b0a17863c7e522f6195593c26f2d693783b2fdd9f4a`.
- WS2 findings and both materialized source archives.
- E4 findings, primary/fallback configs, packets, dual harness receipts, and
  rate receipt.
- The three named core modules plus WS1 receiver, carrier, preuint8, coupled,
  and G1 decode paths.
- Lane registry, subagent progress, per-arm inbox, and broadcast inbox.
- Frozen upstream clone, scorer weights, target cache, exact raw checkpoints,
  and SSD evidence roots.

## Machine receipts

- `.omx/research/ddm_e5_e4_ws1_exporter_adapter_20260724/ddm_e5_e4_ws1_adapter_receipt.json`
- `.omx/research/ddm_e5_e4_ws1_exporter_adapter_20260724/W_seg/brotli/ddm_e4_ws1_runtime_export_receipt.json`
- `.omx/research/ddm_e5_e4_ws1_exporter_adapter_20260724/W_seg/brotli/ddm_e4_ws1_upstream_harness_receipt.json`
- `.omx/research/ddm_e5_e4_ws1_exporter_adapter_20260724/W_seg/ddm_e4_ws1_batch32_remeasure_receipt.json`
- `.omx/research/ddm_e5_e4_ws1_exporter_adapter_20260724/W_joint/brotli/ddm_e4_ws1_runtime_export_receipt.json`
- `.omx/research/ddm_e5_e4_ws1_exporter_adapter_20260724/W_joint/brotli/ddm_e4_ws1_upstream_harness_receipt.json`
- `.omx/research/ddm_e5_e4_ws1_exporter_adapter_20260724/W_joint/ddm_e4_ws1_batch32_remeasure_receipt.json`

## Independent review

Round 1 disposition: **PASS the bounded primary adapter and evidence; BLOCK the
fallback formulation; BLOCK promotion.**

Three consecutive clean passes each ran the 25 focused exporter/WS1 tests.
Pass 1 also checked lint, compile, canonical configs/receipts, emitted-runtime
byte identity, and whitespace. Pass 2 independently re-derived the two source,
packet, harness, and endpoint receipts. Pass 3 deterministically rebuilt both
packets against immutable publication, then repeated lint, compile, tests, and
whitespace. The machine record is
`.omx/research/ddm_e5_e4_ws1_exporter_adapter_round1_review_20260724.json`.

MAIN must independently:

1. verify the V15 literal route is unchanged and the WS1 route is a second
   schema, not a wildcard;
2. re-derive the gap-free stream/SHA/consumer bijection and source re-emit;
3. inspect the generic source-bundle import closure and emitted-runtime
   cleanliness;
4. recompute both packet rate rows and both exact batch32 endpoint equalities;
5. confirm the fallback blocker reaches a real inner Brotli decompression and
   is scoped at FORMULATION;
6. keep all rows advisory-only and the contest pointer unmoved.

`main_review_required=true`.
