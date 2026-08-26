Completed #1303: the live no-MPS-decision gate fell from 21 violations to 0 without weakening the gate.

- Classification: 14 rule false positives, 4 historical records, 1 real CUDA-backed claim, 2 honestly downgraded advisory claims, 0 waivers.
- Targeted suite: 19/19 passed.
- Positive control: exactly one synthetic violation detected; strict mode raised correctly.
- All 17 touched Python files passed two review passes and policy checks.
- No scorer or Modal work ran; no frontier score was measured or moved.
- Full evidence and per-site table: [MG1 burndown memo](/Users/adpena/Projects/pact/.omx/research/ddm_mg1_mps_gate_burndown_20260826.md).

The managed sandbox denied Git index/object writes. The final self-contained #1293 custody artifact is fallback commit `94cad1e4e836b34069bc1e87fcbc607d874812d6`, based on unchanged HEAD `95a14876a276595516af4dcc10ac14bfe32177c4`. Its 22 declared files match the current working tree with zero hash mismatches:

- [intended-commit.bundle](/Volumes/VertigoDataTier/pact/ddm_mg1_serializer_fallback/20260826T214713.391623Z-23094/intended-commit.bundle), SHA-256 `510f7fdcdcfb156d246a2dcee7413e78441b94cf4e804eb84b92da3bc418920a`
- [receipts.jsonl](/Volumes/VertigoDataTier/pact/ddm_mg1_serializer_fallback/20260826T214713.391623Z-23094/receipts.jsonl)

`[contest-CUDA T4 n600] own-vehicle frontier: GB1 — S=0.14811799921260607 @ 180,215 B, archive sha256 ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4; MG1 measured no score and did not move the pointer.`

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN/operator; consumer store: main Git object store/ref at `/Users/adpena/Projects/pact/.git`; fire trigger: the first writable Git landing window with a clean shared index. Land fallback commit `94cad1e4e836b34069bc1e87fcbc607d874812d6` from the bundle, then rerun the strict gate and targeted test.

## LIVE-HYPOTHESES

- None remain for #1303. The strict scanner is clean, the synthetic violation still fires, and the complete targeted suite passes.

## DEAD-ENDS

- Blanket exemptions: every site was individually curable.
- New waivers: no site was irreducible.
- Fabricated CUDA support: only Wave N+6 had verified CUDA custody.
- Direct generated-report edits: the registry was corrected and the report regenerated.
- Gate relaxation: unnecessary; the positive control proves the decision signature remains active.

