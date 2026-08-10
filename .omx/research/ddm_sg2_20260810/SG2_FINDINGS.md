# DDM-SG2 — PR130 Seg-axis source audit and scorer-gated disposition

**Status:** scorer-free work complete; official PR130 error-edge decomposition
`UNMEASURED_SCORER_GATED_AND_SOURCE_PAYLOAD_NOT_RETAINED` · **axis:** mixed,
explicit per table · **score claim:** false · **pointer moved:** false

## Result

The rounded PR130 report gives `d_seg = 0.00028609`, or `0.028609 S`, but it
does not preserve an exact error count: multiplying by the n600 pixel
denominator `600 × 384 × 512 = 117,964,800` gives the non-integer
`33,748.549632`. The searched PR130 reproduction, quantized reproduction, SD1
CPU screen, and intake roots retain scalar mismatch counts but no PR130
candidate argmax payload. Consequently, there is no honest way to reconstruct
the shipping candidate's per-edge errors from existing bytes. Running a new
scorer job would race `ddm_ai1`, which this charter forbids. The missing
decomposition is queued with retained argmax and decoded-frame outputs as P0
requirements.

The nearest exact per-edge table available is useful but answers a different
question: it is the class transition table between retained AV and DALI
**source-target caches**, not between PR130 predictions and the DALI target.
On the same Tesla T4 host, the decoder caches differ in `20,671` of
`117,964,800` pixels, `d = 0.00017523023817274304`.

| symmetric source-target edge | pixels | share of the 20,671 disagreements |
|---|---:|---:|
| Road ↔ MyCar | 5,824 | 28.1747% |
| Road ↔ Lane | 5,695 | 27.5507% |
| Road ↔ Undrivable | 5,668 | 27.4201% |
| Undrivable ↔ Movable | 2,068 | 10.0044% |
| Road ↔ Movable | 1,344 | 6.5019% |
| Lane ↔ MyCar | 25 | 0.1209% |
| Lane ↔ Undrivable | 22 | 0.1064% |
| Lane ↔ Movable | 19 | 0.0919% |
| Movable ↔ MyCar | 6 | 0.0290% |
| Undrivable ↔ MyCar | 0 | 0% |

Road participates in `18,531 / 20,671 = 89.6473%` of these decoder-induced
target flips. This resemblance to retired-vehicle Road-heavy tables is a lead,
not transfer evidence.

## #906 correction at source

Commits `afa34a0860` and `38e08900c3`'s stored 120-pair rows re-reduce exactly:

- selection: seeded stratified-random n120, never a prefix;
- denominator: `120 × 384 × 512 = 23,592,960` pixels;
- disagreements: `5,377`;
- fraction: `0.00022790696885850695` `[macOS-CPU advisory]`;
- ratio to the rounded PR130 `0.00028609`: `79.6626827%`;
- positive control: centered conversion is byte-identical to the upstream path
  on pair 3.

The number therefore reproduces, but the causal claim requires correction. It
measures a hypothetical centered-vs-left chroma siting perturbation on original
source frame 1. It is neither PR130 candidate error nor an actual DALI-vs-AV
comparison. The later retained n600 same-host T4 AV-vs-DALI caches give
`0.00017523023817274304`, only `61.2500%` of the rounded PR130 value. Neither
decoder-difference number may be subtracted from PR130's Seg term without a
matched candidate replay.

## Top two measured current-vehicle levers

At the PR130 denominator, `1,000 B` costs exactly
`25,000 / 37,545,489 = 0.000665858953 S`.

| lever | bytes | d_seg change | Seg ΔS | rate ΔS | semantic-leg ΔS | disposition |
|---|---:|---:|---:|---:|---:|---|
| stage-07 → stage-08 expected-flip tail QAT | +12 | −0.0000208791097005 | −0.002087910970 | +0.000007990307 | **−0.002079920663** | Measured, but already consumed by the PR130 base |
| SD1 selected mixed q3/q4 | −848 | +0.00000140719943576 | +0.000140719944 | −0.000564648392 | **−0.000423928449** | New semantic-leg survivor; pose unmeasured |

The tail-QAT Seg benefit is `261.3×` its exact 12-byte rate cost, but it is not
a new arm: the 191,052-byte base already ships stage 08. Its retained stage-07
counterfactual is 191,040 bytes, SHA-256
`ab11494c15a320dfcc005e950b028ee5630c3bff985ce39fafe3a1a03880ba51`;
independent archive builds are byte-identical and parse back to the packed
state exactly.

The mixed q3/q4 archive saves four times more rate `S` than it loses on the
matched Seg axis. It remains only a semantic-leg win because PoseNet consumes
the changed rendered frame. The exact retained 190,204-byte archive is SHA-256
`010a8a5273ae87595191ffc03447fa36e61978ae9f827c2def46dea7075dfa67`.
The landed public receiver support is pinned at `58f62cd22f`; the paired n600
component replay is queued rather than inferred.

## Payload custody

New bulk evidence lives only below
`/Volumes/VertigoDataTier/pact/ddm_sg2_20260810/source_audit_v2/`:

| artifact | bytes | SHA-256 |
|---|---:|---|
| source-audit result | 11,362 | `4459179fc9e7367b32561c29ac95b827ad95639c81f6c269dd8031bb9caf33e3` |
| chroma pair IDs, int16 LE | 240 | `19db1c1b47494a9742f923957e322c9337109eca901cab895a2017120de7ec9b` |
| stage-07 semantic payload | 40,252 | `81058169865ffc7d1a400feba7dbe174d3610b5d55af78d13aa595062ecc1ea9` |
| stage-07 archive | 191,040 | `ab11494c15a320dfcc005e950b028ee5630c3bff985ce39fafe3a1a03880ba51` |
| stage-07 archive repeat | 191,040 | `ab11494c15a320dfcc005e950b028ee5630c3bff985ce39fafe3a1a03880ba51` |
| stage-07 decoded state | 266,745 | `5a1ce07cafa810208489fdce3bcddf9f682d941fdbfeb1b9e3c8fea08027aaaf` |

The audit is resumable from `progress.json`; each stage is recorded atomically.
The targeted `tac.payload_retention_gate` scan reports 1 Python file scanned
and 0 findings.

## RECALL EVIDENCE

Searched the full `.omx/research/` corpus by content for `PR130`, `d_seg`,
`per-edge`, `Road`, `chroma siting`, `DALI`, `AV`, `q3`, `q4`, and the base
archive/checkpoint hashes; searched the canonical research index, sub-0.15 DAG
FEED blocks, task bridge rows `#906` and `#917`, the 429-entry canonical
equations registry, the intake artifact manifest, and the actual evaluator.

Beyond the charter seeds, this found:

- the completed retained n600 AV-vs-DALI cache pair, which corrects #906's
  interpretation and supplies an exact source-target edge control;
- SD1's measured n600 mixed q3/q4 semantic-leg survivor and the later public
  receiver support at `58f62cd22f`, which replaces an unnecessary new lever
  sweep with a ready byte-closed candidate;
- SM3's six retained representation candidates, which remain unmeasured on
  `d_seg`/`d_pose` and therefore stay hypotheses rather than top-two priced
  levers;
- the earlier SG2/RR2 proof that stage 08 is the shipping packed object, which
  prevents misreporting the tail-QAT gain as a new PR130 improvement;
- retired TR1/burn Road↔Lane evidence, which is retained only as
  formulation-scoped history and is not transferred to PR130.

This changed the plan from inventing a new Seg actuator to (1) correcting
#906, (2) exact byte-pricing the two already measured current-vehicle moves,
and (3) queueing the missing prediction-edge payload and paired pose replay.

## Boundaries and verdicts

- **INSTANCE:** the official PR130 `0.028609 S` per-edge table was not measured;
  exact source bytes were not retained and the scorer slot is unavailable.
- **MECHANISM:** #906 reproduces as chroma sensitivity, not as causal attribution
  of PR130 error.
- **INSTANCE:** stage-08 tail QAT is strongly favorable but already in the base.
- **INSTANCE:** mixed q3/q4 survives the semantic-leg price, but no full-score
  verdict exists until matched pose replay.
- No SegNet, PoseNet, candidate rendering, contest evaluator, Modal job, or
  upstream mutation ran in this arm.

The PR130 reference remains `S=0.172141297491896447 @ 191,052 B`
`[contest-CUDA, DALI GT, n600]`; it is a bar, not a new score from this arm.
Own-vehicle frontier remains `S=0.7539807296911207 @ 357,836 B`
`[macOS-CPU advisory] n600`.
