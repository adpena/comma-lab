# Codex findings — DDM M7 relaxed receiver realization

Date: 2026-07-23
Lane: `ddm_m7_relaxed_receiver_realize_byteclose`
Verdict scope: one explicitly named 177,169-byte PR110-lineage receiver
member, measured locally on the macOS CPU advisory axis.
Authority: local `$0` only. No Modal, remote, GPU, paid dispatch, score claim,
promotion, or pointer mutation was authorized or performed.

## Outcome

**MEASURED — `BYTE-CLOSED_CANDIDATE_FOR_MODAL_EXACT_EVAL`**

The actual counted archive survived exact receiver parse-back, its native
integer lattice, the full renderer `R`, uint8 realization, and the frozen
SegNet/PoseNet scorers for all 600 pairs:

| quantity | value |
|---|---:|
| archive bytes | 177,169 |
| archive SHA-256 | `cb6cf0ba719a535bf8874b31675a4ec66a893423d320f1e4071a2012cd88a56f` |
| d_seg | 0.000545578002735662 |
| d_pose | 0.00002930755865188909 |
| Seg term | 0.054557800273566194 |
| Pose term | 0.017119450532037846 |
| rate term | 0.11796956486570198 |
| **S** | **0.18964681567130603** |
| delta vs 0.1910828242 pointer | -0.0014360085286939661 |
| delta vs strict 0.19108 fork | -0.001433184328693965 |

This is `[macOS-CPU frozen-scorer advisory]`, `score_claim=false`,
`promotion_eligible=false`, and `ready_for_exact_eval_dispatch=false`.
The canonical frontier pointer remains `0.1910828242`. MAIN must review this
landing and owns any Task #381 exact-eval dispatch.

## Receiver and lattice custody

**MEASURED.** The ZIP is one `ZIP_STORED` member `x`, 177,069 member bytes,
member SHA-256
`e9ebce6e7b89eb2d611641df9112c45aa4014197dfbbcf70f0c4a82e064abe18`.
The existing PR110 receiver parsed the native editable state as `(600, 28)
uint8`. `FrozenPacket.verify_roundtrip()` proved all three legs true:
latent-raw, member, and archive byte identity. No continuous re-solve and no
solve-then-round substitution occurred.

**MEASURED.** `Renderer` consumed that native table through decoder, sidecar,
optional DQS1, bicubic camera resize, offsets, clamp/round, and selector. The
tool called the frozen upstream `DistortionNet.compute_distortion` directly.
Thirty-eight immutable batch checkpoints cover pair IDs 0–599 exactly once;
every checkpoint body hash independently revalidated. A second invocation
resumed from the complete prefix and reproduced the same 17,804-byte receipt,
file SHA-256
`429baa120a54af0246a7cb45bef53b326c40ac2a317e18e1f6081590ab7f8718`.

## Arithmetic counterfactual resolved

**DERIVED.** Combining the exact-C1 solve distortions
`d_seg=0.00015196`, `d_pose=0.00010184` with this archive's 177,169-byte rate
produces `S=0.1650779449085631`. Those distortions belonged to a different
high-byte object; they were never properties of this receiver member.

The realized-minus-counterfactual gap closes exactly by score term:

| term | realized − counterfactual |
|---|---:|
| Seg | +0.03936180027356619 |
| Pose | -0.014792929510823262 |
| rate | 0.0 |
| total | +0.024568870762742945 |

The instance-level realization-transfer ratios are
`d_seg=3.590273774254159×` and `d_pose=0.2877804266681961×`. They are
diagnostics for these two named objects, not universal transfer coefficients.

## Provenance and re-derivation

- implementation commit: `e961b08fad12e2c7446efdb1db01a89fe0c497a2`
- runtime manifest:
  `0e0702e374546b00a0169b00c836e54bc2c4b7268e096bf1dd34776e64335d1b`
- frozen-upstream manifest:
  `826399224af95aec46bdc797b2ee93804d8b30574c05e95f50f9bce0c4688f69`
- source video SHA-256:
  `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`
- hardware: Apple M5 Max arm64; Python 3.13.12; Torch 2.12.1; NumPy 1.26.4
- durable SSD receipt:
  `/Volumes/VertigoDataTier/pact/evidence/ddm_m7_relaxed_receiver_realize_byteclose_20260723/ddm_m7_relaxed_receiver_realization_receipt.json`

Re-derive or resume:

```bash
PYTHONPATH=src /Users/adpena/Projects/pact/.venv/bin/python \
  tools/realize_ddm_m7_relaxed_receiver.py \
  --config .omx/research/ddm_m7_relaxed_receiver_realize_byteclose_20260723.config.json
```

## Review disposition

Three bounded clean passes independently re-derived: (1) archive/receiver and
checkpoint custody, (2) all score terms and fork comparisons from per-pair
rows, and (3) authority/scope and resumability. The lane is research-only and
requires MAIN landing review. The only canonical validation debt observed was
pre-existing global lane-registry debt: `lane_maturity.py validate` reports
110 missing historical evidence paths unrelated to this lane; this lane's
evidence paths exist and its own record is internally consistent.
