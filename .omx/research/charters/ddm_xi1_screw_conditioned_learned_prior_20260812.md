# ddm_xi1 — the ξ-CONDITIONED LEARNED PRIOR (learn the prior via the screw; store nothing new)

OPERATOR STEER (2026-08-12, binding): *"What about learn prior via screw or
something like that instead of storing directly? Seems a bit naive and toy."*
The critique is CORRECT under the charter-time optimal-form law, twice over:

1. **tf1's F1 verdict is FORMULATION-scoped, not family-scoped.** tf1 killed
   ξ/XOR transport with FROZEN generic coders (LZ pays for match structure;
   XOR fragments it — the #859 law). That is a MECHANISM-reduced form of the
   real hypothesis: a LEARNED conditional model
   p(partition_t | warp_ξ(partition_{t-1})) trained jointly. Cross-entropy
   under a trained conditional model is a different quantity from
   LZ-on-XOR bytes. That cell is UNMEASURED. (SR1's post-hoc FROZEN context
   losing 2 B/43 B does not measure it either — post-hoc ≠ joint.)
2. **The pose stream is STORED directly** (exact B1 carriage 6,864 B on tf1's
   ladder; PR135-family pose carrier 23,384 B per pk2). A learned se(3)
   DYNAMICS prior (Chasles screw smoothness, tac.lie; am1 acceleration-
   matching crosswalk) predicting pose_t from pose history and coding only the
   residual has never been raced against direct storage at matched fidelity.

KEY PROPERTY (why this is clean and high-EV): a conditional ENTROPY MODEL over
the SAME losslessly-decoded tokens changes ONLY the rate term. No R/uint8
survival questions, no realization gap, no scorer distortion risk. Pure bytes.

## MISSION (two legs, cheapest-first)

**Leg A — ξ-conditioned HPAC context (the seg/partition rate leg).**
Extend the cl1 HPAC trainer (tools/fit_ddm_cl1_hpac_capacity.py + its attested
train/pack/encode/decode chain — REUSE, do not rebuild; MAIN has proven the
chain end-to-end today: pack byte-equal, encode 116,716 B, decode
verified_exact=True) with ONE additional context input: the ξ-warped previous
decoded partition plane (warp from the CARRIED pose via tac.lie — rule-118
free at decode, since the receiver already has both). Train spatial-only vs
spatial+ξ-context at MATCHED capacity (same λ rung), same seed/schedule.
MEASURE: coded token bytes (real range coder, the cl1 encode path) both arms.
Falsifier FA: conditional ≥ 0.98× spatial-only bytes at 2 capacity rungs →
temporal context closed at the LEARNED scope too (then tf1's F1 upgrades from
formulation to family, honestly).

**Leg B — learned screw-dynamics pose prior (the pose rate leg).**
Race THREE pose codings at matched decoded fidelity on the cp135-family pose
stream: (i) direct storage (incumbent, pk2 receipts); (ii) B-spline/AR
ξ-dynamics prediction + coded residual (tac.lie + am1 acceleration prior);
(iii) tiny learned dynamics model (counted) + residual. Bytes vs realized
d_pose through the REAL decode. Recall pz4a variable-precision receipts +
pk2's 23,384 B / 0.0155704 S row — extend, never re-derive.

## BINDING LAWS
Payload P0 (sha256+bytes to /Volumes/APDataStore/pact/ddm_xi1_20260812/ —
Vertigo at 99%); matched-capacity A/B (same seed/schedule — the px1 update-RMS
fairness lesson); real coders only; n600 or stratified per m88/m96; serializer
--no-co-author post-edit shas [no-triality] [p0-ledger-ok]; 2 review passes per
.py; resumable + per-stage ckpts P0; bounded runs ≤~40 min per smoke (the cl1
60-epoch scale is the reference form); NO scorer needed for Leg A (lossless);
Leg B pose on custody planes. Skeleton annexes queued in run dir.

## OPTIMAL FORM
Reference: cl1 trainer/chain (landed, attested) + tac.lie SE(3) + am1/pz4a/pk2
receipts. SCOPE reductions legal (fewer epochs, stratified pairs).
MECHANISM reductions TOY-BRACKETED: frozen post-hoc context (that's SR1, done);
entropy estimates instead of coded bytes; pose leg without real decode.

## FALSIFIERS
FA (above) · FB: screw-dynamics pose ≥ incumbent bytes at matched d_pose on
both formulations → direct storage vindicated, family closed with the curve.
