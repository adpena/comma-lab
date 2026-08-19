# ck2 fire-order DRAFT — for MAIN. Nothing here was fired by `ddm_ck2`.

**Gate:** the local n600 CPU advisory at
`/Volumes/APDataStore/pact/ddm_ck2/advisory/attempt_0001` must land CLEAN **and** must
reproduce ck1's advisory distortion to 8dp. This candidate is a pure container transform,
so the decoded state is bit-identical and the advisory is a falsifier, not an estimate:

Read from ck1's own advisory receipt
(`/Volumes/APDataStore/pact/ddm_ck1/advisory_rebased/attempt_0002/contest_auth_eval.json`),
not from prose:

| field | ck1 advisory (attempt_0002) | ck2 required | if it differs |
|---|---|---|---|
| `avg_segnet_dist` | 0.00043336 | **identical** | bit-identity claim is FALSE — void the candidate |
| `avg_posenet_dist` | 0.00014829 | **identical** | bit-identity claim is FALSE — void the candidate |
| `score_rate_contribution` | 0.11797822103209257 | **0.11754075169989130** | rate arithmetic is wrong |
| `canonical_score` | 0.19982266166528362 | **0.19938519233308236** | as above |
| `archive_size_bytes` | 177,182 | **176,525** | wrong bytes under test |

The advisory delta is **−4.3746933e-04**, identical to the T4-axis delta, because the rate
term is axis-independent and the two distortion terms do not move.

Do not fire on a clean-but-drifted advisory. A drift on either distortion axis means the
receiver overlay is not the permutation it claims to be, and the whole basis of the row is
that it changes nothing except layout.

---

## 1. Seal

```bash
.venv/bin/python tools/make_candidate_seal.py \
  --candidate-id ck2_plane2_container_r1 \
  --runtime-dir /Volumes/APDataStore/pact/ddm_ck2/generations/ck2_plane2_r1 \
  --axis contest_cuda \
  --pointer-axis contest_cuda \
  --out /Volumes/APDataStore/pact/ddm_ck2/seal/CANDIDATE_SEAL_ck2_r1.json \
  --admit-bar-net-ds -3.5e-6 \
  --archive-member p \
  --verify-archive-sha 0aa1cada2ca79ad43a11bfa72be69a5240315e35cf5b4c94665d60d0c3583933 \
  --retained-path /Volumes/APDataStore/pact/ddm_ck2/compile/build_r1/SA3_REBASE.json \
  --retained-path /Volumes/APDataStore/pact/ddm_ck2/overlay/CK2_RECEIVER_OVERLAY.json \
  --retained-path /Volumes/APDataStore/pact/ddm_ck2/probe/ceiling_r1/CK2_RATE_CEILING.json \
  --retained-path /Volumes/APDataStore/pact/ddm_ck2/advisory/attempt_0001 \
  --falsifier "F1 RATE: archive is exactly 176,525 B and the receipt's rate term is 0.11754075169989130; any other value falsifies the byte arithmetic" \
  --falsifier "F2 SEG: d_seg is UNCHANGED at 0.00030309 — the decode is bit-identical, so any move falsifies the container-transform claim" \
  --falsifier "F3 POSE: d_pose is UNCHANGED at 7.77e-06 — same reason as F2" \
  --falsifier "F4 NET: net dS = -4.374693e-04, which is 125.0x the -3.5e-6 bar and 131.1x ck1's own report-8dp error bound 3.336608e-06" \
  --sealed-by MAIN \
  --notes "Eleventh move. Parameter-free whole-section 2-plane container transform on the ck1 base: semantic -613 B, compensated carrier -44 B, -657 B total at zero distortion. sz1's PINNED split constants measure +59 B (a loss) on this body; the parameter-free form beats even the fitted (offset,length) argmax by 55 B. Identity control reproduces the ck1 pointer archive 35c318d5... byte-identically."
```

The seal producer derives every hash from disk. Do not hand-type shas into it — the two
values above are passed only as CHECKED expectations (`--verify-archive-sha`), which
refuse on mismatch and are never stored as the value.

## 2. Fire

```bash
.venv/bin/python tools/fire_modal_auth_eval.py \
  --seal /Volumes/APDataStore/pact/ddm_ck2/seal/CANDIDATE_SEAL_ck2_r1.json
```

Single-axis (contest-CUDA T4) under the standing F26 waiver (#1049/#1054), as ck1's r4 row
was. ~$0.16, ~1,300 s expected — the decode cost is unchanged from ck1 apart from one
O(n) byte scatter over 58 KB, which is not measurable against a 941 s inflate.

## 3. Expected receipt

| term | ck1 (pointer) | ck2 (expected) | delta |
|---|---|---|---|
| d_seg | 0.00030309 | 0.00030309 | 0 |
| d_pose | 7.77e-06 | 7.77e-06 | 0 |
| rate | 0.11797822103209257 | 0.11754075169989130 | −4.374693e-04 |
| **S** | **0.15710198138050818** | **0.15666451204830689** | **−4.374693e-04** |
| bytes | 177,182 | 176,525 | −657 |

Gap to 0.15 after this row: 0.00666451 (from 0.00710198) — **6.16% closed**.

## 4. Not done by this arm

No paid dispatch, no seal executed, no push, no PR, no hosting. The packet re-stage at the
`ddm_pq1` boundary (generation 5) is a separate consumer and is not touched here.

STORES CONSULTED: run-artifacts (ck1 T4 receipt + advisory_rebased/attempt_0002 contest_auth_eval.json; CK2_CUSTODY_MANIFEST.json + six compiled-archive shas; fx1/fx2 rate receipts; mz2 retained-candidate payload sha 156112d0; t1h/iv1/qs5 T4 refusal rows) · memories (cpu-to-cuda-seg-transfer-has-no-law — seg legs discounted as upper bounds; probability-model-axis-live-fx1-sweep-prior — ctx≫members>lr inherited, not re-measured; cross-regime-constant-transfer — sz1 pinned (49, 8284) re-measured on ck1 body, not transferred; the-counted-byte-is-not-fungible) · research memos (ddm_bp1 section-coding closure re-read with its INSTANCE scope on sz1 body; ddm_sz1/ck1 landing memos) · tasks ledger (#1128 tenth-move row, #1111 packet state). Not consulted: canonical_equations evaluators, graph-memory reconstruct (candidate ranked from receipts directly).

## VERDICT (appended by MAIN, 2026-08-19 ~02:45Z) — ELEVENTH POINTER MOVE ADMITTED

Fired per §1–2 (r2; r1 was correctly refused by the Modal single-flight gate while the ck1
contest-CPU row was live — that row then TIMED OUT at 1800s on Modal-CPU hardware, recorded as
a CPU-axis disclosure fact, no score). Call fc-01M0BVWYZWS9VY23G5Z24EYG0Q, wall 1,303 s, ~$0.16.

ALL FOUR FALSIFIERS HOLD EXACTLY [contest-CUDA T4, n600]:
- F1 RATE: 176,525 B; score_rate_contribution 0.1175407516998913 — exact.
- F2 SEG: avg_segnet_dist 0.00030309 — UNCHANGED from ck1 (bit-identity confirmed at authority).
- F3 POSE: avg_posenet_dist 7.77e-06 — UNCHANGED.
- F4 NET: canonical_score 0.1566645120483069 vs ck1 0.15710198138050818 = -4.3746933e-04,
  matching the projection to the last digit. 125x the -3.5e-6 bar; 65.6x the summed two-row
  8dp bound (6.673e-06 — bounds ADD for deltas and are unequal per row).

Pointer auto-moved with ZERO manual steps (firer anchor mirror ->
experiments/results/modal_auth_eval_mirror/contest_auth_eval_ck2_plane2_t4_r2_20260819.json ->
refresh_canonical_frontier effective 0.1566645120). Gap to 0.15: 0.0066645.
Successor: ma1 within-miss law (-105 B, priced on THIS body) fires as the twelfth-move rate
candidate once #1131's per-candidate rc64 recipe lands.

STORES CONSULTED: run-artifacts (ck2 t4_row_r2 MODAL_REMOTE_RESULT.json + report.txt; ck1
advisory receipt; ck1 cpu_row_r2 timeout receipt; CK2_CUSTODY_MANIFEST) · memories
(concavity/bounds-add exact margins; single-flight binding; bp1 INSTANCE-scope) · tasks ledger
(#1129/#1130/#1131).
