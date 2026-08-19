# ck2 fire-order DRAFT — for MAIN. Nothing here was fired by `ddm_ck2`.

**Gate:** the local n600 CPU advisory at
`/Volumes/APDataStore/pact/ddm_ck2/advisory/attempt_0001` must land CLEAN **and** must
reproduce ck1's advisory distortion to 8dp. This candidate is a pure container transform,
so the decoded state is bit-identical and the advisory is a falsifier, not an estimate:

| axis | ck1 advisory (attempt_0002) | ck2 required | if it differs |
|---|---|---|---|
| d_seg | 0.00043336 | **identical** | bit-identity claim is FALSE — void the candidate |
| d_pose | 0.00014829 | **identical** | bit-identity claim is FALSE — void the candidate |
| canonical_score | 0.19982266166528362 | ck1's minus 25·657/37,545,489 = **0.19938519…** | rate arithmetic is wrong |

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
