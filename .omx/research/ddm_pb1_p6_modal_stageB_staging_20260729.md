# ddm_pb1 P6 — Modal contest-CPU Stage-B flight: STAGED, NOT FIRED (operator-GO required)

> **REPOINTED 2026-07-29 (ddm_pfs1): the staged flight now targets the pfs1 D1 recomposed
> archive** — the pb1 archive with its 6-cosine pose member swapped for the warp-base carrier
> (grammar v3; D1 solve receipt d_pose 0.22155 → pose contribution 1.4884; instrument S_pred
> 2.2570 vs the prior 20.2746 row; local Stage-A eval receipt at
> `ddm_pfs1_20260729/d1/d1_eval_receipt.json`). The operator's <$2 flight buys the calibration
> on the BEST local row. The command block below is updated in place; the superseded
> pb1-archive parameters are preserved in this file's git history (commit c91c27ca30) per
> APPEND-ONLY provenance.

**Status: STAGED ONLY. No dispatch fired. No paid spend. Pointer 0.1910828242 [contest-CPU] UNMOVED.**

Per pn1 S1 Stage-B (single-flight dual-ledger, est <$2, hard envelope <=$20) and the charter P6:
this file is the complete dispatch configuration for ONE Modal contest-CPU exact flight on the
pb1 composed archive. The operator fires it by appending `--execute` to the staged command below
(and nothing else).

## Preconditions (all satisfied at staging time or blocking-noted)

1. Composed archive byte-closed with parse-back + exact-consumption asserts (pfs1 D1 build
   receipt: `/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/d1_build_receipt.json`
   (archive 569,996 B; DR7T tokens 557,253 B via the r7 SMEVR winner; receiver-reconstructed
   packet BYTE-IDENTICAL; vendored-warp byte-identity max_abs 0)).
2. Local Stage-A advisory-exact row measured through the locked evaluate.sh path (pfs1 D1 eval
   receipt: `/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/d1_eval_receipt.json`).
3. Lane claim BEFORE dispatch (the staged command sequence includes it).
4. Bare-venv bootstrap smoke (r5 lesson) is handled by the dispatch tool's runtime-tree upload +
   the submission dir's self-contained receiver (stdlib+numpy+scipy+brotli+torch only — all
   proven importable in the locked env; Modal CPU image installs from the runtime tree).

## The staged command sequence (operator runs from /Users/adpena/projects/pact)

```bash
# 1) lane claim (24h TTL)
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id lane_ddm_pfs1_composed_paired_auth_20260729 \
  --agent ddm_pfs1 \
  --notes "pfs1 recomposed tr1+warp-pose archive; single-flight contest-CPU (CUDA optional inside envelope)"

# 2) the ONE staged flight (DRY-RUN as written; append --execute for the paid dispatch)
.venv/bin/python tools/dispatch_modal_paired_auth_eval.py \
  --archive /Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/submission/archive.zip \
  --expected-archive-sha256 624ffe57000c6fe4a6802a6d8b9a5d6002617f29b0bbb9e186d1273fa996600c \
  --submission-dir /Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/submission \
  --label ddm_pfs1_composed_20260729 \
  --lane-id-base lane_ddm_pfs1_composed_paired_auth_20260729 \
  --expected-runtime-tree-sha256 auto \
  --gpu T4 \
  --claim-agent ddm_pfs1 \
  --json-out /Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/p6_dispatch_result.json
# 3) harvest within 24h (Modal .spawn HARVEST OR LOSE)
```

Flags verified against `tools/dispatch_modal_paired_auth_eval.py` argparse at staging time
(never-invent-flags): `--archive/--expected-archive-sha256/--submission-dir/--label/`
`--lane-id-base/--expected-runtime-tree-sha256 auto/--gpu/--claim-agent/--json-out/--execute`.

## Expected drift band (pre-registered)

- Stage-A (local advisory) vs contest-CPU: per pn1 B2, typed constants (delta_seg, delta_pose,
  delta_rate === 0) are MEASURED by this flight, never assumed; the PR107 precedent is 6e-6 on
  the CPU axis for a like-for-like torch pipeline, but it is NOT assumed to transfer.
- GREEN: |S_contestCPU - S_localAdvisory| <= 5e-4 with rate term byte-identical.
- OUTSIDE band => STOP: a deploy-parity bug was found; that discovery is the value (no second
  flight until root-caused).

## Cost + budget

Single flight, CPU axis (T4 CUDA optional within the same envelope): estimated <$2 total;
hard envelope <=$20 per pn1. The Modal budget exists to buy exact rows; this staging note is
the GO surface — the operator's `--execute` is the only remaining action.

## Honest framing

The composed local row (pfs1 D1, S ≈ 2.26-class) is still far above the 0.172 bar — pose
~1.49 + seg 0.389 + rate 0.380 all individually exceed it. This flight's value at the CURRENT
composed S is calibration (the Stage-B drift constants + first own-vehicle exact row on
contest hardware), NOT a competitive row. If the operator prefers to hold the <$2 until a
composed row is within striking distance of the bar, that is a legitimate routing decision —
this staging file remains valid for whichever composed sha is current when fired (re-stage the
sha + rerun the local Stage-A first). The pfs1 D2 pose-field ladder (memo §5) names the next
S movers on this same archive family.
