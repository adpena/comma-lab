# Provider recovery + PR106 PacketIR identity tranche (codex, 2026-05-12)

## Scope

Goal-linked tranche for score lowering under the active `/goal`: clear stale
provider WIP, prevent repeated Modal recovery/claim bugs, and add a reusable
PR106/R2 PacketIR identity layer for byte-closed sidecar work.

## Provider harvest / classification

- `t1_balle_cheap_v2_20260512T130830Z` recovered from Modal call
  `fc-01KRE4XPREYGDHQ80XR2H2T54J`.
  - Result: `returncode=21`, no score/components/sample count.
  - Root cause: remote script aborted before training because the dispatch
    claims ledger was missing inside `/tmp/pact/.omx/state/`.
  - Classification: infrastructure/custody failure, not model evidence.
  - Local summary:
    `experiments/results/lane_t1_balle_cheap_v2_20260512T130830Z_modal/harvest_summary.json`.

- `scpp_stage1_20260512T130830Z` recovered from Modal call
  `fc-01KRE4YCY88E8H6XEY0F9BVG7N`.
  - Result: `returncode=21`, no score/components/sample count.
  - Root cause: same missing remote dispatch claims ledger.
  - Classification: infrastructure/custody failure, not model evidence.
  - Local summary:
    `experiments/results/lane_scpp_stage1_20260512T130830Z_modal/harvest_summary.json`.

- PR106 residual Modal CPU rows were already recovered:
  - C3 residual: `score_recomputed=0.22810213271134513`, `n_samples=600`,
    archive SHA `eafb1a027f7065751bc6eb8716f9cc92e0e86c290c68eaa529c59b0066f0960a`,
    `score_claim=false`, `promotion_eligible=false`.
  - Wavelet residual: `score_recomputed=0.22810213271134513`, `n_samples=600`,
    archive SHA `ed90a2250e948ed70086162a84ab907a866099b9d2a7d13fd13870e5fa29fe81`,
    `score_claim=false`, `promotion_eligible=false`.
  - Classification: Modal CPU advisory only; not GHA contest-CPU and not
    promotion evidence.

- Kaggle PR106 y-shift score-table run failed immediately:
  - Log: `reports/raw/kaggle_logs/pr106_yshift_latest/comma-lab-pr106-yshift-score-table.log`.
  - Root cause: missing `pact_pr106_yshift_source_bundle.tar.gz` under
    `/kaggle/src` or `/kaggle/input`.
  - Classification: source-bundle attach/materialization bug, not model
    evidence.

Active dispatch ledger was reduced from 11 active rows to 6 active staged rows
after terminal classification.

## Hardening landed

- `experiments/modal_t1_balle_endtoend.py`
  - Recovery now resolves both canonical result dirs and legacy
    `lane_<label>_modal` dirs.
  - T1 recovery refuses non-T1 metadata, preventing SCPP or other generic
    Modal calls from being closed under `t1_balle_128k_endtoend`.

- `experiments/modal_train_lane.py`
  - Generic Modal training now copies `.omx/state/active_lane_dispatch_claims.md`
    into the writable remote workspace before running lane scripts.
  - Generic Modal training exports `T1_DISPATCH_CLAIMS_PATH` and
    `SCPP_DISPATCH_CLAIMS_PATH` to the copied ledger path.
  - Generic Modal training records/exports mounted git head and branch to both
    T1 and SCPP remote scripts.
  - Generic Modal training ensures an active dispatch claim before `.spawn()`;
    known lane scripts map to canonical lane IDs.

- `experiments/modal_recover_lane.py`
  - CUDA recovered scores are now printed as `UNADJUDICATED, NON-PROMOTABLE`
    until archive SHA, runtime tree SHA, sample count, evaluator schema,
    component recomputation, and terminal claim evidence are checked.

## PR106 PacketIR identity layer

Added `src/tac/packet_compiler/pr106_sidecar_packet.py` and exported it through
`tac.packet_compiler`.

Capabilities:

- Parse PR106/R2 sidecar wrappers into typed sections:
  `format_id`, `pr106_bytes`, `sidecar_payload`, and optional PR101 grammar
  `framing_meta`.
- Emit `0.bin` bytes identity for format `0x01` brotli sidecar and format
  `0x02` PR101 grammar sidecar.
- Read and emit single-member stored ZIP archives while preserving metadata.
- Emit non-promotional identity manifests with `score_claim=false`,
  `promotion_eligible=false`, and `ready_for_exact_eval_dispatch=false`.

Identity tests prove byte-for-byte archive re-emission for:

- `submissions/pr106_latent_sidecar_r2/archive.zip`
  (`7f926bc3e213af1c3ea4be0608c63d041d455eb6b988562b64465e81b25f3a3f`).
- `submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip`
  (`c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383`).

## Verification

- `pytest src/tac/tests/test_modal_train_lane_hardening.py src/tac/tests/test_modal_t1_balle_endtoend.py src/tac/tests/test_remote_auth_eval_hardening.py::test_modal_recover_labels_non_cuda_scores_advisory src/tac/tests/test_remote_auth_eval_hardening.py::test_modal_recover_labels_cuda_scores_as_unadjudicated_non_promotable -q`
  - 45 passed.
- `pytest src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py src/tac/tests/test_packet_compiler_golden_vectors.py -q`
  - 8 passed.
- `py_compile` on edited Python modules passed.

## Next score-lowering actions

1. Relaunch T1 and SCPP only after this hardening commit is on `main`, because
   the previous calls never reached training.
2. Rebuild/fix the Kaggle y-shift source bundle attach path before any new
   Kaggle PR106 y-shift dispatch.
3. Use `tac.packet_compiler.pr106_sidecar_packet` as the only PR106/R2 wrapper
   parser for future PR106 sidecar mutations, including sidecar grammar swaps,
   bit-packing variants, and arithmetic/range-coded sections.
4. Promote any transformed PR106 archive only after PacketIR identity/non-no-op
   proof, same-runtime inflate smoke, exact CUDA adjudication, and axis-labelled
   custody are present.
