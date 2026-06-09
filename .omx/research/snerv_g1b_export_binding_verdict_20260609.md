# SNeRV G1b export-binding verdict — fidelity SURVIVES the receiver; the failure is 100% RATE

UTC 2026-06-09 · subagent `snerv_g1b_export_binding_20260609` (task #45) · `[macOS-CPU advisory]` /
`exact_cpu_advisory` — **NON-PROMOTABLE** (promotion requires paired Linux x86_64 contest-CPU + CUDA per
CLAUDE.md). Run: `/Volumes/VertigoDataTier/pact/snerv_mistake_b_g1a_20260609T201221Z/` (PID 42886,
600ep/600pair path-B official conv MFU/HFR/TUB, real SegNet+PoseNet direct-live VJP, skip=full, commit
`f5c66f43c`+; completed 22:28:35Z, 6489s, best ckpt ep273 policy `best_total_checkpoint_for_archive_export`).

## THE ONE-MEASUREMENT ANSWER (the decisive fork of `snerv_b_first_scorer_probe_verdict_20260609.md`)

**The export/receiver binding PRESERVES the live fidelity essentially exactly, and the archive bytes are
catastrophic.** SNeRV path-B is the first carrier in the fleet whose live scorer-fidelity survives the full
trained-state → packet → inflate → camera-res → uint8 chain (link-5 of the operator's chain CROSSES on the
fidelity axis). The score-killer is purely the RATE term: the stored skip_high LL planes ride as raw float64
LZMA at 554.6 MB. **Routing: the LF entropy-coding research front** (V3 campaign decision:
`binding=rate, shares seg=0.00 pose=0.00 rate=1.00`).

| Surface | d_seg | sqrt(10·d_pose) | non-rate score | rate | total |
|---|---|---|---|---|---|
| LIVE (batch, ep273 step-guard post) | 0.00269 | 0.1655 | ~0.43 | — | — |
| LIVE (batch, ep599) | 0.00268 | 0.1295 | ~0.40 | — | — |
| **ARCHIVE (real inflate, uint8, camera-res, N=48, real scorers)** | **0.002468** | **0.142602** | **0.389** | **387.25** | **387.64** |

**Survival ratio ≈ 1.0 on both axes** (archive d_seg 0.002468 sits inside the live 0.0023–0.0027 band;
archive pose term 0.1426 sits inside the live 0.113–0.166 band). The receiver-replay is bit-near-exact:
receiver-vs-export MSE 2.87e-12; receiver-vs-target MSE 9.7591 vs live-final recon MSE 9.7585 (0.006% rel).

## Export outcome: BOUND-mechanically + BLOCKED-for-authority

1. **Mechanically BOUND.** The run's own export tail bound the trained official-conv MLX receiver state
   (best ckpt ep273) into a byte-closed SNAR2 packet via `model.export_official_components()` →
   `_build_official_mfu_hfr_tub_packet_from_components`
   (`src/tac/substrates/snerv_inverse_steg_carrier/mlx_native_train_export.py:8107-8144`,
   `allocation_mode=official_mfu_hfr_tub_trained_mlx_receiver_payload`). The real scorer-free inflate
   (`decode_snerv_archive_frames`, official dual-mode dispatch `archive.py:2545`) decodes it: 600 pairs in
   57.7s. `trained_receiver_payload_exported=true`, `trained_receiver_state_bound=true`.
2. **Authority-BLOCKED (honest, fail-closed; NOT hacked around).** The carrier's
   `official_mfu_hfr_tub_export_blockers` (`carrier.py:309-316`) + the run's
   `official_mfu_hfr_tub_train_export.authority_blockers` hold: `weight_mapping_missing`,
   `state_dict_mapping_missing`, `source_forward_replay_missing`. Scope honestly recorded as
   `mlx_receiver_component_state_not_upstream_official_state_dict`. **The demanded proof cannot be honestly
   produced from this run:** it requires the UPSTREAM torch SNeRV_T trained checkpoint
   (qwertja/SNeRV `0844a08f`) mapped + numerically source-forward-replayed; this run trained the MLX-native
   renderer, and the contract explicitly forbids fixture-config stand-ins
   (`snerv_official_trained_checkpoint_source_config_fixture_forbidden`). Closing it = train upstream-torch
   SNeRV_T itself, or land an MLX→upstream-torch weight-mapping + numerical parity replay (not $0-local).
3. The contest `archive.zip` packaging + receiver proof was skipped by launch flag
   (`--skip-snerv-native-mlx-archive-export` → blocker
   `snerv_mlx_native_archive_export_disabled_packet_only_smoke`); the G1b measurement wrapped the packet in
   a single-member STORED zip (`x`) for the rate numerator: 581,583,207 B (+100 B over the packet).

## Bytes (the rate decomposition; LF share ≈ 100%)

- Packet `snerv_mlx_native_packet.snar`: **581,583,107 B** (554.6 MB), SNAR2, sha256 `934349b0efd15b39…`.
- Sections: `decoder_payload` 581,582,145 B (**99.9998%**) · `lf_payload` 388 B
  (**dummy**: `lf_payload_receiver_usage=unused_dummy_zero_official_payload_frame_decode_uses_decoder_payload`)
  · `step_map_packet` 499 B · `metadata_payload` 4 B.
- Inside `decoder_payload` (codec `official_numpy_float64_lzma` preset6; raw 1,415,583,232 B → 581,565,620 B):
  - **`inputs.mfu.skip_high` (1200, 3, 192, 256) float64 = 1,415,577,600 B raw = 99.9996%** — the stored
    per-frame Haar LL planes ARE the archive. This is the Z8 disease in pure form: near-lossless float64
    storage of the half-res video.
  - **ALL trained conv weights ≈ 5.6 KB** (MFU residual blocks 2×1,296 B + HFR heads 3×(648+72+24) B +
    upsamplers 2×288 B + biases).
- Rate term = 25·581,583,107/37,545,489 = **387.25** (3258× PR101's 178,493 B frontier bytes).
- The carrier's OWN LF quantization + step-map machinery (path-A) is NOT applied to skip_high — that is the
  concrete in-repo headroom lever, not a research unknown.

## Routing (the three-way fork resolved)

- ~~Receiver destroys fidelity → binding bug hunt~~ — FALSIFIED: survival ≈ 1.0 (receiver replay 2.9e-12 MSE).
- ~~Live fidelity survives at small bytes → first complete carrier, route to composition~~ — NOT YET: bytes
  are 3258× frontier.
- **RATE DOMINATES → the LF entropy-coding research front** (probe-verdict branch (b); V3 ingest
  `campaign_decision: INSPECT_BINDING_CONSTRAINT / early_high_score_binding_is_rate`; next-action router
  `case=A seg_descended_pose_sane_carrier_works`). Quantified headroom ladder for the 554.6 MB skip_high:
  1. float64→float32/fp16: 2–4× free (receiver math tolerance to be verified; receiver replay currently
     bit-exact at float64).
  2. Route skip_high through the EXISTING carrier LF quantizer + step-map codec (Z8 sister anchors: ~90.4%
     cut at near-lossless Δ; 99.7% @Δ=0.25 → ~1.7 MB → rate ~1.16) — the cheapest structural lever; the
     d_seg cost per Δ must be measured against the 0.002468 baseline.
  3. The LL planes are a 192×256 video: temporal prediction / real video-codec-class coding for the
     LL sequence + HF-head generation already in place. PR101-class rate (0.119) demands ~178 KB total —
     i.e. ~3258× — which is video-codec territory, not quantizer-only territory.
- Non-rate floor check: even at rate→0, the archive-surface non-rate score 0.389 is ~2× the PR101-class
  TOTAL (~0.193). The seg axis at 0.002468 contributes 0.247 — further seg descent (or pose-term descent
  below 0.14) is needed for frontier contention AFTER rate is fixed. Training was flat on seg (probe
  finding 2: stored-LF design starts evaluator-close; the residual learnable surface is small).

## Audit-provenance records (per `src/tac/optimization/audit_provenance.py`; surface field MANDATORY)

1. claim: "G1b archive-surface d_seg = 0.002468 (N=48)" · file:
   `/Volumes/VertigoDataTier/pact/snerv_mistake_b_g1a_20260609T201221Z/g1b_path_b_trained_packet_score.v1.json`
   · field: `archive_surface_d_seg` · candidate_id:
   `snerv_path_b_official_conv_trained_mlx_receiver_payload_g1a_ep273` · observed_value: `0.0024678972016166276`
   · **surface: receiver** · reproduce_command: `.venv/bin/python
   /Volumes/VertigoDataTier/pact/snerv_mistake_b_g1a_20260609T201221Z/g1b_path_b_score_fix.py`
2. claim: "G1b archive-surface d_pose = 0.002034 → pose score term 0.142602" · same file · fields:
   `archive_surface_d_pose`/`archive_surface_pose_score_term` · same candidate · **surface: receiver** ·
   same reproduce_command.
3. claim: "packet bytes 581,583,107; rate 387.25; advisory total 387.64" · same file · fields:
   `packet_bytes_total`/`rate_term_packet`/`advisory_score_with_packet_rate` · **surface: export** ·
   reproduce: `ls -l …/snerv_mlx_native_packet.snar` + the score script.
4. claim: "skip_high = 99.9996% of raw payload (1,415,577,600 B float64)" · file:
   `…/snerv_mlx_native_packet.snar` (decoder_payload tensor manifest) · **surface: export** · reproduce:
   `python -c "from tac.substrates.snerv_inverse_steg_carrier.archive import unpack_snerv_archive,
   _decode_official_mfu_hfr_tub_payload_tensor_manifest; …"` (per-tensor nbytes dump).
5. claim: "live d_seg ep273 = 0.00269 / ep599 = 0.00268; pose term 0.1655/0.1295" · file:
   `…/long_training/telemetry.jsonl` · fields:
   `scorer_space_step_guard_post_segnet_argmax_disagreement_before_restore` +
   `…post_pose_direct_live_score_term_before_restore` at epochs 273/599 · **surface: live** (batch-sampled,
   4 pairs/step) · reproduce: jq/python scan of the telemetry rows.
6. claim: "receiver replay preserves the export bit-near-exactly (MSE 2.87e-12) and the live render
   (9.7591 vs 9.7585)" · files: `compact_renderer_mlx_spine_runner_report.json`
   (`snerv_mlx_native_receiver_export_mse_nchw255`, `…receiver_target_mse_nchw255`) +
   `snerv_score_aware_long_training.json` (`final_recon_mse_nchw255`) · **surface: receiver** vs **live** ·
   reproduce: read those fields.
7. claim: "path-B authority export blocked by 3 carrier blockers; proof not producible from this run" ·
   files: `src/tac/substrates/snerv_inverse_steg_carrier/carrier.py:309-316` +
   `…/snerv_mlx_native_train_export.json` (`official_mfu_hfr_tub_train_export.authority_blockers`) +
   `…/snerv_official_mfu_hfr_tub_source_forward_replay_contract.json` (13 blockers incl.
   `source_config_fixture_forbidden`) · **surface: export** · reproduce: read those fields / the carrier
   property dry-test in `g1b_export_binding_measure.py::probe_path_b_export_blocker`.

Caveats: N=48-of-600 pairs scored (uniform first-48; per-pair spread 0.0019–0.0026 d_seg, no degenerate
constants); the live values are batch-sampled training telemetry (4 pairs/step), so the survival ratio is
band-vs-point, not pair-matched; everything is macOS-CPU advisory — contest-axis (Linux x86_64 / T4) replay
is the promotion gate and is NOT claimed.

## V3 ingest (landed rows)

`candidate_action_evaluation_g1b_pathb_ep273.v1.json` + `campaign_decision_g1b_pathb_ep273.v1.json` in the
run dir (vehicle=snerv, authority_tier=exact_cpu_advisory, metric_family=exact_pair_scorer,
pays_rent=False, ΔS_vs_frontier=+387.45, decision=INSPECT_BINDING_CONSTRAINT binding=rate, next-action
case=A `seg_descended_pose_sane_carrier_works`). Bridge:
`g1b_path_b_bridge_exact_eval.v1.json`. Mechanism-update-eligible ONLY (Vehicle-OS rule 5).

## Supplementary (in flight at memo time)

The path-A carrier advisory (600-pair `run_snerv_advisory`, source-LF + linear-HF baseline — the
training-independent small-bytes reference) runs detached
(`…/g1b_export_binding/g1b_daemon.log`; output flushes into `snerv_g1b_export_binding_verdict.v1.json`).
It quantifies the carrier's OWN LF-quantized rate point for headroom-ladder step 2. The G1b verdict above
does NOT depend on it.

## Cross-refs

`snerv_b_first_scorer_probe_verdict_20260609.md` (the fork this resolves) ·
`snerv_fullstack_extreme_scrutiny_vs_evaluate_py_20260609.md` (B3 export-blocker map; B1 closed by the G1a
run; B3 now measured-around mechanically) · `feedback_z8_600pair_byte_closed_contest_score_advisory_landed_20260531.md`
+ `feedback_z8_detail_entropy_headroom_report_landed_20260531.md` (the sister wavelet-storage rate anchors)
· `docs/vehicle_operating_system.md` (L3 archive-real reached mechanically at the packet level; L4 advisory
row landed; L7 requires contest-axis) · operator chain name→mechanism→gradient→effect→authority: links 1–4
held, link-5 fidelity now CROSSES at the receiver surface; the remaining gap to authority is rate + the
contest-axis replay.
