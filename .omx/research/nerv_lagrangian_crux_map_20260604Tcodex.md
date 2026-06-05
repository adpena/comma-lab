# NeRV Lagrangian Crux Map - 2026-06-04 Codex

Axis: `[planning/control:false-authority]`. No score, promotion, rank, kill, or exact-eval dispatch authority.

## SNeRV

- SNAR2 header-pruned packet input: `139123` bytes.
- `SNSA2` binary step-map candidate packet: `53596` bytes (`-85527` delta).
- Step-map section: `105591` -> `20064` bytes (`-85527` delta).
- Candidate `archive.zip`: `142134` bytes; `0.bin` deflates to `51586` bytes.
- Runtime closure audit: runtime members deflate to `87364` bytes, runtime Python to `87038` bytes, and whole-file unreachable runtime bytes are `0`.
- Upstream-shaped data-only submission bundle: `51694` archive bytes (`-90440` versus self-contained archive) with full receiver proof passed.
- Automated upstream CPU eval gate: return code `0`, score `90.61`, PoseNet `162.09104919`, SegNet `0.50314105`, rate `0.00137684`, with inflated raw output hashed then deleted.
- Upstream eval feedback row now feeds the long-training campaign planner as
  `family_upstream_eval_gate_context`; the SNeRV queue row is `disabled` with
  `queue_launch_blockers=["snerv_upstream_eval_gate_score_bad"]`.
- Post-`SNSA2` generic section optimizer saved `0` bytes.
- Proof: decoded step maps exact equal = `True`, runtime consumption proof passed = `True`, packaged `inflate` import smoke passed = `True`.

Next SNeRV crux: upstream `evaluate.py` charges only `submission_dir/archive.zip`, while `inflate.sh` runs from the submission directory, and the data-only archive is evaluator-runnable. The immediate byte crux is closed for this packet, but the scorer crux is now confirmed and pipeline-enforced: the current representation is scorer-bad despite excellent rate and must not launch as more same-family long training. If runtime source is treated as charged by a stricter internal compliance view, the next runnable byte lane is candidate-specific branch/symbol specialization of `inflate.py` and receiver modules, blocked by `runtime_source_minification_not_materialized`, `identifier_renaming_not_implemented`, and `minified_runtime_receiver_replay_missing`.

## HiNeRV

- Epoch-7749 `int16_brotli_q11` archive: `124481` bytes.
- Epoch-7749 `int16_hi_ac_brotli_q11` archive: `123625` bytes (`-856` delta).
- Generic section optimizer saved `0` bytes for both byte-closed exports.
- Receiver proof ready: Brotli `True`, high-byte arithmetic `True`.

Next HiNeRV crux: the arithmetic latent codec is a small measured win, but the authority blocker is fit-scale/scorer/exact-eval closure, not another generic compressor pass.

## Artifacts

- JSON: `.omx/research/nerv_lagrangian_crux_map_20260604Tcodex.json`
- SNeRV materialization: `/Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_materialized_20260604Tcodex/snerv_step_map_compaction.json`
- SNeRV runtime closure audit: `/Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_materialized_20260604Tcodex/snerv_runtime_closure_audit.json`
- SNeRV upstream-shaped bundle: `/Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_upstream_submission_20260604Tcodex/snerv_upstream_submission_bundle.json`
- SNeRV upstream eval gate: `/Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_upstream_submission_20260604Tcodex/snerv_upstream_eval_gate.json`
- SNeRV upstream eval planner feedback: `/Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_upstream_submission_20260604Tcodex/upstream_eval_gate_20260604Tcodex/snerv_upstream_eval_candidate_feedback_row.json`
- Planner smoke consuming feedback: `.omx/research/nerv_long_training_campaign_plan_20260604Tupstream_feedback_gateblocked_codex.json`
- SNeRV upstream-shaped runtime audit: `/Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_upstream_submission_20260604Tcodex/snerv_upstream_submission_runtime_audit.json`
- SNeRV section optimizer: `/Volumes/VertigoDataTier/pact/snerv_step_map_snsa2_materialized_20260604Tcodex/snerv_post_snsa2_section_payload_grammar_optimizer.json`
- HiNeRV Brotli export: `/Volumes/VertigoDataTier/pact/hinerv_epoch7749_reexport_section_telemetry_20260604Tcodex/hinerv_checkpoint_archive_export.json`
- HiNeRV high-byte arithmetic export: `/Volumes/VertigoDataTier/pact/hinerv_epoch7749_hi_ac_latent_codec_20260604Tcodex/hinerv_checkpoint_archive_export.json`
