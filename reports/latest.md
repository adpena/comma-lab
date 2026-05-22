<!--
generated_at: 2026-05-22T15:02:28Z
from_state_hash: frontier_scan_dqs1_gap_uleb_exact_cpu_cuda_20260522T1502Z
regenerated_by: codex:dqs1_gap_uleb_exact_cpu_recovery_20260522
last_refreshed_at: 2026-05-22T15:02:28Z
last_refreshed_by: codex:dqs1_exact_recovery_and_gap_uleb_20260522
last_refreshed_head: a12c231f9
last_refreshed_note: |
  Current frontier/status refresh after DQS1 top32 raw-u16 exact Modal recovery
  and compact gap-ULEB runtime hardening. The raw-u16 DQS1 top32 archive is now
  superseded on the contest-CPU axis by the compact gap-ULEB DQS1 archive at
  0.19202894881608987. MLX remains non-authoritative local research signal.
  The compact gap-ULEB packet passed raw-output locality and exact Modal
  recovery on both CPU and CUDA; CPU improves the frontier, CUDA regresses.
-->

# Comma Lab - Current Frontier Snapshot - 2026-05-22 UTC

> **2026-05-22 refresh note**: compact DQS1 top32 `sorted_gap_uleb` exact
> Modal recovery moved the scanner-derived `[contest-CPU]` best to
> `0.19202894881608987`. The same archive regresses on `[contest-CUDA T4]`
> at `0.22619043540195719`, so this is a CPU-axis frontier move only.
> Historical roadmap sections below remain retained for context; use `.omx/state/current_focus.md`,
> `.omx/state/next_experiments.md`, and the dated `.omx/research/` ledgers for
> detailed queue routing.

## FRONTIER (scanner-derived from canonical state)

> This section is the scanner-derived citation surface. Catalog #316 checks it
> against `.omx/state/continual_learning_posterior.json`,
> `.omx/state/active_lane_dispatch_claims.md`, and
> `.omx/state/modal_call_id_ledger.jsonl`. Regenerate/check manually with:
>
>     .venv/bin/python tools/scan_best_anchor_per_axis.py
>
> This is a strict preflight citation gate, not score authority. Per CLAUDE.md
> "Submission auth eval — BOTH CPU AND CUDA" non-negotiable, only 1:1
> contest-compliant hardware (Linux x86_64 + recognized GPU class) qualifies.
> macOS-CPU advisory / MPS rows are excluded.

### Current best - last rechecked 2026-05-22T15:02Z

| Axis | Best score | Archive sha256 (first 12) | Hardware | Lane |
|---|---|---|---|---|
| **`[contest-CPU Linux x86_64]`** | **0.1920289488** | `e12f5cfe93f9` | linux_x86_64_cpu | `lane_dqs1_top32_gap_uleb_selective_decoderq_exact_cpu_20260522` |
| **`[contest-CUDA T4]`** | **0.2053300290** | `9cb989cef519` | linux_x86_64_t4 | `lane_pr106_format0d_latent_score_table_20260516_contest_cuda` |

### 2026-05-22 MLX portable-local-substrate refresh

- MLX scorer-response remains **local candidate-generation and spend-triage
  signal only**. It is not auth-eval evidence, not rank/kill evidence, and not
  promotion evidence.
- The public-frontier calibration gate is `strict_pass` on PR110/101/103/102:
  certified pairwise decisions `6/6`, uncertain `0/6`, calibration uncertainty
  `1.7603544242461577e-05`, and recommended minimum MLX gap for spend triage
  `8.801772121230789e-05`.
- Auth-side MLX calibration targets must now pass the strict contest-auth-axis
  contract in `tac.auth_eval_schema`: full-sample `contest-CPU` /
  `contest_cpu` or `contest-CUDA` / `contest_cuda` only. Advisory/proxy/local
  diagnostic payloads are debug or historical signal, not transfer authority.
- The OOF response dataset is same-axis local signal with `600` rows across
  `mlx_scorer_response` and `mlx_decoder_q`; it is explicitly
  `score_claim=false`, `promotion_eligible=false`,
  `rank_or_kill_eligible=false`, `ready_for_exact_eval_dispatch=false`, and
  `promotable=false`.
- Rich-identity MLX production contracts are now required before planner rows
  can be used for exact-eval spend triage. The dataset-verified PR101 rich
  bundle at
  `experiments/results/mlx_strict_score_calibration_pr101_pose_axis_20260522/candidate_production_contract_bundle_v2_dataset_verified_rich_identity.json`
  has SHA-256
  `18e1cc7d78318fed7f50cc4702ac89d4d32d9727ea716f665c25f5936dc676ae`,
  `dataset_coverage_gate.status=strict_pass`, and one matched MLX row.
- `tools/plan_ll_scorer_response_next.py` now has
  `--require-effective-mlx-spend-triage`. Automation can fail closed with exit
  code `2` unless `ll_effective_mlx_spend_triage_gate.v1` is a strict pass,
  while still writing JSON/Markdown outputs for inspection.
- MLX scorer-response cache loading now verifies cached array hashes and `.npy`
  artifact hashes before scoring, so mutated local tensors cannot silently feed
  planner rows or calibration artifacts.
- The decoder-q response surface found useful high-leverage atoms, but the
  first surface-guided waterbucket batch is now **blocked** for exact-eval
  spend: five unique fixed-length archives changed all 600 frames, preserved
  archive byte count, and all regressed on `[macOS-CPU advisory decoder-q]`.
- Current decoder-q first probe is
  `ll_decoder_q_surface_sign_calibration_repair`, not exact CUDA dispatch.
  Treat the advisory-negative batch as labeled sign-calibration data.
- The strict full-600 MLX decoder-q selector now has a bridge work order:
  `decoder_q_selective_window_bridge_plan.v1` at
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/decoder_q_selective_window_bridge_plan_top32.json`
  (SHA-256
  `f98e3017e8568bf36b304b5c053b6c7b4afc0e7ece8ef6d3f927849114765202`).
  It maps the top 32 observed MLX singleton windows to the existing
  `d1f1e56e042692f2` materialized decoder-q mutation. Runtime grammar is now
  implemented as an archive-member `DQS1` trailer. Planner and local advisory
  artifacts remain non-authoritative; exact recovered auth payloads below carry
  the score axis.
- Decoder-q selective packet planning and materialization now have L0 byte
  accounting for the same bridge. Singleton pair `501` maps to frames
  `[1002, 1003]` under `pair_all_frames` with a charged `DQS1` payload of `13`
  bytes and estimated rate-score delta `0.000008656166390588228`; top32 maps to
  `64` affected frames with a charged `DQS1` payload of `75` bytes and
  estimated rate-score delta `0.00004993942148416285`. Plan artifacts:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/decoder_q_selective_runtime_packet_plan_singleton_pair501.json`
  and
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/decoder_q_selective_runtime_packet_plan_top32.json`.
  The canonical top32 materialized local packet is
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/selective_runtime_candidate_append_tail_top32/submission_dir/archive.zip`
  with archive SHA-256
  `3c4e15bfe7ae1004ad23e89a52c2836e609c1f99e25b58f45c01747226705d59`,
  member `x` length `178492`, and `+75` bytes versus FEC6.
  Official raw-output locality controls pass for top32:
  selected-frame hash `cff4cc7400011f2d1cbe0e4a19c7aeb66af1b409d0cdad9868b86edcb23e2cea`
  equals the full global decoder-q runtime, unselected-frame hash
  `d99cf5d5f735249a66bea6a8ed02bd45d44abbf9e79e20cfa801756cb95d0654`
  equals the parent, and all mismatch counts are `0`.
  Control artifact:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/selective_runtime_candidate_append_tail_top32/locality_controls_top32.json`.
  The stale-path local scorer smoke has the same archive SHA and raw-output SHA
  as the canonical top32 locality output; as `[macOS-CPU advisory]` only, it
  reported canonical score
  `0.1920602563025898`, archive bytes `178592`, PoseNet distortion
  `0.00002943`, SegNet distortion `0.00055988`, rate `0.004756683286239793`,
  artifact
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/dqs1_top32_cpu_advisory_venv.json`.
  Against the matching local FEC6 advisory baseline
  `0.19206131688110561 [macOS-CPU advisory]`, this is `-0.0000010605785158157577`
  lower despite the `+75` byte trailer, driven by a `-0.000000510000000000007`
  SegNet-distortion shift. Against the Linux x86_64 `[contest-CPU]` frontier
  it was not an apples-to-apples claim; exact CPU/CUDA replay is recorded
  below.
  These local advisory artifacts remain `score_claim=false`,
  `promotion_eligible=false`, `rank_or_kill_eligible=false`, and
  `ready_for_exact_eval_dispatch=false`.
- Exact Modal recovery for the canonical raw-u16 DQS1 top32 archive
  (`3c4e15bfe7ae1004ad23e89a52c2836e609c1f99e25b58f45c01747226705d59`,
  `178592` bytes) passed both axes on `2026-05-22`:
  `[contest-CPU]` score `0.1920502563025898` with SegNet distortion
  `0.00055978`, PoseNet distortion `0.00002943`, and rate component
  `0.11891708215599484`; `[contest-CUDA T4]` score `0.2262117428884571`
  with SegNet distortion `0.00066252`, PoseNet distortion `0.00016845`,
  and the same rate component. The CPU result was a temporary CPU frontier
  improvement, superseded by compact gap-ULEB below; the CUDA result is a
  regression versus the PR106 CUDA anchor.
- Compact DQS1 `sorted_gap_uleb` pair encoding is implemented and tested in
  `fb14164d6`. It reduces the top32 DQS1 payload from `75` bytes to `43`
  bytes and materializes archive
  `e12f5cfe93f9dbf624549466cda62d00a01e10bee8d1e0ea8a635af69247908a`
  (`178560` bytes). Official raw-output locality controls pass with the same
  selected/unselected frame hashes and all mismatch counts `0`. Exact
  `[contest-CPU]` recovery passed at `0.19202894881608987`, with SegNet
  distortion `0.00055978`, PoseNet distortion `0.00002943`, and archive bytes
  `178560`. That is `0.000022368065015737626` below the prior PR101/FEC6
  `[contest-CPU]` frontier. Exact `[contest-CUDA T4]` recovery passed at
  `0.22619043540195719`, with SegNet distortion `0.00066252` and PoseNet
  distortion `0.00016845`; CUDA remains a regression.

### Top-5 per axis (sanity / promotion-candidate queue)

**`[contest-CPU]`**:

1. **0.1920289488** — `e12f5cfe93f9` — `lane_dqs1_top32_gap_uleb_selective_decoderq_exact_cpu_20260522`
2. 0.1920289488 — `e12f5cfe93f9` — duplicate canonical anchor for current CPU frontier
3. 0.1920502563 — `3c4e15bfe7ae` — `lane_dqs1_top32_selective_decoderq_exact_cpu_20260522`
4. 0.1920502563 — `3c4e15bfe7ae` — duplicate canonical anchor for superseded raw-u16 DQS1 top32
5. 0.1920513169 — `6bae0201fb08` — `lane_pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515`

**`[contest-CUDA T4]`**:

1. **0.2053300290** — `9cb989cef519` — `lane_pr106_format0d_latent_score_table_20260516_contest_cuda`
2. 0.2053300290 — `9cb989cef519` — duplicate canonical anchor for current CUDA frontier
3. 0.2063163866 — `56cdd10bdc43` — `lane_pr106_format0c_exact_radix_paired_20260515`
4. 0.2063163866 — `56cdd10bdc43` — duplicate canonical anchor for format0c
5. 0.2063257086 — `5c9ef623a089` — `lane_pr106_hdm11_hlm3_magicless_packetir_format0b_20260515`

### 2026-05-20 operations refresh

- **PR #110** (`hnerv_fec6_fixed_huffman_k16`) is open, non-draft, and
  mergeable-clean at head `ec6cc7f98c16b6ad2db8bc7cde65757bb7993004`. The
  live diff is correctly under
  `submissions/hnerv_fec6_fixed_huffman_k16/`; the root README is restored.
- **FEC6 PR submission custody**: hosted archive release, public source pin,
  D-3 strict compliance clearance, and runtime equivalence are recorded in
  dated `.omx/research/` ledgers. Public maintainer/bot feedback is still
  pending.
- **PR101/FEC6 PacketIR closure**: runtime-consumption proof shows all
  `178417` member bytes consumed, no-op detector passed, and candidate queue
  rebuilt. Authority remains local/compiler-only; no score or promotion claim.
- **VQ K=2 diagnostic**: call `fc-01KS21XSVGM2KJ5ET0ET3YCCFN` completed and was
  terminalized at `2026-05-20T12:10:19Z`; diagnostic CPU score
  `[diagnostic-CPU; score_claim=false]`
  `78.07586900258559`, archive SHA
  `fea2cd8af897fcc22525b86a4a6bc9745b47a385cc83c392e01e56fdb93dda76`,
  `score_claim=false`, `promotion_eligible=false`.
- **Next chosen frontier-moving path**: Rule #6 / FEC6 component-moving packet
  path, recorded at
  `.omx/research/codex_chosen_frontier_path_rule6_fec6_component_moving_20260520T121606Z_codex.md`.
  Same-runtime byte polish is below the ~78 charged-byte threshold, so the next
  useful artifact must change Seg/Pose components or add a consumed procedural
  residual layer before paired exact eval.

### 2026-05-21 MLX local-acceleration refresh

- **Frontier unchanged after current scan**:
  `tools/scan_best_anchor_per_axis.py --format json` was rerun at
  `2026-05-21T23:23Z`. Drift list is empty; best existing `[contest-CPU]`
  remains `0.1920513169` and best existing `[contest-CUDA T4]` remains
  `0.2053300290`.
- **MLX full-cache local advisory identity evidence landed**:
  `.omx/research/codex_findings_mlx_full600_local_advisory_fec6_transfer_calibration_20260521T231339Z_codex.md`
  records the FEC6 full-600 local MLX response against the matching macOS CPU
  advisory payload. After the 2026-05-22 auth-axis contract hardening, this is
  historical advisory/identity evidence only, not an auth-side transfer
  authority payload. Corrected archive bytes are `178517`; `[macOS-MLX
  research-signal]` advisory score is `0.19206194316409206`; delta versus
  `[macOS-CPU advisory]` is `6.26282986443405e-07`. This is not a contest
  score.
- **MLX deterministic pair-windowing landed**:
  `.omx/research/codex_findings_mlx_scorer_response_windowing_20260521T232100Z_codex.md`
  adds `--start-pair` / `--max-pairs` to
  `tools/run_mlx_scorer_response_cache.py`, enabling deterministic slice-level
  scorer-response runs without rebuilding full caches.
- **MLX profiler landed**:
  `.omx/research/codex_findings_mlx_scorer_response_profiler_20260521T232300Z_codex.md`
  adds `tools/profile_mlx_scorer_response_cache.py`. A real FEC6 window smoke
  over pairs `[16, 20]` measured CPU `batch_pairs=2` at
  `1.1174859826486507` pairs/s versus `0.94424426733706` pairs/s for
  `batch_pairs=1`; SegNet SHA was identical and PoseNet drift was low-order.
  The profiler output is candidate-generation only.

### 2026-05-22 MLX auth-cache full-600 contract addendum

- **FEC6 plus decoder-q full-sample parent contracts closed**:
  `.omx/research/codex_findings_mlx_decoderq_full600_parent_contract_20260522T123316Z_codex.md`
  records strict full-600 local MLX parent contracts for both auth-cache-backed
  FEC6 and decoder-q. Bundle
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/mlx_parent_contract_bundle_full600_fec6_decoderq.json`
  passed with `2/2` strict contracts; refreshed parent plan
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/parent_production_contract_plan_full600_fec6_decoderq.json`
  is `strict_pass` over `1200` MLX rows. This is `[macOS-MLX
  research-signal]` only: no score claim, no rank/kill authority, no
  promotion, and no exact-eval dispatch readiness.
- **Full-parent ordering is certified for spend triage, not promotion**:
  FEC6 full-parent MLX score is `0.1920527920355189 [macOS-MLX
  research-signal]`; decoder-q full-parent MLX score is `0.1924459939299716
  [macOS-MLX research-signal]`. Decoder-q is worse than FEC6 by
  `0.00039320189445271603 [macOS-MLX research-signal]`, matching the
  `[contest-CPU]` ordering and clearing the calibrated spend-triage minimum
  gap `7.375772066442465e-06`.
- **Window signal remains useful**:
  The full-600 same-axis dataset has `170` decoder-q singleton windows that
  improve over the FEC6 baseline; best singleton delta is
  `-0.0020326847010743165 [macOS-MLX research-signal]`. Those windows may feed
  local byte-closed candidate construction, but every promoted candidate still
  requires claimed exact CPU/CUDA auth eval.
- **OOF selector hardening landed**:
  `.omx/research/codex_findings_mlx_oof_selector_gate_20260522T131934Z_codex.md`
  records the expanded grouped OOF pass. Overall validation still passes
  (`r=0.32517898715374266`), but decoder-q candidate-family utility fails
  (`r=-0.0756861363816352`, zero negative decoder-q predictions, top-8 overlap
  `0/8`). Automation now records `prediction_spend_triage_usable=false` and
  the planner emits `do_not_use_oof_predictions_for_spend_triage_selection`.
  Observed strict-gated MLX windows remain usable as candidate-generation
  inputs only: the top-32 selector has `148` eligible rows above the calibrated
  MLX gap, `32` selected rows, and `0/32` prediction agreement, so OOF
  predictions are explicitly not the selector yet.
- **Grayscale LUT timeout recovery produced archive bytes**:
  `.omx/research/codex_findings_grayscale_lut_export_recovery_20260522T123316Z_codex.md`
  records local export-only recovery from the Modal A100 `best.pt`. The
  recovered archive ZIP SHA-256 is
  `99203f6b0858e8bd54bbc8b88b0a1583ed49f4c75d75590c1ce1951ecfcfda13`; it is a
  local archive artifact with `score_claim=false`, pending normal contest-axis
  auth eval.

### 2026-05-21 zero-spend cascade refresh

- **Frontier unchanged after scan**: `tools/scan_best_anchor_per_axis.py`
  scanned `222` anchors (`202` qualifying). Best existing `[contest-CPU]`
  remains `0.1920513169` at archive SHA
  `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`;
  best `[contest-CUDA T4]` remains `0.2053300290` at archive SHA
  `9cb989cef519ed1771f6c9dc18c988ee93d01a2925da1913d63f9015d6247cf4`.
- **Null-byte matrix surfaced a real but not-yet-materialized budget**:
  `.omx/research/null_byte_matrix_across_11_anchors_20260520.md` records
  `16,292` fec6 frontier null-space bytes (`9.131%`) at the master-gradient
  surface. This is candidate routing signal only; runtime-consumed deletion or
  procedural reconstruction still needs a byte-closed adapter.
- **Simple selector-seed adapter is closed for now**:
  `.omx/research/pr101_seeded_selector_adapter_probe_20260520T232000Z_codex.md`
  and `.omx/research/pr101_seeded_selector_adapter_profile_20260520T232000Z_codex.md`
  show the best charged seed/residual selector candidate is `326` bytes, a
  `-77` byte regression versus the live `249` byte FEC6 selector.
- **FEC7 selector entropy replacement is closed for now**:
  `.omx/research/pr101_fec7_selector_entropy_profile_20260521T000900Z_codex.md`
  shows a `241` byte zero-overhead global entropy floor, but the best charged
  FEC7 prototype is `268` bytes (`-19` byte saving versus FEC6). It does not
  meet the `78` charged-byte materiality threshold.
- **Sidecar-only grammar re-encode is closed for now**:
  `.omx/research/pr101_fec6_sidecar_reencode_probe_20260520T235543Z_codex.md`
  proves the live PR101/FEC6 sidecar re-encodes byte-identically at `607` bytes
  after correcting the exact dim/no-op rank width semantics. Byte movement
  needs a semantic/runtime adapter or a newly produced residual stream, not
  post-hoc wrapping of the current sidecar.
- **Magic-codec DWT detail residual rescue path is falsified**:
  `.omx/research/magic_codec_dense_streams_dwt_residual_smoke_landed_20260520.md`
  records direct empirical detail-subband brotli at `131,779` bytes versus
  procedural-plus-dense-stream residuals at `187,054` bytes (`+55,275` bytes;
  empirical score-direction regression), so that pair is not the next paid
  path.
- **Magic-codec pair #2 sparse PacketIR/SRL1 on FEC6 null-byte residuals is
  falsified**:
  `.omx/research/magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_smoke_landed_20260521T002120Z_codex.md`
  corrects the key interpretation bug (`null byte` = master-gradient-null
  position, not literal `0x00`) and records the full-scale result: `16,292`
  in-place charged bytes versus `97,473` procedural-plus-SRL1 residual bytes
  (`-81,181` bytes saved, empirical advisory Delta S `+0.054055`, z-score
  `101.18`, `CARGO-CULTED`). This pair should not receive paid eval.
- **Residual-hybrid magic-codec accounting is now separated from replacement
  savings**:
  `.omx/research/procedural_predictor_residual_savings_equation_landed_20260521T010524Z_codex.md`
  registers canonical equation
  `procedural_predictor_plus_residual_correction_savings_v1`. It uses
  `Delta S_rate = 25 * (K_predictor + R_residual + H_envelope - N_original) / 37,545,489`
  and carries the pair #1 and pair #2 smokes as zero-residual anchors under
  the corrected formula. This preserves the magic-codec residual paradigm but
  prevents future residual-hybrid work from reusing equation #26's direct
  codebook-replacement savings predicate.
- **PACT-NERV-DistilledScorer now has a fail-closed LL scorer-response dataset
  hook**:
  `.omx/research/pact_nerv_distilled_scorer_ll_dataset_hook_landed_20260521T011316Z_codex.md`
  wires the PDS substrate to the Codex LL scorer-response surface with
  `CONSUMES_SCORER_RESPONSE_DATASET = True` and
  `load_scorer_response_distill_rows(...)`. The loader only accepts
  `scorer_response_dataset.v1` data with explicit false authority on the
  dataset, authority block, and rows; row-level
  `authority_source_score_claim` must also be explicit false. This provides
  fail-closed plumbing for future Quantizr/Hinton KL-T=2.0 training-signal use;
  empirical usefulness remains gated by paired smoke / Stage 1 work. Historical
  PR110 datasets missing only the later `rank_or_kill_eligible`/`promotable`
  fields require an explicit legacy flag and still fail closed on missing core
  authority or source-score-claim ambiguity.
- **Legacy LL scorer-response datasets now have an explicit-authority
  normalization path**:
  `.omx/research/scorer_response_dataset_explicit_authority_normalizer_landed_20260521T012921Z_codex.md`
  adds `tools/normalize_scorer_response_dataset_authority.py` and
  `normalize_legacy_response_dataset_authority(...)`. The normalizer only
  backfills historical missing `rank_or_kill_eligible`/`promotable` fields as
  false; missing core authority or ambiguous source-score authority still
  fails closed. The historical PR110 scorer-response dataset was normalized to
  an explicit-authority JSON artifact and strict-loaded by the PDS hook without
  legacy flags.
- **Magic-codec pair #4 procedural-seed orthogonality is closed**:
  `.omx/research/magic_codec_pair_4_procedural_seed_orthogonality_smoke_landed_20260521T004054Z_codex.md`
  tests seed lengths `16/32/64/128/256` across six reversible byte-orderings
  plus non-free ordering controls. Raw seed bytes dominate all `30/30`
  canonical reversible rows; the best non-raw wrapper is still `+4` bytes.
  Value-dependent sorted controls are excluded because their inverse
  permutation is not free. Verdict:
  `PAIR_4_BOUNDARY_VALIDATED_RAW_SEED_DOMINATES`; keep procedural seeds raw
  and route magic-codec only to residual streams.
- **LL scorer-response planner now consumes the pair #4 guard**:
  `.omx/research/ll_scorer_response_pair4_guarded_plan_landed_20260521T004628Z_codex.md`
  wires the pair #4 boundary smoke into `tools/plan_ll_scorer_response_next.py`
  via `--magic-codec-seed-boundary-smoke`. The fresh guarded plan adds
  `do_not_wrap_procedural_seed_bytes_with_magic_codec` alongside the existing
  sparse-residual widening prohibition, while preserving null-byte priority
  rows as non-promotional LL training-harvest priors.
- **LL pair #4 authority boundary hardened after adversarial review**:
  `.omx/research/ll_pair4_authority_hardening_landed_20260521T005829Z_codex.md`
  makes pair #4 seed-boundary and null-byte-matrix consumers require explicit
  false authority fields by default. The historical null-byte matrix can still
  be consumed only with
  `--allow-legacy-null-byte-matrix-missing-authority`, and the regenerated plan
  records the accepted missing authority fields. Pair #4 empirical result is
  unchanged: raw procedural seed bytes dominate `30/30` canonical reversible
  rows, while the best non-raw wrapper remains a `+4` byte regression.
- **LL frame/pair curriculum regenerated from the guarded plan**:
  `.omx/research/ll_frame_pair_curriculum_pair4_guarded_landed_20260521T004853Z_codex.md`
  records the pair #4 guarded curriculum. Top frames are
  `15,13,5,9,11,7,1,3` (all odd pair-last SegNet-visible frames); top pairs
  are `7,6,4,2,5,3,1,0`. The artifact emits `11` masked adjustment layers and
  carries both prohibitions through the response policy.
- **DP1 procedural paired-harvest bridge is now the concrete next gate**:
  `.omx/research/dp1_procedural_paired_harvest_planner_landed_20260521_codex.md`
  adds `tools/plan_dp1_procedural_paired_harvest.py` and
  `tac.optimization.dp1_procedural_paired_harvest_plan`. The planner refuses
  false authority, requires byte-closed `archive.zip` + `submission/inflate.sh`
  + manifest/provenance custody, and emits only the real paired dispatcher
  surface (`tools/dispatch_modal_paired_auth_eval.py`) with
  `--expected-runtime-tree-sha256 auto` and
  `--skip-axis-if-promotable-anchor-exists`. Current probe status is correctly
  blocked on missing harvested DP1 output directories for the baseline and
  procedural arms; no spend or score claim was attempted.
- **Current highest-signal follow-on**: preserve the master-gradient,
  per-frame/per-pair, byte/pixel sensitivity surfaces as routing authority, but
  stop direct FEC6 member-byte substitution attempts unless they change
  components or supply a consumed residual/runtime adapter. Next byte-closed
  work should run the operator-gated DP1 baseline + procedural recipes, feed
  their harvested output dirs into the DP1 paired-harvest planner, and then run
  the emitted paired CPU/CUDA exact-eval commands. LL scorer-surrogate/frame-
  pair planner work should continue consuming the null-byte matrix and
  per-frame decomposition as prioritization signal; residual-hybrid work should
  keep using the predictor-plus-residual accounting predicate rather than
  equation #26 direct-replacement savings.

### Comparison vs public PRs

- **PR101 GOLD** (`qpose14_qzs3_filmq9g_slsb1_r55`): 0.193 `[contest-CPU]` → **we beat by 0.00095 on CPU axis** with `6bae0201`.
- **PR102 (bronze)**: 0.19538 `[contest-CPU]` / 0.22839 `[contest-CUDA]` → **we beat by 0.0033 CPU + 0.023 CUDA**.
- **PR103 (silver)**: ~0.195 `[contest-CPU]` → **we beat by 0.0028 CPU**.

### Technique inventory of frontier lanes

- **fec6** (`pr101_frame_exploit_selector_fec6_fixed_huffman_k16`): PR101 GOLD decoder + frame-conditional K=16 codec selector with fixed-Huffman entropy coding.
- **format0d** (`pr106_format0d_latent_score_table`): PR106 latent score-table grammar with exact-radix sidechannel + autohash runtime closure.
- **format0c sister** (`pr106_format0c_exact_radix_paired`): same family, prior format version.
- **format0b** (`pr106_hdm11_hlm3_magicless_packetir_format0b`): PR106 magicless PacketIR variant.

### Strategic implications

- The Top-5 CUDA bucket is dominated by PR106 `format0*` family (alien-tech routing per `feedback_a1d3dd050fc09dc54`) — a clear architectural cluster worth deeper investigation.
- The CPU/CUDA bifurcation is real: best-CPU is fec6 (PR101 family) but best-CUDA is format0d (PR106 family). Different lanes win different axes. **A binding artifact that combines fec6's selector + PR106 format0d's score-table grammar + FP4 codebook + Ballé hyperprior + Cheng2020 has every primitive we'd need for a dual-axis frontier sweep.**
- **Per-byte sensitivity comparative analysis (Wave 3, 2026-05-20).** The 21-pair cross-candidate matrix empirically established the HNeRV-class backbone is class-saturated at the 0.19xxx medal cluster (Pearson seg ρ=0.961 between PR101 and fec6 on shared 178,158 backbone bytes); the entire +794 ppm fec6 advantage is concentrated in +259 bytes of orthogonal selector overhead. PR106-vs-HNeRV-family pairs classify SUPER_ADDITIVE (top-K Jaccard 0.000) per canonical equation `cross_codec_super_additive_orthogonality_predictor_v1`, structurally supporting the dual-axis stack candidate. Discipline + methodology: [`docs/per_byte_sensitivity_comparative_analysis_methodology.md`](../docs/per_byte_sensitivity_comparative_analysis_methodology.md). Selector-extensions class enumeration: [`docs/asymptotic_floor_candidate_inventory.md`](../docs/asymptotic_floor_candidate_inventory.md) §C.12.

### G1 CPU-axis optimization — 2026-05-18 Codex local rerank

Per `codex_routing_directive_rate_attack_vector_2_g1_cpu_axis_optimization_20260518.md`, Codex added a $0 CPU-axis-only reranker over canonical `tac.frontier_scan` anchors. Result: no existing qualifying Linux x86_64 CPU anchor beats the current CPU frontier.

Evidence report: `experiments/results/g1_cpu_axis_re_rank_20260518T214250Z/report.json`.

| Metadata bucket | CPU score | Archive sha256 (first 12) | Lane / source metadata |
|---|---:|---|---|
| pr101 | 0.1920513169 | `6bae0201fb08` | `lane_pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515` |
| pr106 | 0.2271259174 | `9cb989cef519` | `lane_pr106_format0d_latent_score_table_20260516_contest_cpu` |
| pr103 | 0.1948697174 | `7d1e46331a04` | `pr103_global_combo_mid32_latent_hi_brotli_retune_exact_cpu` |
| other | 0.1928475774 | `87ec7ca5f2f3` | `hnerv_ft_microcodec` |

**G1 verdict:** `current_frontier_remains_cpu_optimal`; delta vs current CPU frontier = `+0.0000000000` (CPU-axis delta, not a frontier score citation). The 2026-05-21 recheck scanned 222 canonical anchors, including 67 qualifying CPU anchors from the live `accepted_anchor_history` posterior plus dispatch-claim surfaces. This does not retire CPU-axis-specific optimization as a class; it closes the zero-cost existing-anchor rerank probe and leaves any future G1 improvement dependent on new paired Linux x86_64 CPU anchors.

### ASYMPTOTIC PURSUIT pivot (2026-05-17)

Per operator directive 2026-05-17 verbatim *"Want to keep pushing to asymptotic as top priority"* + Option B falsification (WZ-on-existing-archives DEFERRED; top aggregate ΔS = 0.000421 << 0.001 leaderboard precision floor). ASYMPTOTIC pivot must come from **substrate-class-shift methods** per CLAUDE.md "Substrate retirement discipline" + HORIZON-CLASS council 2026-05-17 Stage 2 deferred queue.

**Canonical readiness assessment**: `tools/asymptotic_pursuit_candidate_readiness_assessment.py` + dispatch queue `tools/asymptotic_pursuit_dispatch_queue.py`. Artifact: `.omx/state/asymptotic_pursuit/readiness_assessment_*.json`.

**Current 7-candidate readiness matrix** (refreshed 2026-05-18 by Codex; artifact `.omx/state/asymptotic_pursuit/readiness_assessment_20260518T040149Z.json`):

> Historical matrix retained from 2026-05-18. Superseding status: Z6 candidate
> 4c is not currently launchable until the driver/full-mode fix and a fresh
> smoke-before-full pass land; see the DEFER-PENDING-DRIVER-FIX section below.

| Substrate | Verdict | Council | Recipe | Predicted ΔS | Cost | Blocking |
|---|---|---|---|---|---|---|
| `z6_v2_candidate_4c_scorer_logit` | **READY** | NO_DELIBERATION | dispatch-enabled | [0.11, 0.17] *asymptotic_pursuit planning prior* | $1.08 full / $2.18 paired sequence | L1 registered; score_claim=false; promotion_eligible=false |
| `time_traveler_l5_autonomy` | NEEDS_FIX | NO_DELIBERATION | dispatch-enabled | unknown | $27.43 | stale Dykstra/composition blockers cleared; current blockers are C1/Z5/TT5L probe-gate semantics, prediction-band custody, and provider capacity/runtime readiness |
| `time_traveler_l5_z6` | DEFER | PROCEED_WITH_REVISIONS | research_only=true | [0.13, 0.16] *frontier_pursuit* | $1.08 | Catalog #313 Z6 Wave 2 DEFER + Catalog #315 + recipe blockers |
| `atw_codec_v2` | DEFER | PROCEED_WITH_REVISIONS | research_only=true | unknown | $1.92 | **Catalog #313 probe BLOCKED** (INDEPENDENT verdict 2026-05-16) |
| `c6_e4_mdl_ibps` | DEFER | PROCEED | dispatch_enabled=false | suppressed | $0.76 | **Catalog #313 smoke DEFER** + Catalog #324 falsified random-init band suppressed |
| `nscs01_nullspace_split_renderer` | DEFER | PROCEED_WITH_REVISIONS | research_only=true | unknown | $0.43 | Catalog #315 + Phase 2 council |
| `nscs03_end_to_end_balle_joint_codec` | DEFER | PROCEED_WITH_REVISIONS | research_only=true | unknown | $1.01 | Catalog #315 + recipe blockers |

**TOP-1 recommendation: `z6_v2_candidate_4c_scorer_logit`** (READY for an operator-authorized smoke-before-full probe; no spend fired by this report). **TOP-2 (Stage 2 stacking candidate): `time_traveler_l5_autonomy`** (still NEEDS_FIX).

**Candidate 4c is launchable, not promotable.** Its `[0.11, 0.17]` band is tagged `pending_post_training`; it is a planning prior only until paired exact-eval evidence, post-training Tier-C validation, and the full-vs-identity disambiguator land. It is registered as `lane_z6_v2_candidate_4c_scorer_logit_conditioning_20260518` at L1 with implementation, strict-test, memory, and runbook gates marked; no empirical score gates are marked. Byte-consumption/no-op proof for the scorer-logit ego section now passes at `experiments/results/z6_candidate4c_paired_auth_stats_paircap2_codex_20260518T0256Z/scorer_logit_ego_motion_byte_mutation_proof.json` (`0.bin@192390:16`, 2/4 mutations changed output; score_claim=false).

**TT5L blocker refresh:** Dykstra score-axis sanity, move-level feasibility, contest full-frame side-info consumption, first-anchor timing smoke, Lightning preflight, execution bundle, dry-run verification, route-unblock packet, and doctor plan validate through `l5_v2_tt5l_campaign_readiness`; these are not score/rank/promotion authority. Codex removed a false probe-gate `probe_verdict_sha256_mismatch` caused by builder/validator hash-contract drift. Remaining blockers are real fail-closed semantics: `requires_c1_z5_tt5l_probe_disambiguator_before_architecture_lock`, architecture lock still false, probe blockers still nonempty, missing eligible C1/Z5/TT5L observations, `prediction_band_not_dispatch_ready` with missing baseline/anchor custody, Modal workspace billing limit, and missing Lightning target/teamspace/inventory/source/runtime probes.

**ATW V2 is BLOCKED** by Catalog #313 predecessor probe outcome (`atw_v2_d4_h_latent_given_scorer_class_20260516` verdict INDEPENDENT; MI=0.006385 bits/symbol << 0.5 threshold). Per CLAUDE.md "Forbidden premature KILL" + Catalog #308: NOT a class-kill; reactivation requires richer side-information per probe notes.

**C6 original recipe is DEFER, not the cheapest path forward.** Catalog #313 recorded the 3.04 smoke miss, and Catalog #324 suppresses the disabled recipe's stale predicted band until a reactivation branch supplies post-training validation or operator waiver.

**Stage 2 stacking design** (Q9 per HORIZON-CLASS council): TOP-1 ⊕ TOP-2 composition deferred until first ASYMPTOTIC empirical anchor lands. Pre-entropy pairwise composition_alpha probe (OP-3 from HORIZON-CLASS council) is the dependency for any 2+ substrate composition. Per Catalog #227 substrate composition matrix.

**5 op-routables for operator** (NO commits / NO paid dispatch fired by this assessment per operator NON-NEGOTIABLE):

1. If spend is approved, claim `lane_z6_v2_candidate_4c_scorer_logit_conditioning_20260518`, then run `tools/run_modal_smoke_before_full.py --recipe substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch` per Catalog #167.
2. Keep Candidate 4c output non-promotional until paired CPU/CUDA harvest, full-vs-identity disambiguator deltas, archive/runtime custody, and terminal claim rows exist.
3. Run pre-entropy pairwise composition_alpha probe per HORIZON-CLASS council OP-3 ($0 GPU; ~2h editor) before any Stage 2 stacking
4. Resolve `time_traveler_l5_autonomy` probe-gate / prediction-band / provider-capacity blockers if Candidate 4c smoke is weak or if Stage 2 stacking needs a second launchable substrate.
5. Treat C6 original as deferred; work only a named reactivation branch (beta sweep / latent widen / Phase 2 redesign) unless an operator waiver explicitly supersedes Catalog #324.
- A1 (`87ec7ca5f2f3`) at 0.19285 CPU is #3 — anchor for a separate (PR101-grammar minimalist) line.

## Q4 STATUS + PIVOT — REDO+PIVOT 2026-05-17

> Last refreshed 2026-05-17 by REDO+PIVOT subagent (lane `lane_redo_pivot_fix_all_phantom_score_substrate_class_shift_q4_budget_redirect_20260517`). Per CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable.

### Wyner-Ziv on existing archives — DEFERRED-pending-pivot

**Empirical falsification**: Option B archive-member sweep landed 2026-05-17 at `.omx/state/wyner_ziv_deliverability/option_b_archive_member_sweep_20260517T221034.json`:

- **All 8 VALIDATED contest archives at entropy floor** (compression ratio ≥ 0.997 across lzma + brotli + zlib).
- **Top aggregate savings: 0.000332** (apogee_int6_archive) — **below 0.001 leaderboard precision floor**.
- **Verdict**: `recommended_q4_target=None`, `q4_verdict=DEFER_Q4`.
- **Implication**: Pre-entropy Wyner-Ziv hoist of currently-shipping contest archive members offers NO measurable score gain. The archives are already at compression-ratio saturation.

### Phantom-score class extincted by Catalog #321 + #322

The PRIOR Q4 path (sister `a891faa328f9e4754` HALTED) cited 3 candidates (`pr101_state_dict`, `pr106_state_dict`, `posenet_class_sensitivity`) with positive `deliverable_score_savings_estimate` values (0.477 / 0.470 / 11.608) that were **phantom**: those bytes are research sidecars (`.pt` state dicts under `.omx/tmp/`), NOT contest `archive.zip` members. Catalog #321 (entry-side phantom revert) + Catalog #322 (downstream autopilot-consumer phantom revert) STRICT-from-byte-one extinct this class structurally at 5 surfaces. The 2 phantom-provenance `pairwise_alpha_*.json` artifacts (Q6 OP-3 extended sweep) are quarantined to `.omx/state/wyner_ziv_deliverability/quarantine_phantom_pre_catalog_322/`.

### Q4 BUDGET REDIRECT — top-1 substrate-class-shift candidate

Per HORIZON-CLASS Stage 2 deferred plan (CLAUDE.md Catalog #309 horizon_class taxonomy) + Catalog #310 (PRIMARY class-shift not bolt-on) + Option B `q4_alternative_paths`, Q4's redirected $0.70 budget should target **substrate-class-shift methods** that are structurally distinct from the within-class (PR101/PR106/A1) plateau cluster:

| Rank | Substrate | Lane | Predicted band | Status |
|------|-----------|------|----------------|--------|
| **#1 redirect target** | **Z6 ego-motion-conditioned predictive coding** (Rao-Ballard + Atick-Redlich + FiLM) | `lane_time_traveler_l5_z6_l1_scaffold_substrate_build_20260516` | [0.13, 0.16] [Dykstra-validated] | Phase 1B lift landed 2026-05-16; recipe `substrate_time_traveler_l5_z6_modal_t4_dispatch.yaml` exists `research_only=true` pending Phase 2 council |
| #2 | C6 IBPS MDL Information Bottleneck Per Pair | `lane_c6_e4_mdl_ibps_substrate_20260514` | Tier-C density pending | L1 substrate scaffold |
| #3 | ATW V2 cooperative-receiver loss (Atick-Redlich) | `substrate_atw_codec_v2_modal_a100_dispatch.yaml` | (V2 D4 probe INDEPENDENT 2026-05-16) | Bolt-on at loss surface |
| #4 | TT5L L5 autonomy (broader staircase) | `lane_time_traveler_l5_autonomy_substrate_20260513` | L5 deferred queue | L1 |
| #5 | Z7 wavelet-multi-scale Daubechies | (Phase 1A scaffold) | HIGHER engineering risk per Z6/Z7/Z8 design memo §22 | L0 scaffold |
| #6 | Z8 Hafner DreamerV3 latent dynamics | (Phase 1A scaffold) | HIGHEST risk; requires Z6+Z7 anchors first | L0 scaffold |

**Q4 redirect stub** at `.omx/operator_authorize_recipes/q4_substrate_class_shift_redirect_z6_predictive_coding_20260517T222800Z.yaml` documents the decision. The actual dispatch routes through the Z6 recipe AFTER Phase 2 council consensus + Catalog #167 smoke-before-full pattern.

### Lanes DEFERRED-pending-pivot (per CLAUDE.md "Forbidden premature KILL")

| Lane | Reason | Reactivation criteria |
|------|--------|------------------------|
| `lane_super_additive_lane_g_v3_siren_topology_integration_20260517` | Sister #823 self-flagged α=4.74 as FALSE_SIGNAL ARTIFACT (byte-identical renderer.bin SIREN timeout); v2 cascade structurally landed for FUTURE genuine SUPER_ADDITIVE discoveries | Re-run pairwise_alpha probe against VALIDATED_CONTEST_MEMBER substrates only |
| `lane_q6_preprobe_pairwise_composition_alpha_20260517` | Phantom-provenance probe inputs (pr101_state_dict / pr106_state_dict / posenet_class_sensitivity) per Catalog #321/#322 | VALIDATED_CONTEST_MEMBER inputs only |
| `lane_batched_815_consumer_15_amendment_plus_816_meta_meta_cleanup_plus_q6_op3_extended_20260517` | Q6 OP3 extended sweep produced phantom-provenance α values | Catalog #321 revalidation |

### Current frontier remains (no Q4 anchor landed)

- **CPU**: 0.19205 [contest-CPU] fec6 (unchanged from 2026-05-15)
- **CUDA**: 0.20533 [contest-CUDA] pr106 format0d (unchanged from 2026-05-16)
- **Mission alignment** (CLAUDE.md): this REDO is `frontier_protecting` (preventing phantom anchor pollution that would corrupt dispatch ranking) + queues `frontier_breaking` substrate-class-shift work via Q4 redirect to Z6.

### First ASYMPTOTIC empirical anchor — C6 IBPS smoke 2026-05-17 — DEFER

**Empirical anchor**: C6 IBPS smoke 50ep Modal A10G `fc-01KRW353MJJ9A6QW8H99QWZEMH` 2026-05-17T23:08:18Z returned final_score=**3.04** [diagnostic-CPU advisory; score_claim=false; archive_sha=`be06a4b0972e6c...`].

- **Predicted smoke band**: `[0.10, 0.30]` (recipe `smoke_score_band`)
- **Actual**: 3.04 — **10× OUTSIDE band**
- **Mechanism (SegNet collapse)**: `score_seg_contribution=2.60` dominates (avg_segnet_dist=0.0260, vs PR101 baseline ~0.001). IB bottleneck (24-dim z) compresses pose well (avg_posenet_dist=0.0081 OK) and rate is fine (0.0060 OK), but the bottleneck destroys segmentation.
- **Training converged** (50ep, A10G, 33min): loss 53.49 → 4.91 (10× reduction)
- **Archive**: 225,157 bytes zip / 226,388 bytes raw; 128K params (encoder=35K + decoder=78K + latents=14K)
- **CPU axis (auth_eval_device=cpu, advisory_only=true)** — Modal injected CPU axis NOT CUDA; rc=1 because dispatch contract required `contest_cuda` axis. Paired full dispatch **NOT FIRED** per briefing "Score outside band → ABORT before full dispatch + emit landing memo + register Catalog #313 probe outcome verdict DEFER."
- **Catalog #313 probe outcome registered DEFER** (`c6_e4_mdl_ibps_smoke_modal_a10g_50ep_fc01krw353mjj9a6qw8h99qwzemh_20260517`; blocker_status=blocking; expires_at_utc=2026-06-16); per CLAUDE.md "Forbidden premature KILL" + "Substrate MUST be at OPTIMAL FORM" — this is **implementation-level falsification** of `beta_ib=0.01 + latent_dim=24`, NOT a paradigm kill.
- **Stage 2 reactivation gate clause #1** (anchor within ±10% of L5 codex band 0.138): **NOT SATISFIED** (3.04 / 0.138 = 22× outside).
- **Frontier beat**: NO — 3.04 ≫ 0.19205 CPU baseline; no axis update.
- **Reactivation queue** (3 paths): (1) `beta_ib` sweep `[0.0001, 0.001, 0.01, 0.1, 1.0]` to balance pose vs seg; (2) `latent_dim` widen 24 → 48/96/192 to give SegNet more capacity; (3) Phase 2 sextet cargo-cult-unwind redesign of SegNet-collapse mechanism per Catalog #303 (decoder output resolution? IB KL too tight? Phase 2 redesign).
- **Mission alignment**: `mission_questioned` — produced a real empirical upper bound for first ASYMPTOTIC class-shift smoke; downstream operator decision: invest in C6 redesign (reactivation queue) or pivot Q4 budget to Z6 / TT5L L5 autonomy per the readiness matrix above.
- Memory: `feedback_c6_ibps_first_asymptotic_dispatch_smoke_before_full_paired_landed_20260517.md`.
- Lane: `lane_c6_ibps_first_asymptotic_dispatch_smoke_before_full_paired_20260517` (L1; gates: real_archive_empirical + memory_entry).

### Z6-v2 Wave 2 first DISPATCH attempt 2026-05-18 — DEFER-PENDING-DRIVER-FIX (implementation-level falsification)

**Wave 2 dispatch attempt** (task #847): Z6-v2 Candidate 1 (Multi-layer FiLM depth=3 ~300K params per Phase 3 council §9 binding spec) fired TWO Modal T4 dispatches via paired-env Catalog #199 bypass after operator "All approved be aggressive" 2026-05-17 + Phase 3 council §17 op-routable #1 sign-off + C6 IBPS DEFER outcome 2026-05-17 satisfied council §9 cost-adjusted recommendation.

- **Smoke** (`fc-01KRW7RHFHP640BHTQ0FZM3M38`, T4, 50ep, $0): rc=0 in 10.3s; both archives emitted (52,458 B main + 52,610 B identity-predictor disambiguator at SAME archive bytes per Catalog #229 paired-control marker); `evidence_grade=smoke-no-scorer`; `score_claim_valid=false`; smoke-before-full wrapper marked SMOKE RED per band-enforcement vs synthetic-mode contract mismatch.
- **Full canary** (`fc-01KRW7ZCYK5XF6MSHD24R71A46`, T4, 100ep, $0): rc=0 in 9.1s; SAME `_smoke_main` path executed (`evidence_grade=smoke-no-scorer`, `smoke=1`, `smoke_epoch_cap=3`); produced 27,850-param synthetic-cfg architecture NOT the council-binding ~300K depth=3 hidden_dim=96 spec.
- **Diagnosis**: driver-level bug in `scripts/remote_lane_substrate_time_traveler_l5_z6.sh` `stage_4_trainer_begin` hardcodes `smoke=1` regardless of dispatch env vars (Z6_EPOCHS=100 + Z6_PREDICTOR_ARCHITECTURE=multi_layer_film_depth_3_300k + Z6_PREDICTOR_PARAM_COUNT_TARGET=300000 + Z6_ENABLE_AUTOCAST_FP16=true were threaded correctly but ignored by driver).
- **Implementation-level falsification per Catalog #307**: the depth=3 ~300K Multi-layer FiLM spec was NEVER empirically tested. Atick Council Revision #6 trigger NOT MET (ΔS unmeasured). Per CLAUDE.md "Forbidden premature KILL without research exhaustion": substrate paradigm NOT killed; cargo-cult unwind methodology preserved.
- **Catalog #245**: both call_ids registered in `.omx/state/modal_call_id_ledger.jsonl` with terminal `harvested` events.
- **Catalog #313 probe outcome**: `z6_v2_wave_2_dispatch_smoke_before_full_paired_2026_05_18` verdict DEFER blocker_status=blocking expires_at_utc=2026-06-17T00:30:00Z.
- **Catalog #324 post-training Tier-C re-measurement**: NOT APPLICABLE — neither archive carries the architecture-as-spec'd; post-training Tier-C cannot validate the council-binding spec against synthetic-cfg outputs.
- **Stage 2 reactivation gate clause #1**: NOT SATISFIED.
- **Frontier beat**: NO axis update (no empirical contest-CUDA / contest-CPU score produced; frontier remains `0.19205 [contest-CPU]` / `0.20533 [contest-CUDA T4]`).
- **Total spent**: ~$0.50 across both fires (both 10s well under cost band).
- **Reactivation queue** (3 paths): (1) **driver-fix subagent**: modify `scripts/remote_lane_substrate_time_traveler_l5_z6.sh` to accept `$Z6_TRAINER_MODE` env var (smoke|full) and dispatch trainer accordingly + re-fire $3 envelope per Phase 3 council §9; (2) trainer-side fix: gate `smoke=1` flag in `_smoke_main` invocation on env var rather than hardcoded; (3) Phase 4 council on Wave 2-outcome with empirical anchor (BLOCKED on path 1 or 2 landing first).
- **Mission alignment**: `mission_questioned` — second consecutive ASYMPTOTIC dispatch attempt frustrated by infrastructure-level bug (C6 IBPS by SegNet collapse mechanism; Z6-v2 by driver mode hardcoding). Per CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" + Catalog #315: BOTH attempts demonstrate the iteration discipline IS structurally protecting against premature paradigm KILL by deferring; the next operator action is operator-routed driver-fix (path 1).
- Memory: `feedback_z6_v2_wave_2_dispatch_smoke_before_full_paired_cpu_cuda_landed_20260518.md`.
- Lane: `lane_z6_v2_wave_2_dispatch_smoke_before_full_paired_cpu_cuda_20260518` (L1; gates: impl_complete + memory_entry).

## Executive summary

Substrate canvas grew to 12 wired trainers + 20 operator-authorize recipes
across HIGH-target / MEDIUM-target / TRADITION 2 / EXPLORATORY traditions
during the 2026-05-12 session. Phase B-2 dispatch sweep is BUILD-COMPLETE
for all 6 HIGH-target substrates (sane_hnerv α, balle_renderer β, SIREN,
Cool-Chic, VQ-VAE, self_compress_nn, hybrid_renderer_residual) but BLOCKED
on Modal source-staleness investigation (Catalog #166 STRICT @ 0 — HEAD
parity ledger landed; smoke-before-full pattern Catalog #167 warn-only
pending strict-flip). The NV7 cost-band posterior fix corrected the Modal
A100 prediction by 312× (from `weak_posterior $0.016` to
`hand_calibrated_fallback $5.00`); two failed-dispatch anchors are now
correctly excluded from successful-dispatch percentile bands. Net session
GPU spend: $0.10 (5 failed Modal A100 attempts). $19.90 envelope reserved
for Phase B-2 dispatch wave once the Modal source-staleness verification
gate clears.

## Substrate canvas state (12 trainers + 20 recipes)

### HIGH-target traditions (sub-PR101-gold attack vectors; predicted ≤ 0.18)

| Substrate | Tradition | Predicted score | Recipe wired | Trainer LOC | Status | First-anchor |
|---|---|---|---|---|---|---|
| **siren** | TRADITION 1 (coordinate-MLP) | 0.145 [predicted] | yes | 1085 (45.1K) | L1 SCAFFOLD; dispatch-ready | NOT DISPATCHED |
| **cool_chic** | TRADITION 1 (multi-scale latent + AR prior) | 0.165 [predicted] | yes | ~1100 (45.1K) | L1 SCAFFOLD; dispatch-ready | NOT DISPATCHED |
| **vq_vae** | TRADITION 2 (discrete codebook) | 0.17 [predicted] | yes | 870 (47.9K) | L1 SCAFFOLD; dispatch-ready | NOT DISPATCHED |
| **self_compress_nn** | δ (Selfcomp/Quantizr block-FP) | 0.17 [predicted] | yes | ~1200 (55.0K) | L1 SCAFFOLD; dispatch-ready | NOT DISPATCHED |
| **hybrid_renderer_residual** | composite (α + β residual) | 0.17 [predicted] | yes | ~1200 (55.1K) | L1 SCAFFOLD; dispatch-ready | NOT DISPATCHED |
| **wavelet** | TRADITION 1 (Daubechies-4) | 0.175 [predicted, byte-floor blocked] | yes (research_only=true) | ~1100 (49.3K) | L1 DEFERRED-research-only (byte-floor 471M raw subband bytes ≫ N=37,545,489) | DEFERRED |
| **balle_renderer** β | TRADITION 1 (entropy bottleneck + scale hyperprior) | 0.18 [predicted] | yes | ~52K LOC | L1 SCAFFOLD; full_main wired | DISPATCH HALTED on Modal source-staleness |
| **sane_hnerv** α | TRADITION 1 (HNeRV substrate engineering) | ~0.19 [predicted] | yes | ~46K LOC | L1 SCAFFOLD; full_main wired | DISPATCH HALTED on Modal source-staleness (operator: "~0.19 not useful") |

### MEDIUM-target HNeRV-family (DEFERRED pending HIGH-target empirical signal)

| Substrate | Predicted score | Recipe wired | Trainer LOC | Status |
|---|---|---|---|---|
| **tc_nerv** | 0.19 [predicted] | yes | 47.1K | L0 SCAFFOLD (no full_main wire-in) |
| **block_nerv** | 0.19 [predicted] | yes | 45.8K | L0 SCAFFOLD |
| **ff_nerv** | 0.19 [predicted] | yes | 46.3K | L1 SCAFFOLD |
| **ds_nerv** | unknown — first empirical needed | (pending recipe) | 42.8K | L1 (memory-only entry) |

### TRADITION 2 EXPLORATORY (operator decision pending: production target?)

| Substrate | Predicted score | Recipe wired | Status |
|---|---|---|---|
| **lane_12_v2_nerv** | <0.21 sub-Quantizr [predicted] | yes (dispatch_blockers: remote driver missing) | recipe-only |
| **nervdc** | unknown | yes (dispatch_blockers: remote driver missing) | recipe-only |
| **e_nerv** | unknown | yes (dispatch_blockers: remote driver missing) | recipe-only |
| **ego_nerv** | unknown | yes (dispatch_blockers: remote driver missing) | recipe-only |
| **cnerv** | unknown | yes (dispatch_blockers: remote driver missing) | recipe-only |
| **quantizr_faithful** | 0.33 [predicted, first-anchor replication target] | yes (dispatch_blockers: trainer not wired) | recipe-only |
| **mlx_mask_renderer** | [macOS-MLX research-signal — never promotable to contest-CUDA] | yes | local Apple Silicon only |
| **dp_sims_renderer** | unknown | yes (dispatch_blockers: library deps) | recipe-only |
| **diffusion_renderer** | research_only=true, never produces contest archive | yes | research-only |
| **grayscale_lut** | unknown | (pending recipe) | L0 SKETCH |

## Cost-band posterior state (NV7-corrected)

Posterior at `.omx/state/cost_band_posterior.jsonl` — 4 anchors (2 successful_dispatch + 2 failed_dispatch).

| (platform, gpu, epochs_bucket) | n_anchors (success) | n_failed | Confidence | p10 / p50 / p90 USD |
|---|---|---|---|---|
| (modal, A100, 2000-3000ep) | 0 | 2 | hand_calibrated_fallback | 3.00 / 5.00 / 8.00 |
| (local, local_cpu, build-only) | 2 | 0 | weak_posterior | 0.00 / 0.00 / 0.00 (build-only probes) |

**NV7 fix (commit `1e7e1b0d`)**: failed-dispatch anchors NO LONGER poison
percentile bands. The earlier Modal A100 prediction was
`weak_posterior $0.016` (treating a 14.77-second crash as a successful
3000-epoch run); post-fix, `predict('modal', 'A100', 3000)` returns
`hand_calibrated_fallback p50=$5.00` until a real successful dispatch
appends an empirical anchor. Migration script
`tools/migrate_cost_band_posterior_failed_anchors.py` ran under fcntl
LOCK_EX with `.pre_nv7_migration.<sha>.bak` backup.

**Continual-learning posterior** at `.omx/state/continual_learning_posterior.json`:
21 accepted anchors, 11 refused. Schema
`tac_continual_learning_posterior_v1`; evidence_grade
`[continual-learning posterior; non-authoritative]`.

## Catalog # STRICT preflight inventory (165 numbered gates active)

Numbered catalog entries: **82 distinct check function entries** in CLAUDE.md
(numerals span 1-166 with intentional gaps for retired/renumbered checks).
Highest active gates this session:

| Catalog # | Check | Purpose | Status |
|---|---|---|---|
| **#166** | `check_modal_dispatch_verifies_worker_source_matches_head` | HEAD-parity ledger + worker-side source-parity ledger in `experiments/modal_train_lane.py` | STRICT @ 0 (this session) |
| **#167** | `check_smoke_before_full_pattern` | Refuses operator-authorize Modal wrappers that fire `--full` before `--smoke` validation | warn-only initial; strict-flip pending 4 legacy wrapper backfill |
| **#164** | `check_substrate_score_aware_loss_calls_preprocess_input_before_scorer` | AST-walks substrate score_aware loss; refuses bare `self.<scorer>(...)` forward without paired `preprocess_input(...)` | STRICT @ 0 (FIX-H Part 1) |
| **#163** | `check_remote_lane_script_uses_sentinel_when_sourcing_bootstrap` | `scripts/remote_lane_*.sh` sourcing `remote_archive_only_eval.sh` MUST prepend `REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1` | STRICT @ 0 |
| **#161** | `check_quantize_degenerate_range_clamped_correctly` | Substrate archive `_quantize_intN` degenerate-range branch must fill with `-(MAX_LEVELS // 2)` not zero | STRICT @ 0 |
| **#159** | `check_claude_md_catalog_text_matches_preflight_strict_value` | CLAUDE.md catalog text must match `strict=` value wired in preflight | STRICT @ 0 |
| **#158** | `check_deterministic_compiler_canonical_use` | Refuses new packet-compilation surfaces bypassing `tac.packet_compiler.deterministic_compiler` | STRICT @ 0 |
| **#157** | `check_commit_serializer_pre_lock_hash_against_head` | Refuses bare `git commit` outside canonical serializer; sister to `--expected-content-sha256` discipline | STRICT @ 0 |
| **#156** | `check_gc_helper_refuses_delete_on_tracked_paths` | `tools/gc_experiments_results.py` external callers must accept `TrackedDeleteRefusedError` defense | STRICT @ 0 |
| **#154** | `check_experiments_results_gc_helper_is_canonical` | Refuses ad-hoc `shutil.rmtree(experiments/results/...)` outside the canonical helper | STRICT @ 0 |
| **#153** | `check_modal_dispatcher_uses_canonical_mount_builder` | `experiments/modal_*.py` must route through `tac.deploy.modal.mount_manifest.build_training_image` | STRICT @ 0 |
| **#152** | `check_operator_wrapper_validates_required_input_files_pre_dispatch` | Dispatch wrappers must validate `required_input_file=True` flag values exist BEFORE GPU dispatch | STRICT @ 0 |
| **#151** | `check_operator_wrapper_threads_trainer_tier_required_flags` | Operator wrappers must thread env→CLI ladder for trainer's `TIER_<N>_OPERATOR_REQUIRED_FLAGS` | STRICT @ 0 |
| **#150** | `check_phase_b_auth_memo_in_repo` | `phase_b_preconditions_status(auth_memo_path=...)` must point under git repo root | STRICT @ 0 |

Catalogs #1-#149 (sister gates spanning device defaults, eval roundtrip,
EMA, mount manifests, lifecycle, custody validation, lock discipline,
trainer manifests) all STRICT @ 0 per session-cumulative landing memo
ledger.

## Modal source-staleness pivot (Phase B-1)

The 2026-05-12 canary subagent reported "Modal worker mounted stale source"
after two consecutive Modal A100 dispatches of `train_substrate_sane_hnerv`
crashed at `score_aware_loss.py:129 unsqueeze(1)`. Investigation of the
chronology revealed:

| Event | UTC |
|---|---|
| Dispatch [redacted-private-custody] (WWW4) fires | 2026-05-12T17:12Z |
| Dispatch [redacted-private-custody] fires | 2026-05-12T20:26:47Z |
| Commit `6048d690` (FIX-H Part 1) lands | 2026-05-12T20:44:00Z |

Both dispatches fired BEFORE the fix landed; the Modal worker faithfully ran
the broken pre-fix code. Post-mortem could not distinguish "stale snapshot"
from "pre-fix dispatch" because `modal_metadata.json` did not record:
- `mounted_code_git_head` (dispatch-time HEAD SHA)
- `working_tree_dirty` state
- sentinel-file SHA-256 ladder

**Fix landed (commit `4ada3a59`):** Catalog #166 — HEAD parity ledger
serialized on dispatch + worker-side source-parity ledger written to
`modal_worker_head_ledger.json`. `--require-clean-head` opt-in fail-closed
gate added. 6th-class diagnosis (`HOK-at-dispatch-but-fix-needed-after-dispatch`)
documented; runtime sister tool
`tools/diagnose_modal_worker_source_staleness.py` provides H1-H5 verdict
taxonomy (image cache / eval-timing / deploy stale / manifest gap /
PYTHONPATH ordering).

**Smoke-before-full pattern (commit `5056c70f`)**: Catalog #167 — refuses
operator-authorize Modal wrappers that fire `--full` before `--smoke`
validation. Initial warn-only; strict-flip pending 4 legacy wrapper
backfill.

## Outstanding operator decisions (consolidated from this session)

1. **Vast.ai balance reload OR commit Modal/Lightning-only.** Account at
   $0 balance; Modal credits exhausted as of 2026-04-15; Lightning T4 free
   pool is the remaining no-cost option. The Phase B-2 dispatch wave has
   $19.90 of envelope reserved but no provider with non-zero balance.
2. **HNeRV-family Wave 3 build (TCNeRV, BlockNeRV, FFNeRV, DSNeRV, HiNeRV,
   ego_nerv).** DEFER until Wave 2 HIGH-target empirical signal arrives.
   Operator framing: "sane_hnerv at ~0.19 is not useful — we need to BEAT
   0.193, not match it."
3. **TRADITION 2 substrate production targeting.** 8 substrates have
   recipes but no trainers (lane_12_v2_nerv, nervdc, e_nerv, ego_nerv,
   cnerv, quantizr_faithful, mlx_mask_renderer, dp_sims_renderer,
   diffusion_renderer). Are these production-targeted, or research-only?
   `mlx_mask_renderer` is `[macOS-MLX research-signal]` only by construction;
   `diffusion_renderer` is `research_only=true` permanently.
4. **`recovered_*/` body-cleavage (~106 MB).** WAVE-8 audit landed but
   DEFERRED-pending-operator. Per Catalog #110 + #154:
   PRESERVE-METADATA-DELETE-BODIES verdict eligible but requires explicit
   operator handle + UTC timestamp.
5. **Wavelet substrate reactivation criteria.** Dense full-grid WLV1 is
   byte-floor blocked (471M raw subband bytes ≫ N=37,545,489). Highest-EV
   future work: sparse-subband compiler (top-k coefficients per subband +
   run-length entropy coder); OR redesign as residual-over-champion packet;
   OR coarser quantization (int4 with learned codebook). NO KILL verdict.

## Next dispatch event horizon

Phase B-2 substrate ranking sweep (6 HIGH-target dispatches @ Modal A100
2000ep, ~$5 each = $30 total — currently EXCEEDS $19.90 envelope):

**Gating events (in dependency order):**
1. **Modal source-staleness verification** — operator runs
   `tools/diagnose_modal_worker_source_staleness.py` once; HOK verdict
   unblocks all future Modal dispatches.
2. **Smoke-before-full validation** — every HIGH-target substrate runs
   `--smoke` ($0.30, 100 epochs) before `--full` ($5, 2000 epochs);
   confirms trainer + Catalog #166 + Catalog #167 chain on real Modal.
3. **Cost-band first-anchor** — first successful_dispatch anchor lands
   into `.omx/state/cost_band_posterior.jsonl`, triggering automatic
   posterior recalibration; subsequent dispatches use the empirical
   prior instead of `hand_calibrated_fallback`.
4. **Provider balance** — operator decision required (item #1 above)
   before ANY paid Modal dispatch. Lightning T4 free pool path is
   alternative if Modal/Vast.ai both unfunded.

## 2026-05-12 (session) — earlier landings (preserved from prior generations)

[The "Substrate scaffolds + Catalog flurry + state hygiene" generation
covering α `sane_hnerv`, β `balle_renderer`, Catalog #98 strict-flip,
#117/118/119 commit-machinery, #150 phase-B, #151 wrapper-TIER, #152
required-input validation, #153 Modal mount, #154 GC helper canonical,
T1-D GC, claim-ledger prune, MMM PR101 GOLD primitive port, T1 Balle
engineering audit landings still applies; this WAVE-9 audit supersedes the
top-of-file generated-at marker only and adds the Phase B-2 readiness +
substrate canvas + cost-band NV7 + Modal source-staleness pivot summary.]

## Cumulative session subagent landings

**2026-05-12 LANDED memos:** ~88 distinct `feedback_*_20260512*.md` files
written this session (mix of `_LANDED_` / `_landed_` / `_DEFERRED_pending_*`
suffixes). Ten of these match the operator's referenced subset:

- `feedback_phase_b1_nv7_fix_canary_pair_LANDED_20260512.md`
- `feedback_phase_b2_build_3_high_target_trainers_LANDED_20260512.md`
- `feedback_wave1_vq_vae_trainer_build_LANDED_20260512.md`
- `feedback_wave1_self_compress_nn_trainer_build_LANDED_20260512.md`
- `feedback_wave1_hybrid_renderer_residual_trainer_build_LANDED_20260512.md`
- `feedback_wave8_gha_contest_cpu_workflow_LANDED_20260512.md`
- `feedback_wave8_vastai_phantom_cleanup_LANDED_20260512.md`
- `feedback_wave8_recovered_cleanup_DEFERRED_pending_operator_decision_20260512.md`
- `feedback_phase_b1_pivot_modal_source_staleness_fix_LANDED_20260512.md`

**Lane registry growth:** 490 lanes total (this WAVE-9-PROGRESS-AUDIT lane
included; from ~384 baseline before the multi-wave session).

## Disclosure hygiene compliance

Per CLAUDE.md "Public Disclosure Hygiene":
- No Vast.ai instance IDs published (private-custody artifact only)
- No Modal FunctionCall IDs published ([redacted-private-custody] used inline)
- No `~/.claude/projects/...` paths in this report
- No operator email
- All score claims tagged `[predicted]` / `[contest-CUDA]` / `[contest-CPU]`
  / `[macOS-MLX research-signal]` / `[macOS-CPU advisory decoder-q]`
  / `[empirical:<short-ref>]` per axis discipline
- Lane IDs, commit SHAs, substrate names included as canonical references

---

*Generated by `subagent:wave9_progress_audit_20260512` per operator-orchestrated
parallel deployment WAVE-9; META audit + public-publishable summary.*

## Session closure 2026-05-12

The 2026-05-12 orchestrator session is FORMALLY CLOSED. Across 9 deployment
waves the session landed 19+ subagent batches, wired 11 substrate trainers,
flipped 13 new Catalog # STRICT preflight gates (cumulative 82 distinct
catalog-numbered checks active), grew the lane registry to 496 lanes, and
held net GPU spend to **$0.10** (5 failed Modal A100 attempts whose anchors
are now correctly classified as `failed_dispatch` per the NV7 cost-band
posterior fix at commit `1e7e1b0d`). All Catalog # gates audited via
`tac.preflight --scope dev` returned `PREFLIGHT PASSED`; lane registry
audited via `tools/lane_maturity.py validate` returned `OK — 496 lane(s)
validated cleanly`.

### Substrate canvas Phase B-2 readiness

11 HIGH-target substrate trainers wired to dispatch (sane_hnerv α,
balle_renderer β, SIREN, Cool-Chic, wavelet [research_only=true],
VQ-VAE, self_compress_nn, hybrid_renderer_residual, plus DSNeRV, HiNeRV,
grayscale_lut from later waves). 5 MEDIUM-target HNeRV-family substrates
hold `dispatch_blockers: remote driver missing` recipe-only entries
(TCNeRV, BlockNeRV, FFNeRV plus 2 prior). 9 TRADITION 2 substrate recipes
landed via WAVE-6 (cnerv, e_nerv, ego_nerv, lane_12_v2_nerv, nervdc,
quantizr_faithful, mlx_mask_renderer, dp_sims_renderer, diffusion_renderer)
with documented `dispatch_blockers` per substrate (trainer-not-wired vs
remote-driver-missing vs library-deps vs research-only). 1 of 5 TRADITION 2
remote drivers landed (cnerv via WAVE-6-FOLLOWUP at commit `d1397618`).

The Modal source-staleness pivot (Catalog #166 STRICT @ 0; Catalog #167
warn-only pending strict-flip) closed the structural failure mode
documented in the Phase B-1 canary harvest. WAVE-8-VASTAI-CLEAN reduced
phantom Vast.ai tracker entries from 230 to 0. WAVE-8-GHA landed the
`.github/workflows/contest_cpu_eval.yml` Linux x86_64 contest-CPU axis
plumbing for `lane_g_v3` per CLAUDE.md "Submission auth eval — BOTH CPU
AND CUDA on 1:1 contest-compliant hardware" non-negotiable.

### Memory hygiene compliance (5-memo audit sample)

| Memo | Council | Internal-consistency | Reactivation | 6-hook wire-in | /tmp | KILL |
|---|---|---|---|---|---|---|
| `phase_b1_nv7_fix_canary_pair` | 0 | 0 | 0 | 5 | 0 | 0 |
| `phase_b1_pivot_modal_source_staleness_fix` | 0 | 0 | 0 | 3 | 0 | 0 |
| `wave1_vq_vae_trainer_build` | 1 | 1 | 0 | 2 | 0 | 0 |
| `wave5_b1_4_composition_cells_on_a1` | 1 | 1 | 1 | 3 | 0 | 2* |
| `meta_catalog_152_annassign_fix` | 1 | 1 | 1 | 5 | 0 | 0 |

*The two KILL/FALSIFIED matches in `wave5_b1` are negation context (memo
explicitly states "This memo does NOT carry a KILL/FALSIFIED verdict per
CLAUDE.md 'KILL is LAST RESORT'"). NOT actual KILL verdicts.

PCC4 compliance is uneven across in-flight build memos (NV7 + Modal
source-staleness pivot landed without explicit Grand Council /
Internal-consistency / Reactivation sections). DEFERRED-to-future-cycle:
backfill PCC4 sections on those 2 memos in the next maintenance pass; both
landings are reversible and carry implicit reactivation criteria via the
Catalog # gates that wrap them. NOT raised as KILL or strict-rejected per
"Bugs must be permanently fixed AND self-protected against" — the 2 affected
memos are operationally subsumed by Catalog #166 + #167 STRICT gates which
ARE the structural protection.

### Operator decisions awaiting routing (consolidated, 9 items)

| # | Decision | Cost | Information value | Status |
|---|---|---|---|---|
| 1 | Vast.ai balance reload OR commit Modal/Lightning-only | Account at $0; Modal credits exhausted; Lightning T4 free pool remains | Unblocks all paid GPU dispatches | OPERATOR-GATED |
| 2 | sane_hnerv smoke validation on Modal A100 (validates Catalog #166 end-to-end + first real cost-band anchor) | $0.30 (100 epochs) | Highest information density per dollar this session | OPERATOR-GATED |
| 3 | HNeRV-family Wave 3 build (TCNeRV, BlockNeRV, FFNeRV remaining; DSNeRV+HiNeRV landed) | ~$15-25 dev + $5-15 dispatch | Medium-target substrates; framing "sane_hnerv at ~0.19 not useful, must BEAT 0.193" | DEFERRED-pending-Wave-2-empirical |
| 4 | TRADITION 2 substrate production targeting (8 substrates) | Recipes landed; trainers/drivers vary | Operator framing decision: production-target OR research-only? Affects dispatch eligibility | OPERATOR-GATED |
| 5 | `recovered_*/` body-cleavage (~106 MB) | Free; ~30 min review | Disk hygiene; PRESERVE-METADATA-DELETE-BODIES verdict eligible per Catalog #110 + #154 | DEFERRED-pending-operator (WAVE-8 audit landed) |
| 6 | Wavelet substrate reactivation criteria | Sparse-subband compiler ~5-8h dev; OR redesign as residual; OR int4 codebook | Currently byte-floor blocked (471M raw subband bytes ≫ N=37,545,489) | NO KILL; documented reactivation path |
| 7 | T10 IB Lagrangian dispatch | ~$40 | Information bottleneck Lagrangian-coupled training | OPERATOR-GATED (cost cap) |
| 8 | PR95 Phase 2-4 (8-stage curriculum + Muon optimizer + dual-RGB-head) | ~700 LOC dev; dispatch TBD | Substrate decision required first | DEFERRED-pending-substrate-choice |
| 9 | B1 5 composition cells on PR106 r2 substrate | ~3-5h dev per cell + $1-3 dispatch | Predicted -0.00750 max Δ; WAVE-5 cell 4 on A1 substrate landed only -234B savings (1/4 cells positive) | DEFERRED-pending-grand-council-after-Wave-5-evidence |

### Recommended next move

**Single highest-information-density next event:** sane_hnerv smoke ($0.30,
100 epochs) on Modal A100 via the Catalog #167 smoke-before-full pattern
in `tools/run_modal_smoke_before_full.py`. This validates the Catalog #166
HEAD-parity ledger fix end-to-end, lands the first real successful-dispatch
anchor in the cost-band posterior (collapses the
`hand_calibrated_fallback p50=$5.00` band into a measured percentile), and
de-risks the entire 6-substrate Phase B-2 dispatch wave for $0.30 instead
of $30. Operator decision #1 (provider balance) is the only blocking
gating event. Lightning T4 free pool is the no-balance fallback.

---

*Closure section appended by `subagent:wave9_session_closure_20260512` per
operator-orchestrated parallel deployment WAVE-9-CLOSURE; FORMALLY CLOSES
the orchestrator queue with documented routing for all 9 outstanding
operator decisions.*
