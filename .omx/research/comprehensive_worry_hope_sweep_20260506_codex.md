# Comprehensive Worry/Hope Sweep - 2026-05-06 Codex

Purpose: capture the current cross-paradigm sweep into durable project state,
separating exact guardrails from score-lowering opportunities. This is not a
score ledger. Evidence labels follow `AGENTS.md`.

## External Anchors

- comma10k/SegNet labels: comma documents a 5-label SegNet contract in three
  broad motion classes: road, lane markings, undrivable including sky, movable,
  and my car. Source: https://blog.comma.ai/crowdsourced-segnet-you-can-help/
- SPADE: semantic image synthesis uses spatially adaptive normalization from
  semantic layouts; useful as a renderer-design prior, not score evidence.
  Source: https://github.com/NVlabs/SPADE
- CLADE: class-adaptive normalization preserves much of SPADE's benefit with
  lower parameter overhead and is a better fit for byte-budgeted renderers.
  Source: https://arxiv.org/abs/2012.04644
- openpilot: MIT-licensed public stack with real hardware/camera/road priors,
  useful for geometry and label priors but not a substitute for charged archive
  bytes plus exact CUDA eval. Source: https://github.com/commaai/openpilot

## Worries Closed In This Tranche

1. LCT archive contract gap.
   - Finding: `--learnable-class-targets` changes the scored inverse grayscale
     LUT, but preflight did not statically require the charged
     `class_targets.fp16` payload and `SEGMAP_CLASS_TARGETS_FILENAME` inflate
     config.
   - Fix: added strict `check_segmap_lct_archive_contract()` and focused tests.
   - Evidence: [empirical:src/tac/tests/test_segmap_lct_archive_contract.py]

2. Categorical label drift in live code comments.
   - Finding: several live surfaces still described the contest classes as
     vehicle/sky/background. That is dangerous for CLADE/SPADE/openpilot-label
     lanes because stale prose can drive wrong class-specific weight design.
   - Fix: corrected live comments and widened
     `tools/audit_semantic_label_contract.py` advisory coverage.
   - Evidence: [empirical:tools/audit_semantic_label_contract.py --fail-on-advisory]

3. AQv1 corrupt-container handling.
   - Finding: arithmetic qint tests covered happy-path roundtrip and entropy,
     but not truncated headers, truncated payloads, or trailing bytes. A
     malformed AQv1 stream should fail closed before a repack can be trusted.
   - Fix: exact-read header/payload parsing, nonempty frequency checks, trailing
     byte rejection, and corruption tests.
   - Evidence: [empirical:src/tac/tests/test_arithmetic_qint_codec.py]

4. SegMap/LCT exact-eval whitelist gap.
   - Finding: `contest_auth_eval` rejected valid SegMap archive members such as
     `segmap_weights.tar.xz` and `class_targets.fp16` before inflate. Diet-packed
     `.br` logical members such as `grayscale.mkv.br` were also blocked.
   - Fix: extended the whitelist to SegMap/LCT members and logical `.br`
     members while preserving rejection of unknown debug payloads.
   - Evidence: [empirical:src/tac/tests/test_contest_auth_eval.py]

5. SegMap raw-output naming drift.
   - Finding: SegMap inflaters wrote `0.mkv.raw`, but auth eval expects
     `Path(video_name).with_suffix(".raw")`, e.g. `0.raw`.
   - Fix: all SegMap inflaters now use a shared local raw-path helper and
     preserve subdirectories safely.
   - Evidence: [empirical:src/tac/tests/test_train_segmap_lct.py]

## Active Worries Still Open

1. AQ/Huffman optimality is not proven.
   - Current AQv1 is static-model arithmetic coding over flattened qint streams.
     It is deterministic and now better-validated, but it is not necessarily
     optimal for byte frontier: per-layer/per-block contexts, canonical Huffman
     for small alphabets, rANS/tANS, or delta/context models may beat it after
     table overhead.
   - Next exact step: build a deterministic codec bake-off over the same
     `payload.tar.xz` streams: xz, AQv1 global, AQv1 per-layer, canonical
     Huffman, rANS/tANS if dependency-free or Rust-contained, and brotli. Rank
     by charged bytes plus decode parity, then only dispatch an archive that
     passes exact inflate parity.

2. SHv1 record-level corruption is only partially hardened.
   - AQv1 now rejects truncated/trailing containers, but the outer SHv1 record
     reader still deserves exact-read helpers, per-record length checks, and
     trailing-byte rejection.

3. Canonical archive builder/manifest does not yet model SegMap/LCT/AQ as
   first-class archive shapes.
   - Current lane scripts are improved, but `ArchiveManifest` still centers
     renderer/mask/pose shapes. This keeps SegMap/AQ lanes more ad hoc than
     they should be.

4. `validate_archive()` should call strict ZIP member metadata validation.
   - `safe_extract_zip()` already uses the strict member-info validator, but
     the generic archive validator should not collapse duplicates through a
     `set(zf.namelist())` path.

5. Archive preflight should fail closed for explicit archive paths.
   - `preflight_check(archive_path=...)` still warns instead of failing on
     archive validation exceptions.

6. LA-POSE is planning-only, not paper-faithful.
   - Current modules correctly fail closed as planning-only and tag
     `lapose_lite_is_not_paper_faithful_lapose_model`.
   - Next exact step: keep LA-POSE as atom allocation/profile feedback until a
     charged archive builder consumes selected motion atoms and emits no-op
     controls plus exact CUDA eval.

7. Categorical/CLADE/SPADE is still mostly a scaffold.
   - The semantic quantization lane is honest that Lane A is not SPADE/CLADE,
     so it falls back to uniform backbone quantization.
   - Next exact step: produce a tiny class-conditioned renderer profile whose
     class-specific parameters are actually consumed at inflate time, then run
     the class-bit bake-off under exact archive custody.

8. Telescopic/foveation remains a prior until consumed.
   - Geometry priors and openpilot camera priors can rank atoms, but any foveal
     transform must be charged in `archive.zip`, consumed by inflate, and pass
     pose/seg component gates.

9. Public-HNeRV frontier stacking remains the best score-lowering path.
   - HNeRV-class public submissions established the low score frontier. Current
     HNeRV wavelet/repack work should prioritize payload-changing, exact
     replayable micro-stacks over planner-only complexity.

## Hopeful Next Experiments

1. `SH-BAKEOFF`: deterministic entropy terminal bake-off for SegMap payloads.
   - Entry points: `src/tac/arithmetic_qint_codec.py`,
     `scripts/remote_lane_sh_shannon_arithmetic.sh`,
     `src/tac/balle_hyperprior_renderer.py`.
   - Evidence target: byte/parity first, then exact CUDA archive eval.

2. `CLADE-LITE`: class-adaptive normalization renderer with canonical comma10k
   labels and LCT grayscale targets.
   - Entry points: `src/tac/semantic_label_contract.py`,
     `src/tac/semantic_quantization.py`, `src/tac/segmap_renderer.py`,
     `src/tac/learnable_class_targets.py`.
   - Evidence target: deterministic local train/export smoke, archive closure,
     exact CUDA eval.

3. `LA-POSE-ALLOC`: use LA-POSE-lite only to rank charged atoms.
   - Entry points: `src/tac/analysis/lapose_*`,
     `tools/build_lapose_*`, `src/tac/optimization/meta_lagrangian_allocator.py`.
   - Evidence target: planner artifact with `score_claim=false`, then concrete
     archive builder with no-op controls.

4. `HNeRV-MICROSTACK`: prioritize small payload-preserving deltas on PR101/102/103
   anatomy.
   - Entry points: `src/tac/hnerv_*`, `tools/audit_hnerv_frontier_scorecard.py`,
     `tools/build_hnerv_*`.
   - Evidence target: exact replay parity, charged byte deltas, exact CUDA eval.

5. `OPENPILOT-PRIORS`: use public openpilot/camera priors only as allocation
   features, not as score claims.
   - Entry points: `src/tac/openpilot_seeding.py`,
     `src/tac/lane_mark_pose_v2.py`, `src/tac/camera.py`.
   - Evidence target: sourced priors in manifests, charged archive consumption
     before dispatch.

## Parallel Score-Lowering Sweep

Read-only score sweep on `main` ranked the immediate exact-evaluable work as:

1. Promote PR106x low-level brotli repack as the current custody base.
   - Evidence: exact T4 archive evidence reported for score
     `0.20935073680571203`, `186080` bytes, a `151` byte improvement over
     PR106x.
   - Next: run the same low-level repack path against the non-`x` PR106 q10
     candidate, then exact CUDA.
   - Risk: pure-rate gain is tiny; every archive SHA still needs exact replay.

2. Build HNeRV per-tensor/context entropy recode.
   - Evidence: empirical entropy floor suggests per-tensor zero-order coding is
     the plausible path; global AQ/Huffman-style recodes remain byte-negative.
   - Next: parity-first per-tensor range-coded decoder fixture with compact
     charged headers, then archive rebuild and exact CUDA.
   - Risk: model/header/runtime overhead can erase the floor.

3. Dispatch WR01 HNeRV wavelet transform after packet blockers are fixed.
   - Evidence: best current byte-different candidate is `1/2` strength with
     `-9` archive bytes and one payload section changed.
   - Next: fix exact-eval packet blockers, claim lane, run exact CUDA on the
     `1/2` candidate.
   - Risk: decoded payload changes can dominate the byte gain.

4. Canonicalize PR102 archive provenance before ranking it locally.
   - Evidence: PR101/PR103 exact replays diverge from public rounded claims,
     and current PR102 local archive provenance is not trustworthy enough to
     rank.
   - Next: identify the actual PR102 submission archive, exact CUDA replay it,
     then add it to the HNeRV scorecard.

5. Use beta/LA-POSE/categorical/foveation/meta-lagrangian work as selectors
   for archive-producing lanes until they consume charged bytes directly.
   - Evidence: these are valuable planning/profile surfaces, but not score
     claims without byte-closed archive consumers and exact CUDA.

Immediate highest-EV stack: PR106x low-level repack custody base ->
per-tensor HNeRV entropy recode -> WR01 exact replay. Categorical/openpilot,
LA-POSE, foveation, beta sensitivity, and joint ADMM should feed those archive
builders until they have their own charged consumers.

## Verification Run

- `.venv/bin/python tools/audit_semantic_label_contract.py --format json --fail-on-advisory`
- `.venv/bin/python -m pytest -q src/tac/tests/test_arithmetic_qint_codec.py src/tac/tests/test_balle_hyperprior_codec.py src/tac/tests/test_inflate_segmap_arithmetic.py src/tac/tests/test_segmap_lct_archive_contract.py src/tac/tests/test_audit_semantic_label_contract.py src/tac/tests/test_semantic_quantization.py src/tac/tests/test_semantic_label_contract.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_contest_auth_eval.py src/tac/tests/test_train_segmap_lct.py src/tac/tests/test_inflate_segmap_arithmetic.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_arithmetic_qint_codec.py src/tac/tests/test_balle_hyperprior_codec.py src/tac/tests/test_inflate_segmap_arithmetic.py src/tac/tests/test_segmap_lct_archive_contract.py src/tac/tests/test_audit_semantic_label_contract.py src/tac/tests/test_semantic_quantization.py src/tac/tests/test_semantic_label_contract.py src/tac/tests/test_contest_auth_eval.py src/tac/tests/test_train_segmap_lct.py src/tac/tests/test_remote_lane_segmap_lct_scripts.py`
