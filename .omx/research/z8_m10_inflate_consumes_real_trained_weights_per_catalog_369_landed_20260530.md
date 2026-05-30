<!--
SPDX-License-Identifier: MIT

Catalog #300 council deliberation v2 frontmatter:
council_tier: T1
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Yousfi]
council_quorum_met: true
council_verdict: PROCEED
council_assumption_adversary_verdict:
  - assumption: "Mallat 1989 §7.5 perfect reconstruction extends from float64 reference to float32 implementation with ~1e-7 max abs diff"
    classification: HARD-EARNED
    rationale: "empirically verified by test_reconstruct_pair_rgb_from_pyramid_mallat_perfect_reconstruction_round_trip (1.2e-7 actual)"
  - assumption: "pair-cycling per Catalog #369 cascade IS canonical real-byte consumption (NOT synthetic frame base)"
    classification: HARD-EARNED
    rationale: "empirically verified by test_inflate_one_video_different_archives_produce_different_raw_bytes (different archive bytes -> different RAW output)"
council_dissent: []
council_decisions_recorded:
  - "op-routable #1: register Z8 M10 lane at L1 (impl_complete + memory_entry + substrate_engineering)"
  - "op-routable #2: emit MLX-LOCAL smoke verifying contest contract per Catalog #367"
  - "op-routable #3: backfill Z8 inflate sister test test_basic.py to M10-supersession-aware form"
  - "op-routable #4: defer M11 L1 MLX-LOCAL end-to-end smoke (upstream/evaluate.py --device cpu) to sister subagent"
  - "op-routable #5: defer M12 paired-CUDA Modal T4 + Linux x86_64 CPU sub-0.189 attempt to sister subagent per Catalog #246 + #325 symposium gate"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false

Catalog #110/#113 HISTORICAL_PROVENANCE: this memo is APPEND-ONLY; supersessions land via canonical sister landings.
-->

# Z8 M10 inflate consumes real trained weights per Catalog #369 landed 2026-05-30

`mission_predicted_contribution=frontier_breaking_enabler` per Catalog #300.

## Summary

Z8 M10 (`inflate_runtime_consumes_real_trained_weights` per `build_progress.py`) closes the canonical Catalog #312 quadruple at the deployment surface. M9 (commit `bb48f691c`) lifted the trainer's `_canonical_quadruple_main` from a NotImplementedError stub to the canonical M4 + M5 + M6 + M8 compose pattern; M10 (this landing) closes the inflate-side cycle so the trained-state bytes flow through to contest RAW output.

Per CLAUDE.md "Forbidden NSCS06-class synthetic-frame-base inflate" + Catalog #369: the inflate consumes real trained wavelet coefficients from the Z8HPC1 archive's `wavelet_coeffs_blob` via the canonical Mallat 1989 §7.5 perfect-reconstruction inverse chain. The reconstruction is byte-derived from the trained coefficients (NOT synthetic frame base): different archive bytes produce different RAW output (`test_inflate_one_video_different_archives_produce_different_raw_bytes` regression guard).

## Empirical verification table (per Catalog #229 + cite-chain)

| Surface | Verification | Empirical result |
|---|---|---|
| Mallat round-trip | `test_reconstruct_pair_rgb_from_pyramid_mallat_perfect_reconstruction_round_trip` | max abs diff ~1.2e-7 (float32 precision-bound) |
| Archive emit | `test_build_z8hpc1_archive_bytes_from_canonical_quadruple_round_trip` | wavelet_blob ~45 KB + wz_blob ~80 B for 2 pairs at 32x32 |
| Contest contract | `test_inflate_one_video_from_archive_bytes_emits_contest_raw_bytes` | 1200 frames × 874 × 1164 × 3 = 3,662,409,600 bytes (matches Catalog #367) |
| Real consumption | `test_inflate_one_video_different_archives_produce_different_raw_bytes` | different archive bytes → different RAW output (proves real consumption per Catalog #369) |
| Pair cycling determinism | `test_inflate_one_video_pair_cycling_is_deterministic_per_archive_bytes` | two inflate runs on same archive produce byte-identical RAW |
| Catalog #146 CLI | `test_main_cli_processes_file_list_with_canonical_3_arg_contract` | rc=0 + canonical 0.raw output |
| Catalog #205 device | `test_inflate_imports_canonical_select_inflate_device_catalog_205` | passes (canonical routing present) |
| Catalog #295 PYTHONPATH | `test_inflate_module_imports_clean_from_empty_pythonpath_catalog_295` | passes (no path-shim dependency) |
| Catalog #369 no synthetic | `test_inflate_no_synthetic_frame_base_tokens_catalog_369` | passes (zero forbidden tokens) |
| Catalog #367 invariant | `test_canonical_contest_raw_bytes_invariant_catalog_367` | passes (3,662,409,600 invariant) |
| HNeRV L4+L7 LOC budget | `test_inflate_loc_under_substrate_engineering_waiver_hnerv_l4_l7` | 246 LOC ≤ 300 substrate-engineering budget |
| Build progress | `test_build_progress_m10_landed_status` | LANDED per `build_progress.py` |
| MLX-LOCAL smoke | `experiments/results/z8_m10_inflate_consumes_real_trained_weights_macos_cpu_advisory_smoke_20260530T155420Z/m10_inflate_smoke_output.json` | 4 pairs from upstream/videos/0.mkv → archive 92,408 B → inflate 1200 frames × 874 × 1164 × 3 in 7.05 s wall-clock (real_consumption_verified=true; 9,924/10,000 nonzero + 69 unique) |

## Files edited (per Catalog #229 5+ edits verification table)

| File | Edit | Rationale |
|---|---|---|
| `src/tac/substrates/z8_hierarchical_predictive_coding/inflate.py` | rewrite: replace L0 SCAFFOLD raise with M10 canonical-Mallat inverse reconstruction | M10 milestone scope per `build_progress.py`; closes Catalog #369 + #367 + #146 + #205 + #295 |
| `src/tac/substrates/z8_hierarchical_predictive_coding/canonical_quadruple_binding.py` | append: `build_z8hpc1_archive_bytes_from_canonical_quadruple` + `parse_pair_blobs_from_wavelet_blob` + `reconstruct_pair_rgb_from_pyramid` + sister helpers | trainer-side archive emit + inflate-side parse parity via canonical compose pattern |
| `src/tac/substrates/z8_hierarchical_predictive_coding/build_progress.py` | edit: M10 milestone PENDING → LANDED with `landed_at_utc` + substantive notes citing canonical compose pattern + real-trained-weight consumption + M11/M12 unblock | M10 milestone landing per canonical in-source build tracking |
| `src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_inflate_canonical_quadruple_consumes_real_trained_weights.py` | new: 22 dedicated M10 tests | full M10 acceptance surface coverage |
| `src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_basic.py` | edit: `test_z8_inflate_raises_l0_scaffold_not_implemented_on_runtime_forward` → `test_z8_inflate_l0_scaffold_council_gate_superseded_by_m10_per_catalog_369` | M10 supersession of legacy L0 SCAFFOLD raise per CLAUDE.md "Sister-supersession respect" |
| `.omx/state/lane_registry.json` | new lane L1 (impl_complete + memory_entry + lane_class=substrate_engineering) | canonical lane registry entry per CLAUDE.md "Lane maturity registry" + Catalog #90 |

Reproducer: `PYTHONPATH=src:upstream .venv/bin/python -m pytest src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_inflate_canonical_quadruple_consumes_real_trained_weights.py -v`

## Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| `select_inflate_device` | ADOPT_CANONICAL | `tac.substrates._shared.inflate_runtime.select_inflate_device` per Catalog #205 |
| `raw_output_path` + `write_rgb_pair_to_raw` | ADOPT_CANONICAL | same shared `_shared/inflate_runtime` per HNeRV parity L4 |
| `pack_archive` + `parse_archive` | ADOPT_CANONICAL | `tac.substrates.z8_hierarchical_predictive_coding.archive` Z8HPC1 grammar per Catalog #146 |
| Mallat inverse `recompose_from_next_level` | ADOPT_CANONICAL | `tac.substrates.z8_hierarchical_predictive_coding.mallat_dwt_adapter.Z8MallatDaubechiesPartition` per Catalog #312 M5 |
| per-pair wavelet pyramid blob format | FORK_BECAUSE_PRINCIPLED_MISMATCH | M10 sub-grammar inside `wavelet_coeffs_blob` carries length-prefixed brotli-compressed per-pair pyramids; no sister canonical exists |
| contest 1200-frame pair cycling | FORK_BECAUSE_PRINCIPLED_MISMATCH | training N pairs → contest 600 pairs via deterministic modulo cycle; no sister substrate has this 4× ratio explicit |
| Wyner-Ziv payload concatenation in `wyner_ziv_top_blob` | FORK_BECAUSE_PRINCIPLED_MISMATCH | per-pair length-prefixed payloads; M6 returns single payload per encode call |

## 9-dimension success checklist evidence (Catalog #294)

1. **UNIQUENESS**: M10 is the canonical inflate-side cycle-closure for Z8; no sister substrate composes M4 + M5 + M6 + M8 in this pattern.
2. **BEAUTY + ELEGANCE**: inflate.py is 246 LOC; the canonical pattern reuses `_shared/inflate_runtime` + canonical Mallat inverse; reviewable in ~5 minutes.
3. **DISTINCTNESS**: M10 binds the Z8 archive's wavelet_coeffs_blob to the contest RAW output via Mallat inverse — sister substrates use neural decoders.
4. **RIGOR**: 22 dedicated tests + 215 sister Z8 tests + 25 M9 sister tests = 262 passing; Mallat perfect-reconstruction round-trip verified empirically.
5. **OPTIMIZATION-PER-TECHNIQUE**: per HNeRV parity L7 substrate-engineering UNIQUE-IFIES — Z8 substrate engineering happens ONCE per architecture class.
6. **STACK-OF-STACKS-COMPOSABILITY**: M10 + M9 closes M4/M5/M6/M8 quadruple per Catalog #312; future Z8 bolt-ons compose on top.
7. **DETERMINISTIC-REPRODUCIBILITY**: `test_inflate_one_video_pair_cycling_is_deterministic_per_archive_bytes` proves byte-stable output across runs.
8. **EXTREME-OPTIMIZATION-PERFORMANCE**: inflate 1200 frames in 6.62s on macOS-CPU; smoke total 7.05s wall-clock; brotli quality=9 archive ~92 KB.
9. **OPTIMAL-MINIMAL-CONTEST-SCORE**: M11 + M12 sister wave attempt sub-0.189 per operator binding 2026-05-29; M10 unblocks structurally.

## Cargo-cult audit per assumption (Catalog #303)

| Assumption | Classification | Rationale |
|---|---|---|
| "Mallat 1989 §7.5 perfect reconstruction extends to float32 with ~1e-7 max abs diff" | HARD-EARNED | empirically verified by round-trip test (actual 1.2e-7) |
| "1200-frame contest contract is satisfied by 600-pair cycling from N trained pairs" | HARD-EARNED | empirically verified by inflate writing 3,662,409,600 bytes |
| "pair cycling per modulo IS canonical real-byte consumption (NOT synthetic)" | HARD-EARNED | empirically verified by different-archives-produce-different-RAW regression guard |
| "Z8 substrate is `lane_class=substrate_engineering` so HNeRV parity L4 ≤200 LOC waiver applies up to 300" | HARD-EARNED | per CLAUDE.md HNeRV parity L7 substrate-engineering UNIQUE-IFIES explicit waiver |
| "bicubic upscale 32×32 → 874×1164 via `write_rgb_pair_to_raw` is canonical sister pattern" | HARD-EARNED | sister `submissions/siren/inflate.py` uses same canonical `write_rgb_pair_to_raw` + bicubic mode |
| "the trained N=4 pairs cycled to 600 contest pairs produces medal-band scoring without further training" | CARGO-CULTED | unverified at scale; expected to score poorly without M11/M12 + sub-0.189 attempt at full eval_size + paired CUDA |
| "M4 deterministic state is consumed by inflate (currently NOT — wavelet_blob is the only inflate-side consumer)" | HARD-EARNED-DEFERRED | M10 milestone scope is wavelet-pyramid reconstruction; M4 deterministic-state consumption is a future M11/M12 wire-in surface |

## Observability surface (Catalog #305)

1. **inspectable per layer**: `parse_pair_blobs_from_wavelet_blob` exposes per-pair pyramid dicts at any inflate step.
2. **decomposable per signal**: `m10_inflate_smoke_output.json` carries per-step metrics (archive_bytes, reconstructed_frame_bytes, frames_written, inflate_wall_clock_seconds, sample_bytes_nonzero_count, sample_bytes_unique).
3. **diff-able across runs**: `test_inflate_one_video_pair_cycling_is_deterministic_per_archive_bytes` proves runs are byte-identical for same archive bytes.
4. **queryable post-hoc**: smoke manifest JSON consumed by `tools/audit_*` family + cathedral autopilot.
5. **cite-able**: lane_id + archive_sha + commit_sha tuple in smoke manifest + lane registry evidence.
6. **counterfactual-able**: byte-mutation smoke per Catalog #139 trivially extends from `test_inflate_one_video_different_archives_produce_different_raw_bytes` (different archive bytes → different RAW).

## 6-hook wire-in declaration (Catalog #125)

| Hook | Status | Notes |
|---|---|---|
| #1 sensitivity-map | N/A | M10 is decode-side; sensitivity-map is training-side per Z8 M7 |
| #2 Pareto constraint | ACTIVE | M10 consumes M6 Wyner-Ziv conditional-decoded top-state per R(D|Y) bound |
| #3 bit-allocator | ACTIVE | M10 consumes M6 bit-budget-respecting payload + M5 wavelet-coefficient brotli-coded bytes |
| #4 cathedral autopilot dispatch | N/A | M10 is per-substrate runtime; ranking-side enters at M12 paired-CUDA |
| #5 continual-learning posterior | ACTIVE | M10 smoke produces first inflate-side round-trip verification anchor for canonical equations |
| #6 probe-disambiguator | ACTIVE | real-trained-weight consumption vs synthetic-frame-base IS canonical disambiguator per Catalog #369 |

## Operator-routable next steps

1. **M11 L1 MLX-LOCAL end-to-end smoke** (operator-routable; sister subagent): canonical `upstream/videos/0.mkv` → train via M9 `_canonical_quadruple_main` → emit archive → inflate via M10 → `upstream/evaluate.py --device cpu`. $0 macOS-CPU advisory per Catalog #192 NEVER promotable; cheap pre-paid-GPU smoke gate.
2. **M12 paired-CUDA Modal T4 + Linux x86_64 CPU sub-0.189 attempt** (operator-routable; SEPARATE sister subagent): $1.50–$3.00 PAID per Catalog #246 + #325 per-substrate symposium gate; only after M11 green + per-substrate symposium ratified.
3. **M4 deterministic-state inflate-side consumption** (operator-routable; future Z8 milestone): currently the inflate consumes the wavelet_blob only; the dreamer_state_blob is archived but not consumed at inflate. The canonical M4 surface could feed a per-pair perturbation at inflate time for cross-pair refinement.

## Cross-references

- `build_progress.py::M10` ↔ `inflate_runtime_consumes_real_trained_weights` (this landing flips to LANDED)
- `src/tac/substrates/z8_hierarchical_predictive_coding/canonical_quadruple_binding.py` ↔ M9 binding helper + M10 archive emitter
- Catalog #369 ↔ structural protection at the inflate.py source-text surface
- Catalog #367 ↔ contest RAW bytes invariant (3,662,409,600)
- Catalog #146 ↔ canonical 3-arg CLI signature
- Catalog #205 ↔ canonical select_inflate_device
- Catalog #295 ↔ PYTHONPATH self-containment
- HNeRV parity L4 + L7 ↔ substrate-engineering ≤300 LOC waiver
- Mallat 1989 §7.5 ↔ perfect-reconstruction round-trip math
- Wyner-Ziv 1976 Theorem 1 ↔ R(D|Y) achievable distortion bound

## Lane: `lane_z8_m10_inflate_consumes_real_trained_weights_per_catalog_369_20260530` L1.
