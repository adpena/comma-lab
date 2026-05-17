<!--
generated_at: 2026-05-12T00:00:00Z
from_state_hash: 07f25a7046e72c9504cd9e053a378d0a4da86837533ed9f0ed0bd864df431b26
regenerated_by: subagent:wave9_progress_audit_20260512
last_refreshed_at: 2026-05-13T00:00:00Z
last_refreshed_by: subagent:FIX-COMBINED-Z-PLUS-R5
last_refreshed_head: b1552977
last_refreshed_note: |
  R5 finding R5-3 (Low, 2026-05-13) flagged this report as stale by ~24h
  + 35+ commits since the 2026-05-12 regen. The body content below is
  unchanged (no substantive regen this pass); this header records the
  re-acknowledgement and points future agents at the canonical regen
  driver. Per CLAUDE.md "Required durable state" non-negotiable,
  reports/latest.md is on the post-cycle update list and should be
  re-generated against current state on the next session-close gate
  per the WAVE-9-CLOSURE pattern. Reactivation criterion: if this header
  is again >24h or >25 commits stale at session close, R5-3 reactivates
  and FIX-WAVE-R5+ should fully regenerate the body. See
  feedback_recursive_review_r5_LANDED_20260513.md + the FIX-COMBINED-Z-PLUS-R5
  closure memo (feedback_fix_combined_z_plus_r5_LANDED_20260513.md).
-->

# Comma Lab — Substrate Canvas + Phase B-2 Readiness — 2026-05-12

> **R5-3 closure note (2026-05-13)**: report body below remains the 2026-05-12
> snapshot. Header refreshed by FIX-COMBINED-Z-PLUS-R5 to acknowledge staleness
> without unilaterally rewriting the body. Full regen deferred to next
> session-close gate per the WAVE-9-CLOSURE pattern.

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

### Current best — last regenerated 2026-05-17

| Axis | Best score | Archive sha256 (first 12) | Hardware | Lane |
|---|---|---|---|---|
| **`[contest-CPU GHA Linux x86_64]`** | **0.1920513169** | `6bae0201fb08` | linux_x86_64_cpu | `lane_pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515` |
| **`[contest-CUDA T4]`** | **0.2053300290** | `9cb989cef519` | linux_x86_64_t4 | `lane_pr106_format0d_latent_score_table_20260516_contest_cuda` |

### Top-5 per axis (sanity / promotion-candidate queue)

**`[contest-CPU]`**:

1. **0.1920513169** — `6bae0201fb08` — `lane_pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515`
2. 0.1920978868 — `8866ebb655e9` — pr101 fec sibling
3. 0.1948697174 — `7d1e46331a04`
4. 0.1982817145 — `14190c2aac3d`
5. 0.1983654164 — `ac8b7681cde0`

**`[contest-CUDA T4]`**:

1. **0.2053300290** — `9cb989cef519` — `lane_pr106_format0d_latent_score_table_20260516_contest_cuda`
2. 0.2063163866 — `56cdd10bdc43` — `lane_pr106_format0c_exact_radix_paired_20260515`
3. 0.2063257086 — `5c9ef623a089` — `lane_pr106_hdm11_hlm3_magicless_packetir_format0b_20260515`
4. 0.2063310355 — `186a3d59f203`
5. 0.2063330331 — `09bcd867c277`

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
- A1 (`87ec7ca5f2f3`) at 0.19285 CPU is #3 — anchor for a separate (PR101-grammar minimalist) line.

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
| **mlx_mask_renderer** | [macOS-MLX advisory only — never promotable to contest-CUDA] | yes | local Apple Silicon only |
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
   `mlx_mask_renderer` is `[macOS-MLX advisory]` only by construction;
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
  / `[macOS-MLX advisory]` / `[empirical:<short-ref>]` per axis discipline
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
