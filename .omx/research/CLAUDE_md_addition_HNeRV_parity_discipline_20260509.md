# CLAUDE.md addition — operator paste-in (drafted 2026-05-09)

The following section is drafted to be pasted into `/Users/adpena/Projects/pact/CLAUDE.md` as a NON-NEGOTIABLE section. Suggested location: immediately after the existing "Race-mode rigor inversion + parallel-dispatch first" section and before "Main branch source of truth" (i.e., the highest-emphasis tier).

The section title and body below are the EXACT text proposed; operator may edit before pasting.

---

## HNeRV / leaderboard-implementation parity discipline — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Source:** `feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509.md` (operator-mandated retrospective, 2026-05-09). Cross-refs `.omx/research/representation_integration_gap_audit_20260508_codex.md` (codex parallel finding) + `feedback_substrate_vs_codec_composition_meta_pattern_20260508.md` (claude-side framing) + `feedback_grand_council_fields_medal_phase2_floor_REBASELINE_with_integration_discipline_20260509.md` (Phase 2 floor rebaseline).

**The 2026-04-30 → 2026-05-04 retrospective:** we had HNeRV/MNeRV/NeRV/SIREN/Cool-Chic/C3 representation primitives in the repo BEFORE PR #95/#100/#101/#103 ever hit the contest. We never got sub-0.20 with them. The leaderboard HNeRV-family won not because of architectural novelty but because PR #100's hnerv_lc_v2 (268 LOC) bound architecture + score-aware training + archive grammar + inflate runtime + export contract simultaneously, and PR #101 (337 additional LOC of entropy bolt-ons) won gold at 0.193 by stacking on the verified substrate. Each layer was reviewable in 30 seconds.

Our internal NeRV/HNeRV/Cool-Chic/C3 work had every architectural ingredient but never bound them simultaneously. The integration loop was always 5-7 separate research artifacts that never converged into a single packet. Lane 12 NeRV mask codec targeted the WRONG slot (mask only, not full RGB renderer). Cool-Chic / C3 hit the FP4A export gate AFTER training, not before.

### The 13 inviolable lessons

Every representation/codec lane (NeRV / HNeRV / Cool-Chic / C3 / wavelet / VQ-VAE / grayscale-LUT / SIREN / coordinate-MLP / hyperprior / nonlinear-transform-coding / time-varying-FiLM / shared-codebook / etc.) MUST honor all 13 of these from byte zero:

1. **Substrate must be score-aware.** Train against the contest's actual `upstream/videos/0.mkv` with gradient-through-SegNet/PoseNet, not extracted masks, not L²/KL on raw frames, not synthetic data. Default loss `((mask_pred - mask_gt) ** 2).mean()` is FORBIDDEN as the primary training signal for any representation entering the archive.
2. **Export-first design.** Declare the archive grammar + parser-section manifest BEFORE writing the training script. If the variant cannot export into the contest packet format (e.g., Cool-Chic / C3 non-FP4A), the run is research-only by construction; tag `research_only=true` and ungate `--auth-eval-on-best` only after the export contract lands.
3. **Archive grammar = monolithic single-file `0.bin`** (or explicitly justified multi-file). Fixed offsets declared in `codec.py` source (e.g., `DECODER_BLOB_LEN = 162_164`, `LATENT_BLOB_LEN = 15_387`). ZIP-member-budget rows are invalid unless the packet really has separate ZIP members.
4. **Inflate.py ≤ 100 LOC** (default budget; explicit waiver for ≤ 200 with rationale). ≤ 2 external dependencies declared in the runtime tree. CUDA-or-CPU agnostic. Reviewable in 30 seconds.
5. **Architecture must be the FULL renderer** (RGB out), not a single-component slot (mask only / pose only). The contest scorer derives masks from frames; replacing the masks slot is dominated by replacing the frames the masks are derived from. Lane 12-style "mask codec only" lanes are DEFERRED-pending-research-with-renderer-rescope.
6. **Score-domain Lagrangian** (not weight-domain proxies like rel_err²). The Lagrangian must be `α·B(θ)/N + β·d_seg(θ) + γ·√d_pose(θ)` with `d_seg` and `d_pose` computed via the actual scorer (or a Hinton-distilled co-trained surrogate per Phase 2 / Phase 3). rel_err²-as-objective is FALSIFIED at rms ≥ 0.04 per `feedback_three_lossy_anchors_show_rel_err_squared_objective_falsified_20260508.md`.
7. **Bolt-on size ≤ 350 LOC** (substrate engineering may exceed; tag `lane_class=substrate_engineering` explicitly). Substrate engineering happens ONCE per architecture class; bolt-ons happen many times. PR101 was 605 total LOC = 268 substrate + 337 bolt-on. The kitchen_sink anti-pattern (PR105: 1776 LOC, 21 files, LOST to rem2's 241 LOC silver) is what happens when you violate this.
8. **Eval-roundtrip-aware and differentiable scorer-preprocess training.** The uint8 bottleneck (384 → 874 → uint8 → 384) MUST be simulated in the proxy loss. `eval_roundtrip=False` produces 2-11x proxy-auth gap and is FORBIDDEN per existing CLAUDE.md non-negotiable. The scorer preprocess must also be gradient-reachable: PR #95/#106 monkey-patched `rgb_to_yuv6` because the upstream challenge helper is `@torch.no_grad()` / in-place and otherwise severs PoseNet gradients. New NeRV/HNeRV/Cool-Chic/C3 renderer trainers need a PoseNet/SegNet gradient-reachability check before GPU dispatch.
9. **Runtime closure.** Run the exact contest `inflate.sh` signature in a clean environment BEFORE dispatch. Dependency closure failures (missing brotli, wrong wrapper signatures, hidden sidecars, local paths, CPU/CUDA mismatches) are runtime blockers, not method negatives. PR106 belt_and_suspenders FAILED its first replay due to missing `brotli` — exactly this bug class.
10. **Mask/pose coupling gate.** Any mask change requires pose regeneration + geometry diagnostics + decoded mask SHA-256s + mask disagreement record. Smaller mask bytes alone are insufficient.
11. **No-op detector.** Prove the targeted bytes changed AND were consumed by inflate. Reuse, decode/re-encode, provenance-only changes, and cosmetic ZIP repacks stay forensic until this proof exists.
12. **Single-LOC-per-LOC review discipline.** Every line in the bolt-on must be reviewable in 30 seconds. PR101's `codec.py` is 480 LOC of pure codec code (no training scaffold, no profile dispatch, no smoke/full mode flags). Our internal `nerv_mask_codec.py` is 1000+ LOC and includes coordinate sampling + training scaffolds + sample components + magic-byte versioning + ... — NOT a packetized codec.
13. **KILL/FALSIFIED is LAST RESORT.** Per the existing CLAUDE.md non-negotiable: if a representation lane returns negative, the default verdict is DEFERRED-pending-research-with-XYZ-applied with reactivation criteria, not KILLED. Lane 12 NeRV is DEFERRED-pending-renderer-rescope; Cool-Chic / C3 are DEFERRED-pending-export-design.

### The 8th forbidden pattern (named here)

**Forbidden representation-without-archive-grammar (the "research-substrate trap"):**

Building a representation (NeRV / Cool-Chic / C3 / wavelet / VQ-VAE / grayscale-LUT / SIREN / coordinate-MLP / hyperprior / etc.) WITHOUT simultaneously building (a) the `archive.zip` builder that emits scored bytes, (b) the `inflate.sh` runtime that reads them, (c) the parser-section manifest that locates them, (d) the export contract that converts trained weights → archive bytes, and (e) the score-aware training loop that backprops through SegNet/PoseNet on the contest video — is a research-only path by construction. The bytes never enter the contest packet; the score never moves.

This is the dominant representation-lane integration meta-bug from the 2026-04-30 → 2026-05-04 gap. It does not by itself explain the full miss: the postmortem also requires (a) failure to consume PR #95's open training stack during the race window, (b) failure to measure the CPU public-leaderboard axis early enough, and (c) missing differentiable scorer-preprocess training in our NeRV/HNeRV loops. STRICT preflight check #124 (`check_representation_lane_has_archive_grammar_at_design_time`) enforces the archive-grammar part; trainer-specific grad-reachability guards must cover the scorer-preprocess part.

### Five forbidden code patterns

1. **Forbidden NeRV-style coordinate MLP that targets the masks.mkv slot without rescope to the renderer.** Lane 12 mistake. If your representation's output shape is `(T, H, W, 5)` of mask logits and not `(T, 3, H, W)` of RGB frames, the lane is DEFERRED-pending-renderer-rescope.

2. **Forbidden `--auth-eval-on-best` gate bypass for non-FP4A export variants.** Cool-Chic / C3 mistake. `train_renderer.py:2099-2122` blocks `--auth-eval-on-best` for variants that lack full archive/export support — this is correct fail-closed behavior. NEVER add a workaround that runs auth eval against a non-exportable variant; instead, land the export contract first.

3. **Forbidden `make_synthetic_pair_batch` calls in any non-smoke training path.** Per `feedback_codex_finding_pr101_synthetic_targets_FIXED_20260508.md`. Train against `upstream/videos/0.mkv` decoded via pyav, not random Gaussian noise. Smoke-only mode does not generalize to non-smoke.

4. **Forbidden representation-lane Level 1+ promotion without `archive_grammar` / `parser_section_manifest` / `inflate_runtime_loc_budget` / `runtime_dep_closure` / `export_format` / `score_aware_loss` / `bolt_on_loc_budget` / `no_op_detector_planned` declared in lane-registry evidence.** STRICT preflight check #124.

5. **Forbidden cross-archive composition (HStack/VStack/cross-paradigm) without a single verified [contest-CUDA] substrate anchor.** Per substrate-vs-codec meta-pattern. T9 (cross-archive multi-substrate composition) is the kitchen_sink anti-pattern under a new name. DEFER until a verified composable substrate exists; or re-scope to single-axis branching from the ONE verified score-aware substrate (currently A1).

### Enforcement

- STRICT preflight check #124 `check_representation_lane_has_archive_grammar_at_design_time` lands warn-only initially; flip to STRICT after in-flight Phase 2 lanes (T1/T6/T10/T15/T17/T18) backfill the blueprint.
- `tools/lane_maturity.py` audit refuses to mark a representation lane as Level 1+ without the 8 declared fields.
- Council review of any new representation/codec lane MUST cite this section and walk through all 13 lessons.
- Memory file `feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509.md` is the canonical retrospective; future agents should re-read it before starting any new representation lane.

---

## End of CLAUDE.md addition.

Operator: paste the section above (between the `## HNeRV /` header and the `## End of CLAUDE.md addition.` marker, exclusive of the marker line itself) into `/Users/adpena/Projects/pact/CLAUDE.md` at the suggested location.
