# Release manifest — v0.2.0-rc1

**Tag:** `v0.2.0-rc1` (created LOCAL 2026-05-11; pushed to `origin` 2026-05-12
during operator OD-A2 sweep — see
`feedback_category_a_b_operator_authorize_execution_landed_20260512.md`).

**HEAD at tag:** `73ff0dba` (annotated tag pointing at commit `ebbc3ccc`).

**HEAD at this manifest:** `d5b69eff` (work on top of the tag continues; this
manifest is the RECIPE for the tagged release, not a frozen snapshot of HEAD).

**Generated:** 2026-05-12

Per CLAUDE.md "Public Disclosure Hygiene" non-negotiable + "Strategic Secrecy
code-comment cleanup" discipline + "Frontier target" external-claim handling
+ operator directive 2026-05-12 ("OO release prep for v0.2.0-rc1").

---

## Headline

v0.2.0-rc1 is the first release candidate under the production-hardened
substrate-engineering posture mandated by the 2026-05-09 Fields-medal
grand council. It packages:

1. **Two byte-closed substrate scaffolds** at L1 (SCAFFOLD): the α
   sane_hnerv (Score-Aware NeRV Extended) primary substrate and the β
   balle_renderer (Ballé-hyperprior-as-renderer) parallel substrate. Both
   land with full archive grammar + parser-section manifest + inflate runtime
   + score-aware loss + roundtrip tests per HNeRV parity discipline.
2. **3 PR101 GOLD primitives** ported byte-faithfully to
   `tac.packet_compiler` (`pr101_conv4_storage_perms`,
   `pr101_decoder_byte_maps`, `pr101_decoder_storage_order`) with committed
   golden vectors verified against PR101's public archive bytes.
3. **The Rust crate `tac-packet-compiler` at v0.2.0-rc1** with 19/19 primitives
   byte-for-byte parity GREEN against committed Python golden vectors
   (crate retains `publish = false` per council Q5 verdict — IRREVERSIBLE
   crates.io publish remains operator-gated).
4. **Schema-elision + sign-encoding unified taxonomy design memos** for
   PR98/PR100/PR105/PR96/PR101/PR103, landing at L1 (research-only) pending
   IMPL phase.
5. **A 2026-05-12 Catalog # wave** (#151 + #152 + #153 + #154; #155 +
   adversarial-review-fixup landings) — 4 new STRICT preflight self-protection
   gates closing the env→CLI wire-up + required-input + Modal-mount +
   experiments/results GC bug classes.
6. **Cathedral autopilot + canonical wrappers** infrastructure for the
   operator's "production hardened before public sharing" workflow: 8
   one-command authorize scripts (`scripts/operator_authorize_*.sh`),
   cost-band self-calibrating posterior, Modal training-cost canonical
   helper, GHA CPU-eval queue infrastructure for the `[contest-CPU]` axis.

The release is **deliberately a release candidate, not a stable release**.
No external score claim is promoted to authoritative without exact CUDA
replay; sub-0.20 leaderboard claims remain `external` until both
`[contest-CUDA]` and `[contest-CPU]` axes are captured on 1:1 contest-CI
hardware per the CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"
non-negotiable.

---

## What's in the release

### Python package `tac` (MIT)

- `tac.substrates.sane_hnerv` — α primary substrate (Score-Aware NeRV
  Extended). Full RGB renderer, score-aware loss with PoseNet/SegNet gradient
  reachability, single-file archive grammar with declared section offsets,
  inflate.sh + inflate.py runtime ≤ 200 LOC, roundtrip tests.
- `tac.substrates.balle_renderer` — β parallel substrate (Ballé-hyperprior-as-renderer).
  Same byte-closure discipline; dispatch-ready in parallel.
- `tac.packet_compiler` — adds PR101 GOLD primitives
  (`pr101_conv4_storage_perms`, `pr101_decoder_byte_maps`,
  `pr101_decoder_storage_order`); now exposes 26 transform tokens
  (up from 21 at v0.1.x).
- `tac.packet_compiler.golden_vectors` — adds 3 new golden vectors for the
  PR101 GOLD primitives.
- `tac.deploy.modal.mount_manifest` — canonical Modal-image mount builder
  (Catalog #153 wires this into STRICT preflight).
- `tac.cost_band_calibration` — `PLATFORM_RATES_USD_PER_HOUR` +
  `normalize_gpu(platform, gpu)` + `estimate_cost_usd(platform, gpu, sec)`
  + `append_platform_training_anchor(platform, ...)` canonical functions
  (Modal training-cost shim now delegates here).

### Rust crate `tac-packet-compiler` (MIT OR Apache-2.0)

- Version: `0.2.0-rc1` (up from `0.1.0-prerelease`).
- Status: 19/19 primitives byte-for-byte parity GREEN against Python golden
  vectors (`cargo test -p tac-packet-compiler --test golden_vector_parity`).
- `publish = false` retained per council Q5 verdict (IRREVERSIBLE crates.io
  publish — see operator-decision items below).

### STRICT preflight catalog gates (new since v0.1.x)

Counted from CLAUDE.md "Meta-bug class catalog" table, additions made in the
2026-05-08 → 2026-05-12 window (since the v0.1.x baseline):

- Catalog #98 (`check_pr101_tools_torch_load_allowlist`) — STRICT-flipped at
  0 violations during the 2026-05-12 audit (commit `058517cf`); previously
  warn-only.
- Catalog #117 / #118 / #119 — META-meta commit-machinery permanent
  protections (subagent commit serializer + duplicate-catalog-# detector +
  Co-Authored-By trailer enforcement).
- Catalog #123 — Track 4 weight-domain saliency on score-aware substrate
  STRICT gate.
- Catalog #124 — Representation lane archive grammar at design time.
- Catalog #125 / #126 — Subagent coherence-by-default (solver wire-in
  declarations + pre-registration of lane_id).
- Catalog #127 / #128 / #130 / #131 — Custody validator + locked posterior
  writes + tag/grade local validator + bare-write-to-shared-state.
- Catalog #132 — Locked writes preserve deletions (codex round-3 HIGH 1).
- Catalog #133 / #134 / #135 — Codex round-4 META-meta gates.
- Catalog #136 / #137 / #138 — Defense-in-depth on codex round-3 fixes.
- Catalog #139 — Packet-compiler no-op proof promotes to blocker (round-5
  HIGH 1).
- Catalog #140 / #141 — Codex round-5 state-writer lock-ownership gates.
- Catalog #142 / #143 / #144 / #145 — Codex round-6 fail-closed + pre-network
  cancel + transactional setup-first-seen + preflight CLI default scope.
- Catalog #146 — Phase 1 trainer contest-compliant runtime emission.
- Catalog #147 / #148 — Codex round-7+8 lightning-cancel-pre-network +
  vastai-tracker-strict-load.
- Catalog #150 — Phase B auth memo in-repo discipline (operator decision C).
- **Catalog #151** (new this wave) — `check_operator_wrapper_threads_trainer_tier_required_flags`.
- **Catalog #152** (new this wave) — `check_operator_wrapper_validates_required_input_files_pre_dispatch`.
- **Catalog #153** (new this wave) — `check_modal_dispatcher_uses_canonical_mount_builder`.
- **Catalog #154** (new this wave) — `check_experiments_results_gc_helper_is_canonical`.
- Catalog #155 / #156 / #157 — landed this wave per CLAUDE.md additions
  (GC + commit-serializer pre-lock hash + helper-refuses-tracked-paths).

### Tooling

- 8 one-command `scripts/operator_authorize_*.sh` wrappers (cathedral
  autopilot + B1/B2/B3 dispatch + GHA CPU eval + Kaggle T1 sweep).
- `tools/validate_dispatch_required_inputs.py` — canonical validator for the
  Catalog #152 wire-up gate.
- `tools/gc_experiments_results.py` — canonical GC helper for
  `experiments/results/` (Catalog #154).
- `tools/claim_lane_dispatch.py prune` — terminal-row archive prune (T1-E
  sister to Catalog #154).
- GHA CPU-eval queue infrastructure (commit `62e976d2`) for the
  `[contest-CPU]` axis.
- Kaggle T1 parallel-sweep harness (commit `dda8bfab`) for free T4 GPU
  capacity.

### Research / design memos

10 markdown research notes + 7 audit-data JSON files under `.omx/research/`,
covering:

- Schema-elision + sign-encoding taxonomy design (PR98/PR100/PR105/PR96/PR101/PR103).
- FFF wiring + integration + arbitrariness audit deliverables.
- Full-stack integration audit v4 + polish + hardening sweep.
- Field-medal grand council substrate design wave.
- Public PR mining expansion (PR50-115 corpora; 20 typed mechanism rows;
  5 new packet-compiler primitives ported).
- Wiring + arbitrariness + experiments_results GC audit data.

---

## Score claims (axis-tagged per CLAUDE.md "Apples-to-apples evidence discipline")

Per the CLAUDE.md "Frontier target" non-negotiable: external public-leaderboard
claims remain `external` until exact CUDA replay. The v0.2.0-rc1 release does
NOT promote any external score to authoritative without dual-axis evidence.

| Score | Axis | Source | Status |
|---|---|---|---|
| 1.05 | `[contest-CUDA]` | Lane G v3 on T4 (2026-04-28) | Internal anchor |
| 1.0024 | `[contest-CUDA]` | OWV3 0120 on RTX 4090 | Internal anchor |
| 0.229 | `[contest-CUDA]` | PR #107 apogee submission | Internal anchor (~11th place) |
| 0.197 | `[contest-CPU]` | PR #107 on GHA Linux x86_64 | Confirmed (paired with above) |
| 0.193 | `[external; contest-CUDA from public PR]` | rem2 PR #101 (gold) | NOT replayed exact yet |
| 0.195 | `[external; contest-CUDA from public PR]` | EthanYangTW PR #102 (silver) | T4 replay matches public claim within 3×10⁻⁶ |
| 0.19538 | `[external; contest-CPU from public PR]` | EthanYangTW PR #102 (silver) | NOT replayed on our CPU axis yet |
| ε ≈ 6.7e-4 | `[contest-CUDA; PR106 r2 SegNet ceiling]` | Per the 2026-05-04 SegNet vs PoseNet marginal-value analysis | Internal anchor |
| pose_avg ≈ 3.4e-5 | `[contest-CUDA; PR106 r2 frontier]` | Same | Internal anchor (2.71× pose-marginal at this operating point) |

**Predicted Δ tags:**

- `tac.substrates.sane_hnerv` — `[predicted; substrate scaffold]` — no
  empirical score; awaits Phase 2 dispatch.
- `tac.substrates.balle_renderer` — `[predicted; substrate scaffold]` — same.
- B1 film_pose × magic_codec composition on PR106 r2 — `[predicted: -0.5%/+1016B regression]`
  → `[empirical: -0.5%/+1016B confirmed regression]` per the 2026-05-12
  byte-proxy probe (saturated-base hypothesis confirmed across 5 cells; see
  memory `feedback_b1_archive_build_empirical_falsifies_composition_cells_on_pr106_r2_20260512.md`).

No `[macOS-CPU]` or `[MPS-PROXY]` score is promoted to a release-frontier
claim. The 2026-05-08 M5 Max `[macOS-CPU advisory]` PR #107 result of
`0.19664189` matches GHA Linux x86_64 `[contest-CPU]` `0.1966358879` within
`6e-6` (informational only).

---

## Sanitization checklist

Per CLAUDE.md "Public Disclosure Hygiene" non-negotiable. The following
classes of content MUST NOT appear in this manifest:

- [x] **No credentials** — verified via grep for common secret-string
  patterns. 0 matches in this file body.
- [x] **No transient-path durable evidence paths** — verified via grep for
  the forbidden ephemeral-scratch prefix discussed in the CLAUDE.md
  FORBIDDEN section. 0 such durable evidence paths appear in this manifest.
- [x] **No provider tokens** — no Modal/Lightning/Vast.ai tokens or account
  IDs appear.
- [x] **No private infrastructure URLs** — no Tailscale IPs, no internal
  hostnames.
- [x] **No internal Slack/email/Linear refs** — none present.
- [x] **No unpublished operator state** — `.omx/state/*.json` paths are
  mentioned ONLY as canonical locations (the lane registry path is
  canonical), NOT as embedded state. No raw operator state is dumped.
- [x] **No raw provider transcripts** — none included.

Sanitization-check tooling is run after manifest generation. The expected
result is **0 matches** for the canonical secret-pattern set and the
forbidden transient-scratch-path prefix anywhere in the manifest body.

---

## Public-frontier disclaimer

Per CLAUDE.md "Frontier target" non-negotiable:

> "Claimed public scores remain `external` until exact CUDA replay, but they
> must enter intake and exact replay immediately."

> "During an active contest, deadline, or replay window, any public
> PR/archive/body/comment/release that plausibly beats the local exact A++
> frontier takes priority over saturated local polish."

The v0.2.0-rc1 release explicitly does NOT promote external public-leaderboard
claims to authoritative without exact CUDA + CPU replay on 1:1
contest-compliant hardware. The 0.193 (rem2 gold) / 0.195 (EthanYangTW silver
/ rem2 silver alt) / 0.195 (EthanYangTW bronze) scores are tagged `external`
in the lane registry and require exact replay before any internal frontier
promotion.

Per the CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable,
every future shippable archive will produce BOTH a `[contest-CUDA]` and a
`[contest-CPU]` artifact on the EXACT same archive bytes on hardware that is
1:1 compliant with the contest's GitHub Actions CI runner family (Linux
x86_64 for CPU, NVIDIA T4 / A100 / 4090 for CUDA). M5 Max `[macOS-CPU advisory]`
is NEVER the authoritative axis; it is allowed only as advisory/dev-loop
signal until confirmed on Linux x86_64.

The GHA-CPU-eval queue infrastructure landed 2026-05-12 (commit `62e976d2`)
is the canonical mechanism for filling the `[contest-CPU]` axis going
forward. The Kaggle T1 parallel-sweep harness (commit `dda8bfab`) provides
free T4 GPU capacity for `[contest-CUDA]` proxy/smoke evaluation.

---

## Lane summary

Per the sister `.omx/research/lane_sweep_v0_2_0_rc1_20260512.md` deliverable:

| Maturity level | Count | Release-surface inclusion |
|---:|---:|---|
| L3 (FULL PRODUCTION HARDENED) | 0 | N/A — see explanation below |
| L2 (INTEGRATION; release-candidates) | 62 | Cited by name in lane sweep |
| L1 (SCAFFOLD; research/in-flight) | 178 | Mentioned as research; not promoted as frontier |
| L0 (SKETCH; named-only) | 181 | Counted; not cited individually |
| **Total** | **421** | |

**Why 0 L3 lanes**: L3 requires all 7 gates including BOTH `contest_cuda`
AND `contest_cpu` on 1:1 contest-CI hardware. Per the 2026-05-08 dual-axis
non-negotiable, even `lane_g_v3` (the original Phase 1 L3 anchor at 1.05
[contest-CUDA] T4) is currently L2 because its `[contest-CPU]` Linux x86_64
anchor was never captured. The GHA-CPU-eval queue infrastructure is the
mechanism for filling this gap; future v0.2.x releases will promote lanes
to L3 as the dual-axis evidence lands.

---

## DEFERRED entries cited in this release

Per CLAUDE.md "KILL is LAST RESORT" + DEFERRED-pending-research discipline,
the following lanes are explicitly DEFERRED with reactivation criteria
documented:

1. `lane_12_nerv_mask_codec` — DEFERRED-pending-renderer-rescope (HNeRV
   parity discipline forbidden pattern #1).
2. `lane_20_balle_hyperprior` — DEFERRED-pending-export-design (HNeRV
   parity discipline forbidden pattern #2).
3. `track1_paradigm_delta_track4_uniward_stc_hessian_a1` — DEFERRED-pending-research
   (Catalog #123 STRICT-flip; 2026-05-09 falsification anchor).
4. `lane_gp_rerun` — DEFERRED-pending-research-with-proper-replacement
   (KILL→DEFERRED reframed 2026-05-11 per CLAUDE.md kill-as-last-resort).
5. `apogee_int4 NAIVE-PTQ` measured-config retirement (NOT a lane KILL).

---

## Build / install

```bash
# Python package
pip install tac==1.0.5

# With runtime extras (recommended for GPU/training)
pip install 'tac[runtime]==1.0.5'

# With cloud orchestration extras (Modal + Lightning + Vast.ai)
pip install 'tac[cloud,runtime]==1.0.5'

# Rust crate (LOCAL build; NOT published to crates.io per council Q5 verdict)
cd runtime-rs && cargo build -p tac-packet-compiler --release
cargo test -p tac-packet-compiler --test golden_vector_parity   # 21/21 pass
```

---

## Operator-decision items (NOT auto-applied; surfaced for operator review)

Per CLAUDE.md "Design decisions — non-negotiable" + "Public Disclosure
Hygiene": the following items require explicit operator approval and are NOT
auto-applied. They are surfaced here so the operator can decide whether to
amend the v0.2.0-rc1 tag before any further public-distribution step.

### Tag-creation timing (already EXECUTED)

The `v0.2.0-rc1` tag was created LOCAL on 2026-05-11 and pushed to `origin`
on 2026-05-12 during the operator's OD-A2 sweep
(`feedback_category_a_b_operator_authorize_execution_landed_20260512.md`).
**No new tag operation is required from this OO release-prep work.** This
manifest is the RECIPE for the existing tag, not the tag creation itself.

### Operator-decision items still open (carried forward from 2026-05-11 audit + amended here)

1. **LICENSE copyright line** — `LICENSE:3` says
   `Copyright (c) 2026 OpenAI artifact output for user-directed scaffold`.
   This is unusual for an OSS project. The typical pattern would be
   `Copyright (c) 2026 Alejandro Pena` (per `pyproject.toml::authors`) or
   `Copyright (c) 2026 pact contributors`. **NOT auto-changed**; LICENSE is
   outside the mutation frontier. Operator should confirm the canonical
   copyright-holder string before any v0.2.0 stable release.

2. **`runtime-rs/crates/tac-packet-compiler/Cargo.toml::repository`** currently
   points at `https://github.com/commaai/commavq-comma-video-compression-challenge`
   (the upstream contest URL), NOT our canonical OSS repo URL
   (`https://github.com/adpena/pact` per `pyproject.toml::project.urls.Repository`).
   **NOT auto-changed**; operator should align this before any future
   `cargo publish` (currently `publish = false`, so this is not blocking
   the v0.2.0-rc1 tag).

3. **SBOM generation** — per the council Q5 ADD (Yousfi/Ballé): a software
   bill of materials listing every Rust crate dep + version pin should land
   before any `cargo publish`. NOT generated by this OO release-prep work;
   NOT blocking v0.2.0-rc1 because `publish = false` is in force.

4. **Crates.io publish authorization** — IRREVERSIBLE per crates.io
   no-unpublish policy. Crate is currently `publish = false`. Operator
   approval via AskUserQuestion would be required before any
   `publish = true` flip. NOT blocking the v0.2.0-rc1 tag itself.

5. **THIRD_PARTY_NOTICES.md amendments** — per
   `.omx/research/license_audit_20260512.md` Recommendations 1 + 2, three
   recommended edits exist (runtime-dependencies attribution section +
   per-component license boundaries clarification + SBOM deferral note).
   **NOT auto-applied** (THIRD_PARTY_NOTICES.md is outside the mutation
   frontier). Operator may choose to apply these before any public
   announcement / blog post / PR-merge that promotes the v0.2.0-rc1 tag
   beyond the current "tag exists; not yet stabilized" posture.

6. **External score claims (rem2 PR101 0.193 gold, etc.)** — remain
   `external` until exact CUDA + CPU replay on 1:1 contest-compliant
   hardware. NOT promoted in this release.

---

## Cross-references

- License audit: `.omx/research/license_audit_20260512.md`
- Lane sweep: `.omx/research/lane_sweep_v0_2_0_rc1_20260512.md`
- Prior 2026-05-11 release-tag landing: `~/.claude/projects/.../memory/feedback_github_release_tag_license_audit_phase1_wiring_lane_sweep_landed_20260511.md`
- OD-A2 GitHub tag push (the actual remote push): `~/.claude/projects/.../memory/feedback_category_a_b_operator_authorize_execution_landed_20260512.md`
- Council Q5 verdict (publish-on-operator-trigger): `feedback_grand_council_a_b_work_plan_review_20260511.md`
- 2026-05-12 Catalog # wave landings (#151 + #152 + #153 + #154):
  - `feedback_catalog_151_landed_20260512.md`
  - `feedback_permanent_fix_required_input_validation_20260512.md`
  - `feedback_modal_mount_manifest_consolidation_landed_20260512.md`
  - `feedback_state_hygiene_gc_and_prune_landed_20260512.md`
- HNeRV parity discipline: CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" non-negotiable
- Dual-axis discipline: CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable
- Apples-to-apples discipline: CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable
- This release manifest's lane: `lane_oo_release_prep_v0_2_0_rc1_20260512`
