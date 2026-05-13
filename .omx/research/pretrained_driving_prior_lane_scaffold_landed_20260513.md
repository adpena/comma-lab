# Pre-trained driving prior side-lane scaffold — forensic landing record 2026-05-13

**Lane**: `lane_pretrained_driving_prior_lane_scaffold_20260513` (Phase 2, L1)
**Verdict**: SCAFFOLD-COMPLETE / NO-GPU-SPEND / OPERATOR-GATED-FULL-DISPATCH
**Predicted Δ contest-CPU**: `[-0.005, -0.012]` `[time-traveler-prediction]`
**Cost**: $0 (CPU-only smoke test)
**Tests**: 25/25 pass
**Catalog #124 archive-grammar gate**: clean (8 fields declared inline)

## Forensic note on commit attribution

This substrate's source files (14 files, ~2000 LOC including tests) physically
landed in commit `a5a6f30e` ("Wire cooperative receiver substrates into solver
stack") as a side-effect of a concurrent sister-subagent commit. The
intent-of-record + authorship trail for these files is THIS memo plus the
companion landing memo
`~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_pretrained_driving_prior_lane_scaffold_LANDED_20260513.md`.

This is the canonical commit-swap class the Catalog #157
(`check_commit_serializer_pre_lock_hash_against_head`) protection is designed
to extinct. The asymmetric-protection failure mode (only one subagent uses
`--expected-content-sha256`) is documented as Catalog #174
(`check_subagent_commit_serializer_always_uses_expected_content_sha256`,
landed 2026-05-13).

## Source

Operator approval 2026-05-13 of MEDIUM-priority lane from 4th-team memo
`.omx/research/expert_team_hardware_physics_future_alien_tech_20260513.md`
§2.3 (NASA Goddard PASS-AI prior) + §7.4 #2 (Time-Traveler pre-trained prior).

Operator strategic framing (verbatim): *"would the new dispatch direction help
convergence and outcomes? remember we can also overfit to the contest video.
would the new dispatch direction including other companies and other datasets
help for comma ai production deployment and our contest score? we are only
interested in comma ai right now"*

My honest pre-scaffold analysis (informing the prediction band):

> For contest-only impact: Δscore predicted -0.005 to -0.012, NOT -0.020 to
> -0.030. The contest scorer (FastViT-T12 + EfficientNet-B2) was ALREADY
> trained on driving data — it implicitly contains the dashcam prior. Adding
> ANOTHER prior on top is riding the same statistics. BUT for
> production-deployment-shaped contest entry, the prior IS the architecture
> that scales. Build as MEDIUM-priority side-lane, dual-purpose.

## What landed

**Substrate package** `src/tac/substrates/pretrained_driving_prior/`:

| File | LOC | Role |
|---|---|---|
| `__init__.py` | 99 | Public API; Catalog #124 archive-grammar 8 fields declared inline |
| `codebook.py` | 320 | `DashcamCodebook` dataclass + int8 quant + brotli serialize/parse + 4 PCA sections + validation + deterministic-zero fallback |
| `distillation.py` | 420 | `distill_codebook` (PCA from frames) + `aggregate_local_codebooks` (federated edge merge) + `check_no_contest_video_leakage` guard |
| `prior_application.py` | 175 | `DashcamPriorLoss` differentiable soft prior; codebook as non-persistent buffers |
| `archive.py` | 245 | DP1 archive grammar (28-byte header + 4 sections); pack/parse byte-stable |
| `inflate.py` | 210 | Catalog #146-compliant 3-arg CLI; ≤200 LOC substantive |
| `architecture.py` | 150 | Tiny SIREN-style coord-MLP renderer (~12K params, 15-20 KB after brotli) |
| `score_aware_loss.py` | 194 | Catalog #164-compliant via `score_pair_components`; eval-roundtrip mandatory |
| `tests/test_pretrained_driving_prior_substrate.py` | 290 | 25 dedicated tests, 100% pass |

**Trainer**: `experiments/train_substrate_pretrained_driving_prior.py` (256 LOC)
- `TIER_1_OPERATOR_REQUIRED_FLAGS` manifest (Catalog #151/#152)
- `_smoke_main` distills synthetic + packs + parses — VERIFIED ($0)
- `_full_main` raises `NotImplementedError` (operator-gated)

**Recipe + drivers**:
- `.omx/operator_authorize_recipes/substrate_pretrained_driving_prior_modal_t4_dispatch.yaml` (Modal T4, $3 p50)
- `scripts/remote_lane_substrate_pretrained_driving_prior.sh` (Catalog #163 sentinel)
- `scripts/operator_authorize_substrate_pretrained_driving_prior_modal_t4_dispatch.sh` (Catalog #167 smoke-before-full)

## License verification

| Dataset | License | Default? | Verified |
|---|---|---|---|
| **Comma2k19** | **MIT** | ✅ primary | `github.com/commaai/comma2k19` |
| **BDD100K code** | BSD-3-Clause | code-only | `raw.githubusercontent.com/bdd100k/bdd100k/master/LICENSE` |
| **BDD100K images** | UC-Berkeley research/academic | ⚠️ OPT-IN `--allow-bdd100k-dataset-images` | docs |
| **Waymo Open** | Non-commercial | ❌ SKIPPED by design | `waymo.com/open/terms/` |

License attribution baked into codebook metadata (`license_tags`).

## 3-round adversarial review summary

Per CLAUDE.md "Adversarial council review of design decisions" non-negotiable;
full council deliberation in companion landing memo.

- **Round 1 (Yousfi/Fridrich/Contrarian/Quantizr/Hotz)** — strategic adversarial. Verdict: MEDIUM-EV at contest; HIGH-EV for production; prediction band ratcheted to [-0.005, -0.012].
- **Round 2 (Shannon/Dykstra/MacKay/Carmack/Hinton)** — math + implementation. Verdict: Shannon bound consistent with prediction; Dykstra feasibility 3-block decomposition; MacKay MDL bound respected; Carmack minimal runtime; Hinton roadmap.
- **Round 3 (van den Oord/Boyd/Selfcomp/Tao/Contrarian SUPER-VETO)** — production + paranoid. Verdict: SCAFFOLD-COMPLETE; full dispatch DEFERRED-pending-Phase-2-council-approval.

Counter-advance: 3 clean rounds → SEAL.

## 6-hook wire-in declared (Catalog #125)

1. Sensitivity map — N/A for L1 scaffold (registers in Phase 2)
2. Pareto constraint — codebook adds `(rate_codebook, rate_residual, rate_renderer)` axis
3. Bit-allocator — N/A (codebook FROZEN)
4. Cathedral autopilot dispatch hook — recipe registered; full dispatch gated
5. Continual-learning posterior update — N/A (no empirical anchor; `[proxy]` evidence_grade)
6. Probe-disambiguator — N/A (single PCA-basis interpretation)

## Self-protection

- Catalog #124 ARCHIVE-GRAMMAR-AT-DESIGN-TIME — passes clean
- Catalog #146 PHASE1-TRAINER-RUNTIME — N/A until `_full_main` lands
- Catalog #151/#152 OPERATOR-WRAPPER-FLAGS — passes clean
- Catalog #163 REMOTE-LANE-SENTINEL — passes clean
- Catalog #164 SCORER-PREPROCESS — passes clean
- Catalog #167 SMOKE-BEFORE-FULL — wrapper delegates canonically

Catalog #198 was claimed via the canonical `tools/claim_catalog_number.py
claim --commit-via-serializer` path (now in commit `6b3e5559` "state: claim
catalog #198 (git-transactional)") but the named placeholder gate is
DEFERRED-pending-Phase-2 when the real `Comma2k19FrameIterator` lands and
gives the gate live coverage.

## Operator-routable decisions

1. **Phase 2 council approval** — fire `_full_main` only after the
   inner-quintet pact signs off on the Phase 2 training design memo.
2. **Which dataset** — Comma2k19 (MIT, primary) recommended. BDD100K opt-in
   only if commercial-use is acceptable. Waymo SKIPPED by design.
3. **Federated rollout** — `aggregate_local_codebooks` math contract is in
   place; production-deployment federated infrastructure (differential
   privacy + auth + transport) is a separate ~3-week project.
4. **Catalog #198 gate placement** — defer to Phase 2 when the real
   distillation iterator lands.

## Composition with the time-traveler substrate

Both substrates share Catalog #164 `score_pair_components`. Composition is
a typed sum of archive grammars in a future `tac.substrates.dp1_x_tt5l_composition/`
module. The two atomic substrates remain independently shippable.

## Cross-refs

- `.omx/research/expert_team_hardware_physics_future_alien_tech_20260513.md`
- `src/tac/substrates/time_traveler_l5_autonomy/`
- `src/tac/substrates/score_aware_common.py`
- `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_pretrained_driving_prior_lane_scaffold_LANDED_20260513.md`

**Per CLAUDE.md "Subagent coherence-by-default"**: 6 wire-in hooks declared.
**Per CLAUDE.md "KILL is LAST RESORT"**: NO KILL verdicts; full dispatch
DEFERRED-pending-Phase-2. **Per CLAUDE.md "Apples-to-apples evidence
discipline"**: every score-impact tag is `[time-traveler-prediction]` or
`[mathematical-derivation]`; NEVER `[contest-CUDA]` / `[contest-CPU]`.
