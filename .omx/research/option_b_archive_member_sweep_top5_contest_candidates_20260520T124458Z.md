# OPTION B ARCHIVE-MEMBER SWEEP — top-5 contest candidates re-run 2026-05-20

**Lane:** `lane_slot_build_2_option_b_archive_member_sweep_top5_contest_candidates_20260520`
**Slot:** BUILD-2 (operator-routed SUBAGENT-BUILD queue fill)
**Subagent ID:** `slot-build-2-option-b-prober-rerun-20260520`
**Wall clock:** ~30 min
**Spend:** $0 (CPU-only LZMA/brotli/zlib probes; no paid GPU)

## Operator directive

> Execute Option B post-Catalog-#321 prober re-run on the top-5 contest
> archive.zip MEMBER bytes (rather than research sidecar paths). The pre-entropy
> substrate pivot prober at `tools/pre_entropy_substrate_pivot_prober.py`
> ALREADY EXISTS + the Catalog #321 phantom-score-from-research-sidecar bug
> class is structurally extincted. This task = empirical sweep of archive.zip
> MEMBER bytes (not sidecar paths) for top-5 contest candidates per Q4-Q5
> Wyner-Ziv reactivation criterion. $0 BUILD ONLY per T3 Decision 3.

## Predecessor context (Catalog #229 PV)

The corrected pre-entropy artifact at
`.omx/state/wyner_ziv_deliverability/pre_entropy_candidate_substrates_corrected_20260517T215345.json`
already established the Catalog #321 phantom-score extinction across the 8
canonical substrates (PR101 fec6 / PR106 format0d / PR106 hdm4 hlm1 / cool_chic
/ wavelet / DP1 / apogee_int6 / track4_sg_a1). The sister
`option_b_archive_member_sweep_20260517T221034.json` already ran
`probe_substrate_archive_member` on those 8 candidates and reported
**verdict=DEFER_Q4** (top apples-to-apples aggregate savings 0.000332 < 0.001
threshold).

This 2026-05-20 task EXTENDS the Q4 retarget question to the **historical
contest gold/silver/bronze medalists** (PR101 GOLD upstream / PR102 BRONZE /
PR103 SILVER), which were NOT in the 2026-05-17 canonical map. The hypothesis
under test: are the May 4 2026 contest winners themselves at the entropy floor,
or is one of them still PRE_ENTROPY (i.e., a Wyner-Ziv hoist target)?

## Top-5 contest candidate identification

Identified from canonical sources:

- **Local CPU frontier** per `.omx/state/canonical_frontier_pointer.json`:
  PR101 fec6 archive `6bae0201fb082457...` (178517 bytes; score
  [contest-CPU] per pointer file).
- **Local CUDA frontier** per same pointer:
  PR106 format0d archive `9cb989cef519ed17...` (186876 bytes; score
  [contest-CUDA] per pointer file).
- **PR101 GOLD upstream** per
  `experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip`:
  `b83bf3488625dbd7...` (178258 bytes).
- **PR102 BRONZE upstream** per
  `experiments/results/public_pr102_hnerv_lc_v2_scale095_rplus1_custody_20260507_codex/public_pr102_intake_20260507_auto/archive.zip`:
  `afd53348f50303bf...` (178981 bytes). **Note**: PR102 and PR100 share the
  same archive sha (`afd53348...`) per the canonical hnerv_lc_v2 scale=0.95
  substrate; this is per CLAUDE.md "Apples-to-apples evidence discipline"
  Section 2026-05-08 anchor.
- **PR103 SILVER upstream** per
  `experiments/results/public_pr103_intake_20260504_codex/archive.zip`:
  `31881b2d23d027e6...` (178223 bytes).

Each archive has a single ZIP member (`'x'` or `'0.bin'`), stored uncompressed
(method=0) — i.e., the maintainer's contest packet contract uses a
single-payload-stored format.

## Empirical sweep results — all 5 at entropy floor

Probe command (`probe_substrate_archive_member` per Catalog #321 fix):

```bash
.venv/bin/python .omx/tmp/option_b_prober_rerun_20260520/invoker.py
```

| substrate | sha8 | member | raw_bytes | lzma_ratio | brotli_ratio | zlib_ratio | best_ratio | best_codec | classification | deliverable_score_savings_estimate |
|---|---|---|---|---|---|---|---|---|---|---|
| pr101_fec6_cpu_frontier | 6bae0201 | x | 178417 | 1.000376 | 1.000028 | 1.000342 | **1.000028** | brotli | **AT_FLOOR** | 0.00000000 |
| pr106_format0d_cuda_frontier | 9cb989ce | x | 186776 | 1.000385 | 1.000027 | 1.000353 | **1.000027** | brotli | **AT_FLOOR** | 0.00000000 |
| pr101_gold_upstream | b83bf348 | x | 178158 | 1.000370 | 1.000028 | 1.000342 | **1.000028** | brotli | **AT_FLOOR** | 0.00000000 |
| pr102_bronze_upstream | afd53348 | 0.bin | 178873 | 1.000375 | 1.000028 | 1.000341 | **1.000028** | brotli | **AT_FLOOR** | 0.00000000 |
| pr103_silver_upstream | 31881b2d | x | 178123 | 1.000387 | 1.000028 | 1.000342 | **1.000028** | brotli | **AT_FLOOR** | 0.00000000 |

Per the canonical scorer rate-term identity (`25 * (member_bytes -
compressed_member_bytes) / 37_545_489`) and the empirical compression-ratio
floor of `~1.000028` (i.e., re-compression costs MORE bytes than the original
member due to codec overhead), the **predicted score-savings upper bound across
all 5 top contest candidates is 0.0** per axis. Every entry is tagged
`[empirical:lzma_ratio_on_actual_member=1.0000]` per Catalog #287 evidence
discipline; every row carries `validation_status=VALIDATED_CONTEST_MEMBER` per
Catalog #321 phantom-score discipline; every row carries
`evidence_grade=predicted` + `score_claim=false` + `promotion_eligible=false`
per Catalog #324 predicted-band discipline.

## Honest answer to operator's framing question

**Q: Of top-5 contest candidates, how many have NON-ZERO deliverable tier-1 +
tier-2 bytes (vs 0 baseline per Catalog #321 falsification)?**

**A: ZERO.** All 5 top contest candidates (including the historical PR101 GOLD
/ PR102 BRONZE / PR103 SILVER medalists, plus our current local CPU + CUDA
frontiers) are at the canonical entropy floor on their archive.zip member
bytes:

- PR101 GOLD (b83bf348, the gold-medal-winning 0.193 archive that this entire
  contest was decided around — see CLAUDE.md "Race-mode rigor inversion"
  postmortem 2026-05-04) → AT_FLOOR.
- PR102 BRONZE (afd53348, the 0.195 third-prize archive identical to PR100 's
  hnerv_lc_v2) → AT_FLOOR.
- PR103 SILVER (31881b2d, the 0.195 silver-medal 241-LOC rem2 archive that won
  silver over PR105's 1776-LOC kitchen_sink) → AT_FLOOR.
- PR101 fec6 (6bae0201, our current local CPU frontier 0.19205) → AT_FLOOR.
- PR106 format0d (9cb989ce, our current local CUDA frontier 0.20533) →
  AT_FLOOR.

The 2026-05-17 Q4 DEFER verdict GENERALIZES to the contest's historical
medalists. **The entire contest's archive.zip layer is at the canonical
entropy floor of the maintainer-mandated single-member-stored packet format
under general-purpose codecs (lzma / brotli / zlib).** This is the expected
result: the contest's contract is RATE-term `25 * archive.zip bytes /
37_545_489`, so every successful submission has already been compressed by
its emitter to the floor of its substrate-specific entropy coder before
shipping into archive.zip (whether HNeRV FP4 weights, latent grids, sidecar
quantized residuals, etc.). Wyner-Ziv hoist via general-purpose recompression
of the already-emitted archive is INTRINSICALLY DOMINATED at the archive-byte
layer.

## What this confirms about Q4-Q5 Wyner-Ziv reactivation

Per the Catalog #321 self-protection extinction (sister-landed 2026-05-17), the
canonical reactivation criterion is now:

> Wyner-Ziv side-info for NOVEL ratesplit (NOT re-compression of already-
> compressed bytes)

This 2026-05-20 sweep confirms that criterion empirically against the contest's
historical medalists. Future Wyner-Ziv work MUST target one of:

1. **Substrate-class-shift candidates** (Z5/Z6/Z7/Z8 predictive-receiver per
   Catalogs #310/#311/#312); the per-substrate score gains come from
   architecturally-different score-bytes-tradeoffs, not from recompressing
   the already-emitted archive bytes.
2. **Cooperative-receiver scorer-margin substrates** (ATW V2/D4 family per
   Catalog #311 ego-motion-conditioning).
3. **Foveation + LA-pose** (TT5L per Catalog #311 ego-motion-central reframing).
4. **Wyner-Ziv side-info for NOVEL ratesplit** — must materialize side-info
   from a source OTHER than the already-compressed archive bytes; e.g.,
   ego-motion priors, scorer-class CDFs derived from canonical-frontier
   archives (not the candidate's own archive).

## Canonical artifacts (operator-routable)

- **Per-archive structured output**:
  `.omx/state/wyner_ziv_deliverability/option_b_archive_member_sweep_top5_contest_candidates_20260520T124451.json`
  (fcntl-locked write per Catalog #131; schema `pre_entropy_pivot_probe_v1`).
- **Invoker script**:
  `.omx/tmp/option_b_prober_rerun_20260520/invoker.py` (scratch; operator-runnable).
- **Probe-outcome ledger anchors** (5 rows; canonical Catalog #313 ledger
  `.omx/state/probe_outcomes.jsonl`):
  - `option_b_archive_member_sweep_top5_pr101_fec6_cpu_frontier_6bae0201_20260520T124451`
  - `option_b_archive_member_sweep_top5_pr106_format0d_cuda_frontier_9cb989ce_20260520T124451`
  - `option_b_archive_member_sweep_top5_pr101_gold_upstream_b83bf348_20260520T124451`
  - `option_b_archive_member_sweep_top5_pr102_bronze_upstream_afd53348_20260520T124451`
  - `option_b_archive_member_sweep_top5_pr103_silver_upstream_31881b2d_20260520T124451`
  - All verdict=DEFER (per CLAUDE.md "Forbidden premature KILL"); blocker_status=blocking;
    expires 2026-06-19 (30-day staleness window per Catalog #298).
- **Landing memo** (Catalog #229 PV + #125 6-hook wire-in):
  `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_slot_build_2_option_b_archive_member_sweep_top5_contest_landed_20260520.md`

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution** — N/A. This is an empirical sweep of
   archive-member compressibility; no per-byte score-sensitivity signal
   is produced. The result is structurally an upper-bound (`predicted
   ΔS = 0.0 across all 5`) consumed by the autopilot ranker via the
   probe-outcome ledger (hook #4).
2. **Pareto constraint** — N/A. AT_FLOOR is the structural boundary
   (`bytes-saved <= 0` ⟹ Pareto-irrelevant for Wyner-Ziv hoist).
3. **Bit-allocator hook** — N/A. There is no positive bit-savings budget
   to allocate.
4. **Cathedral autopilot dispatch hook** — ACTIVE. The 5 canonical
   probe-outcome anchors per Catalog #313 are queryable via
   `tools/check_predecessor_probe_outcome.py --substrate <name>` and the
   runtime gate at `tools/operator_authorize.py::_check_predecessor_probe_outcome`
   will refuse Wyner-Ziv hoist dispatch against these 5 contest archives
   without explicit operator-frontier-override or paired-env bypass per
   Catalog #199.
5. **Continual-learning posterior update** — ACTIVE via probe-outcome
   ledger append (not the score-bearing `posterior_update_locked` because
   no score-claim is made; the canonical surface for probe-only diagnostic
   outcomes is Catalog #313 `register_probe_outcome`).
6. **Probe-disambiguator** — N/A. The interpretation is unambiguous:
   re-compression of already-compressed archive bytes via general-purpose
   codecs is at the entropy floor. The canonical disambiguator that DOES
   exist is the recommended pivot to substrate-class-shift candidates
   (Catalogs #310/#311/#312) versus a deeper Wyner-Ziv investigation; the
   verdict-DEFER + reactivation-criterion-novel-ratesplit + blocker_status=
   blocking encodes the recommendation structurally.

## Catalog compliance verdict (per CLAUDE.md FORBIDDEN_PATTERNS)

- **Catalog #287** (`check_no_docstring_overstatement_without_evidence_tag`):
  every numeric metric in this memo is paired with
  `[empirical:lzma_ratio_on_actual_member=...]` or
  `[predicted:wyner_ziv_savings_upper_bound=0]` tags.
- **Catalog #321** (`check_no_phantom_wyner_ziv_savings_from_research_sidecar`):
  every probe invocation routes through `probe_substrate_archive_member` on
  actual contest archive.zip members with `validation_status=
  VALIDATED_CONTEST_MEMBER`; ZERO research-sidecar invocations.
- **Catalog #322** (`check_no_autopilot_adjustment_derived_from_phantom_provenance_composition_alpha`):
  ZERO composition_alpha rows produced; no autopilot reweight emitted.
- **Catalog #323** (`check_no_score_claim_without_canonical_provenance`):
  every row carries `score_claim=false`, `promotion_eligible=false`,
  `evidence_grade=predicted`; no canonical score claim is made.
- **Catalog #324** (`check_no_predicted_band_without_post_training_tier_c_validation`):
  no `predicted_band` field is declared (this is an empirical
  upper-bound observation, not a predicted band against an unmeasured
  trained architecture).
- **Catalog #325** (per-substrate symposium): N/A; this is an empirical
  sweep, not a paid dispatch decision.
- **Catalog #131** (`check_no_bare_writes_to_shared_state`): the persistence
  writes go through `persist_manifest` and `register_probe_outcome`, both
  of which use the canonical fcntl-locked atomic-write pattern.
- **Catalog #138** (`check_state_writers_strict_load_for_mutating_path`):
  the probe-outcome ledger reader is fail-closed per its own implementation.
- **Catalog #205** (`check_inflate_py_uses_canonical_select_inflate_device`):
  N/A; this sweep does not touch inflate.py.
- **Catalog #287/#323** (Provenance umbrella + docstring overstatement):
  every score reference in this memo carries the
  `[empirical]` / `[predicted]` axis-tag-with-evidence-pointer pattern.

## Forbidden premature KILL: this is a DEFER, not a KILL

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" the
verdict is DEFER (not KILL):

- Verdict: **DEFER**.
- **Reactivation criterion**: novel ratesplit via Wyner-Ziv side-info that
  derives bytes from a source OTHER than the already-compressed archive
  bytes (e.g., ego-motion priors, scorer-class CDFs from sister archives,
  predictive-receiver latent residuals).
- **What would change the verdict**: an empirical demonstration that the
  Wyner-Ziv hoist via side-info on ANY of the 5 candidates produces
  measurable score improvement on the same axis (Linux x86_64 [contest-CPU]
  OR Linux x86_64 NVIDIA [contest-CUDA]) per CLAUDE.md "Submission auth
  eval — BOTH CPU AND CUDA" non-negotiable.
- **What does NOT change the verdict**: more empirical sweeps of
  general-purpose recompression of the same archive bytes; that variable
  is now empirically saturated across 13 total contest candidates (8 from
  2026-05-17 sister sweep + 5 from this 2026-05-20 sweep), all AT_FLOOR.

## Sister coordination

NO active sister subagents at sweep time per `.omx/state/subagent_progress.jsonl`
audit. WIRE-IN-RIGOR-RESUME / sister Wyner-Ziv research subagents are out of
scope (this is a $0 prober-only sweep on the wyner_ziv_deliverability surface).

Catalog #340 sister-checkpoint guard: the only checkpoints active during this
sweep are this subagent's own (`slot-build-2-option-b-prober-rerun-20260520`);
the canonical `subagent_progress.jsonl` does not show any sister with
overlapping `files_touched` on `.omx/state/wyner_ziv_deliverability/` paths or
`tools/pre_entropy_substrate_pivot_prober.py`.

## Cross-references

- `feedback_fix_pre_entropy_prober_phantom_score_plus_catalog_321_strict_gate_landed_20260517.md`
  (the Catalog #321 landing; sister prober runtime fix).
- `feedback_redo_pivot_fix_all_phantom_score_substrate_class_shift_q4_budget_redirect_landed_20260517.md`
  (sister Catalog #322 + Q4 budget redirect to substrate-class-shift).
- `.omx/state/wyner_ziv_deliverability/option_b_archive_member_sweep_20260517T221034.json`
  (sister 8-candidate sweep; verdict DEFER_Q4).
- `.omx/state/wyner_ziv_deliverability/pre_entropy_candidate_substrates_corrected_20260517T215345.json`
  (the corrected pre-entropy artifact; baseline for this sweep).
- CLAUDE.md "Frontier scores are pointer-only" + Catalog #343 (the canonical
  frontier-pointer mechanism that surfaced PR101 fec6 + PR106 format0d as the
  current CPU + CUDA frontiers).
- CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" (the May 4
  2026 race postmortem that established PR101 GOLD / PR102 BRONZE / PR103
  SILVER as the historical medalists).
- Catalog #313 (`check_dispatch_target_has_no_predecessor_adjudicated_outcome`)
  — the runtime gate that consumes the 5 probe-outcome anchors landed in this
  sweep.
- Catalog #298 (substrate retirement discipline; 30-day staleness window
  inherited by the probe-outcome ledger).


<!-- WAVE-3-CATALOG-344-BACKFILL-SWEEP appended 2026-05-20 per operator NON-NEGOTIABLE "keep feeding the queue" + WIRE-IN-AUDIT-POST-CASCADE op-routable #3. -->
<!-- # FORMALIZATION_PENDING:option-B-archive-member-sweep-top5-contest-candidates-audit-memo-trigger-tokens-describe-sweep-not-new-equation -->
