# SYSTEM_MAP.md — What this system is, structurally

**Audience:** Same as `HANDOFF.md` (read that first). This file is the
structural diagram — how the pieces fit. Not a tutorial.

**Last refreshed:** 2026-05-16 (premortem consolidation wave). Stale
against the live repo after 30 days; re-run
`tools/regenerate_system_map.py` (deferred; not yet landed).

---

## 1. Top-level directory map

```
pact/                               # Historical/internal checkout alias.
├── src/tac/                        # The canonical Task-Aware Compression
│                                   # library and compression engine.
│   ├── preflight.py                # ~63K LOC. ~295 STRICT gates. The
│   │                               # single source-of-truth for repo-wide
│   │                               # bug-class extinction.
│   ├── substrates/                 # Per-substrate trainer + architecture
│   │   │                           # + score-aware loss + archive packer.
│   │   ├── _shared/                # Canonical helpers used across
│   │   │                           # substrates (trainer_skeleton,
│   │   │                           # smoke_auth_eval_gate,
│   │   │                           # score_aware_common,
│   │   │                           # inflate_runtime).
│   │   ├── d1_segnet_margin_polytope/
│   │   ├── d4_wyner_ziv_frame_0/
│   │   ├── pretrained_driving_prior/  # DP1: pretrained codebook over
│   │   │                              # comma2k19 OOD data.
│   │   ├── pr101_lc_v2_clone_enhanced_curriculum/
│   │   ├── nscs01_nullspace_split_renderer/
│   │   ├── nscs03_end_to_end_balle_joint_codec/
│   │   └── ... (43+ substrate packages today)
│   ├── deploy/
│   │   ├── modal/                  # Modal SDK wrappers: mount_manifest,
│   │   │                           # call_id_ledger, runtime env block.
│   │   ├── lightning/              # Lightning SDK wrappers + active jobs
│   │   │                           # JSONL store (fcntl-locked).
│   │   ├── azure/                  # Azure VM dispatcher + active VMs JSONL.
│   │   └── claims.py               # Cross-agent dispatch lane claims.
│   ├── continual_learning.py       # Posterior anchor JSONL store.
│   ├── cost_band_calibration.py    # Dispatch cost-band posterior + the
│   │                               # Time-Traveler provider routing
│   │                               # amendment (Catalog #237 / #239).
│   ├── differentiable_eval_roundtrip.py
│   ├── unified_action.py           # Where the meta-Lagrangian /
│   │                               # GR-style S_total target lives.
│   └── tests/                      # ~1450 tests; pytest discovers via
│                                   # pyproject testpaths.
│
├── experiments/                    # Per-experiment trainer entry points.
│   ├── train_substrate_*.py        # One per substrate (43+ files).
│   ├── modal_train_lane.py         # Canonical Modal dispatcher.
│   ├── contest_auth_eval.py        # The authoritative auth-eval CLI
│   │                               # (every substrate trainer routes
│   │                               # through `gate_auth_eval_call`).
│   ├── pipeline.py                 # Canonical end-to-end training
│   │                               # pipeline (the "openpilot standard").
│   └── results/                    # DERIVED_OUTPUT trove (249 GB today).
│                                   # gitignored; per Catalog #113
│                                   # HISTORICAL_PROVENANCE for committed
│                                   # build manifests only.
│
├── submissions/                    # Per-submission packet (archive +
│   │                               # inflate.sh + inflate.py).
│   ├── exact_current/              # The pinned upstream-compatible
│   │                               # submission (DO NOT EDIT without
│   │                               # explicit operator approval).
│   ├── robust_current/             # The actively-improved submission.
│   ├── a1/                         # Internal apogee 0.1928 [contest-CPU].
│   ├── pr106_*_residual_sidecar/   # 13 pr106 family clones (premortem
│   │                               # #F flagged for vendoring dedup).
│   └── _vendored/                  # (Planned per premortem #F):
│                                   # all non-canonical PR intakes here.
│
├── upstream/                       # PINNED contest snapshot. Source of
│   │                               # truth for SegNet/PoseNet weights +
│   │                               # evaluate.py scoring formula.
│   │                               # NEVER EDIT.
│   ├── evaluate.py                 # The authoritative scoring formula.
│   ├── modules.py                  # SegNet + PoseNet defs (`smp.Unet
│   │                               # tu-efficientnet_b2`; FastViT-T12).
│   └── videos/0.mkv                # The contest test video.
│
├── tools/                          # ~545 one-off + canonical helpers.
│   ├── operator_authorize.py       # Canonical dispatch entry point.
│   ├── subagent_commit_serializer.py  # fcntl-locked commit serializer.
│   ├── subagent_checkpoint.py      # Crash-resume checkpoint store.
│   ├── claim_catalog_number.py     # Atomic catalog # claim (with
│   │                               # --commit-via-serializer for
│   │                               # git-transactional claim).
│   ├── lane_maturity.py            # Lane registry + audit log writer.
│   ├── harvest_modal_calls.py      # Walks modal_metadata.json + pulls
│   │                               # Modal result-cache artifacts.
│   ├── canonical_dispatch_optimization_protocol.py  # Catalog #270
│   │                               # umbrella protocol helper.
│   ├── local_pre_deploy_check.py   # 30s pre-dispatch harness (8 checks).
│   ├── run_codex_review_for_dispatch.py  # Catalog #271.
│   ├── audit_stale_l1_substrates.py # Catalog #298 sister (NEW
│   │                               # 2026-05-16).
│   ├── audit_memory_file_freshness.py  # Memory-rotation hygiene
│   │                               # (NEW 2026-05-16).
│   ├── cluster_summarize_memory_category.py  # Quarterly cluster
│   │                               # summarization (NEW 2026-05-16).
│   └── archive_jsonl_state.py      # State-JSONL archival (NEW
│                                   # 2026-05-16).
│
├── scripts/                        # Lane drivers + bootstrap helpers.
│   ├── operator_authorize_substrate_*.sh  # Per-substrate dispatch
│   │                               # wrappers (the operator-facing
│   │                               # entry points).
│   ├── remote_lane_substrate_*.sh  # Modal worker-side lane drivers
│   │                               # (must carry canonical 3-export
│   │                               # NVML/CUDA env block per Catalog
│   │                               # #244).
│   ├── remote_archive_only_eval.sh # Canonical remote bootstrap (uv
│   │                               # install + torch driver-version
│   │                               # auto-pin per CLAUDE.md
│   │                               # "Forbidden uv torch install
│   │                               # without driver-version pin").
│   └── ensure_remote_uv.sh         # Sister uv-install bootstrap.
│
├── .omx/                           # Operator-MX state + research ledgers.
│   ├── state/                      # LIVE_STATE (most files gitignored
│   │                               # per Catalog #113). Critical:
│   │   ├── lane_registry.json      # 758+ lanes; canonical mutated only
│   │   │                           # via tools/lane_maturity.py.
│   │   ├── cost_band_posterior.jsonl  # Dispatch cost-band anchors.
│   │   ├── modal_call_id_ledger.jsonl  # Catalog #245 append-only.
│   │   ├── commit-serializer.log   # Catalog #117/#157/#174 audit.
│   │   ├── subagent_progress.jsonl # Catalog #206 crash-resume.
│   │   └── continual_learning_posterior.jsonl  # Catalog #128.
│   ├── research/                   # Dated research ledgers (commit
│   │                               # OK). Per CLAUDE.md "tac stays
│   │                               # clean".
│   ├── operator_authorize_recipes/ # Per-substrate dispatch recipes
│   │                               # (YAML).
│   └── calibration/                # Anchor JSONs.
│
├── .ralph/                         # Ralph run-log + experiment timeline.
├── .agents/                        # Inter-agent coordination.
├── docs/                           # Public-facing docs (must be
│   │                               # sanitized per "Public Disclosure
│   │                               # Hygiene"; no local absolute paths
│   │                               # per Catalog #208).
│   └── paper/                      # Paper/writeup draft; no venue commitment.
├── reports/                        # Per-cycle reports.
│   ├── latest.md                   # The session-warm summary (often
│   │                               # stale; premortem Category N).
│   └── lane_maturity.md            # Auto-generated from lane_registry.
├── HANDOFF.md                      # ← you are here.
├── SYSTEM_MAP.md                   # ← you read this next.
├── CLAUDE.md                       # Agent-binding contracts (~2.4K
│   │                               # LOC; 41 NON-NEGOTIABLE markers;
│   │                               # 295 cataloged STRICT gates).
└── PROGRAM.md                      # High-level mission + architecture
                                    # statement.
```

## 2. Substrate lifecycle

Per CLAUDE.md "Lane maturity registry" non-negotiable, every substrate
goes through:

```
SKETCH (L0)              SCAFFOLD (L1)              INTEGRATION (L2)         FULL PRODUCTION HARDENED (L3)
   │                         │                          │                           │
   ▼                         ▼                          ▼                           ▼
Operator + council     `experiments/train_       Real-archive empirical    All 7 gates true:
verdict; no code.      substrate_<id>.py`        measurement (Lane G v3    impl_complete +
Lane registered via    `_full_main` is           reference anchor or       real_archive_empirical +
`tools/lane_maturity.  IMPLEMENTED (not          equivalent); recipe       contest_cuda +
py add-lane`.          raise NotImplementedError); declared dispatch-      strict_preflight +
                       archive grammar           enabled (not              three_clean_review +
                       declared; runtime         research_only).           memory_entry +
                       contract honored;                                   deploy_runbook.
                       MAY be `research_only=
                       true` per HNeRV parity                              ONLY `lane_g_v3` is
                       L2.                                                 currently L3.
```

**Promotion is gate-by-gate** via `tools/lane_maturity.py mark <lane>
--gate <gate> --evidence <path>`. Computed level is NEVER set by hand.

**Catalog #220** refuses L1+ scaffolds with byte addition >1 KB without
operational mechanism. **Catalog #272** refuses L2+ substrate promotion
without the 4-field Distinguishing-Feature Integration Contract.
**Catalog #233** enforces the 4-gate canonical for L1→L2 promotion
(smoke green + Tier C MDL density measured + 100ep auth-eval anchor +
Catalog #127 custody validated). **Catalog #298** (NEW 2026-05-16)
refuses L1 SCAFFOLD substrate lanes with no dispatch in 30 days unless
opted-out as `research_only=true` / `lane_class=substrate_engineering`
/ moved to archived state / waiver.

## 3. Dispatch flow diagram

```
   operator                                                                     paid GPU meter
      │                                                                              ▲
      │ ./scripts/operator_authorize_substrate_<id>_modal_a100_dispatch.sh           │
      ▼                                                                              │
┌──────────────────────────────────────────────────────────────────────┐             │
│ tools/operator_authorize.py                                          │             │
│                                                                      │             │
│   1. Catalog #152: validate required input files (cheap; pre-spend)  │             │
│   2. Catalog #243: tools/local_pre_deploy_check.py (30s harness;     │             │
│      8th check is Catalog #270 umbrella protocol)                    │             │
│   3. Catalog #271: tools/run_codex_review_for_dispatch.py            │             │
│      (codex adversarial review; only if cost > $1)                   │             │
│   4. Catalog #199: paired-env operator confirmation                  │             │
│   5. Catalog #166: Modal HEAD-parity ledger (clean-tree gate)        │             │
│   6. Catalog #167: tools/run_modal_smoke_before_full.py              │             │
│      (canary $0.30 smoke validates integration BEFORE $5-15 full)    │             │
│   7. Catalog #143: tac.deploy.modal.call_id_ledger.                  │             │
│      register_pending_job_locked (before submit)                     │             │
│                                                                      │             │
│   --> experiments/modal_train_lane.py::fn.spawn(...)                 │─────────────┘
│                                                                      │             $
│   8. Catalog #245: register_dispatched_call_id (post-submit)         │
└──────────────────────────────────────────────────────────────────────┘
      │                                                                              │
      ▼                                                                              │
  Modal worker runs scripts/remote_lane_substrate_<id>.sh                            │
  (sourced via REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 sentinel per                   │
   Catalog #163; carries Catalog #244 NVML env block).                               │
      │                                                                              │
      │ ~30 min to 4 hr training                                                     │
      ▼                                                                              │
  Worker writes archive.zip + inflate.sh + auth_eval JSON to /modal_results/         │
      │                                                                              │
      ▼                                                                              │
  Worker returns artifact dict to FunctionCall return cache (24h TTL).               │
      │                                                                              │
      ▼                                                                              │
  Local operator runs tools/harvest_modal_calls.py within 24h per CLAUDE.md          │
  "Modal .spawn() HARVEST OR LOSE".                                                  │
      │                                                                              │
      ▼                                                                              │
  Harvested artifacts → experiments/results/lane_<label>_modal/harvested_artifacts/  │
      │                                                                              │
      ▼                                                                              │
  Update lane_registry.json gate evidence via tools/lane_maturity.py mark.           │
```

## 4. Catalog gate categories

The CLAUDE.md "Meta-bug class catalog" table now lists ~295 strict-mode
preflight gates organized by class:

- **Bug class gates (#1-#100)** — Per-bug-pattern extinction. Examples:
  `#1 check_no_mps_fallback_default`, `#5 check_no_eval_roundtrip_false`,
  `#12 preflight_arity` (CLI flag arity check), `#88
  check_training_paths_use_ema_correctly`.
- **META-bug class gates (#100-#200)** — Cross-surface bug-class
  extinction. Examples: `#113 check_artifact_lifecycle_compliance`
  (umbrella over the 5 provenance-vs-state surfaces), `#117 / #157
  / #174 commit-serializer trio`, `#118 check_claude_md_catalog_no_duplicate_numbers`,
  `#125 check_subagent_landing_has_solver_wire_in`,
  `#126 check_lane_pre_registered_before_work_starts`.
- **META-meta gates (#159 / #176 / #185 / #186 / #289)** — Gates that
  protect OTHER gates / the catalog table itself. The "phantom row"
  protection layer.
- **Substrate-specific gates (#220 / #233 / #240 / #270 / #272 / #298)**
  — Substrate-lifecycle enforcement.
- **Council / discipline gates (#229 / #241 / #242 / #265 / #290 /
  #291 / #292)** — Council deliberation discipline, substrate META
  contract, design-memo discipline.
- **Recipe schema gates (#170 / #171 / #181 / #182 / #215)** — Per-recipe
  YAML field declarations.

**Gate consolidation quota (NEW 2026-05-16, Catalog #299):** no new
strict-mode preflight callsite may be added with catalog # > 400 without
operator approval. The premortem projects 500-700 gates by 2027-05-16 at
current cadence; the #400 quota is the operator-visible brake.

## 5. Council apparatus

Per CLAUDE.md "Council conduct" non-negotiable:

- **Sextet pact (binding):** Shannon LEAD + Dykstra CO-LEAD + Yousfi +
  Fridrich + Contrarian + Assumption-Adversary. Every council-grade
  decision needs all 6 explicit positions with operating-within
  assumption statement (per Catalog #292).
- **Inner ten (extended):** + Quantizr + George Hotz + Selfcomp +
  MacKay (memorial) + Ballé. Non-conservative skunkworks.
- **Grand council (advisory, 20 seats):** Boyd, Tao, Filler, Mallat,
  van den Oord, Carmack, Hassabis, Hinton, Karpathy, Schmidhuber +
  Jack-from-skunkworks + 8 new seats added 2026-05-15 (Atick +
  Redlich + Rao + Ballard + Tishby memorial + Zaslavsky + Wyner +
  Time-Traveler protégé).

The **Assumption-Adversary** seat (added 2026-05-15) is structurally
distinct from the Contrarian: where Contrarian challenges weak
ARGUMENTS, Assumption-Adversary challenges the FRAMING all arguments
share. Per CLAUDE.md "META-ASSUMPTION ADVERSARIAL REVIEW" non-negotiable
(landed because 10+ codex reviews + multiple grand councils never caught
the canonicalization-by-default reflex suppressing substrate-optimal
engineering across the entire contest).

## 6. Canonical helpers (the 7 you'll touch most)

1. **`tools/subagent_commit_serializer.py`** — fcntl-locked commit
   serializer. EVERY subagent commit goes through this. Use
   `--expected-content-sha256 <file>=<post-edit-sha>` per Catalog #157.
2. **`tools/claim_catalog_number.py`** — Atomic catalog # claim.
   Always pass `--commit-via-serializer --reason "<purpose>"` per
   Catalog #186 to make the claim git-transactional.
3. **`tools/operator_authorize.py`** — Canonical dispatch entry point.
   Wires all 8 pre-dispatch gates in the right order. Never call
   `modal run` / `modal_train_lane.py` directly.
4. **`tools/lane_maturity.py`** — Lane registry mutation. Use
   `add-lane` / `mark` / `set-field` / `audit` / `report` / `validate`
   subcommands. Audit log appends to
   `.omx/state/lane_maturity_audit.log`.
5. **`tools/subagent_checkpoint.py`** — Crash-resume checkpoint store
   per Catalog #206. Every long-running subagent (>5 tool uses)
   checkpoints every ~10 tool uses.
6. **`tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call`**
   — Canonical auth-eval routing helper. Per Catalog #226: every
   substrate trainer routes through this (no hand-rolled
   `experiments/contest_auth_eval.py` subprocess invocations).
7. **`tac.deploy.modal.call_id_ledger`** — Canonical Modal call_id
   ledger. Use `register_dispatched_call_id` immediately after
   `fn.spawn()` returns; harvest via `query_unharvested`.

---

**For more depth:** read `CLAUDE.md` cover-to-cover (the 41
NON-NEGOTIABLE markers ARE the system's bone structure). Then drill into
`src/tac/preflight.py` for the gate definitions. For per-substrate
implementation reference, `src/tac/substrates/_shared/` is the canonical
helper layer all substrates must compose against.
