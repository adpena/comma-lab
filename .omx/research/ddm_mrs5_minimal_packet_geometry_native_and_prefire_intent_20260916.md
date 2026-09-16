# ddm_mrs5 — the minimal packet at the sealed receiver's MEASURED native set, a same-host interleaved cost ratio, and the pre-fire intent that the frozen contract REFUSES (2026-09-16)

<!-- # FORMALIZATION_PENDING: arm memo; the measured ratio and the three typed refusals are the objects. No canonical equation is registered here — the projection is explicitly non-authoritative and the refusals are contract facts, not laws. -->

**Axis of everything local in this memo: `[macOS-CPU advisory]`, `score_claim=false`, `authority=false`.
No score, no promotion, no timing clearance. The exact pointer did not move and this arm never fires.**

Produced by codex (astra xhigh) through step 4 and **RESUMED on Opus** at 19:10Z after the codex arm hit
its usage limit. Everything below is either re-verified here against primary artifacts or explicitly
labelled as inherited from a receipt.

---

## 1. The packet — seven files (MEASURED)

`submissions/mrs5/`, runtime digest `609ed1efb1bd465cbe1cafd07c38de8cab3c707fa27a70a4043102450ba9d5db`
(`tac.candidate_seal.measure_runtime_digest`, 7 files, 370,308 B):

| file | LOC | bytes | sha256 |
|---|---:|---:|---|
| `inflate.sh` | 8 | 470 | `f7f1b608640bc9e5bdebfa6a1fe08fe89bd13d6c01374b00b83357dd8f8d399e` |
| `inflate.py` | 2,891 | 147,776 | `f79f1e3f0ef897df0f2f010646b0fe27fb453c6a7585e98e2769df1ece413e12` |
| `corrector.c` | 693 | 31,190 | `71f632bc59893cb33673e44f8baa9bac588586463ab057f917fc0fa16b5bed77` |
| `geometry.c` | 145 | 5,643 | `ac99ab49efbc9fce9bab177b70a62b8e32b576caea16a03d78326b61bf9a0e63` |
| `range_decoder.c` | 100 | 4,018 | `d70a494987ff9ea720b9150a5baf3343299139c6ce8c090c388974d03c14baec` |
| `README.md` | 28 | 1,925 | `d9d426b0288cf09a43d217686c920ca5d215c2e2fb1cceea605fa3d934795277` |
| `archive.zip` | — | 179,286 | `aab908d3b32c65582b7de1a3855b67a4f6fdb9373b0dc81e3049ee8164ec3957` |

The archive is **move 53's exact bytes** — MEASURED equal to the live pointer's
`pointer_archive_sha256` and to the pd6 sealed runtime's archive. Charter budgets honoured: geometry.c
145 ≤ 150, inflate.sh 8 ≤ 9, README 28 ≤ 30. `geometry.c` is the third native file that restores the
configuration the 1,185.9 s T4 leg actually ran (`rc64_backend.c` + `f26_corrector_native.c` +
`rlc1_geometry.c` natively, native HPAC off); mrs4 reproduced two of the three.

## 2. What was re-verified here vs inherited from a receipt

**Re-verified by execution on Opus (MEASURED):**

- All seven file shas + LOC above, against the working tree, and against the timing run's own
  `INPUTS.json` binding — identical.
- `tac.candidate_seal.measure_runtime_digest(submissions/mrs5)` → `609ed1ef…`, 7 files, 370,308 B.
- `tac.decode_wall_clock.measure_receiver_digest(submissions/mrs5)` → `6ee7beee…` (accepts the tree).
- `tac.candidate_seal.measure_prefire_risk_receiver_digest(submissions/mrs5)` → **REFUSES**, see §5.
- The same two digests on the sealed pd6 runtime → `fdef3b00…` (legacy) and `9f6e7168…` (risk) — the
  latter is exactly the reference digest the frozen pr14 amendment names, so the risk normalizer is
  behaving as pinned on a pin-bearing tree.
- `check_pin_consistency(submissions/mrs5)` → `PIN_ABSENT`, see §5.
- Live pointer: `S = 0.1361014714463198`, archive `aab908d3…` (move 53).
- The A/B/A/B ratio, re-derived from the four run rows rather than read from the summary (§4).
- **The 600/600 raw identity, re-verified against the PRIMARY pd6 artifact, not the mrs5 receipt:**
  mrs5's cold n600 public decode produced `0.raw`, 3,662,409,600 B, sha
  `8a14f55a6a8b141501836f511f4222dde75dca21714d923ad865b5f4757ef66b`. The same sha appears in
  `/Volumes/APDataStore/pact/ddm_pd6/parseback2/PARSEBACK_RESULT.json` at both
  `inflate_report.raw_sha256` and `rendered_raw.sha256`, over the same archive `aab908d3…`, with
  `decoded_field_matches_admitted: true` — and in the pd6 memo's own table at lines 364 and 398. I also
  compared the 600-element `pair_sha256` list element-wise against the mrs1 reference
  (`/Volumes/APDataStore/pact/ddm_mrs1/public_smoke/IDENTITY.json`, sha `5fa221bb…`): **600/600 equal,
  no differing index**, and that reference's own full-file sha is the same `8a14f55a…`. So the rewritten
  seven-file receiver reproduces move 53's rendered output **byte for byte at both granularities** —
  the whole 3,662,409,600-byte file and every one of the 600 pairs.

**Inherited from the predecessor's receipts (not re-run; each names its artifact):**

- 48-pair stratified geometry/token/raw parity: 48/48, 9,120 geometry groups, 9,437,184 geometry bin
  values and 9,437,184 token bytes per path, 292,992,768 raw bytes per path —
  `PARITY_SUMMARY.json` → `/Volumes/APDataStore/pact/ddm_mrs5/parity/RESULT.json`
  sha `58264ea793d6b665a2f9a93ec1d23d89a4f64bb30164c89b3b21cb1a99e04cc0`.
- **Both bare-venv smokes (PASSED):** compiler **present** → the cold n600 public run itself
  (`public_native/IDENTITY.json` sha `ec5c963a…`, 600 matched pairs); compiler **absent** → 48 pairs
  with stale libraries removed (`public_fallback/RESULT.json` sha `0f12a8c9…`), i.e. the Python
  fallback carries the identical result when `cc` is missing. Entry-point smokes retained separately
  (`public_native/help.DONE.json`, `public_fallback/help.DONE.json`, `public_native/default_guard.DONE.json`).
- The cold n600 wall clock of 1,631.44 s is recorded as **correctness-run metadata only**
  (`wall_is_comparative_timing_evidence: false`) — pd7 loaded the host throughout.

**A fresh-eyes review of those three instruments found six issues; I verified each myself against the
primary artifacts.** Full detail in `.omx/research/ddm_mrs5_20260916/INSTRUMENT_REVIEW_FINDINGS.md`.
The two that matter to a reader of this memo:

- **One of the two geometry comparisons in the parity proof cannot fail** (`ddm_mrs5_proof.py:67`
  compares two class planes that both sides write with the *same* pure-Python line). The 9,437,184
  figure quoted above comes from the *other* comparison (`:62`, inside `ShadowGeometry.contexts`), and
  I confirmed `NativeGeometry.contexts` reads the **C library's own moment state through the handle**
  — so the geometry C arithmetic is genuinely covered, by `bins` and not by `plane`.
- **The smoke accepts the parity result on pair count alone, without binding it to the receiver it
  smokes.** The parity ran in `development_public/`, the smoke in `submissions/mrs5/`. I compared all
  five source files sha-for-sha: **identical**. So the 48-pair parity was measured on exactly the
  shipped bytes — verified by me, not by the gate.

Two further findings are latent resume-path hazards (`stale_library_removed` and
`cold_public_subprocess` are written unconditionally while the stages that earn them can be served from
cache). I checked `attempts/` in both smokes: **exactly one attempt per stage, nothing cached**, so both
claims were earned here. The remaining two are cosmetic (a missing `promotable=False`, and a "bare venv"
flag asserted over a three-package probe). None invalidates a number in this memo.

## 3. Host state at measurement time (MEASURED)

`/Volumes/APDataStore` free: **15 GiB** (the tier is at 100% capacity; the arm's own store is 1.6 GiB,
under the 2 GiB cap). Nothing was written to `/Volumes/VertigoDataTier`. pd7 held the host and the
scorer throughout; this arm needed no scorer and never touched `/Volumes/APDataStore/pact/ddm_pd7/`.
1-minute load average sat between **13.50 and 16.04** across the whole interleave.

## 4. The instrument — same-host interleaved A/B/A/B (MEASURED)

Four cold runs, fixed A/B/A/B order, the same 48 seeded stratified pairs, same threads and device
policy. **A** = the sealed receiver at `/Volumes/APDataStore/pact/ddm_pd6/candidate2/candidate_runtime`
through its own `inflate.sh` and native builds; **B** = `submissions/mrs5` through its own. Each run
built its own native libraries and ran in a cold process. rc 0 on all four.

| # | arm | wall (s) | pair-compute (s) | load 1-min before → after | window (UTC) |
|---:|:--|---:|---:|:--|:--|
| 1 | A | 107.7548 | 91.8073 | 15.92 → 15.63 | 19:06:22 → 19:08:10 |
| 2 | B | 101.2751 | 86.1973 | 15.63 → 16.04 | 19:08:10 → 19:09:51 |
| 3 | A | 106.3497 | 89.6407 | 16.04 → 15.35 | 19:09:51 → 19:11:38 |
| 4 | B | 98.3643 | 83.0811 | 15.35 → 13.50 | 19:11:38 → 19:13:16 |

median(A) = **107.052233 s**; median(B) = **99.819741 s**.

**ratio = median(B)/median(A) = 0.932440.** Spread: adjacent ratios (B₂/A₁, B₄/A₃) =
**0.939867** and **0.924914**, half-range **±0.007476**; the extreme range over all four B/A pairings is
**[0.912854, 0.952284]**. A pair-compute-only ratio (excludes per-run setup) gives **0.932931** — the
ratio is not an artifact of the setup slice.

**Projected T4 = 1,185.899645171 s × 0.932440 = 1,105.7798 s**, against the 1,260 s policy limit and the
1,800 s hard timeout. **It clears both.** (DERIVED from a MEASURED ratio and a MEASURED T4 leg.)
Assumption, stated as the charter requires: serial-Python scaling transfers to the T4 host, the GPU
stages are unchanged code, and host load is approximately stationary across the interleave. The
interleaving is what makes the *ratio* meaningful under a loaded host; the absolute times are not
evidence and no confidence interval is claimed from n=2 per arm.

Because the projection clears 1,260 s, the charter's STOP-before-producer branch did **not** fire, and
the producer was attempted for real. **verdict_scope: this ratio is one host (M-series macOS, 48
restored-state pairs, pd7 co-resident). It is not a T4 measurement and confers no timing clearance.**

**The rewritten receiver measured FASTER than the sealed one, not slower** — 6.8% less wall clock on
identical work. That is the opposite of the risk the charter was guarding against, and it is the one
number a reader should carry: restoring the measured native set did not cost decode time.

## 5. The pre-fire intent — three real refusals, and the STOP

The producer was run for real, with truthful evidence, at the pointer's exact bytes:

```
.venv/bin/python tools/make_candidate_seal.py --first-fire-intent \
  --candidate-id ddm_mrs5_minimal_packet_geometry_native \
  --runtime-dir $PWD/submissions/mrs5 --axis contest_cuda --pointer-axis contest_cuda \
  --bar-tolerance 0 --admit-bar-net-ds 0.0 --public-entrypoint-smoke …/PUBLIC_SMOKES.json \
  --candidate-manifest … --manifest-validation … --twin-encode-receipt … \
  --archive-parseback-receipt … --raw-identity-receipt … --literal-census … \
  --retention-manifest … --timing-risk-evidence …/TIMING_RISK.json \
  --retained-path … --falsifier … --out …/prefire_attempt/CANDIDATE_PREFIRE_INTENT.json
```

**It refused (MEASURED), and NO intent file survives.** Typed refusal retained at
`.omx/research/ddm_mrs5_20260916/prefire_attempt/PREFIRE_REFUSAL_43ca2b00aebd4c4296d8356283327f0c.json`
(324 B, sha `8df50f1c0ee52069bf3eafef2060b7d63d0d0da68a07571e9ad80ee30986b346`):

```json
{"schema": "candidate_first_measurement_refusal.v1",
 "code": "PREFIRE_CONTRACT_DRIFT_REFUSED",
 "detail": "live file differs from committed blob: /Users/adpena/Projects/pact/tools/modal_harvest_poller.py"}
```

There are **three** distinct refusals, and the layering matters for MAIN.

### Gate 0 — candidate-independent, and it fires first (MEASURED)

`PREFIRE_CONTRACT_DRIFT_REFUSED: live file differs from committed blob: tools/modal_harvest_poller.py`.

The frozen contract's newest amendment row (`prefire_contract_consumer_fix.v1`, implementation commit
`a9fd1270495522f5335737ac56643cf8ace8e670`) lists 15 consumer files whose **live** bytes must equal that
commit's blobs. I checked all 15: **14 are live-clean; `tools/modal_harvest_poller.py` is not.** The
freeze record itself is internally consistent — every row's sha still matches its own committed blob —
it is the working tree that moved on: commit **`da66353b7`** landed a later fix to that consumer (+57/−1
lines; blob at the amendment `2a98ab62…` vs live `bc733834…`), and `git status` is clean, so live == HEAD.

**Consequence: no pre-fire intent can be produced today for ANY candidate**, not just mrs5, until a new
`prefire_contract_consumer_fix.v1` row re-pins that consumer at `da66353b7` or later. That is a MAIN
landing — `src/tac/candidate_seal.py` and the freeze record are read-only to this arm, and re-pinning is
exactly the kind of "patch around the control" the charter forbids me. **verdict_scope: this is a
staleness fact about the freeze record relative to HEAD on 2026-09-16, not a defect in the control.**

### Gate 1 — the risk digest refuses an unpinned receiver (MEASURED)

`PREFIRE_RISK_EVIDENCE_REFUSED: both archive pins required`, raised by
`tac.candidate_seal.measure_prefire_risk_receiver_digest(submissions/mrs5)`.

`tac.candidate_seal._materialize_prefire_receiver_rows` normalizes the *values* of the top-level
`ARCHIVE_SHA256` / `ARCHIVE_BYTES` assignments in `inflate.py` to `<ARCHIVE_PIN>` so that re-pinning an
archive cannot change a receiver's behaviour digest — and it **requires both assignments to exist**. The
minimal packet is unpinned by charter, so the required field
`candidate_receiver.sha256` of `candidate_prefire_timing_risk.v1` is not merely missing, it is
**uncomputable for this tree**. `validate_prefire_risk` checks the receipt's exact field set first, so a
receipt carrying honest extra fields refuses on shape; the root control fires under any shape.

The sister normalizer `tac.decode_wall_clock._receiver_rows` performs the *same* pin normalization but
does **not** require the pins, and accepts this tree (`6ee7beee…`). Two normalizers in one contract
family therefore disagree about the same directory — but not silently and not by environment: the
difference is one explicit `_pf_require` line, and the two digests carry different definition names that
the contract never interchanges. My reading (INFERRED, MAIN decides): this is a working control, not a
defect. A receiver that declares no archive pin makes the pin-normalization step vacuous, and the
`check_pin_consistency` message says so in its own words — "this tree is UNPINNED … this check is
vacuous for it (reported, never silently passed)". Pricing a pinless receiver from a pin-bearing
measured leg would be a transfer across an identity class the contract deliberately leaves undefined.

### Gate 2 — a receiver-only change cannot satisfy the admit bar (MEASURED + source)

`_pf_pointer` requires `net_dS_threshold < 0` **strictly** and `derived_net_dS < net_dS_threshold`. For
this candidate `derived_net_dS = 25 × (179,286 − 179,286) / 37,545,489 = **0.0 exactly**` (MEASURED).
No negative bar can be less than zero from above, so the condition is unsatisfiable by construction.
`_pf_identity` independently requires `check_pin_consistency(...).ok`, which is `False` here
(`PIN_ABSENT`) → `PREFIRE_IDENTITY_DRIFT_REFUSED: receiver/archive pins disagree`.

This is the deepest of the three. tc4/pr19 established that *a receiver change needs its own measured
leg, and the intent chain exists for exactly this*. But the intent object as frozen is shaped for a
candidate that **improves the score**: it demands a strictly negative admit bar and a rate delta below
it. A receiver-only change at the pointer's exact bytes improves nothing by construction — its whole
purpose is decode cost and reviewability — so it has **no expressible admit bar**. The contract's own
`completed_t4_receiver_delta` arithmetic points the same way: it projects
`source_t4 × (1 + max(0, ceiling/base − 1))` from cold-n600 `candidate_decode_wall_clock` diagnostics
whose retained verdict is `REFUSED`, a one-sided clamp with **no field for a ratio below 1**. Feeding it
this arm's measured 0.932440 yields **1,185.899645 s unchanged** — the instrument's actual finding, that
the rewrite is 6.8% cheaper, is not representable. **verdict_scope: this is a statement about the frozen
`candidate_prefire_intent.v1` / `candidate_prefire_timing_risk.v1` shapes as of commit `a47543199` plus
its landed amendments, on a receiver-only candidate at identical archive bytes. It does not say the
contract is wrong for score-moving candidates, which is what every prior intent has been.**

### The downstream controls, and what was and was not reached

- **`validate_seal` REFUSES an intent (MEASURED).** No mrs5 intent exists, so I demonstrated the control
  on the one real committed intent in the repo, rlc5's `CANDIDATE_PREFIRE_INTENT_v4.json`:
  `validate_seal(...)` → verdict **`PREFIRE_INTENT_SCHEMA_REFUSED`**, problems
  `('candidate_prefire_intent.v1 is not a completed seal',)`. The control is live.
- **The `--seal` path (DERIVED from source, deliberately NOT executed).**
  `tools/fire_modal_auth_eval.py` calls that same `validate_seal` at line 943, and on a non-ok verdict
  takes `refuse_seal(..., rc=7, "seal invalid: PREFIRE_INTENT_SCHEMA_REFUSED")` at line 959 — before any
  staged-tree mutation or Modal call — with line 396 stamping `code=PREFIRE_INTENT_SCHEMA_REFUSED,
  score_claim=False, promotion_eligible=False` into a receipt written beside the seal. I did not run the
  fire tool: "no Modal, no fire" is a boundary, and a refusal that happens before dispatch is still a
  dispatch-tool invocation.
- **Authorization, first measurement, harvest, completion: NOT REACHED and not attempted.** They are
  MAIN's, and they have no valid intent to consume.

## 6. Artifacts

| artifact | path | bytes | sha256 |
|---|---|---:|---|
| timing-risk receipt | `.omx/research/ddm_mrs5_20260916/TIMING_RISK.json` | 8,013 | `65def7311f524be7547b1252657875eb7951448195b457c8ba51ee8f03c160c2` |
| retention manifest | `.omx/research/ddm_mrs5_20260916/RETENTION_MANIFEST.json` | 1,379 | `89c225e18c58988f1a4c1fc533deca0baef503dd1d196dd957ba9b42cdcad270` |
| producer refusal | `.omx/research/ddm_mrs5_20260916/prefire_attempt/PREFIRE_REFUSAL_43ca2b00aebd4c4296d8356283327f0c.json` | 324 | `8df50f1c0ee52069bf3eafef2060b7d63d0d0da68a07571e9ad80ee30986b346` |
| interleaved A/B/A/B | `/Volumes/APDataStore/pact/ddm_mrs5/timing_interleaved48/RESULT.json` | — | (run rows retained per-run under `sequence_0001/`) |
| move 53 t4_direct leg | `/Volumes/APDataStore/pact/ddm_pd6/SEAL_ddm_pd6_price_first_generator_contest_cuda.json.decode_wall_clock.json` | 1,443 | `97373a987b4208860ef8b596c8d0417942d340f82640bfa1932031c093f4c8b4` |

`TIMING_RISK.json` carries the full lineage the charter asked for — the leg reference with `{path, bytes,
sha256}`, the leg's measured 1,185.899645171 s and 1,260 s limit, the sealed receiver's risk digest
`9f6e7168…`, the seven-file delta with a sha256 each, the four-run interleave with its load readings, and
both projections (mine, 1,105.7798 s, and the contract's clamped 1,185.899645 s). It records
`contract_validated: false` with the exact typed refusal in `contract_conformance.root_control`. **It is
an evidence object for MAIN's adjudication; it is not a passing gate and must never be read as one.**

## 7. What MAIN now owns

1. **Re-pin the drifted consumer.** Land a `prefire_contract_consumer_fix.v1` amendment row that brings
   `tools/modal_harvest_poller.py` up to `da66353b7`. Until then the intent door is shut for everyone.
2. **Adjudicate the unpinned minimal packet.** Either the packet gains the two top-level archive pins
   (which changes the seven files MAIN accepted, and makes `check_pin_consistency` non-vacuous), or the
   contract gains an explicit pinless-receiver identity class. I did neither.
3. **Adjudicate the zero-delta admit bar.** A receiver-only candidate at the pointer's exact bytes has
   `derived_net_dS = 0` and no expressible bar. Either the intent object grows a receiver-only mode, or
   receiver changes reach T4 by some other authorized door.
4. Then, and only then: land the source, commit a valid intent, authorize, measure on T4, harvest,
   complete. **That ordering is the contract's, not mine, and this arm performed none of those steps.**

## 8. Prior negatives accounted

PR #140 (seven files / ~4.5k LOC is the measured minimum for this object — said in the README in one
sentence). mrs2/mrs3/mrs4 (host-ratio projections are not evidence — hence the *interleaved same-host*
ratio here, and hence the T4 measurement still being owed). tc4/pr19 (a receiver change needs its own
measured leg — §5 Gate 2 is why the existing door does not open for it). rlc2/ffi1 (contract refusals
are real controls — three of them are recorded above and none was patched around). The ExFAT venv (all
scratch venvs on APFS). pd6's rider-drop law.

---

**composition S 0.1361014714463198 @ 179,286 B [contest-CUDA T4 n600] (move 53) — unchanged by this arm.**
