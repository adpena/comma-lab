# ddm_sp2 — two defect classes, fixed AND self-protected (2026-08-19)

Axis: `[macOS-CPU advisory]` for every measurement below. Nothing here is a score claim, and
the exact pointer is unmoved by this unit — this is apparatus, per the means/ends firewall.

Operator directive: *"Anything that needs to be fixed and self protected against?"* Two classes
had been fixed at a single site each and left without CLASS protection, which the CLAUDE.md
"Bugs must be permanently fixed AND self-protected against" non-negotiable forbids: a fix at one
surface leaves the class active at the others.

Both guards land **WARN-ONLY**, decided by measurement, not by preference — see the strict-flip
conditions in §4. Neither guard claims a catalog number. Both are **scope extensions of an
existing gate**, following the 2026-07-20 precedent recorded in CLAUDE.md's "2026-07-14 catalog
amendments" section: *"This is a Catalog #351 scope extension, not a new gate or number, per the
post-#400 Catalog #299 consolidation rule."* The #299 quota brake stands at **407/400**.

---

## 1. The charter's census was wrong in both directions — corrected here

The charter named 4 unpinned ZIP sites, grep-derived from the literal tokens `create_system` /
`external_attr`. Measured, that census **over-counted by 2 and under-counted by 7**:

| charter said | measured | why the grep was wrong |
|---|---|---|
| `archive_optimizer.py` unpinned | **already pinned** | routes through `write_deterministic_zip_member` — the canonical helper pins on the caller's behalf, so the tokens never appear |
| `pr79_segaction_payload.py` unpinned | **already pinned** | same helper route |
| `archive_codec.py` unpinned | confirmed | bare `writestr(str, data)` |
| `entropy_archive.py` unpinned | confirmed | bare `writestr(str, data)` |
| — | **+ 7 more** | `archive_diet`, `hnerv_lowlevel_packer`, `mask_prior`, `optimal_stack_orchestrator`, `pr95_hnerv`, `stack_compositions`, + 2 `witness_dsl/ep725_*` |

And the dominant real defect was not "no pin" at all — it was the **HALF pin**: a file that sets
`create_system` but not `external_attr` (or the reverse). That is exactly the half that broke the
jg2 seal. A token grep scores a half-pinned file as clean.

Run under the gate's own recursive `src/tac/**` scope the true population was **40 sites**, not 4.
This is the "negative-existence claims are the #1 false-claim class" law: the charter's list was a
grep result presented as a census.

---

## 2. CLASS 1 — environment-sensitive ZIP metadata

### The mechanism, measured on this interpreter (CPython 3.13.12 / darwin)

`ddm_jg2` §S1i re-packed the shipped tail to **exactly** the right length — 176,420 B, matching the
pointer — and the sha256 still MISMATCHED, on 3 central-directory bytes. Verbatim: *"LENGTH-EQUAL
AND BYTE-EQUAL ARE DIFFERENT TESTS."* The cause, re-derived from source rather than assumed:

- `ZipInfo.create_system` defaults to the **platform**: 3 on POSIX, 0 on win32.
- `ZipFile._open_to_write` contains `if not zinfo.external_attr: zinfo.external_attr = 0o600 << 16`.
  Leaving `external_attr` unset does **not** emit 0 — it emits an *interpreter* default.
- A bare `writestr(name, data)` additionally stamps the member with `time.localtime()`.

**Counter-control (the detector is not vacuous).** Two archives differing only in `external_attr`:
**208 B vs 208 B — equal length, 2 differing bytes.** The jg2 signature reproduced from first
principles. A size check can never see it; a seal always breaks on it.

### The fix: pin what the builder ALREADY emits

The repo has no single canonical mode — `0o600`, `0o644`, and `0o100644` all appear legitimately.
So harmonising modes would have **changed shipped bytes**. Each site was instead pinned to the value
it was already emitting, which is byte-identical on this platform and immune to platform drift.
Proven, not asserted (`test_pinning_the_already_emitted_value_is_byte_identical`):

```
add create_system=3   (POSIX): sha 3d377901bc73846e -> 3d377901bc73846e   IDENTICAL
add external_attr=0o600<<16 : sha 85855e049fdf7f82 -> 85855e049fdf7f82   IDENTICAL
```

| # | file | was | pin added | byte effect |
|---|---|---|---|---|
| 1 | `src/tac/archive_diet.py` | `create_system` only | `external_attr = 0o600 << 16` | none |
| 2 | `src/tac/stack_compositions.py` | `create_system` only | `external_attr = 0o600 << 16` | none |
| 3 | `src/tac/hnerv_lowlevel_packer.py` | `external_attr` only | `create_system = 3` | none on POSIX; fixes win32 |
| 4 | `src/tac/pr95_hnerv.py` | `external_attr` only | `create_system = 3` | none on POSIX; fixes win32 |
| 5 | `src/tac/optimal_stack_orchestrator.py` | `external_attr` only | `create_system = 3` | none on POSIX; fixes win32 |
| 6 | `src/tac/mask_prior.py` | bare `writestr` into `mode="a"` | full pin via `_deterministic_prior_member` | **removes build wall-clock** from a live archive |
| 7 | `src/tac/archive_codec.py` | bare `writestr` ×5 | full pin via `_write_archive_member` | removes build wall-clock |
| 8 | `src/tac/entropy_archive.py` | bare `writestr` ×10 | full pin via `_write_archive_member` | removes build wall-clock |

Sites 6–8 were **already non-deterministic** — they stamped the build wall-clock into the member —
so there were no stable bytes to preserve. `mask_prior.save_prior_to_archive` appends into a real
`archive.zip`, so that one was a live latent defect, not a dormant one.

### The guard

`_scan_python_for_env_sensitive_zip_metadata`, wired into
`check_archive_builders_use_deterministic_zip` (Check E — "archive builders produce byte-identical
zips"; the metadata axis is the same invariant on a different field). Scope `src/tac/**` — the
shipping library, the builders whose bytes can reach an `archive.zip` under seal.

A site is CLEAR when the file (a) literally pins **both** fields, (b) routes through a canonical
deterministic-zip helper, or (c) carries a same-line `# ZIP_METADATA_ENV_OK:<rationale>` waiver.
Clearance (b) is what makes the guard agree with reality where the charter's grep did not.

**What the gauge reads if the cure is applied and nothing else changes.** It does *not* count
"files mentioning a pin" — that reaches 100% by adding comments, measuring instrumentation rather
than reality. It counts write-mode `ZipFile` sites whose emitted metadata is chosen by the
environment. Adding a declaration without pinning moves it by zero.

### Live count

| state | sites |
|---|---|
| before (pre-fix, pre-waiver) | **40** |
| after the 8-file fix | 32 |
| after 2 adjudicated waivers | **30** |

Waivers granted (both genuinely lineage-correct, not conveniences):

| file | rationale |
|---|---|
| `src/tac/witness_dsl/ep725_lossless_xcodec_recode.py:315` | metadata is `copy.copy`'d from the SOURCE archive's own `ZipInfo`; pinning would overwrite the framing this lossless recode exists to reproduce |
| `src/tac/witness_dsl/ep725_population_global_recode_v2.py:856` | same clone-from-source pattern |

The 30 remaining are honest debt across `packet_compiler/`, `substrates/`, `optimization/`,
`deploy/`, `analysis/`, `v2_compose/`, `local_acceleration/`, `residual_basis/`. They are listed in
full by the gate's return value.

### Positive control (the detector is not tuned to its own cure)

Two controls, both tests:

- The **real `experiments/ddm_jg2_tail_reencode.py`** — the file whose fix started this — scans
  **CLEAR**. The detector agrees with the known cure rather than being satisfied by a marker.
- A reconstruction of jg2's **pre-fix shape** (fixed `date_time`, no `create_system`, no
  `external_attr`) is **caught**, naming both missing fields. A fixed timestamp alone was never
  enough; that is exactly the state that produced the right length with the wrong sha.

### Scope ladder, measured — and the honest gap

jg2 itself lives in `experiments/`, i.e. **outside** the guard's `src/tac/**` scope. The guard would
therefore not have caught jg2 at its own site. Widening was measured, not assumed:

| scope | findings | scan cost |
|---|---|---|
| `src/tac` (**shipped**) | **30** | 0.79 s |
| `+ tools` | 84 | 1.80 s |
| `+ experiments, scripts, comma_lab` | 180 | 2.84 s |

180 warn-only rows is a gauge nobody reads. The library scope is where bytes reach a sealed
`archive.zip`, so that is where the guard binds. **Widening condition:** `src/tac/**` reaches
live-count 0, then extend to `tools/`, then to `experiments/`.

---

## 3. CLASS 2 — GT decode-lineage objective custody

### The class

`ddm_pi2` measured one advisory instrument reading **two** ground truths: seg from a DALI/nvdec
argmax cache, pose decoded fresh with PyAV. That split *was* the entire 21.4× advisory-vs-CUDA pose
offset the campaign had been charging to hardware drift. A published dither row was 11.4× optimistic.
Fixed at the sites 2026-08-19 (`809199d24f`, qs1 + mt1 repointed to the DALI table) — but nothing
stopped the next tool from loading the PyAV table as its objective. `tac.gt_lineage` is the
content-addressed authority; `cw1` measured that almost nobody consumes it.

### The precision that carries the guard

`ddm_up2` defines its **own local** `verify_gt_lineage()` that resolves lineage by *filename
substring* — the #936 adoption-decay pattern. A clearance test on a bare `gt_lineage` substring
would clear the very file the guard exists to surface. The route test is therefore **module-exact**
(`tac.gt_lineage`), and `test_a_local_verify_gt_lineage_does_NOT_clear_the_file` pins that.

Two further precision decisions, both measured:

- **Comment-only mentions are not consumption.** `ddm_mt1_*` names the PyAV artifact in a comment
  directly above a DALI load. Flagging that is a false positive; the scanner ignores literals that
  occur only after a `#`. This removed 2 of 18 rows.
- **`gt_argmax*.npy` is deliberately OUT of scope.** The charter listed it. Measured: including it
  takes the flagged population from **18 files / 37 sites to 93 files / 153 sites**, for an artifact
  whose lineage pi2 + gl1 already established as a single DALI lineage. A 93-row warn-only gauge is
  one nobody reads — the flood failure, sister of vacuity-equals-pass. **Widening condition:** a
  second seg-argmax lineage is observed, OR the pose scope reaches live-count 0.

### Live count

| state | sites |
|---|---|
| before waivers | **16** |
| after 5 adjudicated waivers | **11** |

Waivers granted — each verified at the source before granting, not taken on the charter's word:

| file:line | rationale |
|---|---|
| `ddm_po1_t4_error_feedback_pose_compensation.py:590` | round-local custody: the T4 round's OWN emitted GT, read from that round's `retained/pose_vectors/` beside the candidate vectors it is differenced against — GT and candidate share one decode by construction (the charter's canonical example) |
| `ddm_qs1_frame0_schur_coupled_solve.py:69` | the PyAV table is bound under an explicitly lineage-labeled name and is NOT the objective; `GT_POSE` on the line above is the DALI table (the `809199d24f` cure) |
| `ddm_pi2_pose_axis_attribution.py:760` | pi2 is the instrument that MEASURED the split; it binds both caches to difference them |
| `ddm_sg2_pr130_seg_axis_source_audit.py:50` | a seg-axis source audit binding both lineages side by side to compare them |
| `modal_dali_av_gt_cache_diff.py:199` | the #906 PRODUCER that builds both caches in order to diff them |

### Named warn-only debt (NOT fixed here, by charter)

- **`experiments/ddm_jg1_seg_solve.py:89`** — loads `gt_cache_av.pt` (PyAV lineage) and clears its
  own lineage through up2's filename-substring resolver, not the registry. **A live arm (`ddm_jg3`)
  is consuming this file**, so it is left untouched and routed to the jg3 boundary. This row is the
  cw1 adoption-decay finding made mechanical.
- **10 × `gt_cache_600_official_ada.pt`** (`ddm_dt1`, `ddm_hp3`, `ddm_pk2`, `ddm_pz2`, `ddm_rg1b`,
  `tools/build_mx2_pose_adapter_caches`, `tools/fit_ddm_cl1_hpac_capacity`,
  `tools/run_ddm_ec2_sparse_event_hpac_conditioning`, `tools/run_ddm_xi1_*`, `tools/run_ddm_xi2_*`).
  I did **not** adjudicate this cache's lineage and will not guess one. Per `tac.gt_lineage`'s own
  evidence ladder, an honest UNKNOWN outranks a guessed label. Adjudicating it is the next rung.

---

## 3b. CLASS 3 (FOLDED) — fp16 cast destroys its own floor, `ddm_fx4`'s owed gate

Routed by the coordinator while this arm owned `preflight.py`. `ddm_fx4` (commit `61c41ab166`)
fixed the class at **35 sites / 22 files** and explicitly recorded the gate as *"Owed, not done:
the STRICT preflight wire-in… `src/tac/preflight.py` is owned by `ddm_sp2` this session, so the
gate lives in the test suite only."* Taken, because a cure without a gate is exactly the half-landing
the non-negotiable forbids.

**The class.** A positive floor on a scale, destroyed by a narrowing cast one line later:
`((maxs - mins).float() / 255.0).clamp(min=1e-8).to(torch.float16)`. fp16's smallest positive value
is `2**-24 = 5.960464e-08`; everything below half of that rounds to zero, so the `clamp` written to
keep the scale non-zero is undone by the very next operation and the dequantiser divides by the
stored 0. The class also spans two statements, which a same-statement detector would half-miss.

**Host: `check_quantize_degenerate_range_clamped_correctly` (Catalog #161).** Exact adjacency —
#161 owns *"the degenerate-value guard must be correct"*; this extension owns the sister half,
*"…and must SURVIVE to the stored value."* Same #351 precedent, no new number.

**ONE detector, not two.** fx4's predicate lived inside its own test sweep. Re-typing it in preflight
would have created two detectors for one class, which drift — the split-bank failure this repo has
already paid for. Instead the predicate was extracted to **`src/tac/fp16_floor_guard.py`**, and BOTH
consumers now import it: fx4's regression sweep (refactored, 29/29 still pass) and the preflight gate.

**Landed STRICT, not warn-only.** Live count measured **0** (fx4 having fixed all 35 sites), which is
precisely the condition the atomicity rule names for flipping in the same batch. This class earns
strict where my two did not — the discipline working, not an exception to it.

| | value |
|---|---|
| live violations | **0** |
| guard-and-cast sites scanned (denominator, reported every run) | **36** |
| negative control | reintroducing either variant in a temp tree is caught; strict gate refuses |

**Precision improvements the shared module adds, and how they were found.** Extraction immediately
exposed defects that the test-only sweep had never had to face, each found by measurement:

1. **Prose read as code.** This module's own docstring, which quotes the pre-fix expression, tripped
   the raw-text predicate — 2 phantom rows, denominator inflated 40 → 36. Exactly the blindness
   `ddm_fx3` named in Catalog #330.
2. **My first cure was itself blind.** I stripped comments with `line.find("#")`. A `#` inside a
   string literal (`"#fff"`) both truncates the line and unbalances its brackets, so
   `_logical_statements` returns NOTHING for the rest of the file — the file then scans clean AND
   contributes zero to the denominator, so even the vacuity guard cannot see it. An instrument that
   cannot tell "checked clean" from "did not check". Replaced with the real `tokenize` pass, which is
   the cure fx3 actually prescribed.
3. **The canonical cure was reported as a violation.** The two-statement form —
   `sf = torch.tensor([scale], dtype=torch.float16)` then `sf = sf.clamp_min(_FP16_MIN_POSITIVE)` —
   is how the fix is naturally written, and a same-statement-only detector calls it a defect. On a
   STRICT gate that refuses an engineer's commit and tells them to do what they have already done,
   which is how a gate earns a waiver it should never have needed. Cured by
   `_cured_in_a_later_statement`, with a substantive-rationale `FP16_POSTCAST_FLOOR_OK` waiver
   (Catalog #287 placeholder rejection) as the last resort rather than the first.

**Wall-clock, measured and fixed twice.** The first working version cost **9.67 s** — 32% of the
30.0 s `DEFAULT_PREFLIGHT_CLI_TIMEOUT_S`, inside a STRICT gate. A ripgrep prefilter plus a two-stage
design (cheap raw pass; pay for tokenization only where a guard-and-cast site actually exists) brought
it to **1.57 s**, with the answer unchanged at 0 violations / 36 sites.

---

## 3c. `ddm_fx3` detector debt — RECORDED, not folded (with the cure, verbatim)

Both items are precision improvements to **Catalog #330**, which is currently GREEN (0 offenders)
after fx3's fix. Neither composes with the ZIP / GT-lineage / fp16 invariants, so folding them would
have been scope creep against a green gate rather than protection of a live class. Recorded with
their cures so the next `preflight.py` owner lands them verbatim.

1. **`_check_330_line_is_comment_or_literal` is not string-literal aware**
   (`src/tac/preflight.py`, fx3 cites `:62234`). It is a line-level lexical guess: it skips a line
   that *starts* with a quote, so a docstring CONTINUATION line beginning with `print(` reads as
   code. Measured consequence: the detector's own two hits on
   `experiments/modal_ot_offset_n600_gate.py` **disagreed with each other** — line 216 (inside an
   f-string, starts with `f"`) correctly skipped, line 33 (docstring continuation) falsely flagged.
   **Cure, now available verbatim:** `tac.fp16_floor_guard._blank_docstrings_and_comments` implements
   exactly the prescribed fix (walk the AST, blank every bare-string `Expr` span, preserve line
   numbers, strip trailing comments, fall back to raw text on parse failure). Lift it or import it.
   Ladder: WRONG-OBJECT (raw text treated as code), sister SILENT-INSTRUMENT.

2. **The detection window is asymmetric, `-20/+180` lines** (fx3 cites `src/tac/preflight.py:62289`).
   A compliant file whose mirroring helper is defined *above* its `.get()` still reads as a
   violation; fx3 hit this directly and worked around it by MOVING `_mirror_terminal_call_state`
   below `_live_call_state`, which it flags as a workaround, not the cure. **Cure:** widen the
   backward window, or resolve the enclosing function's call graph. **Caution I add:** naively
   widening the backward window trades a false positive for a false NEGATIVE — an unrelated
   `append_terminal_call_id_ledger_event(` far above would falsely clear a real offender. The
   call-graph resolution is the sound half; the window widening is not. Do not land the cheap one.

fx3's other named debt is not preflight-surface and stays with its owners:
`tools/codex_companion_spawn.sh:96-100` (`rc=unknown_detached`, blocked on the codex lane being
walled until Aug 20), and the absent shared waiter helper.

---

## 4. Strict-flip conditions

Both extensions are WARN-ONLY, and both host gates run `strict=True` in `preflight_all`. The
extensions are therefore **excluded from the strict raise path** — their findings are returned to
the caller but can never raise. `test_zip_metadata_extension_is_warn_only_not_raising` and
`test_gt_lineage_extension_is_warn_only_not_raising` pin exactly that, the second in the stronger
form (no lineage row may ever appear in a raise message).

The CLAUDE.md "Strict-flip atomicity rule" says flip in the same batch when live count is 0. It is
not 0 — 30 and 11 — so the rule does not license a flip here. Conditions:

CLASS 3 is **already STRICT** (live count 0 at landing), so the table below covers only the two
warn-only extensions.

| extension | strict-flip when |
|---|---|
| CLASS 1 ZIP metadata | live count reaches 0 over `src/tac/**` — i.e. the remaining 30 sites are each pinned-to-emitted-value (verify each with a before/after sha) or waived. Flip the `metadata_warnings` list into the `MetaBugViolation` raise in `check_archive_builders_use_deterministic_zip`. |
| CLASS 2 GT lineage | live count reaches 0 — i.e. `gt_cache_600_official_ada.pt`'s lineage is adjudicated and declared (10 sites), and jg1 is repointed or waived at the jg3 boundary (1 site). Flip `gt_lineage_warnings` into the `PreflightError` in `check_evidence_authority_claims_are_custodied`. |

Landing warn-only was decided by the measurement, not by caution: a strict gate wired into
`preflight_all` blocks every commit repo-wide, and a live arm is committing.

---

## 5. Verification performed

- `41/41` new tests pass (`src/tac/tests/test_ddm_sp2_two_landing_protections.py`) — positive,
  negative, waiver-accept, waiver-placeholder-reject, waiver-too-short-reject, waiver-wrong-line,
  edge (syntax error, missing file, read-mode, no-ZipFile), scope-exclusion, and two live-population
  regressions that fail if either fix rots out.
- `500/500` pass across the host gates' existing suites (`test_v9_provenance_gates`,
  `test_check_351_canonical_producer_identity_scope_extension`, `test_preflight_meta_bugs`,
  `test_build_cross_archive_substrate_composition`, `test_remote_lane_f_v4_script`).
- `48` pass across the 8 edited modules' own suites.
- `ruff check --select F` clean on every edited file. The 2 pre-existing `F841`s in
  `archive_codec` / `entropy_archive` were confirmed identical on HEAD by stash-and-recheck.
- 9 failures in `test_hnerv_lowlevel_exact_eval_packet.py` were confirmed **pre-existing on HEAD**
  (identical count, stash-and-recheck) — not caused by this landing.

## 6. Adjacent finding, NOT mine to fix

`check_evidence_authority_claims_are_custodied(strict=True)` is **already red on HEAD**, on a
Catalog #344 anchor-roundtrip defect:
`.omx/state/canonical_equations_registry.jsonl:750` — `equation=realization_necessity_preimage_per_stratum_v1`
`anchor=3 is not JSON-roundtrip exact at $[3].vs1_rescope_reason, $[3].vs1_rescope_utc`.
Confirmed pre-existing by stash-and-recheck against HEAD. That registry file is dirty in the working
tree and another arm owns it. Flagging, not touching.

Out-of-scope sibling measured and left alone: `src/comma_lab/tracks/exact_current.py:9` is an
unpinned write-mode `ZipFile`, but it writes a literal `"placeholder"` README and sits outside the
`src/tac/**` guard scope.
