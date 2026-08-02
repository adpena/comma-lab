---
schema: ddm_pb2_bracket_direction_and_decode_adapter.v1
date_utc: 2026-08-02
arm: ddm_pb2
lane_id: "lane_ddm_pb2_20260802"
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false
axis: "[macOS-CPU frozen-PoseNet advisory] for row 1; [source + staged-decode-path execution] for row 2. No contest-hardware row; the exact pointer did not move."
verdict_scope: INSTANCE
council_predicted_mission_contribution: frontier_breaking
consumes:
  - .omx/research/ddm_lg2_arity_mismatch_three_rows_20260802.md (§4 pre-registration)
  - .omx/research/ddm_lg2_binary_inventory_20260802.md (the named gap)
  - .omx/research/ddm_pw1_pose_menu_saturation_20260801.md (the vehicle + receipt)
  - /Volumes/VertigoDataTier/pact/ddm_v4d_20260731/pw1/pw1_arms.jsonl (600 probe receipts)
  - experiments/ddm_v4d_resolve.py (the live bracket, :215-221 and :314-319)
  - src/tac/optimization/repair_entropy_coder_runtime_adapters.py (row 2 subject)
  - experiments/ddm_r7_token_coder.py, experiments/stage_wr1_realized_gate.sh
consumers: [MAIN, "#871", ddm_pw1 successor, the #827 post-burn pose re-solve]
tokens: [no-triality, p0-ledger-ok]
---

# ddm_pb2 — two rows: the bracket's `break`, and the unread decode-path adapter

## §0 POINTER HONESTY FIRST

**The exact frontier did NOT move.** `0.1910828242 [contest-CPU]` unchanged. The own-vehicle
advisory line `pw1 0.9476091` is unchanged by this unit — nothing here has been through the
realized gate. Every row-1 number is `[macOS-CPU frozen-PoseNet advisory]`, `score_claim=false`.

Relative-significance denominator, stated once and used throughout: gap to the bar is
`0.9476091 − 0.172141 = 0.7754681`, so **1% of gap = 0.0077547 S**. Every ΔS below is quoted
against the **LIVE BEST (pw1 arm-AB, 0.9476091)**, never against a retired reference.

## §1 STATE CORRECTIONS — my charter was wrong on two mechanical details

Recorded first, because characterizing code instead of reading it is the day's named trap and
my charter contained two instances of it.

| charter claim | re-derived at source |
|---|---|
| "`experiments/repair_entropy_coder_runtime_adapters.py` is staged into the decode path" | **That path does not exist.** The file is `src/tac/optimization/repair_entropy_coder_runtime_adapters.py`. What is staged is a *de-taced copy* that the pfs1 D1 build produced (`ddm_pfs1_d1_build_receipt_20260729.json:39-44`); the gate scripts copy it out of the pfs1 template dir, not out of `experiments/`. |
| "copied by `experiments/stage_wr1_realized_gate.sh:44-48`" | `:44-48` is the `case "$CAND"` archive-selection block. The copy loop is **`:59-63`**. |
| "#900" | **absent from `.omx/state/canonical_task_status.jsonl`.** lg2's pre-registration is registered under **#871**; this memo's ledger note lands there rather than inventing a new id. |

Neither correction changes the substance of either row — both rows were real. They are recorded
because a wrong pointer is how a later agent fails to find the artifact.

## §2 ROW 1 (#871 / "#900") — the pre-registered bracket-direction measurement

### 2.1 The defect, verified at source and reproduced from the receipt

`experiments/ddm_v4d_resolve.py:215-221` (dim0) and `:314-319` (beta) both run:

```python
step, direction = DIM0_STEP0, 0.0
for sign in (1.0, -1.0):
    d, xq = eval_dim0(best_x + sign * step)
    if d < best_d:
        best_d, best_x, direction = d, xq, sign
        break            # <- -1.0 is never evaluated when +1.0 improves at all
```

`tools/pb2_bracket_direction_ab.py --verify-input` reproduces lg2's §4 table **exactly** from
the shipped receipt, at zero compute:

| quantity | lg2 | pb2 re-derivation |
|---|---:|---:|
| arm A short-circuits | 94 | **94** |
| arm B short-circuits | 31 | **31** |
| union pairs | 109 | **109** |
| arm-instances | 125 | **125** |
| of which a *genuine* break (first probe improved) | — | **125 / 125** |
| union share of the live arm-AB d_pose mass | 15.89% | **15.886%** |
| mean d_pose on the union vs population | 0.006685 / 0.007645 | **0.006685 / 0.007645** |

The mass basis is `arm_ab_d` — the **live** pw1 solution at S=0.9476091, not the pre-pw1 row.

### 2.2 The instrument

`tools/pb2_bracket_direction_ab.py` runs **both** bracket variants against **one memoized
evaluator per (pair, arm)**, so the delta is measured inside a single instrument and carries no
cross-run floor, and running both costs barely more than running one.

- **POSITIVE CONTROL:** the asymmetric variant is pw1's semantics verbatim and must reproduce
  the shipped `arm_a_d` / `arm_b_d` / `arm_ab_d`. **MEASURED max abs err = 0.0** (bit-identical),
  and the pw1 canary (`d_ctrl` vs the receipt) is also **0.0**.
- **Tie-breaking:** `+` is scanned first against a running best seeded at `d0`, so `+` wins exact
  ties and any measured delta is a **strict** win for `−`.
- **VERDICT CLEARANCE (L3):** the positive control *gates* the falsifier — a failed control
  returns `INSTRUMENT_UNTRUSTED` and no verdict is admissible.
- **Both denominators required:** lg2's rule ("Δd_pose ≥ −1e-6 over the 109") is ambiguous
  between a per-scope-pair mean and the population mean that enters the score; they differ by
  109/600 and the population reading is the *easier* null. Both must agree, so the convenient
  denominator cannot be chosen after seeing the data.
- Empty scope emits `VACUOUS`, never `PASS`; every count reports its denominator.

19 tests in `src/tac/tests/test_ddm_pb2_bracket_direction.py`, each with a positive control.

### 2.3 RESULT

<!-- ROW1_RESULT -->
n600 complete, 600/600 pairs, 4,832 scorer forwards. **Both positive controls returned exactly
`0.0`** — the asym replay is bit-identical to pw1's stored arms, and the composed asym mean
lands on pw1's own receipt value `0.007645062472871804` to the last digit. The instrument is
trusted, so a verdict is admissible.

**FALSIFIER: `ASYMMETRY_PRICED`** — both required denominators agree it is past threshold:

| reading | measured | lg2 threshold |
|---|---:|---:|
| Δd_pose, scope population mean | **−2.580e-06** | −1e-06 |
| Δd_pose, per-scope-pair | **−1.420e-05** | −1e-06 |

So the row does **not** close at FORMULATION. But the size is the finding, and it is small:

### The deflation — 93% of the untested commitments were inconsequential

Of the **109** pairs that committed to `+` untested, the outcome actually changed on **8**.
**101 of 109 (93%) were inconsequential**: the `−` probe, had it run, would not have altered the
result. And of those 8, **2 pairs carry 96% of the whole effect** (pair 279 at −1.07e-03 and
pair 75 at −5.16e-04; the remaining six sum to ≈ −7.9e-06 net).

### NEITHER ENTRY RULE DOMINATES — the framing correction this row forces

**MEASURED: `+` wins 2 of the 8, `−` wins 6.** The entry probe is a **one-step-lookahead
greedy**, so the direction that *loses* the entry probe can still *win* the doubling
continuation — pair 326 takes `g = −1.5` on a strictly better entry probe and ends **4.58e-05
worse** than asym's `g = +1.5`.

This matters for what the cure is. "Drop the `break`" is **not** automatically an improvement;
both variants are monotone-safe against their common starting point, but neither dominates the
other. The achievable bound is the per-pair **min of both continuations** — what a bracket that
expands *both* directions would reach — and that is what was emitted and byte-closed. A guard
for this is in the test suite (`test_the_entry_rule_does_not_dominate`) so no successor asserts
the false ordering.

Arm A direction flips: **6**; arm B: **0**. The four arm-AB divergences all fell **inside**
lg2's 109 (0 outside) — so lg2's scope, though structurally blind to the shipping arm's probes
(pw1 discarded them), was not an undercount in practice here.

### BYTE-CLOSED ROW — the bytes eat 22% of it

Not reported as a d_pose delta with an assumed rate: pushed through the real builder and the
real receiver.

| | pw1 (LIVE BEST) | pb2 best-of-both |
|---|---:|---:|
| archive bytes | 360,323 | **360,339** (+16) |
| archive sha256 | `0ef9ff7129…` | `6e1b80e901…` |
| 100·d_seg (frame_1 untouched) | 0.4311790 | 0.4311790 |
| √(10·d_pose) | 0.2764971 | **0.2764490** (−4.8103e-05) |
| rate term | 0.2399243 | **0.2399349** (+1.0654e-05) |
| **composed S** | **0.9476004** | **0.9475629** |

**Net composed ΔS = −3.745e−05.** The +16 bytes (the beta section growing 151 → 161 B as the
symbol alphabet widened, plus 3 B of tp) cost 1.07e-05 S and consumed **22.1%** of the pose
gain. Against the gap to the bar (0.7754681) this is **0.0048% of the gap**.

**Parse-back (#417), all four checks pass on the extended archive:** pose reconstruct exact,
(a,b) bit-exact, selector exact, **beta exact**; an independent compose recompute is byte-exact
on 7 sampled pairs with the beta path exercised; rebuild sha stable.

**Regression guard, measured not asserted:** rebuilding from pw1's *own* `final_pw1.jsonl`
reproduces the live archive **byte-identically** — 360,323 B, sha
`0ef9ff7129461f7318f8e8cec8f6579ba651425292f63b31cc9f4f41a8c6963a` — so the builder path used
for the pb2 row is verified against a known answer before it was trusted with a new one.

### Cost — the pre-registered estimate was low, and why

| | forwards |
|---|---:|
| pw1, asymmetric only (its own receipt) | 4,644 |
| pb2, both variants on one memo | **4,832** |
| marginal | **+188** |
| lg2's pre-registered estimate | +125 |

The estimate counted one extra *entry* probe per short-circuited arm-instance (94 + 31). The
realised figure is higher because the shipping arm-AB bracket has short-circuits lg2's scope
could not see, and because a flipped entry sends the continuation down a different chain.
**Named gap:** pb2's emitted rows discard their per-probe traces, so the cost of the shipped
change alone cannot be separated from the cost of replaying both variants — this +188 is an
UPPER BOUND. That is the same "produced and discarded" defect this unit flags in pw1's arm-AB
probes, in my own instrument; it is named rather than hidden.

### VERDICT

**`ASYMMETRY_PRICED`, and the price is ≈ 0.005% of the gap.** The defect lg2 identified is real,
reproduces exactly, and is worth −3.745e-05 S byte-closed at essentially no rate. It is **not a
lever** — 93% of the untested commitments were inconsequential and 2 pairs carry 96% of the
effect. It is a small, permanent, rate-nearly-free correctness improvement that compounds into
every future pose re-solve (including the #827 post-burn re-solve, which would otherwise pay it
again).

**NOT FIRED.** `experiments/stage_v4d_realized_gate.sh:3` reserves the n600 gate for MAIN, one
candidate at a time, and I respected it. The archive is staged and the command is:

```
bash experiments/stage_v4d_realized_gate.sh cpu pb2_bestof
```

Honest cost/benefit for that decision: ~11 min of n600 scorer time to confirm a **−3.745e-05**
move. pw1's gate prediction residual on this exact vehicle was 1.82e-06, so the prediction sits
~20× above the residual — measurable, but marginal. **My recommendation is to NOT spend the
slot on this alone**, and instead fold the best-of-both solution into the next pose re-solve
that fires for a larger reason. The candidate is banked either way.

## §3 ROW 2 — the unread decode-path adapter, read end to end

`src/tac/optimization/repair_entropy_coder_runtime_adapters.py` (285 lines) provides encode +
decode for two prototype packet formats: `TACRNG1\0` (a stdlib-LZMA codestream behind a
sha256-proof header) and `TACANS1\0` (a real rANS coder with an explicit frequency table).
Every claim below is from **reading or executing** it, never from reasoning about it.

### 3.1 Is it consumed? — HARD at import, NEVER at call

| question | MEASURED |
|---|---|
| reachable from `inflate.sh`? | yes: `inflate.sh` → `inflate_runner.py` → `from ddm_r7_token_coder import decode_token_codes` → `ddm_r7_token_coder.py:42` imports `ans_rans_prototype_{decode,encode}` |
| is that import guarded? | **no.** AST over the staged file: **1** top-level import of the adapter, **0** inside any `try`. (`brotli` right above it *is* guarded — the contrast is in the same file.) |
| what if it is removed? | executed: `ModuleNotFoundError: No module named 'repair_entropy_coder_runtime_adapters'` → **the entire decode breaks**. It is a hard dependency, not dead weight. |
| are its functions ever *called* on live bytes? | **no.** The DR7T token codec id is `smevr` (3) on **all four** live archives — pw1 live, wr1 kneeA, wr1 kneeB, pfs1 D1 — and the second DR7T frame (the `s_t` stream inside `pose_warp.stp`) is `smevr` too. The rANS branch needs id 4/5. |
| any adapter packet in any archive member? | **zero** `TACRNG1`/`TACANS1` magics across every member of all four archives. |

**#417 (counted-but-inert) does NOT apply.** #417 is about *counted archive bytes* that the
receiver never consumes. This module is receiver **code**, which the rate term does not charge
(`archive.zip` bytes only). Its cost is exactly **0 bytes**. An unexercised branch in free
receiver code is a hygiene question, not a rate violation — and calling it a #417 violation
would itself be the misclassification.

Exhaustive reference scan over the **5-file** staged decode scope (denominator reported;
positive control found what it should):

| symbol | referenced outside its own module? |
|---|---|
| `ans_rans_prototype_decode` *(POSITIVE CONTROL)* | **yes** — `ddm_r7_token_coder.py` |
| `ans_rans_prototype_encode` | yes — `ddm_r7_token_coder.py` |
| `range_lzma_prototype_decode` / `_encode` | no reference in the 4-file scope |
| `decode_entropy_coder_prototype_member` | no reference in the 4-file scope |
| `entropy_coder_runtime_adapter_manifest` | no reference in the 4-file scope |

So roughly the range/LZMA half of the module ships into the receiver unreferenced — free, but
unreviewed surface on the decode path.

> The first attempt at this scan used `grep -l` over a multi-file list and the tooling mangled
> the file arguments (`No such file or directory`), which would have produced four confident
> "NOT FOUND" negatives from a **broken instrument**. Redone in Python with an explicit
> denominator and a positive control. Reaching for grep is the moment to check the instrument.

### 3.2 Rule-118 — CLEAN

GENERIC ALGORITHM is free in the receiver; VIDEO-DERIVED / LEARNED content must be counted in
`archive.zip`, and hiding data in code is forbidden. AST audit of the **staged** copy:

- **147** AST constants total (denominator), **41** numeric, and only **11 distinct** numeric
  values: `[0, 1, 3, 4, 8, 9, 12, 23, 32, 36, 255]` — all structural (header sizes, bit widths,
  masks, a `<BQH` layout). None is a table.
- **6** literals over the size threshold: **5 are docstrings**, 1 is the 22-key manifest dict of
  schema strings and booleans. **No embedded table, no learned payload, no video-derived data.**
- Imports are **stdlib only**: `__future__, collections, hashlib, lzma, struct, typing`. The
  staged copy replaces `from tac.repo_io import sha256_bytes` with a local `hashlib` shim, so
  the receiver carries no `tac` dependency. No network, no sidecar fetch.
- The rANS frequency table is computed from the payload at encode time and written **into the
  packet** (i.e. into `archive.zip`), which is the counted side of the boundary — correct.

### 3.3 Does it fail closed? — YES, 8/8, zero leaks

Executed against the staged copy on a real packet:

| mutation | result |
|---|---|
| bad magic / empty input | `RepairEntropyCoderRuntimeAdapterError: invalid magic` |
| bad version | `version unsupported` |
| truncated tail / trailing bytes | `encoded length mismatch` |
| flipped payload byte | `decode proof failed` (sha256) |
| flipped digest byte | `decode proof failed` |
| tampered frequency table | `frequencies do not sum to scale` |

**8 of 8 rejected, 0 leaks**, and the untouched packet still decodes. Round-trip: **7/7 rANS**
and **7/7 range**, including the degenerate cases (empty payload, single symbol ×1, single
symbol ×1000, 256-symbol alphabet, 16 KB random).

### 3.4 Two findings worth routing

1. **A stale self-report on the decode path.** `entropy_coder_runtime_adapter_manifest`
   advertises `contest_runtime_decoder_adapter_ready: False`,
   `contest_runtime_decoder_adapter_integrated: False`, and
   `readiness_blockers: ["contest_runtime_decoder_adapter_integration_missing"]` — while the
   module is staged into **5** gate scripts and hard-imported by the receiver. This is an
   **UNDER-claim**, so it is not a NO-FAKE violation (it manufactures no authority), but it is a
   staleness confound: an auditor querying the manifest concludes the module is off the decode
   path when it is on it. The manifest is consumed only offline, by
   `repair_family_byte_transform_executor.py:1907,2433` — never by the receiver.
2. **No drift across staged copies.** All **17** staged copies on the SSD are byte-identical
   (`8ef8520d…`) and the two source-tree copies match the canonical (`7b5820ed…`).

### 3.5 An lg2 row corroborated on the shipped bytes

lg2 flagged that `--token-quant-levels default=16` sits exactly on
`_R7_SMEVR_MAX_LEVELS = 16`. Confirmed on live bytes: every archive's DR7T header carries
`levels = 16`, `src/tac/optimization/ddm_tr1_runtime.py:83` defines the ceiling at 16, and
`:342-343` raises when `levels > 16`. So the default **is** the ceiling on the shipped archive —
the flag cannot be raised without changing codec. lg2's "the default IS the ceiling" reading is
measured, not inferred.

## §4 verdict_scope ledger

| claim | scope |
|---|---|
| 94 / 31 / 109 / 125, 15.886% of arm-AB mass, all 125 genuine breaks | **MEASURED**, shipped pw1 receipt |
| asym replay reproduces pw1's arms bit-identically (0.0) | **MEASURED**, n600 positive control |
| falsifier `ASYMMETRY_PRICED`; Δd_pose −2.580e-06 pop / −1.420e-05 per-scope-pair | **MEASURED**, n600, both denominators |
| 8 of 109 outcomes changed; 6 sym-better, 2 asym-better; 2 pairs = 96% of the effect | **MEASURED**, n600 |
| neither entry rule dominates (one-step-lookahead greedy) | **MEASURED** (pair 326) + **DERIVED** (holds for any doubling bracket) |
| byte-closed net ΔS −3.745e-05 at +16 bytes; parse-back 4/4 | **MEASURED** through the real builder + receiver |
| the builder reproduces pw1's live archive byte-identically | **MEASURED** (360,323 B, sha `0ef9ff7129…`) |
| "this move survives the n600 gate" | **NOT CLAIMED** — predicted, not fired; MAIN owns the slot |
| marginal cost +188 forwards | **MEASURED**, but an UPPER BOUND (per-probe traces discarded) |
| adapter is a HARD import dependency of the receiver | **EXECUTED** (removal → ModuleNotFoundError) + **AST** (1 top-level, 0 guarded) |
| adapter functions never called on live bytes | **MEASURED** on 4 archives (codec id 3 = smevr) + zero packet magics in any member |
| rule-118 clean; stdlib-only; no video-derived content | **AST-AUDITED** with denominator (147 constants, 11 distinct numerics) |
| fail-closed 8/8, round-trip 7/7 + 7/7 | **EXECUTED** on the staged copy |
| 17 staged copies byte-identical | **MEASURED** (sha256) |
| "the rANS path would be better than smevr on this payload" | **NOT CLAIMED** — not measured; the codec race is a separate question |

## §5 STORES CONSULTED

CLAUDE.md · AGENTS.md · `docs/operating_manual_craft_handoff.md` · MEMORY.md (top rows) ·
`.omx/research/{ddm_lg2_arity_mismatch_three_rows,ddm_lg2_binary_inventory,ddm_pw1_pose_menu_saturation}_2026*.md` ·
`.omx/state/canonical_task_status.jsonl` (#871/#821/#822; #900 absent) · primary code:
`experiments/ddm_v4d_resolve.py`, `experiments/ddm_r7_token_coder.py`,
`experiments/stage_wr1_realized_gate.sh`, `experiments/ddm_v4d_build_composed_archive.py`,
`src/tac/optimization/{repair_entropy_coder_runtime_adapters,ddm_tr1_runtime}.py`,
`tools/pw1_pose_menu_saturation_ab.py` · SSD custody
`/Volumes/VertigoDataTier/pact/{ddm_v4d_20260731,ddm_wr1_20260729,ddm_pfs1_20260729}` ·
memories: `boolean_flags_are_a_ui_over_a_continuum_never_binary_judgment` ·
`negative_existence_claims_are_the_days_dominant_error_class` ·
`vacuity_is_indistinguishable_from_pass` · `corpus_first_and_the_recall_instrument_was_down` ·
`built_new_machinery_instead_of_paying_identified_debt` ·
`staleness_is_a_named_confound_class` · `audit_mode_is_not_roadmap_mode_hold_the_frame`.

**Pointer `0.1910828242 [contest-CPU]` UNMOVED; own-vehicle `0.9476091` UNMOVED.**
`[no-triality] [p0-ledger-ok]`
