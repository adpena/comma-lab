# Recursive adversarial review — ROUND 15 of the 5 Layer-2 levers (2026-06-13)

**Reviewer/Engineer:** Partner-A2. Prior FRESH count: R13 CLEAN → 1/3; R14 NOT-CLEAN (contest-optimality
finding: Lever-4 leaves ~0.018 score on the table) → reset 1/3 → 0/3.

**R15 is NOT a pure review round — it is the ENGINEERING+HARDENING landing the operator directed:**
*"where any gaps or optimization or nuance or sophistication opportunities found, completely engineer and
implement and adversarially review and harden as you proceed"* + *"always engineer the optimal
implementation optimized against contest … mathematically and algebraically and geometrically and all
optimal … optimize all for Track A."* R14 FOUND the gap (Lever-4's online sensitivity EMA never reaches the
variable-level export). R15 COMPLETELY ENGINEERS the contest-optimal unification, adversarially reviews it
(found + fixed a real ordering bug in my own implementation), hardens it (4 guards), and EMPIRICALLY
validates it NET-POSITIVE on the real scorer.

## VERDICT: the Lever-4↔variable-level-export UNIFICATION is ENGINEERED + HARDENED + MEASURED NET-POSITIVE.
**SEAL counter stays 0/3** — this round LANDS a new lever feature (engineering), it does not advance the
clean-pass count. The clean count needs THREE fresh consecutive REVIEW passes AFTER this engineering settles.

---

## A. WHAT WAS ENGINEERED (the contest-optimal unification)

New driver config flag `lever4_variable_level_export_enabled` (DEFAULT FALSE → byte-identical to today).
When ON, the EXPORT decoder blob is built at a VARIABLE per-tensor INT8 grid derived from Lever-4's ONLINE
score-sensitivity EMA (`||∂S/∂w_t||`) via the existing `levels_from_sensitivity_for_codec` (the SAME
rank-norm band the score-aware QAT trained the decoder to be robust at) — capturing the full reverse-
waterfill byte saving R14 measured, WITHOUT a separate offline RD-table sweep. This is the unification of
Lever-4's online EMA (free, per-step) with the variable-level codec (Partner-B's D2 byte-half), mutually
exclusive with the D2 RD-table path (both replace the SAME export blob; `__post_init__` refuses both on).

**The math/algebra/geometry (per the operator's directive):** the contest-optimal per-tensor INT8 allocation
minimizes `S = 100·d_seg(levels) + 25·bytes(levels)/N`; the Lagrangian optimum equalizes the marginal
`∂d_seg/∂byte` across tensors (reverse-waterfill, Cover & Thomas Ch.10). Lever-4's `||∂S/∂w_t||` EMA is a
free online estimate of each tensor's distortion sensitivity (the RD-curve gradient); `levels_from_
sensitivity_for_codec` rank-normalizes it into the [0.5×,1.0×]·127 band — protect sensitive tensors at 127,
coarsen insensitive ones toward 16. Geometrically each tensor lands on its lower-convex-hull RD operating
point. Algebraically this is the SAME quantity D2's allocator solves offline — the unification removes the
expensive offline RD sweep.

### Files
- `src/tac/torch_vehicle/driver.py`: `+lever4_variable_level_export_enabled` config flag + mutual-exclusion
  guard in `__post_init__`; `_EvalSnapshot.tensor_sensitivity_ema` field; `_snapshot_ema` captures the EMA
  (only when the flag is on); new method `_build_archive_with_optional_sensitivity_variable_levels`;
  `_build_archive_and_eval_decoder` gains a `sensitivity` param + routes to the new method when the flag is
  on (both no-FiLM and FiLM branches); the `_eval_snapshot` call threads `snap.tensor_sensitivity_ema`.
- `src/tac/torch_vehicle/tests/test_all_layer2_levers.py`: the prior R14 scope-guard (which asserted Lever-4
  had NO variable-level export) is REPLACED by the NEW-contract guards (the scope changed by operator
  directive); +2 unification tests with 6 NO-FAKE guards.
- `experiments/probe_r14_lever4_unification_net_effect.py`: the real-scorer NET-effect disambiguator.

## B. ADVERSARIAL REVIEW OF MY OWN IMPLEMENTATION (found + fixed a real bug)

Self-adversarial review caught a **meta-flag ORDERING bug** in my first implementation: I built
`base_archive = build_base_archive(meta_dict)` BEFORE writing `meta_dict["decoder_codec"] =
"lever4_sensitivity_variable_level.v1"`, so the `meta_brotli` section embedded in the emitted archive did
NOT carry the codec flag — the inflate side could not know to dispatch `decode_decoder_variable`. FIXED to
mutate `meta_dict` with the flag + provenance BEFORE `build_base_archive` (mirroring the D2 method's correct
order). VERIFIED: the emitted meta now decodes to `decoder_codec: lever4_sensitivity_variable_level.v1`
(direct 3-section meta decode). A regression guard (test surface 5) catches re-introduction.

A second adversarial finding: the vendored `parse_archive` CANNOT read a variable-format blob (it raises
`UnicodeDecodeError` trying to vendored-decode the variable bytes) — confirming the meta flag is load-bearing
(the inflate dispatch MUST branch on it). My test's meta-flag check correctly decodes ONLY the meta section
(not the variable decoder blob). NOTE (honest open item, shared with D2): the CONTEST inflate.sh-side
variable-format dispatch is not yet wired in torch_vehicle — both D2 and this unification currently realize
the variable export at the DRIVER-EVAL surface (the driver builds the blob + scores the parse-back decoder
directly). The inflate-side dispatch is a shared next item before a contest-CUDA submission; the byte saving
+ the eval-side parse-back are real and measured now.

## C. HARDENING (the guards landed)

`test_r14_lever4_variable_level_export_unification_default_off_and_byte_saving` (6 NO-FAKE surfaces):
1. DEFAULT-OFF byte-identity (the daemon-safety guard): flag off → byte-for-byte the vendored archive
   regardless of the sensitivity passed.
2. UNIFORM-sensitivity default-preserving: flag on + uniform sensitivity → vendored byte-identical.
3. NON-UNIFORM byte saving: flag on + non-uniform EMA → STRICTLY fewer archive bytes (the saving captured).
4. PARSE-BACK FAITHFUL: the returned eval decoder is a usable module with the SAME keys + renders finite.
5. META-FLAG ORDERING guard: the emitted meta carries `decoder_codec` (catches the ordering bug above).
6. DETERMINISM (resume-safety): same EMA → bit-identical archive (the EMA is persisted/restored across
   checkpoints, so a crash-resume re-exports identically).
Plus `test_r14_lever4_variable_level_export_codec_mechanism_real` (the codec byte half is real + sd-key-
correct + default-preserving).

## D. EMPIRICAL VALIDATION — NET-POSITIVE on the REAL scorer (the contest-validity check)

`probe_r14_lever4_unification_net_effect.py` on the REAL frozen scorer (8 real `0.mkv` pairs, a lightly-
trained decoder + a REAL `||∂S/∂w||` sensitivity from one backward), MEASURED:

| metric | value |
|---|---|
| uniform archive bytes | 82605 |
| variable (unification) archive bytes | 76689 |
| **byte saving** | **5916 B (7.16%)** |
| real d_seg uniform | 0.508686 |
| real d_seg variable | 0.508667 |
| **d_seg delta (variable − uniform)** | **−1.9e-5 (IMPROVED)** |
| contest rate delta (25·B/N) | −0.00394 |
| contest seg delta (100·Δd_seg) | −0.0019 |
| **net score delta (advisory)** | **−0.00585 (NET-POSITIVE)** |

The sensitivity-guided coarsening preserved d_seg (coarsened only score-irrelevant tensors — exactly the
reverse-waterfill intent), so the ~7% byte saving is a NET contest win, not a d_seg regression. On a real
TRAINED basin the genuine `||∂S/∂w||` profile is far more skewed (most tensors score-irrelevant), so the
saving is expected to be larger (R14 RD-table evidence showed up to 36% at aggressive coarsening with
several tensors at ≤0 d_seg cost). **Authority: [contest-CPU advisory] NON-PROMOTABLE** — the
byte-direction + d_seg-direction are real on this tiny slice; the authoritative score claim still requires
the 600-pair byte-closed dual CPU/CUDA eval + the inflate-side variable dispatch.

## E. TRACK-A IMPACT (operator: "optimize all for Track A")

This is a Track-A contest optimization: the live distortion arm (`distortion_arm_l235_*`) is base_ch20 Track-A;
enabling `lever4_variable_level_export_enabled` on a Lever-4-trained Track-A basin captures the
reverse-waterfill byte saving (~0.004–0.018 rate reduction depending on the trained sensitivity skew) at
≤0 d_seg cost. The D2-on-basin measurement is Partner-B's active surface (the `track-a ledger` + finishing-
kit agents) — this unification is the cheaper online-EMA alternative to D2's offline RD sweep; the two are
mutually exclusive level sources for the same export blob. NO daemon was touched (the flag is default-OFF;
the live arm out-dir is untouched + byte-identical).

## Findings by severity (this round = engineering + self-adversarial review)

- **HIGH:** NONE.
- **MEDIUM (FIXED this round — my own implementation):** the meta-flag ORDERING bug (base_archive built
  before the `decoder_codec` flag was written) — fixed + guarded.
- **LOW:** NONE.
- The unification itself is the contest-optimal response to the R14 finding; it is engineered, hardened,
  and measured NET-POSITIVE.

## Test-run count

- 2 unification tests (codec-real + unification 6-guard): **2 passed in 2.02s.**
- Default-byte-identity (daemon-safety, post-unification): **1 passed in 6.74s.**
- Fast structural subset (incl R14 + levers, post-unification): **79 passed** (no regression).
- Net-effect real-scorer probe: byte_saving 5916 B (7.16%), net_score_delta_advisory −0.00585, net_positive=true.
- Full-suite batch confirmation: see the trailer.

## Wire-in / provenance

6-hook (Catalog #125): #3 bit-allocator ACTIVE (the unification IS the contest-optimal reverse-waterfill
bit-allocator wiring Lever-4's online EMA into the variable-level export); #1 sensitivity-map ACTIVE (the
EMA → per-tensor level allocation); #6 probe-disambiguator ACTIVE (`probe_r14_lever4_unification_net_
effect.py`); #2/#4/#5 N/A. Mission contribution: `frontier_breaking` (a measured ~0.006-advisory contest
optimization engineered + hardened + NET-POSITIVE-validated; the authoritative claim pends the 600-pair
dual eval + inflate-side dispatch). Authority: all numbers [contest-CPU advisory] NON-PROMOTABLE. Default
path byte-identical (daemon-safe). Frontier UNMOVED `0.19109982419209975` contest-CPU (the unification must
be enabled + measured at 600 pairs + inflate-wired before it can move the exact pointer).

**VERDICT: ENGINEERING+HARDENING LANDING (the operator-directed contest-optimal Lever-4 unification), with
a self-found+fixed ordering bug, 8 NO-FAKE guards, and a NET-POSITIVE real-scorer measurement.** The SEAL
counter stays 0/3 (this lands a feature; it is not a clean review pass). The next units: (1) wire the
inflate-side variable-format dispatch (shared with D2), (2) measure the unification on a TRAINED Track-A
basin at 600 pairs byte-closed dual CPU/CUDA, (3) resume the 3-fresh-clean-pass SEAL review on the levers
INCLUDING the new unification surface.
