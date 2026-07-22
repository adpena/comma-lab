---
schema: canonical_equations_note.v1
date_utc: 2026-07-22T13:26:15Z
lane_id: lane_splay_dynamic_optimality_crosswalk_20260722
formalization_status: FORMALIZATION_PENDING
research_only: true
score_claim: false
main_landing_review_required: true
---

# Canonical-equations note - splay dynamic optimality versus byte admission

This note prevents a BST runtime theorem from being silently promoted into a source-coding or
contest-rate theorem. All three candidates below remain `FORMALIZATION_PENDING`; no registry row or
executable consumer is landed.

## 1. Published root-BST competitiveness

Let `U=[n]`, `X=(x_1,...,x_m)`, and let

`C_A(X) = sum_i (1 + rotations_A(i))`

be the cost in the paper's root-BST model, in which the accessed ordered key must finish at the
root. Define

`rho(n) = log log n * (log log log n)^2`.

Theorem 1 of the local paper gives the asymptotic relation

`C_splay(X; arbitrary T_0) = O((C_OPT(X) + n) * rho(n))`.                 (E1)

`OPT` knows all of `X` and chooses its initial tree. The additive `n` is necessary because splay's
initial tree is arbitrary.

### Offline comparator simulation used in the proof

For analysis, each access is serialized:

`OPT access/rotations -> freeze OPT -> splay access/rotations`.           (E2)

The normal-form theorem constructs `T*` such that

`C_T*(X) = O(C_OPT(X))`,                                                  (E3)

while `T*` is ribless, has depth `O(log n)`, and rotates only at constant depth. Ranks in the splay
tree are depths in `T*`. Lazy intervals, heap-like pairings, and bends yield the zig-zig/zig-zag
bounds used in (E1).

There is no epoch variable or epoch decomposition in arXiv v1. Any future memo calling the proof
"epoch based" must point to a different source or be corrected to the serialized normal-form
simulation (E2)-(E3).

## 2. Working-set and finger quantities remain time bounds

Let `w_i` be the number of distinct keys accessed since the preceding access to `x_i`, and let
`f_i = |x_i-x_{i-1}|` in the chosen total key order. The appendix records known splay properties of
the form

`access_time_i = O(log(2+w_i))`,                                         (E4)

`access_time_i = O(log(2+f_i))`.                                         (E5)

The paper also explains that competitiveness transfers unified/lazy-finger comparator bounds with
the multiplicative `rho(n)` and initialization term. These are costs in the BST model. Neither
`w_i` nor `f_i` is a normalized probability, and (E4)-(E5) do not satisfy Kraft's inequality by
themselves.

**Consumer verdict:** `ALREADY_HAVE_BETTER_DIRECT_BYTES`. v6 hold24 and the xi temporal coder retain
their measured framed-byte laws; no equation registration changes them.

## 3. Prefix-coder bridge guard

For a deterministic adaptive prefix tree with state `S_{i-1}`, define the emitted path length

`ell_i = depth_{S_{i-1}}(x_i)`

and complete framed length

`L_prefix(X) = H_frame + sum_i ell_i + P_pad`.                            (E6)

It is tempting to substitute (E1) into (E6). That substitution is **REFUSED** unless one proves all
of the following for the concrete codec:

1. symbols and tree nodes map bijectively to the paper's ordered-key root-BST model;
2. the emitted pre-update prefix path is the same path charged by the root-BST access;
3. encoder and decoder perform the same splay rotations without transmitted side state;
4. leaf/internal-node conventions preserve the constant-factor cost relation;
5. `H_frame`, alphabet initialization, EOS, integrity, selector, padding, and decoder dependency
   bytes are counted;
6. the offline comparator is a valid prefix-code comparator with the same framing constraints.

Jones's prefix-trie construction establishes that a splay-based adaptive prefix codec exists; it
does not by itself prove conditions 1-6 for the 2026 ordered-key theorem.

Even if conditions 1-6 land, one still needs a separate inequality relating root-BST OPT to the
actual source-coding comparators. No such relation is in the paper:

`C_OPT^BST(X)  ?  -log2 Q_KT1(X)  ?  8 B_LZMA/Brotli(X)`.                 (E7)

The question marks in (E7) are intentional non-relations, not approximate equalities.

Candidate equation ID: `splay_prefix_bridge_guard_v1`.

Status: `FORMALIZATION_PENDING_UNPROVED_MODEL_AND_OBJECTIVE_BRIDGE`.

## 4. Exact real-stream coder-race admission

For semantic stream `s` and complete codec frame `F_c`, define

`B_c(s) = len(F_c(s))`,                                                   (E8)

subject to

`decode_c(F_c(s)) == s` and strict rejection of truncation/trailers.      (E9)

Let `c_inc(s)` be the measured same-source, same-aggregation-scope incumbent. Local retention is

`keep_local(c,s) := (E9) and B_c(s) < B_c_inc(s)`.                        (E10)

The KT1 number is separately defined as a derived ceiling

`B_KT1_ceiling(s) = ceil((-log2 Q_KT1(s))/8)`,                            (E11)

not as a landed parseable frame. Beating (E11) but not (E10) is not adoption.

For a receiver-closed archive `A_c` with exactly one byte home,

`Delta B_archive(c,s) = len(A_c) - len(A_inc)`.                           (E12)

Final byte admission requires exact semantic identity and

`Delta B_archive(c,s) < 0`.                                               (E13)

For a genuinely lossless substitution, once receiver identity is proved,

`Delta d_seg = 0`, `Delta d_pose = 0`, and

`Delta S = 25 * Delta B_archive / 37,545,489`.                            (E14)

Candidate equation ID: `lossless_coder_race_admission_v1`.

Status: `FORMALIZATION_PENDING_MEASURED_SPLAY_MTF_FRAMES_ABSENT`.

## 5. V10 ordered-set applicability guard

Let `n_live` be the number of elements in a dynamically maintained ordered set and `T_set` its
measured share of encoder wall time. A splay data-structure substitution is even eligible only if

`large_ordered_set := n_live >> 1 and T_set / T_encoder >= tau_profile`.  (E15)

The checked V10 consumer instead has fixed dimensions:

- six RGB box crossings, at most seven intervals;
- a five-class scorer head;
- a rank-at-most-six Pose Gram system;
- dense vectorized NumPy operations.

Thus `large_ordered_set=false` for the inspected functions, independent of asymptotic splay
competitiveness. `tau_profile` is deliberately unassigned because no ordered-set profiler row exists.

Candidate equation ID: `splay_encoder_ordered_set_applicability_guard_v1`.

Status: `N_A_CHECKED_V10_FIXED_DIMENSIONAL_SURFACE`; future profiled large-`n` loops remain open.

## STORES CONSULTED

STORES CONSULTED: delegated authority and its verified SHA-256; local 68-page arXiv PDF and rendered
pages 1/3/13/48/56; Jones University of Iowa primary author page and DOI; paired crosswalk memo and
DAG FEED; truly-optimal coder measurement/survey/equations; v6 receipt/equations; xi source and n600
measurement; V9 findings; V10 solver/tests; `CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`;
`docs/operating_manual_craft_handoff.md`; current lane/progress/frontier-report/inbox surfaces.

No canonical-equation registry was changed. MAIN landing review must verify the objective-separation
guard before merging. `0.1910828242 [contest-CPU]` remains unchanged.
