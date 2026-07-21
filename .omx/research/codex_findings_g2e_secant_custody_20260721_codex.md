# Codex findings — Task #578 G2e realized-secant custody

`lane_id=lane_g2e_secant_custody_578_20260721` ·
`verdict=MEASURED_G2E_SECANT_PREFIX_N16_FAMILY_OPEN` ·
`research_only=true` · `[macOS-CPU advisory]` · `score_claim=false` ·
`promotion_eligible=false` · `pointer=0.1910828242 [contest-CPU] UNMOVED` ·
`MAIN_REVIEW_REQUIRED=true`

## Outcome first

The requested candidate-arrangement realized-secant custody surface now exists
for an exact contiguous n16 prefix. It contains four independent finite-secant
rows per pair, candidate-state logits, exact 144D head-input patches, local RGB
Jacobians, full receiver round-trips, and per-class/per-margin-bucket trust
decisions. This closes the prefix-scoped absence premise
`R1B2_RANK4_FIRST_ORDER_REALIZED_SECANT_CUSTODY_ABSENT`.

It does not authorize the rank-4 correction. All 31 measured class/bucket trust
regions were unusable: all 31 exceeded the declared relative first-order/secant
residual tolerance, and 26 also contained a realized response with the wrong or
zero sign. The exact successor blocker is therefore
`R1B2_RANK4_REALIZED_SECANT_TRUST_REGION_EMPTY_N16_OPENPILOT`. This is an n16,
openpilot-base, signed-amplitude formulation result—not a rank-4 family verdict.

The fail-closed consequence is concrete: all 16 pair solves ended
`TRUST_REGION_REFUSED`; no QP was allowed to fire; no correction packet carried
video-derived bytes; and no score or pointer claim was made. n64/n600 remain
resumable continuations, but an n600 verdict was deliberately not inferred from
the n16 prefix.

## D1 — realized-secant custody

- MEASURED: 64 pair/column rows: 16 rows for each of columns 0, 1, 2, and 3.
- MEASURED: 388 per-write responses across those rows (97 declared writes times
  four columns), with exact 144D feature displacements, first-order margin
  deltas, realized margin deltas, secant ratios, applied RGB norms, and uint8
  saturation counts.
- MEASURED: 31 class/bucket trust regions, all in the `nonpositive` pre-margin
  bucket: target class 0 on 16 pairs and target class 1 on 15 pairs.
- MEASURED: usable `0/31`; `RELATIVE_SECANT_RESIDUAL` on `31/31`;
  `REALIZED_SIGN_OR_ZERO` on `26/31`.
- MEASURED: all four candidate columns were nonzero in this n16 prefix. The
  implementation nevertheless returns an explicit zero-padded fourth column
  for one-write/three-RGB-coordinate pairs, preventing the known 34-pair n600
  low-dimensional tail from crashing or inventing a fourth direction.
- DERIVED: the trust boundary, not QP convergence, is now the first active
  semantic-correction gate for this arrangement.

The first-order and secant rows are kept separate by target class and margin
bucket. No pooled average was used to hide a failed row.

## D2 — receiver-closed QP

The deterministic rank-at-most-four active-set QP, uint8 box inequalities,
KKT residual custody, canonical coefficient packet, #557 parse-back, and hard
oracle are implemented and covered by focused tests. On the real n16 arm the
QP was correctly not invoked because every pair depended on at least one
unusable trust region.

MEASURED hard consequences:

- pair solve status: `TRUST_REGION_REFUSED` on `16/16`;
- admitted pairs: `0/16`;
- correction bytes: `0`;
- factor-2 uint8 exact pairs: `16/16`;
- double-decode-identical pairs: `16/16`;
- receiver-closed declared-write survival after the refused correction: `0/97`.

An empty correction stream is not treated as evidence that the semantic QP
succeeded. The full-frame double-decode predicate passed; semantic admission
did not.

## D3 — semantic and rate ladder

The corrected implementation was rerun from scratch into a fresh SSD custody
root, rather than reusing stages whose config hash bound the pre-correction
code.

| n | whole description exact | declared writes survive | admitted pairs | mean d_seg | correction bytes | total bytes | headroom |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0/16 | 0/97 | 0/16 | 0.3777516682942708 | 0 | 121,128 | +95,094 |

The 121,128-byte base consists of the settled 78,969-byte seed baseline plus
42,159 counted openpilot-base bytes: a 21-byte palette, 835-byte zlib static
chart, and 41,303-byte Brotli lane chart. It fits the 216,222-byte target box.
No marginal score-per-byte row exists because the trust gate admitted no
nonempty correction; the registered `25/37,545,489` stopping threshold was not
bypassed.

The unmodified `predict_project_realization_admissibility_v1` returned false.
It passed factor-2 exactness and double-decode identity, and failed exactly:
`n600`, `semantic_cells_to_rgb_exact`, `pose_within_declared_tube`,
`zero_added_seed_bytes`, and `receiver_derived_rgb`. The last two remain false
because the openpilot base carries 42,159 counted, not receiver-derived, bytes.

The earlier 447,170-byte exact-I-frame result remains a rate-inadmissible
control only. It is not substituted for this 121,128-byte composed arm and is
not a score.

## D4 — Pose honesty

- MEASURED mean frozen PoseNet d_pose: `172.29492246623715`.
- MEASURED mean declared-tube debt: `166.80570309370756`.
- MEASURED tube-contained pairs: `0/16`.
- MEASURED semantic-correction Pose delta: zero, because no correction was
  admitted.

The cross-pair base still uses the declared nearest-target G1 proxy. Therefore
this result does not close exact banked cross-pair motion, learned xi residuals,
or the pose-factorized child.

## Reusable system intelligence and next route

Keep the typed secant rows, exact candidate-state head-input capture, strict
class/bucket trust separation, rank-at-most-four zero padding, deterministic
QP, #557 packet custody, atomic pair/chunk/prefix resume format, and hard-oracle
admission boundary.

Do not spend the +95,094-byte headroom or continue blanket n600 correction
allocation under the failed trust model. The next child should reformulate the
local response model and rerun n16 first—for example, paired bidirectional
secants per effective chart direction and/or a locally smaller amplitude rung—
while retaining the same per-row hard custody. Only after a nonempty measured
trust region exists should the QP emit a candidate and the Fisher/margin
reverse-waterfill rank it against the registered rate threshold. Any residual
carrier remains curvelet/shearlet-only; the separate Pose stream should use xi
factorization.

These are reactivation criteria, not a GO.

## Custody

Final external receipt:
`/Volumes/VertigoDataTier/pact/evidence/g2e_secant_20260721/final_hardened/receipt.json`

- file SHA-256: `e89157bb8dfc6b11b20aecccd4dbe82113ea706c9a1eb054de989c26a740dbc4`;
- embedded canonical receipt SHA-256:
  `cafd3d4d3b1024c47e6cbdfcc74322cfa2a24f2ffc3a7bd02c41727416f542c4`;
- config SHA-256:
  `06887691f57e94ef461b2014ef182f4ab90c9a191d247ba47441c6419c70d20f`;
- external tree: 58 files, 5,200,129 bytes, deterministic manifest hash
  `e6ee0aa4357c634145c0647a6715db053f507202e6dd2d99e9a06566aaddb733`;
- focused verification: 29 tests passed; Python compile, Ruff lint, Ruff
  format, strict receipt parse/validation, and `git diff --check` passed;
- adversarial cross-check: 200 deterministic random QPs matched SciPy SLSQP;
  three independently rehashed corruptions of the real receipt were refused;
- review tracker: two clean passes (`g2e_secant_clean_1` and
  `g2e_secant_clean_2`) cover all four touched Python files. The worktree has
  no `.omx/state/review_policy.json`, so scoped `policy-check` reported that
  absence rather than claiming a policy verdict.

## STORES CONSULTED

Delegated authority SHA
`d90b237d9807ced72ecfb792fb7345b4a590ec903095b721cbcca4207b503a95`;
`CLAUDE.md`; `AGENTS.md`; program/craft/vehicle manuals; v7.5/v8 SPECs;
operator Fisher/margin, corrected first-order+secant+QP, curvelet/shearlet,
xi-factorization, and reverse-waterfill directives; predecessor G2d code,
tests, receipts, findings, DAG, and reuse manifest; rank-4 prototype receipt;
terminal n600 VJP/M1 custody; exact factor-2 receiver; #557 codec; frozen
seed/cache/scorer weights and source; lane/progress state; both delegated
inboxes through `2026-07-21T13:30:00Z`; historical and final n16 SSD trees.

## Triality and landing boundary

- DSL/code: `realized_secant_custody.py` defines typed observations, trust,
  deterministic QP, packet, and receipt contracts; the measurement CLI wires
  them into the settled receiver and scorer path.
- DAG: `g2e_secant_custody_DAG_FEED_20260721.md` routes the measured trust
  blocker into sensitivity, Pareto, allocator, autopilot, posterior, and probe
  consumers.
- Equations: `predict_project_realization_admissibility_v1` was invoked
  unmodified; no threshold or canonical equation was changed.

MAIN must review the branch diff and merge boundary before any result is
treated as landed repository truth.
