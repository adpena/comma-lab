# DDM JS8 implicit decoder-derived edge conditioning

## Outcome

`REFUSED_LOCAL_ADMISSION_POSTHOC_EDGE_GATE_INSTANCE` — JS8 built, decoded, and measured a real
receiver-closed MC36 candidate over all 600 pairs. At scale 0.125 it recovered
only 7 Seg flips while adding 1,749 archive bytes. The realized Seg improvement
is worth `-0.000005934` score units, versus `+0.001164587` in rate, so the
Seg-plus-rate necessary condition is **+0.001158653** before pose. This closes
this post-hoc scalar-gate instance and does not kill implicit conditioning,
trained receivers, or coupled multi-token conditioning.

The candidate derives its spatial edge state from already-decoded semantic
tokens and their four-neighbor class transitions. It ships no edge mask. The
video-derived Road-hub edge weights and the retained EC2 adapter are counted in
the archive. Inactive operation is byte-identical to the MC36 semantic receiver
on a seeded-random n32 identity probe.

## EC2 refusal mechanism

EC2's only retained scored endpoint is the serialized stage-30 EMA adapter on
the full population. It fixed 12,075 of 34,970 CP135 errors but introduced
52,854 new errors, leaving 75,749 errors and a net reduction of **-40,779
flips** `[contest-CUDA T4 frozen-SegNet, n600] COMPONENT-ONLY`. The mechanism
is collateral, not failure to touch the target: 42,184 introduced errors
(79.81%) have GT Road, and the Road-to-Lane cell alone contributes 31,542
(59.68%) of all introduced errors.

The requested stage attribution is **not measured**. The retained local and
harvest receipts contain the terminal stage-30 module and terminal endpoint,
not score fields for stages 10 or 20. The trainer source proves the schedule:
target birth used error/correct mass 4:1, balanced descent 1:1, and collateral
finish 1:4. Those intended pressures do not identify which stage created the
terminal collateral. Assigning the loss to a named stage would be a fake causal
claim. The admissible conclusion is that the terminal EC2 mechanism overcorrects
Road-hub interfaces despite fixing substantial base error.

## Design derived from the retained decomposition

The charter's approximate `87.8% / 49.2%` m91 law is historical context. On
the exact cp135/MC36 base consumed here, `STAGE0_RESULT.json` measures 34,970
flips, with 28,549 Road-incident flips (81.6385%) and 15,178 Road-Lane flips
(43.4029%) `[contest-CUDA T4 frozen-SegNet argmax field, n600] COMPONENT-ONLY`.
JS8 therefore binds the existing EC2 correction only where the decoder's own
token neighborhood crosses a measured Road-hub interface. Its symmetric
class-pair strengths are the measured cp135 counts normalized by Road-Lane:

| decoded interface | measured flips | conditioning strength |
|---|---:|---:|
| Road-Lane | 15,178 | 1.000000 |
| Road-Undrivable | 6,972 | 0.459349 |
| Road-Movable | 4,205 | 0.277046 |
| Road-MyCar | 2,194 | 0.144551 |

The generic neighbor traversal and gate application live in the receiver. The
counted `5x5` float16 table is symmetric with a zero diagonal and carries an
adapter scale. The runtime parses it exactly, rejects trailing bytes, derives
the token-local edge state, and applies the retained 1,369-byte EC2 adapter at
the same pre-TokenBlock receiver site as EC2. This consumes the decomposition
rather than repeating EC2's uniform adapter.

## Admission and custody receipts

- Source base: MC36 archive `f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de`,
  186,269 B.
- Decomposition: `STAGE0_RESULT.json`, SHA-256
  `472fc816f6656ec0cdd37bd475598e8e9683260dc97adeb4163ead5ae90b3e67`.
- Retained EC2 adapter: 1,369 B, SHA-256
  `9559c2ab5128f193c8b0c754c5d61851b7784070fa049e04cf48cfd157eead82`.
- Selected JS8 gate: 170 B, SHA-256
  `02992f3d58b8eb0b244812a6b4d2cd681347f4520408af440284157470245a9c`,
  adapter scale `0.125`.
- Selected archive and independent repeat: 188,018 B, SHA-256
  `7f64f30b09bcb75428125ee11cf0aedbee368d2f6fb1b3707b89542ef6fde448`;
  exact parse-back passed.
- Inactive seeded-random n32 pre-R arrays are byte-identical, SHA-256
  `84d4cdfc476aee64f121da1d754d62fcb332a7a3b8b26870ef9e76ac432f275b`,
  with maximum absolute difference zero.
- The selected runtime decoded all 600 pairs through the real receiver in
  806.059 seconds. The retained 3,662,409,600-byte raw payload has SHA-256
  `d9e976025f27a6597776f1034b6199028fcc197437ebb2573ba866d6075ad0f5`.
- Small routing receipts remain under
  `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/edge_conditioned/js8`.
  Bulk payloads are retained under
  `/Volumes/APDataStore/pact/pr135_joint_solve_20260810/edge_conditioned/js8`
  because the logical Vertigo store had only 1.06 GiB free. No materialized
  payload was discarded.
- Full result: 5,986 B, SHA-256
  `4c764491ee5bde45d6f3f2ab1af68e2a69b2b920ad0317398c48ca24dcc50706`.
- The 3,662,409,600-byte MC36 base raw was rehashed after the run and matches
  its pinned SHA-256
  `e5539653f598a1c31e28900888f450a6de019cb29864674f232ad2f8956b15c9`;
  the runner now fails closed on this pin before future scorer resumes.
- Typed adjudication: 4,075 B, SHA-256
  `5eab37eb910417050e00ba55525f91f26d075272985629b52c39ba735814993c`.
- Provenance receipt: 4,443 B, SHA-256
  `96985d3381b1214ec7bdf2c038bef50947693720c0188e899286c6a377b7ccfc`;
  it pins the run-time source, environment, exact GT video, evaluator source,
  scorer weights, base/candidate raws, archive, progress, and result. The
  canonical full-upstream tree helper refused a pre-existing submission-library
  symlink, so no full-tree SHA is invented; every file consumed here is hashed.

## Realized advisory measurements

The n32 screen used a seeded random sample of eight pairs from each quartile of
the cp135 per-pair flip distribution. It is a toy bracket, not the verdict.

| row | flips / 6,291,456 | d_seg | d_pose | bytes | recomputed S |
|---|---:|---:|---:|---:|---:|
| MC36 base | 2,753 | 0.0004375776 | 0.0001396923 | 186,269 | 0.2051620754 |
| JS8 scale 1, uncompensated | 2,880 | 0.0004577637 | 0.0056426711 | 188,005 | 0.4085042524 |
| JS8 scale 0.125, uncompensated | 2,754 | 0.0004377365 | 0.0002752595 | 188,018 | 0.2214323004 |

The scale sweep retained every table, archive, independent repeat, parse-back,
camera payload, pre-R tensor, scorer input, logit field, argmax field, and pose
vector for scales `0.125, 0.25, 0.5, 0.75, 1.0, 1.5`. Selection used realized
Seg plus exact rate only because frame-0 compensation had not yet been compiled.
Scale 0.125 added one flip on n32; every larger scale added at least four, and
scale 1.5 added 302.

The full row uses all 117,964,800 scored Seg pixels and all 3,600 official
PoseNet values. The candidate S is uncompensated and therefore diagnostic; the
Seg and rate terms are the load-bearing necessary-condition verdict.

| row | flips / 117,964,800 | d_seg | d_pose | bytes | recomputed S |
|---|---:|---:|---:|---:|---:|
| MC36 base | 50,388 | 0.0004271443685 | 0.0001474661936 | 186,269 | 0.2051446455 |
| JS8 scale 0.125, uncompensated | 50,381 | 0.0004270850288 | 0.0003114726302 | 188,018 | 0.2237117046 |
| candidate minus base | -7 | -5.93397e-8 | +0.0001640064366 | +1,749 | +0.0185670591 |

The component deltas are Seg `-0.000005933974`, uncompensated Pose
`+0.017408405775`, and rate `+0.001164587309`. Paying the rate requires
1,373.803 Seg flips; the candidate realizes 7, leaving a 1,366.803-flip
shortfall.

Every S value is recomputed as
`100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37,545,489`; no rounded evaluator
print is used.

## Verdict and boundaries

**INSTANCE REFUSAL.** The tested object is the scale-0.125 Road-hub scalar gate
over the terminal EC2 adapter. It fails the Seg-plus-rate necessary condition
by `+0.001158653335` `[macOS-CPU advisory, n600]`. This margin is exact on the
deterministic local scorer object and is not a noise-floor call.

This is a local-admission verdict, not a platform-transfer claim. The local CPU
MC36 base field has 50,388 flips, while the retained contest-CUDA T4 base field
has 34,970. JS8 was not measured on T4, and the CPU delta is not promoted into
a T4 delta. The charter routes a failed local admission without buying that
row; reopening this exact instance would require new cross-host sign evidence.

QS5 compensation was intentionally folded after that gate. The proven QS5
pattern restores pose leakage induced by a changed frame 1 toward the source
object; it cannot alter the measured frame-1 Seg field or the already-counted
rate. It therefore cannot make this object negative without becoming a new,
unmeasured below-source pose optimizer. No compensated row or joint-score claim
is invented.

Measured: exact counted bytes and deterministic repeats; exact gate parse-back;
inactive real-receiver identity; stratified-random n32 frozen CPU Seg/Pose; real
full-n600 receiver decode; and full-population frozen CPU Seg/Pose in twenty
retained 30-pair chunks. Not measured: EC2 per-stage scorer endpoints, a QS5-compensated JS8
archive, contest-CPU, contest-CUDA for JS8, or `upstream/evaluate.py`. The n32
rows do not support an instance, formulation, or family verdict.

No pointer moved. The current own-custody effective frontier remains MC36
Variant C: `S = 0.1619344578804448 @ 186,269 B [contest-CUDA T4, n600]`, archive
SHA-256 `f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de`.

## RECALL EVIDENCE

Before design and adjudication, the full research/equations/memory/DAG/council/
task/doc corpus was queried for:

- `implicit decoder-derived edge-state conditioning Road hub EC2 collateral`;
- `Road Lane edge gate trained receiver coupled multi-token 978 982`;
- `EC2 target birth balanced descent collateral finish stage fixed introduced errors`.

The source set included the JS1C stage-0 memo and retained result, EC1/EC2 source
and receipts, BG1/BG2 mechanism postmortems, SR1, GCA1, GV2, PC2, the per-edge
optimality directive, canonical equations, live DAG FEED blocks, task ledger,
hot state, and lane registry.

Beyond the charter seeds, recall found four plan-changing constraints:

1. EC1 already measured decoded-token oriented context at AUROC 0.995655 but did
   not materialize a candidate; context predictiveness alone is not value.
2. BG2 located EC2's failure in Road collateral and found that the existing
   8-D frame state has negative held-out incremental R2 (`-0.018448`, p
   `0.897810`) for this residual. That closed a generic bilinear frame gate and
   forced JS8 to use token-local per-edge state.
3. SR1's standalone additive decoder-known edge-state calibration saved only
   2 B, so JS8 could not claim a rate-only win.
4. GV2's sparse Road-Lane token edits were closed by collateral blast radius;
   JS8 therefore gated the retained adapter without shipping or directly
   editing an explicit edge mask/token event stream.

The canonical equation registry fixed the marginal comparison at one Seg flip
equal to `8.477105e-7` score units and one archive byte equal to
`6.6585895e-7` score units, or about `1.273108 B/flip`.

## Verification

- `ruff check` and `ruff format --check` passed for all six JS8 Python files.
- `py_compile` passed for all six JS8 Python files.
- Focused tests: `3 passed`.
- Strict payload-retention audit over all six files: zero findings.
- Review pass 1 found and fixed two resumability/provenance gaps: future scorer
  entry now rehashes the 3.662 GB base raw, and a resume after completion
  validates/reuses `FULL_RESULT.json` instead of overwriting it. A direct reuse
  check preserved SHA-256 `4c764491...`.
- Review pass 2 challenged the shared assumption that an adapter trained for
  uniform application remains useful after post-hoc scalar edge gating. The
  n600 necessary-condition failure rejects that assumption for this instance;
  violating it through gate-aware joint training is the #982 hypothesis. The
  pass also made the CPU-versus-T4 field boundary explicit.
- Two complete `review_tracker.py` scan/whole-file mark passes finished with
  every entity in all six JS8 Python files marked reviewed.

## NEXT_IF_RESUMED

- `FIRED_EXISTING_OWNER_NO_DUPLICATE` — owner: trained-receiver #982 / RX2; consumer store: `/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac`; fire trigger: the already-live governed MC36-label HPAC training reaches its terminal QAT checkpoint; action: consume its export-fit-encode-receiver identity race and do not launch a duplicate JS8 post-hoc gate.
- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN #978 scorer-lane router; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/multitoken_978/ddm_mt1_20260814/optimal_form_r2/retained/t4_sign_gate_r1`; fire trigger: higher-priority exact rows are terminal, the #978 T4 lane is free, and every sealed request and payload SHA matches; action: fire only the sealed MT1 T4 sign gate and advance to a separately claimed resumable joint train only if `positive_t4_sign=true`.

## LIVE-HYPOTHESES

- A receiver trained with the Road-hub gate active may preserve EC2's corrected
  cells without the terminal Road collateral. This is plausible because the
  present gate is post-hoc over an adapter optimized for uniform application,
  so the adapter has never learned the gated distribution.
- A jointly learned multi-token conditioner may distinguish boundary geometry
  that a four-neighbor class-pair scalar cannot. This is plausible because EC1
  established highly predictive decoded context while BG2 closed only the
  low-rank frame-state gate.

## DEAD-ENDS

- Uniform reuse of the terminal EC2 adapter is closed: it fixed 12,075 base
  errors but introduced 52,854, for net -40,779 flips at the full T4 endpoint.
- A generic bilinear gate on the existing 8-D frame state is closed at the
  formulation scope by negative held-out incremental R2 and the exact BG2
  collateral census.
- Standalone additive implicit-context rate calibration is closed for this
  representation: SR1 found only a 2-byte saving.
- Direct sparse Road-Lane token events are closed on the measured GV2 instance:
  collateral blast radius overwhelmed target reach.
- The tested post-hoc Road-hub scalar gate is closed at INSTANCE scope on the
  charter's local admission rail: the optimal screened scale recovered 7 flips
  but needed 1,373.803 to pay its 1,749-byte cost at n600. No T4 family claim
  follows from that closure.
- QS5 source-restoring compensation is folded for this exact JS8 object because
  Seg plus rate is already positive; compiling it cannot change either term.
