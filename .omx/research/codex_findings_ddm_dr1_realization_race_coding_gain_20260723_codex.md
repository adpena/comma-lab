# Codex findings — DDM DR1 realization race and coding gain — 2026-07-23

`research_only=true`

`execution_allowed=false`

`score_claim=false`

`evidence_axis=[macOS-CPU frozen-scorer advisory]`

`pointer_moved=false`

`main_landing_review_required=true`

## Outcome

The full n600 application-stage race selects exact post-quantization int8
placement for this SHA-bound horizon and move set. It lowers the realized joint
objective by `-0.12083155237745034` for `+235` exact archive bytes. Moving the
same admitted corrections to camera-Q8 before final uint8 nearly erases their
Seg benefit and loses after rate. Ordered Bayer diffusion and the
exact-resize-derived block sigma-delta contender change no camera uint8 value
relative to uniform pre-uint8 placement.

This is an instance result, not a family result. The winning stage row is also
not yet composed-candidate admissible: margin-priced record tolerance and the
ordered cross-stream conditional-byte matrix remain unmeasured.

| application stage | exact bytes | Seg errors | `d_seg` | `d_pose` | joint delta vs control | stage admission |
|---|---:|---:|---:|---:|---:|---|
| horizon control | 135,328 | 3,240,528 | `0.027470296224` | `163.061327281443` | — | control |
| post-int8 lattice | 135,563 | 3,097,769 | `0.026260113186` | `163.061571784114` | `-0.12083155237745034` | accepted |
| pre-uint8 uniform | 137,630 | 3,240,299 | `0.027468354967` | `163.060485284886` | `+0.001234424569241465` | rejected |
| pre-uint8 Bayer8 | 137,633 | 3,240,299 | `0.027468354967` | `163.060485284886` | `+0.0012364221461008316` | rejected |
| pre-uint8 resize-null sigma-delta | 137,650 | 3,240,299 | `0.027468354967` | `163.060485284886` | `+0.0012477417483039083` | rejected |

The positive row's first rung is an n600 rerun after margin-priced pruning of
post-int8 correction records, with each stream conditioned on all earlier
decoded streams and the pairwise redundancy matrix emitted.

## Equal-rate falsifier

The three pre-uint8 correction-program payloads have equal byte length, but
their original ZIPs differ because their manifest strings differ. Uniform and
Bayer were therefore equalized to 137,650 counted bytes with outer ZIP
comments. Exact decoded-member proof plus full-n600 receiver replay proves each
original and equalized archive produces the same camera uint8 SHA-256:
`62ac2e91065c50b840cac5b851046e51140da2483f6aef4bce075675a336c058`.

At equal exact archive bytes:

- operational continuous-vs-uint8 distortion gap: `0.12068964649058955`;
- distortion recovered by ordered dither: `0`;
- recovery fraction: `0`, below the preregistered `0.10` threshold.

Verdict:
`INSTANCE_NEGATIVE_FOR_TESTED_MOVE_CLASSES`. This does not generalize to
another diffusion rule, correction support, formulation, family, or paradigm.

## Exact-resize-derived contender

The late geometry directive was consumed as a receiver mode, not as an
uncustodied heuristic. The implementation derives each disjoint 2x2
bilinear-resize numerator from the exact downstream scorer resize operator and
chooses adjacent integer lattice values that minimize its numerator residual,
directing the remaining error into that operator's nullspace. The bicubic
source-to-camera step is already inside the SHA-bound base; only the downstream
camera-to-scorer operator is actuated by this correction stage.

At equal 137,650-byte archives, generic Bayer8 and the derived mode both recover
exactly zero of the operational continuous-vs-uint8 gap and produce the same
full-n600 uint8 camera hash. The preregistered strict prediction that the
derived contender would recover more is therefore `FALSIFIED_INSTANCE`.

The scope is deliberately narrower than the literal MAIN prediction:

- the available generic arm is Bayer8 ordered dither, not Floyd-Steinberg;
- no counted or receiver-derivable rank-4 margin field is bound, so the
  within-range Fisher tie-break is `BLOCKED_NOT_IMPLEMENTED`;
- other diffusion supports and kernels remain open.

No general sigma-delta or description-family negative follows.

## SDWL1 context fold

Exact decode, causal-context re-encode, and parse-back reproduce the selected
payload byte-for-byte:

- control: 68,464 bytes;
- context fold: 68,464 bytes;
- delta: 0 bytes;
- exact parse-back: true.

The selected syntax already uses decoder-derived same-channel left/upper
contexts. The registered 89,161-byte gain belongs to a different innovation
symbol stream and cannot be subtracted from a complete SDWL1 object. This ties,
rather than loses to, the current syntax row. Pixel-exporter realization is
still owed. Exact fact coding also fails the later sensitivity-priced tolerance
doctrine, and `SDWL1 | decoded G4` pairwise bytes remain unmeasured.

The first rung is margin-priced CELL/SEPR/SCRW quantization followed by exact
syntax parse-back and one receiver-export smoke.

## Persistent Undrivable transfer

The delegated prompt's “unmeasured twin” premise was stale. The SHA-bound DV1
ledger already settles both requested semantic-cell rows, so they were extracted
without remeasuring a settled result:

| scope | Undrivable described | described fraction | static fraction of described | global net errors closed | counted bytes |
|---|---:|---:|---:|---:|---:|
| standalone | 166,085 | `0.5613179489259305` | `0.9930938977029834` | 865,179 | 283 |
| persistent + events | 165,882 | `0.5606318692460559` | `0.9931035314259534` | 2,385,747 | 132,914 |

The joint section costs 308 bytes given the event archive versus 283 bytes
standalone, so the measured ordered-pair redundancy value
`bytes(B)-bytes(B|A)` is `-25` bytes. These are semantic-cell arbitration rows;
RGB receiver survival, Pose collateral, and contest-axis score remain open.
The first rung is exact RGB realization of the persistent partition with
Undrivable and Pose remeasured on the same archive.

## Four-clause stream audit

Every touched row records:

1. scorer visibility;
2. sensitivity-priced tolerance;
3. descriptive form, inherently compact DOF, and coder gain;
4. single-owner facts, cross-stream conditioning, dimension homes, and
   correction-delta status.

Missing measurements fail closed. In particular, the application-stage
template/sparse/track pairwise redundancy matrix and SDWL1 conditioned on the
decoded G4 stream are blockers, not inferred zeroes.

MAIN's optional pre-prox third leg is `BLOCKED_NOT_MEASURED`: this delegated
base predates reviewed receiver commit `1c55f78063` and lacks
`src/tac/optimization/ddm_runtime_receiver.py`. This is a custody blocker only,
not a formulation or family verdict.

## Triality and artifacts

- DSL: `.omx/research/configs/ddm_dr1_realization_race_coding_gain_n600_20260723.json`
  (typed, SHA-bound, seed 1234, batch size 16).
- DAG: `.omx/research/ddm_dr1_realization_race_coding_gain_FEED_20260723.json`.
- Equations: strict realized joint admission reuses the existing pure-priced
  objective; the equal-rate recovery quotient is an operational instance
  diagnostic. No new general law landed, so no canonical-equations row was
  added.
- Machine receipt:
  `.omx/research/ddm_dr1_realization_race_coding_gain_n600_20260723/receipt.json`
  (SHA-256
  `a84112f0fc0063e43711a0e9c0777a51abac85795b990281ca08899b122a1104`).
- Post-int8 archive SHA-256:
  `ad37c5b97b1cab051b8f8d767cae7099c7248a10ccc52691dbad5c14b543c29c`.
- Equal-rate uniform archive SHA-256:
  `86a9bea7c94c74c89eaa1afed006f810a7735e42f3bab3f859780d7913a0f984`.
- Equal-rate Bayer8 archive SHA-256:
  `c893bfdc7b755a3715910a11cda1c54e32847f41fbdd35f4f292087e34b7590a`.
- Equal-rate resize-null sigma-delta archive SHA-256:
  `ab35a38e5467995d5dac418f43956676d96eed56486ca3982ef27145c94b5aef`.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, and
  `docs/operating_manual_craft_handoff.md`;
- all SHA-bound inputs named by the typed config;
- `.omx/state/lane_registry.json` and
  `.omx/state/subagent_progress.jsonl`;
- MAIN inbox directives at `16:36:35Z`, `16:45:43Z`, `16:59:16Z`, and
  `17:13:39Z`.

MAIN landing review must verify input and archive hashes, the 152 preserved
batch receipts plus 38 preserved receipts for the derived arm, the equal-rate
camera hash, all four audit clauses, narrow
verdict scopes, and advisory-only flags. No merge, launch, dispatch, score
promotion, or pointer move is authorized by this arm.
