# DDM GMF1 fitted cross-group G/M verdict — RECALL-CLOSED

## Result

**Typed verdict: `RECALL-CLOSED`.** `verdict_scope:` **FORMULATION — the current SFP1
no-side-stream `G`-edit schema on the three pinned materialized fields.** The schema names
`source_class`, `target_class`, `boundary_distance`, and `position_cell`, but it does not define a
receiver-causal group-membership rule, a serialized schedule/model grammar, or a decoder binding.
Three of those four named values are not available before the replacement field is decoded. A
stage-1 fit would therefore require inventing a different mechanism or hiding video-derived
membership outside counted bytes. I did neither.

This is a source-level formulation closure, not a performance negative. I measured no fitted model
bytes, no coded-field bytes, no distortion, and no score. The nonlinear cross-group HPAC family
remains unmeasured and is not claimed closed.

Axis: **[source-recall / scorer-free / no byte measurement]**. Denominator: **3/3** retained SFP1
proposals have the same missing receiver-causal contract. Scorer runs: **0**. Modal calls: **0**.
Model fits: **0**. Coder runs: **0**.

## Stage 0 — prior-results race adjudication

| Prior result | Mechanism actually priced | Same object as the requested fitted schedule? | Adjudication |
|---|---|---:|---|
| RR9 | Legal within-group permutation under the shipped HPAC; full n600 stream 113,777 B → 113,777 B | No | RR9 says cross-group change means “training a different model” and explicitly leaves re-architected schedule geometry untouched. It closes reorder, not this fitted model. |
| MI1 | Counted probability-model packet **added** to the shipped model, plus code-length screens | No | Its 47.4× law is add-on economics. It cannot be borrowed as replacement economics, and its cross-group conclusion is not a real fitted-model/coded-stream row. |
| LM1 | Replacement by discrete-context models and linear/logistic parametric models over causal token neighbourhoods | No | LM1 explicitly says a nonlinear network or differently architected HPAC is not closed. |
| JF1, found beyond the charter seeds | Full 60-epoch refit of the same HPAC schedule on its retained field ladder | No | Its null row measured 13,463 B model + 114,143 B stream = 127,606 B, but it keeps the shipped schedule/causal geometry. It is a fit result, not a fitted cross-group schedule. |

The mandatory race therefore did **not** supply a same-object performance closure. The stop comes
from the requested object's own source contract, below.

## The receiver-causality gate

SFP1's prep source constructs each proposed field from three independent encoder-side objects:

- `x`: the pre-edit AFR1/DX2 dense field;
- `target`: the retained CUDA terminal scorer argmax;
- `boundary`: a retained boundary-distance map computed from pre-edit `x`.

The selector uses `boundary <= 1` and `x != target`. Its transition ledger then defines
`source_class` as `old = x[mask]` and `target_class` as `new = target[mask]`. The transition order is
merely the frequency ordering of those encoder-observed `old→new` pairs. SFP1 also requires
`stored_side_stream = false`.

That is enough to materialize a whole changed field, but not enough to decode it under the declared
schedule:

| Required context | Decoder-known before the replacement symbol/group? | Reason |
|---|---:|---|
| `source_class` | No | It is defined from the pre-edit field `x`, not from previously decoded state of the replacement field. |
| `target_class` | No | It is defined from retained scorer argmax used at field construction. It is not a pre-symbol decoder input. |
| `boundary_distance` | No | The retained map belongs to pre-edit `x`; the declared G edit carries no map or derivation rule. |
| `position_cell` | Yes | It is a deterministic coordinate function, but using it alone would omit three mandatory contexts and change the mechanism. |

The durable JBP1 blocker independently found the same structural gap for **3/3** proposals:
`declared_executable_keys = []`, no RXC1/JG2 consumer for the operation, and
`MISSING_EXECUTABLE_GM_REFIT`. Inspecting SFP1's producing source sharpens that blocker: the named
contexts are encoder-side edit labels, not an executable causal factorization.

This is also why a trained model is learned by construction if a repaired stage 1 is ever chartered;
there is no closed-form substitute for the fitted coding model itself. The byte arithmetic remains
exact. The sharp-optimum law does not close a schedule-basin change, but it also does not waive the
need to specify a decodable basin before fitting it.

## Stage 1 — fitted-pool table

Stage 1 was not entered because the recall/source gate failed before compute. `NOT RUN` is not a
zero-byte result.

| Proposal | Train/test receipt | Fitted model bytes | Real coded-field bytes | Total | Versus shipped 126,926 B | Versus allowance 87,403.86 B |
|---|---|---:|---:|---:|---:|---:|
| B1 `sfp1_p01_atlas24_boundary1` | NOT RUN | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT ADJUDICATED | NOT ADJUDICATED |
| B2 `sfp1_p02_atlas64_boundary1` | NOT RUN | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT ADJUDICATED | NOT ADJUDICATED |
| B3 `sfp1_p03_mi1_patch12_boundary1` | NOT RUN | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT ADJUDICATED | NOT ADJUDICATED |

Reference arithmetic only, not a new measurement: the charter's shipped replacement pool is
13,515 B model + 113,411 B RC64 stream = **126,926 B**; the body allowance is **87,403.86 B**. MI1's
47.4× add-on law does not apply directly to this replacement pool in either direction.

## Stage 2 — binding

Not entered. There is no model, stream, parser, parse-back receipt, or `GM-READY` row. The three
retained SFP1 field hashes remain unchanged:

| Proposal | Bytes | Verified SHA-256 |
|---|---:|---|
| B1 | 117,964,800 | `75fe37daf8c3f615cd943a76697e9c6e8eabc56cb1c23d55a6b4251fc4553690` |
| B2 | 117,964,800 | `656bd0c5c102109c3327eccd0c6e3a606aac44cbce7d9144396f8c171e24b76e` |
| B3 | 117,964,800 | `fe6a9dd8ce770e308c7c3d1903ea1e40bee44938cc836188e486eefd408f527a` |

The unchanged base field remains 117,964,800 B, SHA-256
`cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`.

## RECALL EVIDENCE

I searched the full local research corpus, not only the charter seeds:

- Content queries across `.omx/research/`, `.omx/tmp/arm_receipts_local/`, the canonical research
  index/DAG surfaces, designs/specs, and hot-state/task rows included
  `refit_cross_group_causal_schedule`, `cross-group.*(refit|schedule)`,
  `source_class.*target_class`, `MISSING_EXECUTABLE_GM_REFIT`, `delta`, `patch`, and task `#1374`.
- I generated the canonical-equations JSON with `tools/list_canonical_equations.py --json` and
  searched it for cross-group/schedule/model-direction laws. The relevant direction-dependence law
  says model bytes and stream bytes must be priced jointly; it supplies no missing schedule.
- I inspected RR9, MI1, LM1, JBP1, SFP1, RXC1, JC1, the SFP1 prep source, the retained candidate
  manifest, and JBP1's machine-readable blocker at source.

Beyond the charter's seeds, the search found JF1's retained full-reference refit and its physical
model/stream payloads. That result changed the recall table but not the verdict because it retains
the shipped group schedule. More importantly, source inspection of SFP1 changed the plan: instead
of treating JBP1's gap as merely “code not written,” it showed that the declared four-context object
has no decoder-causal semantics to implement. I therefore stopped before creating a convenience
subset or a falsely named fitted row.

I did not find an executable implementation, serialized schema, receiver parser, or prior fitted
row for the exact SFP1 cross-group declaration in those searched scopes.

## Custody and boundaries

- APDataStore free space at start: **51,784,843,264 B**; required reserve: **1,073,741,824 B**;
  preflight passed.
- Durable receipt: `/Volumes/APDataStore/pact/ddm_gmf1_fitted_crossgroup_gm/RECALL_CLOSURE.json`,
  5,136 B, SHA-256 `95f90363ea4d58b52bc00cd5370a7996dc3502b971f203d3aded6a6e71b17598`.
- Materialized-payload denominator: **0**; retained-payload numerator: **0**; scalar-only
  measurements: **0**. No model or coded payload ever existed in memory to discard.
- The JBP1 store and `upstream/` remained read-only. No scorer lane was requested. No detached job
  or Modal dispatch was launched.

## GESTALT-DELTA

The charter predicted a likely performance closure near the shipped 126,926 B pool. The source
contract closes earlier and more narrowly: there is not yet one decoder-defined fitted object whose
performance can be measured. This does not strengthen the prior against nonlinear HPAC; it removes
a falsely specified row from the measurement queue until the receiver semantics are made explicit.

## NEXT_IF_RESUMED

- **Disposition: `QUEUED-WITH-A-FIRE-ORDER`. Owner: task #1374 SCMDL causal-state/model builder assigned by MAIN. Consumer store: `/Volumes/APDataStore/pact/ddm_gmf1_fitted_crossgroup_gm/`. Fire trigger: a versioned schema maps all four named contexts to decoder-available values before every coded symbol/group, serializes and counts every video-derived schedule/model byte, defines the receiver parser and parse-back identity contract, and preserves the three pinned SFP1 field hashes. Action: retain that executable contract first, then issue a fresh held-out stage-1 fit charter.**

## LIVE-HYPOTHESES

- A repaired schedule based on previous decoded replacement-field classes, causal boundary state,
  and deterministic position cells may capture some of SFP1's transition structure without an
  address side stream. This is plausible because RR9 and LM1 both leave nonlinear re-architected
  HPAC open, but it is a new formulation and must be named and priced as such.

## DEAD-ENDS

- The current SFP1 four-context/no-side-stream declaration is closed: its transition labels come
  from encoder-only pre-edit/scorer objects and do not define decoder group membership.
- A fixed-G/M re-encode is closed as a stand-in: JBP1 already showed it would price a different
  mechanism.
- A position-only fit is closed as a stand-in: it drops three mandatory contexts and would turn a
  mechanism reduction into a falsely named result.
- RR9's within-group reorder cannot be retried as this fit: it is exactly byte-neutral at full n600
  and explicitly excludes schedule rearchitecture.
- MI1's 47.4× add-on law cannot be used to admit or reject this replacement: the accounting objects
  differ.
- LM1's discrete-context and linear/logistic replacement families are closed at their measured
  scope; they do not close a nonlinear schedule model.

Own-vehicle frontier: **S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600], AFR1 archive SHA-256 `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25` — UNMOVED.**
