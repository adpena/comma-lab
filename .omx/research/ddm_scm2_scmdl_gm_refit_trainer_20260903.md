# DDM SCM2 fitted cross-group G/M trainer — terminal formulation refusal

## Result

**Typed decision: `TERMINAL-REFUSED-MISSING-EXECUTABLE-CAUSAL-GRAPH`.**
`verdict_scope:` **FORMULATION — the four-row roster requested by SCM2 under the admitted SFP1
schedule declarations.** The inputs do not define the receiver-executable schedule that the charter
requires. Candidate A has no distinct `G` edit at all. B1–B3 name a common operation and four
contexts, but supply no coordinate-to-group membership, group scan, reset semantics, causal mask,
receiver parser, or counted transport for decoder-unavailable contexts. Implementing or pricing a
row would therefore require inventing a fourth schedule, silently dropping declared contexts, or
hiding encoder-side state from the byte count. I did none of those.

This is a source-level refusal, not a performance negative. It does **not** close the nonlinear
cross-group HPAC family. No fitted-model bytes, coded-field bytes, archive bytes, distortion, score,
or outer-loop timing were measured. The pre-registered `<= 140,479.86 B` decision gate was not
reached, so no scorer fire order exists.

Axis: **[source-recall / scorer-free / no byte measurement]**. Candidate denominator: **4/4
refused before execution**. Model fits: **0**. Coder runs: **0**. Receiver parses: **0**.
Determinism repeats: **0**. Scorer runs: **0**. Modal calls: **0**.

## Source and storage preflight

All admitted retained objects were checked in place; no input was regenerated or modified.
APDataStore had **32,169,000,960 B free**, above the charter's 1.5 GiB refusal floor. The durable
machine-readable receipt is
`/Volumes/APDataStore/pact/ddm_scm2_scmdl_gm_refit_trainer/RESULT.json`, **10,355 B**, SHA-256
`eb245678a9b110b1321c38a4e1e12cef6877cad3526688a1b45cd32fa1845a33`.

| Object | Bytes | Verified SHA-256 |
|---|---:|---|
| AFR1 archive | 180,002 | `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25` |
| RC64 token stream | 113,411 | `5601d6fd792c60c176e7cb7478e6033c4ed9a7e87404582340ed3f50ed60cfe3` |
| Shipped HPAC section | 13,515 | `602115b323b0e403d08287af9b273a2d4fb23e026d83c1f6e4609ed77ef98f98` |
| A field | 117,964,800 | `e685c4bf7fbea1188b64f521487192196eaae99c8b8b335b770586ab984585fa` |
| A overlay | 792,683 | `7953c9164cc5ac4f3fa59b8715a6eaecdc9cb11e61cba58d20c3d24db38eea63` |
| B1 field | 117,964,800 | `75fe37daf8c3f615cd943a76697e9c6e8eabc56cb1c23d55a6b4251fc4553690` |
| B1 overlay | 34,430 | `9c8bcdc738bc9044e96155cae7c1ae1781b441b87cbda8c23afa0e8456e2a1f8` |
| B2 field | 117,964,800 | `656bd0c5c102109c3327eccd0c6e3a606aac44cbce7d9144396f8c171e24b76e` |
| B2 overlay | 92,409 | `9ae52f56797dbc24ae6d8f83aae53a442c3a26b0b1207308f0be93a35455de9d` |
| B3 field | 117,964,800 | `fe6a9dd8ce770e308c7c3d1903ea1e40bee44938cc836188e486eefd408f527a` |
| B3 overlay | 837,020 | `de5b85a690afb9c092a5fa860c6f0bf121020ce3b5ff88d6d55af65b8b13321a` |

The JBP1 manifest and A receipt also verified at SHA-256
`42b788b3fe8d3e26bb68db0f7e98f38a183c3f24d19493ac87826fb86b44503a` and
`fd09ba7fbd430a468a7e3069cfa14139ddd139c1c790a1c20983f861c779a3da` respectively.

## Why no real schedule can be derived from the declaration

SFP1 assigns the same `GEditSpec` to all three B rows: operation
`refit_cross_group_causal_schedule`, an initially empty `transition_order`, contexts
`source_class`, `target_class`, `boundary_distance`, and `position_cell`, and
`refit_required=true` (`experiments/ddm_sfp1_scmdl_field_proposal_prep.py:303-309`). It forbids a
stored side stream (`:350-360`). After materializing each whole field, it fills
`transition_order` only by sorting the observed old-to-new transition counts (`:639-644`). That is
a label-frequency order, not a decoding schedule.

The deployed HPAC object exposes the missing requirements. Its integer convolution masks depend on
an exact `delta`-derived causal offset (`submissions/semantic_joint_ctxmix/cpr1/hpac_integer.py:73-83`).
The receiver instantiates the model with fixed `HPAC_PATCH` and `HPAC_DELTA`, derives
`grid = columns + HPAC_DELTA * rows`, and visits each resulting mask in order
(`submissions/semantic_joint_ctxmix/cpr1/inflate.py:253-287`). A different group plan therefore needs,
at minimum, an exact coordinate-to-group map, scan order, matching model mask, serialized grammar,
and parse-back binding. SFP1 defines none of them.

Three of SFP1's four contexts are also unavailable when the replacement symbol must be decoded:

| Declared context | Available to the receiver before the symbol/group? | Reason |
|---|---:|---|
| `source_class` | No | It is the class in the pre-edit field, not necessarily previously decoded replacement state. |
| `target_class` | No | It comes from the retained scorer-side terminal argmax used to construct the edit. |
| `boundary_distance` | No | It is computed from the pre-edit field; no receiver derivation or counted map is declared. |
| `position_cell` | Yes | Coordinates are deterministic, but this single context does not define the other three or the group plan. |

The registered decoder-causal transport law gives the operational test: the decoder must reproduce
the exact context class at the causal instant when the model consumes it. A correlated proxy is not
enough. GMF1 had already applied that test to the same three declarations and closed **3/3 at the
formulation boundary**, explicitly leaving the nonlinear HPAC family open
(`.omx/research/ddm_gmf1_fitted_crossgroup_gm_verdict_20260901.md:5-19`). DCC1 independently records
that SFP1's source, target, and boundary labels must be removed, rewritten, or counted before a fitted
row exists (`.omx/research/ddm_dcc1_decoder_causal_conditioning_verdict_20260901.md:81-91`). The new
charter says to build the missing executable, but it supplies no new graph semantics and does not
supersede either causal finding.

Candidate A introduces a second independent blocker. Its retained JBP1 receipt says: **“no distinct
G edit declared by XOV1 candidate 3; shipped G/M used physically.”** SCM2 forbids repricing A under
shipped G/M, asks for A under fitted G/M, and says the only schedules are the SFP1 declarations and
that no fourth may be invented. A therefore has no admissible schedule ID.

The model export machinery is not the load-bearing stop. The current `integer_model_io.py` is a
fixed-schema loader rather than a complete training/export command, but JF1 and CL1 provide actual
same-schedule 60-epoch HPAC training and the exact IHS1 pack path. Those could be reused after a valid
schedule exists. Conversely, the charter's GB1 reference does not define such a refit: its source
fixes the shipped 64-by-`delta=2` plan and explicitly says **NO TRAINING**
(`experiments/ddm_gb1_groupbin8_conditioning.py:14-45`).

## Typed candidate rows

`NOT MEASURED` is not a zero-byte value, and `NOT RUN` is not an identity pass.

| Candidate | Schedule ID | Model bytes | Token bytes | Framing bytes | Archive bytes | Delta vs 180,002 B | Demand fraction | Joint pool vs 87,403.86 B | Parse-back | Repeat | Seconds | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| A `xov1_bhw5506` | **ABSENT** | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT ADJUDICATED | NOT ADJUDICATED | NOT ADJUDICATED | NOT RUN | NOT RUN | NOT RUN | `TERMINAL-REFUSED-MISSING-SCHEDULE` |
| B1 `sfp1_p01_atlas24_boundary1` | `refit_cross_group_causal_schedule` — declaration only | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT ADJUDICATED | NOT ADJUDICATED | NOT ADJUDICATED | NOT RUN | NOT RUN | NOT RUN | `TERMINAL-REFUSED-MISSING-EXECUTABLE-CAUSAL-GRAPH` |
| B2 `sfp1_p02_atlas64_boundary1` | `refit_cross_group_causal_schedule` — declaration only | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT ADJUDICATED | NOT ADJUDICATED | NOT ADJUDICATED | NOT RUN | NOT RUN | NOT RUN | `TERMINAL-REFUSED-MISSING-EXECUTABLE-CAUSAL-GRAPH` |
| B3 `sfp1_p03_mi1_patch12_boundary1` | `refit_cross_group_causal_schedule` — declaration only | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT ADJUDICATED | NOT ADJUDICATED | NOT ADJUDICATED | NOT RUN | NOT RUN | NOT RUN | `TERMINAL-REFUSED-MISSING-EXECUTABLE-CAUSAL-GRAPH` |

JBP1's **separate, previously measured** A-under-shipped-G/M row remains 177,052 B, delta −2,950 B,
7.464170715452149% of the 39,522.14 B demand, and refused by +36,572.14 B. It is not a fitted-G/M
row and is not relabelled as one here.

## Decision and economics

The archive-ceiling rule is **NOT REACHED**: there is no fitted model, exact token stream, complete
archive, receiver parse, or determinism repeat. Consequently neither `REFUSED-WITH-SHORTFALL` nor
`FIRE-ORDER` is supportable for performance. The only honest decision is the typed formulation
refusal above.

Seconds per row and suffix-replay amortization are **NOT MEASURED** because no row could enter the
outer loop. The bounded-reset route remains out of scope and queued; it was not built.

## RECALL EVIDENCE

I searched the full local research corpus rather than stopping at the charter seeds. Content searches
covered `.omx/research/`, `.omx/tmp/arm_receipts_local/`, the canonical research index and `sub015_DAG`
surfaces, design/SPEC files, source, task-ledger and hot-state rows. Queries included
`refit_cross_group_causal_schedule`, `MISSING_EXECUTABLE_GM_REFIT`,
`source_class target_class boundary_distance`, `cross-group schedule refit`, `HPAC_PATCH HPAC_DELTA`,
`group_masks`, `IHS1`, `integer model export`, `#1374`, `GMF1`, and `decoder-causal`. I also generated
the canonical-equations listing with `.venv/bin/python tools/list_canonical_equations.py --json` and
checked the relevant causal-transport and model-plus-stream pricing laws.

Beyond the charter's seeds, the search found:

- GMF1's exact source-level closure of the same 3/3 SFP1 declarations. This changed the work from an
  assumed missing-code implementation into a required supersession check.
- DCC1's registered causal-time availability criterion. This made encoder-side labels inadmissible
  unless removed or carried in counted bytes.
- JF1/CL1's real HPAC training and IHS1 packing precedent. This shows that training/export tooling is
  available and is not the primary blocker.
- The GB1 source contradiction: it uses the shipped plan and no training, so it cannot serve as the
  promised different-plan retraining template.
- CCS1's already-measured receiver-executable sparse schedule: 607,228 B for its exact stream and
  664,770 B for its archive. This remains a closed table-model instance, not the parameter-sharing
  HPAC declaration requested here.

No searched source defined an executable group-membership grammar, parser, or causal mask for the
exact SFP1 operation, and none supplied an A schedule. That bounded absence is why implementation
stopped before a false row existed.

## Custody and handoff boundaries

The handoff follows `docs/operating_manual_craft_handoff.md`: the stop is typed, its consumer and fire
condition are explicit, and no follow-on is left as an ownerless “noted” item. The frozen
`submissions/semantic_joint_ctxmix/` and `upstream/` trees were not modified. The JBP1 and JC1 SSD
inputs remained read-only. No detached process, scorer-lane claim, scorer job, Modal call, model fit,
coder run, or receiver run was launched.

Scientific-payload denominator: **0**; retained scientific-payload numerator: **0**. No trained
weights, coded stream, candidate archive, or other scientific payload existed in memory to discard.
The only new artifact is the durable refusal receipt above.

## NEXT_IF_RESUMED

- **Disposition `QUEUED-WITH-A-FIRE-ORDER`; owner: MAIN-assigned SCM2 schedule-schema successor;
  consumer store: `/Volumes/APDataStore/pact/ddm_scm2_scmdl_gm_refit_trainer/`; fire trigger:** a
  versioned counted schema explicitly defines schedules for A and B1–B3, coordinate/prefix group
  membership, group order, reset semantics, causal mask, receiver parser, and transport or removal of
  every decoder-unavailable context, and explicitly supersedes GMF1/DCC1. Then train/export the real
  integer HPAC per field and exact-price all four receiver-closed repeat-identical archives.
- **Disposition `QUEUED-STRUCTURAL-LOCALITY`; owner: MAIN allocator; consumer store:
  `/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/`; fire trigger:** a separately chartered,
  counted bounded-reset grammar proves exact full-state-versus-restarted receiver identity while
  preserving every adaptive RC64 state component.

## LIVE-HYPOTHESES

- A coordinate-derived group plan with a parameter-sharing HPAC remains plausible because both sides
  can know group membership and causal order without a video-derived membership stream.
- A small counted schedule grammar may pay if useful cross-group changes repeat spatially, but it must
  reproduce the exact decoder context class at the instant of use.
- Joint X/G/M training may beat fitting M after freezing an encoder-side edit because the field can
  avoid changes whose causal transport costs exceed their coding benefit.

## DEAD-ENDS

- Treating SFP1's transition-frequency order as the decoding schedule: it contains no symbol groups,
  coordinate membership, receiver scan, reset rule, or mask.
- Using `source_class`, `target_class`, or `boundary_distance` as free contexts: they are unavailable
  before replacement decoding and no counted transport is declared.
- Pricing A under an unspecified fitted schedule: A declares no `G`, and adding one would invent the
  forbidden fourth schedule.
- Using GB1 as a different-group retraining implementation: its source fixes the shipped plan and
  explicitly performs no training.
- Reusing the CCS1 sparse table: its exact 607,228 B stream is already closed, and the charter requires
  a parameter-sharing HPAC rather than that table.
- Substituting shipped G/M, a float proxy, or estimated lengths: each prices a different object and
  cannot support a real fitted-G/M row.

OWN-VEHICLE FRONTIER UNMOVED: S 0.14797617125559104 @ 180,002 B [contest-CUDA T4, n600], AFR1 archive SHA-256 cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25.
