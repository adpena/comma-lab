---
schema: ddm_dcc1_decoder_causal_conditioning_verdict.v1
date: 2026-09-01
arm: ddm_dcc1_decoder_causal_conditioning
status: DERIVATION-COMPLETE__ONE-CAUSAL-SCMDL-RUNG-QUEUED
axis: "[source proof + scorer-free exact-receipt synthesis]"
score_claim: false
pointer_moved: false
scorer_runs: 0
modal_calls: 0
computed_payloads: 0
canonical_equation_status: FORMALIZATION_PENDING
---

# DCC1 — decoder-causal conditioning is a transport obligation, not a free modelling choice

## Result

**Verdict: `LAW-DERIVED / CENSUS-CLOSED / ONE-FIRST-MEASUREMENT-QUEUED`.** A coder may call
conditioning free only when the decoder can reproduce the exact probability-context equivalence
class before decoding the symbol or group that uses it. QX2 and SFP1 violate this rule: their cheap
contexts come from encoder-only C1, pre-edit fields, scorer argmax, or pre-edit boundary maps. QX4
and GMF1 independently show the two lawful repairs: condition on decoder-produced state, or transmit
and count enough side information to reproduce the missing context. The former is still too large in
QX4; the latter is catastrophically too large in QX3.

The bounded census contains **11 representations: 8 satisfy the transport rule by construction and
3 violate it**. Six satisfying rows have real-coder evidence. Some satisfying rows still die on
economics: QX4 is at least 9,342 B over its archive envelope, OC2 saves only 2 B, and G4's two
decoder-derived charts are either 263,647 B above the current full-archive cap before container costs
or lose 192,417 B versus their control. Causality is necessary, not sufficient.

The ranked first successor is a **repaired causal G/M schedule on the unchanged AFR1 field** whose
context is previous decoded classes, a boundary state updated from the decoded prefix, and
deterministic position cells. It is not the closed SFP1 four-label schedule. The first future
measurement is one full-n600, scorer-free, payload-retaining exact rate row on the byte-identical
AFR1 field after a receiver schema exists; no scorer, evaluator, training run, or payload
materialization occurred in DCC1.

## Canonical conditioning-transport law

For coded symbols or groups `X_i`, let:

- `D_<i` be the complete receiver state before `X_i` is decoded;
- `p_i` be public deterministic coordinates and traversal state;
- `C_i` be the encoder's proposed conditioning value;
- `E_i(C_i)` be the equivalence class of conditioning values that select the same integer CDF,
  alphabet, group membership, or parse action for `X_i`;
- `T` be any counted side message available to the receiver before `X_i`.

The formulation is receiver-closed exactly when a declared function `phi_i` satisfies

```text
E_i(C_i) = phi_i(D_<i, p_i, T)       for every legal prefix and every coded i.
```

It has **free conditioning** only when `T` is empty. Equivalently,

```text
H(E_i(C_i) | D_<i, p_i) = 0.
```

If this conditional entropy is nonzero, the encoder has only three honest choices:

1. transmit and count a sufficient `T` such that
   `H(E_i(C_i) | D_<i, p_i, T) = 0`;
2. replace the conditioning with a decoder-native statistic; or
3. withdraw the formulation because the receiver cannot select the encoder's code.

For lossless carriage of the missing context class, the registered
`wyner_ziv_decoder_side_information_conditional_entropy_savings_v1` supplies the optimistic lower
bound

```text
B_T >= ceil(H(E(C) | D, p) / 8).
```

Finite coder, model, grammar, header, and archive costs are additional. The existing
`argmax_cell_identity_ideal_bytes_v1` likewise remains only a known-site self-information floor; it
does not waive site grammar or receiver transport. The new operational fork is therefore stronger
than merely saying that side information is correlated: **the decoder must possess the exact context
class at the exact causal time it is consumed.** Reproducing raw `C_i` is unnecessary when several
values select the same integer CDF, but reproducing only a correlated proxy is insufficient.

This law predicts both source closures without hindsight:

- QX2 uses a C1 field that the QX1 receiver does not produce; therefore its 22,661 B price has no
  receiver semantics until a bridge is counted. QX3 measures that bridge at 510,404 B.
- SFP1 uses source/target/boundary labels that are unavailable before the replacement field is
  decoded; therefore no fitted row exists until the schedule is rewritten or those labels are
  counted. GMF1 closes 3/3 current proposals at that source boundary.

`# FORMALIZATION_PENDING: decoder_causal_condition_transport_is_a_new_operational_domain_extension_of_wyner_ziv_decoder_side_information_conditional_entropy_savings_v1; this charter permits only the verdict memo, while its qx3_qx4_gmf1 source memos are unlanded sibling working-tree artifacts, so mutating the canonical registry here would either exceed the one-deliverable boundary or absorb sibling custody; register the extension after those SHA-pinned anchors land.`

## Source receipts for the law

| source | content SHA-256 | load-bearing receipt | fact consumed |
|---|---|---|---|
| `ddm_qx3_receiver_closure_20260831.md` | `ac893d741fc34f70c74bf6f0fbe936b0d01c8e9d5e4484720f1df7a143369136` | `/Volumes/APDataStore/pact/ddm_qx3/RESULT.json`, 33,964 B, SHA `f9a71967ec01aa8905aeb31806f29ebb40a8c9729d0db9c58d36a02a540d7867`; exact complete archive `/Volumes/APDataStore/pact/ddm_qx3/retained/complete/archive.zip`, 624,296 B, SHA `5be6693516348f2a25c87fcea65f205477f339d6090c64636ef1c4b98531901c` | QX2 C1 differs from decoder state at 1,669,798 / 117,964,800 sites; exact bridge is 510,404 B and +486,311 B over cap |
| `ddm_qx4_decodable_conditioning_reprice_20260901.md` | `bb98fe45060491599d39e4f3f9ffdb00587518b362ed6563c071cb0742c762d8` | `/Volumes/APDataStore/pact/ddm_qx4/RESULT.json`, 66,841 B, SHA `a147b3d08c7f485a323be3d41388f72e095ef8d3989e8a813993f0f36679d8bf`; selected archive SHA `19809991d47be7856e2aed5570bbcbecaa43e4a2252b7bad526786e27c55cf19` | decoder-native QBT makes all six forms receiver-closed; best is still 33,435 B / 147,327 B |
| `ddm_gmf1_fitted_crossgroup_gm_verdict_20260901.md` | `4317c83335bb3f66ecfa179e0ecbe757c0453cac99a3dbcabb89dc077c688662` | `/Volumes/APDataStore/pact/ddm_gmf1_fitted_crossgroup_gm/RECALL_CLOSURE.json`, 5,136 B, SHA `95f90363ea4d58b52bc00cd5370a7996dc3502b971f203d3aded6a6e71b17598` | source, target, and boundary contexts are decoder-unavailable; 3/3 proposals source-closed |
| `ddm_sg2b_falsifier_verdict_20260901.md` | `ed592b68c663df80247f3f6a14103f93931c4c7a45b38966da9863a449b4a39a` | retained p00-p03 advisory receipts named in the memo | all three SFP1 decoded-field edits worsen Seg and Pose on the matched CPU/PYAV instrument; X-alone is closed |
| shipped HPAC source | `hpac_integer.py` SHA `6e6b4f4d0b293fb60cc1b751958756a4cd6c2ce7bcff68c6f03e20277856803f`; `inflate.py` SHA `e01325d65c42223d5e1ca8169f2bef0f62ae59bdcfeabf321e681fa2cd07d4e2` | `prepare_frame_context`, `cached_context_logits`, and decode loop in `src/tac/pr130_runtime/dv1_cpu_runtime/` | current-group prefix and previous frame are decoded before use |
| HPAC calibration source | `experiments/ddm_hc1_hpac_calibration.py`, SHA `1174b87f8b99cd4d93488bc0073dcbc9ba4c88920341de9a52c7004e66a7ef47` | source comments and context reconstruction | corrector/render features are functions of decoded symbols |

The QX3 hashes above are copied from its own SHA-pinned memo because DCC1 did not mutate or re-author
those retained sibling artifacts. The complete content hashes for all repo sources used by DCC1 are
recorded in this memo.

## Representation census

`SATISFIES` means the context class is reproducible before use; it does not mean the representation
is small, beneficial, receiver-complete as an archive, or score-valid. `VIOLATES` means the declared
free context is not reproducible; the underlying family may be repaired as a new formulation.

| # | representation and conditioning source | transport verdict | measured consequence | typed disposition | receipt |
|---:|---|---|---|---|---|
| 1 | **Shipped HPAC RC64** — previous decoded frame, already-decoded current groups, deterministic group/position | **SATISFIES BY CONSTRUCTION** | shipped token stream 113,411 B + counted model 13,515 B; exact baseline, not a claimed delta | `LIVE-BASELINE` | source SHAs above; AFR1 archive SHA `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25` |
| 2 | **MI1/LB1 patch192** — absolute pixel cell from public `(x,y)` | **SATISFIES BY CONSTRUCTION** | real full-n600 archive 180,192 → 180,083 B, zero token changes | `ADOPTED-IN-LB1-LINEAGE`; magnitude small but real | `ddm_mi1_indicator_model_axis_20260824.md`, SHA `9fef082681a032e6eb522cd4e2ade321f6197130b689adc234086e477e8baf1f`; OC2 physical lineage |
| 3 | **OC2 miss_rank8** — deterministic rank from receiver model probabilities | **SATISFIES BY CONSTRUCTION** | real marginal −2 B both before and after patch192; 28 B below solo fire bar | `FAMILY-DRAINED-ON-LB1 / BANKED-RIDER` | `ddm_oc2_orthogonal_conditioning_charts_20260829.md`, SHA `f7bb2817467ff6394e63da5325f98c3d0d992f56a28c96c01f9ef270ae0838b2`; `RANK_ADJUDICATION.json` SHA `a4e22137b893647300569fd29ac53405456fba1bec8d76d94a29eec200e86ba3` |
| 4 | **G4 aggregate spatial prior** — pixel coordinate + causally decoded earlier-pair flips | **SATISFIES BY CONSTRUCTION** | selected innovation coder 401,633 B vs 490,794 B control, −89,161 B; partial stream only, container and receiver RGB excluded | `RESEARCH-WIN / FULL-ARCHIVE-ECONOMICS-DEAD` because 401,633 B alone exceeds 137,986 B | `ddm_g4.../summary.json`, SHA `1e522f92014388454ee0b01d3e4a6abd8c740cac7a4ddcbf341935cdde3a1361` |
| 5 | **G4 predictor-boundary chart** — distance bins derived from decoder predictor cells | **SATISFIES BY CONSTRUCTION** | 683,211 B vs 490,794 B control, **+192,417 B** | `MEASURED-ECONOMIC-REFUSAL` | same G4 summary SHA |
| 6 | **QX2 enumerative event coder** — exact encoder-only C1 baseline | **VIOLATES** | attractive 22,661 B section price, but receiver mismatch 1,669,798 / 117,964,800; exact repair costs 510,404 B | `FORMULATION-CLOSED` | QX3 memo/result above; C1 field SHA `02a2a3f572d6e0abf039d812330962ae8b1a44f02701661136482759e33ccf34` |
| 7 | **QX4 six event grammars** — freshly decoded QBT native field | **SATISFIES BY CONSTRUCTION** | all 17,926 events parse back; cheapest 33,435 B, complete archive 147,327 B, +9,342 B | `FORMULATION-CLOSED-ON-MAGNITUDE` | QX4 memo/result above; payload SHA `6637733eafee7d57510a3d3738d0222cddcc30fb4e034608f8058521ae9767b3` |
| 8 | **SFP1 four-label G/M schedule** — pre-edit source class, scorer target class, pre-edit boundary, position | **VIOLATES** (3/4 contexts) | no model, stream, or price exists; 3/3 proposed fields share the defect | `RECALL-CLOSED-FORMULATION` | `ddm_sfp1_scmdl_field_proposal_prep_20260901.md`, SHA `af70ab65c258b8700851bdf525ab8dc1c58b41cf34b374403e9c8e67ad48538b`; GMF1 closure above |
| 9 | **Exact GF1 born tuple conditioning** — source/scorer/boundary tuple packet | **VIOLATES AS FREE CONTEXT** | counted tuple packet 47,603 B; later all-live ideal ceiling ~613 B, 77.6× underwater versus its packet | `CLOSED-BY-CEILING` | `ddm_dds1_decoder_derivable_verdict_20260901.md`, SHA `d51f9a07c9fd2c467bcec0d1c7eb782b84e32794468934dd7c5f95549a5a7bc0`; correction SHA `13f33173c38b8ccdbcca6529976740a3fdb117ca9bc8897997dd69f742a3dc38` |
| 10 | **DDS1 weakened M-only born surrogate** — HPAC argmax + predicted boundary from counted model | **SATISFIES BY CONSTRUCTION, DISTINCT OBJECT** | n120 SCREEN wrong-half 10.8575%, but decoder-realizable all-live ceiling is only ~2.08 B | `CLOSED-BY-CEILING`; do not spend a #1374 price | same DDS1 memo/correction; v2 `RESULT.json` SHA `057b073eb874bb74e35915f5ac5939551c442b1d8bad255c5f4c5d5f20aa54d8` |
| 11 | **Repaired causal G/M schedule** — previous decoded classes + causal boundary state + deterministic position cells, first on unchanged AFR1 X | **SATISFIES BY SPECIFICATION; UNBUILT** | no new model bytes or stream bytes measured; unchanged X makes the first row fixed-distortion | `QUEUED-WITH-A-FIRE-ORDER` | GMF1 memo/closure and SG2B source above |

### Census denominator and prior-law check

- Universe requested by the charter plus the two necessary G4/DDS1 control splits: **11 / 11
  classified**.
- Transport outcomes: **8 SATISFY**, **3 VIOLATE**.
- Rows with a physical or real-coder byte outcome: **7 / 11** (HPAC, patch192, OC2, two G4 rows,
  QX2/QX3, QX4); rows without one: **4 / 11** (SFP1, exact born free-context claim, M-only real
  coder, repaired GMF1). SCREEN and ceilings are not promoted to bytes.
- Named byte-improving rows against a control: patch192, OC2, G4 aggregate, and QX2's unclosed
  section price. **3 / 3 receiver-valid wins satisfy the law.** The fourth apparent win violates it
  and disappears after receiver closure. No falsifier to the prior-law prediction was found in this
  bounded census.

## Complete law set and what it does to the ranking

The transport law is applied together with, not instead of, the campaign's binding constraints:

| law | current reading | consequence here |
|---|---|---|
| exact contest objective | `100*d_seg + sqrt(10*d_pose) + 25*B/37,545,489` | final admission is one exact archive, never a section or proxy |
| current rate corner | fixed current distortion requires `B <= 137,986`; hot-state continuous demand is −42,016 B | a rate-only first rung must clear the cap before consuming a scorer |
| zero-distortion corner | `B_max = 180,218.347 B`; AFR1 is 216.347 B under | arithmetic opens the zero-distortion corner but does not make a lossy candidate valid |
| conservative pose corner | absolute `d_pose <= 1.25e-4` on the no2 first rung | use as an early gate, not a global pose law; the allowable pose changes with bytes |
| affine round-trip law | two-body fit gives an intercept near **140,477 B** | above that operating region, token accuracy alone cannot rescue the current field; do not transfer the intercept as a theorem to a new field |
| Cross | `{byte-feasible} ∩ {distortion-feasible}` measured empty at **n=4** | composition labels do not establish a meeting point; a successor must be one parsed object |
| sharp optimum | current same-basin HPAC directions are sharply optimized | do not relabel another fixed-context tweak as SCMDL; only a new causal schedule basin remains |
| token-error amplification | historical BZ2D through-origin **1.157×** | **refuted at LB1** (about 12× wrong there); keep only as a BZ2D-scope historical receipt, use the affine reading for current-field screening |
| generator form | **2.178×** cheaper on its own bit-identical generated object | real but inseparable from the existing form's ~1.12% fit error on a foreign exact target; no transferable byte credit |
| QX4 feasibility map | six decoded-QBT forms span 33,435–38,778 B; best archive 147,327 B | new QX work must change the event object or core, not tune these six forms |

## Ranked successor objects

This is a fire ranking, not a claim that any object exists at the required score.

| rank | successor object | why transport admits it | complete-law falsifier | cheapest honest rung | disposition |
|---:|---|---|---|---|---|
| **1** | **Causal G/M schedule on unchanged AFR1 X**, using previous decoded classes, causal prefix-boundary state, and deterministic position cells | every context is present before use; learned model/schedule bytes are counted; decoded X and renderer output stay bit-identical | schema/parse-back failure closes construction; a complete archive above 137,986 B closes this fixed-distortion schedule instance | one full-n600 fitted-model + RC64 + archive encode on AFR1 X, deterministic repeat, exact parse-back, scorer-free | `QUEUED-WITH-A-FIRE-ORDER`; sole first measurement |
| **2** | **QX target-overwrite grammar on decoded QBT**, dropping the historical C1-syndrome ABI only after consumer semantics are proved | QBT and traversal are decoded; target overwrites become the counted object | semantic proof fails, or a new complete exact section exceeds 24,093 B / archive exceeds 137,985 B; no unchanged QX4 form re-enters | source-level consumer proof, then one full-n600 real-coder race preserving the 8,749 actual changes | `FOLDED-BEHIND-RANK-1`; structurally new QX owner required |
| **3** | **QBW1/QBMIX causal quotient topology + joint renderer** | positions arise by causal topology integration; any regime must be decoder-derived and every model/tag counted | deterministic parse-back failure, projected complete `B_hat>137,986`, `d_pose_hat>1.25e-4`, or `S_hat>=0.12` on the preregistered serialized rung | existing no2 seeded-random n32 serialized gate; no inherited BR2 distortion | `FOLDED-BEHIND-NEARER-EXACT-ROWS` |
| **4** | **RB1 exact changed-object renderer** | renderer conditioning is internal to the counted changed object; the transport law is neutral | complete archive >137,986 B or own-object distortion misses its D56/F64 gate | existing sealed D56 byte gate, then its owned scorer screen | `STILL-ADMISSIBLE / EXISTING-OWNER`; not duplicated by DCC1 |

The exact BR2 born archive is not a successor instance: at 106,832 B it measured
`d_seg=0.1707768843`, `d_pose=115.8374`, and advisory `S≈51.18`; this closes that archive, not the
causal quotient family. Receipt: `ddm_br2_born_object_scorer_realization_20260831.md`, SHA
`558c4741001b64a81f6c6d7f3c4b4c23254d2b60462fa321c59c6fc86273e7e5`, archive SHA
`0e2ffdfaa5fe481d481dd70a9672a67f80b9aad7648f0c775fe2956dd3a4841d`.

## What changed versus the prior rankings

| prior surface | prior leading read | transport-law delta |
|---|---|---|
| **NO1** | learned probability model, alphabet merge, and born-small were ranked as object classes | a learned model is no longer one generic row: every context must be decoder-causal and every video-derived schedule/model byte counted. Same-field fixed-context work remains sharp-optimum closed; only a new schedule basin survives. NO1's pose `1.25e-4` is retained as a corner, not a universal constant. Source SHA `d46b072f72088d42e9a8f65c64b74e543efb75aea9242e384107fb6f4b14aa24`. |
| **NO2** | QBW1, QBMIX, QBCERT, QBFLOW | QBW1 stays admissible because topology integration is causal; QBMIX survives only with decoder-derived regimes; QBCERT survives only when the certificate is counted and available before omission; QBFLOW survives only with counted internal state. BR2's realized distortion downranks the first quotient instance. Source SHA `7e9579a73538200c83306867c5c9687614bbf32579fe8f24c7157cf15bbd2865`. |
| **XO1** | RB1 first, QBFLOW second | neither is killed. They move behind two nearer exact receiver objects: repaired SCMDL and the QX target-overwrite object. The law forbids any XO1 atlas/regime labels that come from GT or scorer state for free. Source SHA `1435c2ce8185a94669388e894c8ee5de132c35d9f4fae5d812de70c51c2e8d30`. |
| **FB2 / RT3** | many renderer, pose, born, and merge combinations remained open; RT3 reopened D3/Lane carriage | the table does **not** become empty. Renderer/pose co-design is not a conditioning claim. The new law only removes free status from encoder-only contexts. D3/Lane or quotient routes remain admissible if topology/state is generated causally and all video-derived carriage is counted. Sources SHA `e5f1cffaa6bf6e562109bf43457671e73bfadff68648de252e3f0266c8b3dc84` and `a11154ba95a04bf56feb7884ad42820d69d05f083ec59ad6e5b83728e03ccf78`. |

The new constraint therefore changes ranking rather than globally killing alternative objects. It
**kills** QX2's exact-C1 free context, SFP1's current four-label schedule, exact GF1 tuple context as
free, fixed-G/M stand-ins, and all unchanged QX4 forms. It **admits** shipped HPAC, deterministic
coordinate charts, decoder-QBT grammars, a repaired causal SCMDL schedule, causal quotient
integration, and internal changed-object renderer state—subject to their independent byte and
distortion gates.

## First measurement — do not run from this arm

The cheapest decisive rung of the top-ranked object is one **full-n600 unchanged-AFR1 rate row**,
not a toy prefix, entropy projection, scorer run, or B1-B3 batch:

1. Author and retain a versioned receiver schema in which the schedule for every group uses only
   previous decoded classes, a boundary state updated from that decoded prefix, and
   deterministic position cells. The schema must define integer CDF selection, parser ordering,
   reset state, and exact parse-back.
2. Preserve the pinned AFR1 field exactly: 117,964,800 B, SHA
   `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`.
3. Fit one seeded held-out nonlinear model, then retain and count the model, schedule, raw/real-coded
   stream, headers, complete archive, deterministic repeat, checkpoints, command/config, and hashes.
4. Decode twice and require AFR1 field and final rendered-output identity. Because distortion is then
   fixed to AFR1, **>137,986 B** closes this schedule instance without scorer. At `<=137,986 B`,
   hand the exact archive to MAIN for authority replay; the same-distortion score arithmetic is a
   projection until `upstream/evaluate.py` runs on those exact bytes.

**Typed fire order:** `QUEUED-WITH-A-FIRE-ORDER`; **owner:** task #1374 SCMDL causal-state/model
builder assigned by MAIN; **consumer store:**
`/Volumes/APDataStore/pact/ddm_gmf1_fitted_crossgroup_gm/`; **fire trigger:** the versioned causal
schema exists, every context passes the equation above at every coding step, all video-derived
schedule/model bytes are counted, parser + deterministic parse-back tests pass, the three SFP1 field
hashes remain pinned, and MAIN confirms no duplicate #1374 fit/encode is active. The measurement
uses unchanged AFR1 X only; B1/B2/B3 remain folded because SG2B already measured their decoded-field
distortion in the wrong direction and a lossless G/M refit cannot change those rendered outputs.

## RECALL EVIDENCE

Recall searched the full `.omx/research/` corpus by content, not only charter seeds, plus canonical
equations, canonical index/DAG surfaces, design/SPEC files, hot state, task/final-message ledgers,
source, and both retained SSD roots. Query families included:

- `decoder-causal|receiver-available|decoder-derived|free conditioning|side information`;
- `QX2|QX3|QX4|C1 baseline|decoded QBT|17,926|enumerative`;
- `SCMDL|cross-group|source_class|target_class|boundary_distance|position_cell|#1374`;
- `HPAC|patch192|miss_rank8|spatial stationarity|causal boundary`;
- `born tuple|M-only|GF1|QBW1|QBMIX|QBFLOW|RB1`;
- `137,986|140,477|1.25e-4|1.157|2.178|sharp optimum|Cross empty`.

The canonical-equation command was
`.venv/bin/python tools/list_canonical_equations.py --json`. It found no registered equation that
states the exact causal-time transport obligation. The nearest registered laws were
`wyner_ziv_decoder_side_information_conditional_entropy_savings_v1` and
`argmax_cell_identity_ideal_bytes_v1`; DCC1 extends their operational interpretation and carries the
specific Catalog #344 waiver above.

Findings beyond the three named source memos changed the verdict:

1. Source inspection of shipped HPAC supplied the positive control: its previous-frame/current-group
   contexts are decoded before use.
2. MI1/OC2 and G4 showed that satisfying contexts can still die on magnitude, so the law is a
   receiver-admission gate rather than an economic predictor.
3. SG2B's matched p01-p03 receipts close the three current X edits on distortion. That changed the
   first rung from a B1 fit to a pure fixed-X causal G/M rate test.
4. DDS1's later ceiling re-adjudication closes the M-only rider at ~2.08 B; it is not a #1374
   successor despite satisfying transport.
5. The through-origin 1.157× transfer was later refuted at LB1, so DCC1 does not combine it with the
   140,477 B affine intercept as though both were current laws.
6. QX4 exposes a new target-overwrite object: 9,177 of 17,926 historical events are already target
   no-ops relative to QBT. That does not authorize dropping them under the old ABI, but it creates a
   precise new semantic question rather than another tuning pass over the closed six forms.
7. BR2's exact 106,832 B born instance is distortion-refused, which downranks quotient work but does
   not close causal topology as a family.

No DCC1-specific prior law, executable repaired SCMDL schedule, or cheaper decoder-native QX grammar
was found in those bounded scopes. Task ownership resolves through current hot state and the #1374
GMF1/JBP1 fire orders; no ownerless parallel instrument was created.

## Authority and custody boundaries

- **Measured in source receipts:** QX3 receiver mismatch and bridge bytes; QX4 six exact real-coder
  rows and parse-back; shipped HPAC byte objects; MI1/OC2/G4 rate rows; DDS1 screens/ceilings; BR2's
  exact retained scorer realization.
- **Derived here:** the causal transport equation, 11-row census, cross-memo ranking, and first
  measurement fire order.
- **Not measured here:** bytes, model fit, distortion, Seg, Pose, score, runtime, CPU/CUDA parity, or
  contest evaluation for any new object. Scorer runs 0; Modal calls 0; coder runs 0; payloads
  materialized 0; upstream writes 0.
- DCC1 wrote only this memo. The source stores and unrelated dirty worktree files remained read-only.
  No artifact directory was needed because no computed payload existed.

## NEXT_IF_RESUMED

- **Disposition: `QUEUED-WITH-A-FIRE-ORDER`; owner: task #1374 SCMDL causal-state/model builder assigned by MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_gmf1_fitted_crossgroup_gm/`; fire trigger: the versioned receiver-causal schedule, counted model grammar, parser, and exact parse-back contract exist while AFR1 X and all three SFP1 field hashes remain pinned and no duplicate #1374 fit is active.** Run the single full-n600 unchanged-AFR1 scorer-free exact rate rung specified above; retain every payload and repeat; do not fit B1/B2/B3 or fire a scorer first.
- **Disposition: `FOLDED-BEHIND-RANK-1`; owner: future MAIN-assigned QX representation owner; consumer store: `/Volumes/APDataStore/pact/ddm_qx4/`; fire trigger: the fixed-X causal-schedule row is terminal or MAIN explicitly reprioritizes, and source proof establishes that QX1 consumers require target-overwrite output rather than historical C1 syndrome identity.** Then price one new decoder-native target-overwrite grammar; do not rerun any of QX4's six forms.

## LIVE-HYPOTHESES

- A causal G/M schedule may recover transition structure without an address stream because previous
  decoded classes and prefix-derived boundary state preserve local class/boundary information while
  satisfying the receiver law. RR9/LM1 close old schedule basins, not this new one.
- A QBT-native target-overwrite object may remove much of QX4's 9,342 B gap because only 8,749 sites
  change the decoded QBT field while 9,177 historical events are target no-ops. It is plausible only
  after the consumer semantics are changed explicitly; the old C1 tuple ABI cannot inherit the win.
- Causal quotient topology may eventually satisfy both transport and address economics because
  traversal generates positions from decoded births, deaths, and boundaries. It remains speculative
  because BR2 shows that a cheap born object can be catastrophically wrong after realization.

## DEAD-ENDS

- QX2's 22,661 B C1-conditioned price is not a receiver-valid section. The exact bridge is 510,404 B;
  do not cite or rerun the unclosed price.
- All six QX4 decoded-QBT formulations are closed unchanged; best is 33,435 B / 147,327 B, 9,342 B
  over. Boundary dilation and distance-rank retuning are not successors.
- SFP1's current source/target/boundary/position schedule is closed at formulation scope for 3/3
  fields. Fixed-G/M, position-only, or hidden encoder-label stand-ins would be different mechanisms.
- B1/B2/B3 are closed as the **first rate rung**: SG2B measured every decoded X edit as Seg- and
  Pose-negative on its matched advisory instrument, and lossless G/M refitting cannot repair their
  rendered output. This is not promoted to a contest-axis family kill; a future jointly proposed X
  under a new causal schedule is a different object.
- The GF1 exact born tuple cannot be free: its required state is encoder-only and its 47,603 B packet
  is 77.6× above the later all-live ideal ceiling.
- The DDS1 M-only surrogate is receiver-causal but economically closed at an optimistic ~2.08 B
  full-population ceiling. Do not spend a #1374 exact-price slot or n600 confirmation on it.
- OC2's current-body free-chart family is drained; miss_rank8 is a 2 B rider only. G4's aggregate
  prior is a partial-stream research win, not a near-cap archive, and its predictor-boundary chart is
  192,417 B worse than control.
- The 1.157× through-origin token-to-seg transfer is refuted at LB1, and the 2.178× generator-form
  advantage is inseparable from the measured foreign-target fit error. Neither is a transferable
  credit for a new conditioner.

Own-vehicle frontier: **AFR1 S `0.14797617125559104` @ `180,002 B` `[contest-CUDA T4 n600]`, archive SHA-256 `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`; UNMOVED.**
