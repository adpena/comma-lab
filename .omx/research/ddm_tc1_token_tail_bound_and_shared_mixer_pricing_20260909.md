# ddm_tc1: current token-tail context bound and shared-mixer pricing

Owner: ddm_tc1. Status: 549 B byte winner; full public n600 field identity passed; contest-CUDA seal ready for MAIN.
Axis: `[macOS-CPU advisory / scorer-free EXACT byte measurement]`.
`score_claim=false`. No scorer or remote evaluation is authorized to this arm.

## Source custody and denominator

**Current base after the pc2 pointer move:** 181,373 B, SHA-256
`e138ee097905902ad6e1d49841b2ff2f043736298079f66c0a1628031bcd8372`,
S = 0.13882326433317044 `[contest-CUDA T4 n600]`. Current generation:
`/Volumes/VertigoDataTier/pact/ddm_tc1_tail_shared_mixer/rebase_pc2/`.
The source census (`PC2_REBASE_CENSUS.json` in the parent store) found only the
carrier/header and the public entrypoint's archive pins changed. HPAC, tail and
every decoder source file are identical. The new carrier is 18,580 B. A strict
rebase receipt retains the completed old trace as immutable hard links/copies;
the full-stream control must still reproduce the current archive before admission.
This pointer improvement belongs to pc2, not tc1.

The original pinned own-vehicle archive was 181,414 B, SHA-256
`c810c2c7f72e57670dc29bde27d584b18aa82feff68b063936a61dca89cf671e`.
Its live source is `/Volumes/VertigoDataTier/pact/ddm_rc2_hpac_semistatic_mixing/candidate_runtime`,
which is read-only to this arm. The arm's complete copy, retained payloads, row traces,
and restart state are under `/Volumes/VertigoDataTier/pact/ddm_tc1_tail_shared_mixer/`.
`INPUTS.json` binds the source files, the pointer snapshot, and the field.

| Component | Measured bytes |
|---|---:|
| RX1 header | 14 |
| HPAC model | 12,112 |
| Semantic model | 30,246 |
| Carrier | 18,621 |
| Tail | 120,321 |
| ZIP overhead | 100 |
| Complete archive | 181,414 |

The tail includes a 96 B residual prefix and a 120,225 B RC64 stream. Its SHA-256
is `af6b0997bf26c446f2b235d8cad7a83b927c17a0bb53b40e18ef4e05ff4832dc`.
The decoded label field contains **600 x 384 x 512 = 117,964,800** uint8 symbols,
SHA-256 `a73289e0a30dd765215fbb615f7802804b298ebdc816aeb9e921f32b2971f4d4`.
Every final statistic must cover this entire field. Prefix receipts are resumable
progress only; they cannot admit or close a formulation.

## RECALL EVIDENCE

Content searches covered `.omx/research/`, the full canonical-equations export
(`recall_equations.json` in the arm store), `CANONICAL_RESEARCH_INDEX*`,
`sub015_DAG_*`, design/SPEC documents under `docs/`, and task-ledger rows.
Queries included `token.tail|shared.mixer|Miller.Madow|run.length`,
`reordering.pays|container.break|token.stream.is.one.binary`, and
`HPAC|RC64|calibration|context.mixing|indicator.model`.
These searches extended beyond the charter's named memos.

Findings that changed the implementation or the claim boundary:

* **hc2, beyond seeds:** `.omx/research/ddm_hc2_wrong_half_flip_location_decomposition_20260905.md`
  records only 23.82 B of old-field causal flip clustering. Local oracle tables
  can overfit nearly singleton flips. The current instrument preserves the actual
  HPAC probability instead of replacing it with a constant probability per cell.
* **mi1, beyond seeds:** `.omx/research/ddm_mi1_indicator_model_axis_20260824.md`
  and the live `runtime/fx2_model_axis_corrector.py::SHIPPED_CONFIG` show that
  temporal/run and spatial/group information already enters the shipped corrector.
  The current source has 23 predictor families. An extra context's name does not
  establish that HPAC lacks that information; only residual gain can do that.
* **hc1:** `.omx/research/ddm_hc1_hpac_calibration_reliability_20260824.md`
  reports 97.8% binary-question bits on an older field, with only 8.44 B held-out
  pure-calibration gain. Its 64-bucket constant-probability replacement lost
  6,987 B. We recompute the binary decomposition on the current field and retain
  full-resolution coding probabilities. The coding-row winner is distinct from
  the neural model's pre-corrector winner.
* **bd1:** its weak counting-model B-pyramid ratios are not a mathematical ceiling
  against HPAC. Its 8% bar paid for a reference/order/training change and roughly
  1,045 B of model expense, with a 5,000 B net gate. The current 500 B charter gate
  concerns a tiny shared mixer with unchanged field and order; no B-pyramid is built.
* **mc1, reorder law, cl2/cl3:** motion compensation, label permutation and HPAC
  capacity changes are already closed at their documented formulation scopes.
  None is included in this arm's candidate family.
* **fs2/fs3:** average token price and marginal edited-field price have different
  denominators. This arm observes the encoder's actual RC64 input rows and measures
  each selected symbol's frequency; no average-price extrapolation is used.
* **rc1/rc2:** the successful model-section form used counted shared parameters,
  twin encodes, receiver identity, and the complete container price. Per-cell
  tables lost. rc2's raw-section and whole-container rankings differed by bytes.
* **fe1:** range-stream edits can change outer compression independently of raw
  byte length. A candidate decision must use a retained complete archive sweep.

Relevant recalled equations: `context_model_reorder_savings_v1`,
`bidirectional_pyramid_context_gain_v1`, `hpac_prior_capacity_slope_v1`,
`token_rate_model_direction_dependence_v1`,
`greedy_set_average_vs_marginal_price_v1`,
`model_section_edit_container_break_fee_v1`,
`model_section_adaptive_recode_ceiling_v1`, and
`coder_strength_substitutes_for_capacity_v1`.

## Bound definitions and limits

The trace is a side-effect-only observer on the copied shipped sparse HPAC
forward, 23-family corrector, and RC64 encoder. The mandatory n600 control compares
the entire resulting stream to the pinned stream byte for byte. At each symbol,
RC64 casts the float32 row to double, multiplies by 2^31, truncates, applies a
minimum frequency of one, and balances the winner to total 2^31. The realized
ideal bit cost is `-log2(selected_frequency / 2^31)`; termination bytes are separate.

The HPAC bucket is coding-row winning class crossed with
`min(floor(-2 log2(miss_probability)), 63)`. The added contexts are spatial order
two (left/up), spatial order three (left/up/up-right), previous-plane co-located
label, clipped spatial/temporal run state, and 32-row band. Spatial neighbours
must belong to a strictly earlier HPAC group. Temporal run state stops at the
previous plane. The joint table measures all five contexts together; marginal
gains are never added.

For cell counts, the plugin entropy is `-sum n_cy log2(n_cy/n_c)` and the
Miller-Madow correction is `(occupied_cell_symbols - occupied_cells)/(2 ln 2)`
bits. **This is an estimated constant-cell oracle, not a universal upper bound
on a full-resolution mixer.** Finite-sample bias can remain after correction;
see [Paninski, Estimation of Entropy and Mutual Information (2003)](https://www.cns.nyu.edu/~lcv/pubs/makeAbs.php?loc=Paninski03).
Free oracle cell parameters are not equivalent to a counted small mixer.
Coarse HPAC buckets also discard useful within-bucket resolution.

A separate finite-family calculation keeps the original categorical probabilities.
Five causal online KT count/expected-mass ratio predictors, a temperature feature,
and a hit/miss feature receive seven coefficients per winning class: **35 shared
weights**, prospective int8 values divided by 32, each in `[-4, 127/32]`.
All online predictor counts update after each complete plane. The generic feature
algorithm is fixed; video-fitted weights would be counted. No entropy codec has
been built before the full-field construction gate.

The convex objective `F(w)` is categorical negative log likelihood. Its supporting
plane gives `min_box F >= F(w) - sum_j g_j (w_j - endpoint_j)`, with the endpoint
chosen to minimize the linear term. Correct near-certain positions may be omitted
from optimization only by assigning them zero loss in this lower bound, giving
the candidate their entire possible saving as explicit optimistic credit. All
wrong predictions are retained, and all symbols update the online predictors.
This bounds the specified fixed-weight family, not arbitrary adaptive weight
trajectories, unspecified predictors, or every conceivable 64-parameter program.

## Results and construction gate

The **all-600-pair control passed** against the current pc2 archive. The retained
120,225 B RC64 stream has SHA-256
`f9978083199a525df0ce91612b00d2da22fde6e9af8bd247e34f5638635e9f7a`.
Receipt: `rebase_pc2/TRACE_0600.json` in the arm store. The final trace stage's
measured peak RSS was 1,787,871,232 B. Its inherited jg2 progress-bit counters are
pre-quantization float costs; final entropy and mixer calculations instead use
the exact RC64 frequencies.

| Added context | MM information given bucket (bits) | MM saving vs actual HPAC (B) | Weight + header B | Optimistic net (B) | Dense fp16 table B, excluding indices |
|---|---:|---:|---:|---:|---:|
| spatial2 | 134,843.380 | -8,489.771 | 10 | -8,499.771 | 18,184 |
| spatial3 | 151,691.532 | -6,383.752 | 10 | -6,393.752 | 36,240 |
| previous | 77,681.712 | -15,634.980 | 10 | -15,644.980 | 9,656 |
| run | 18,394.587 | -23,045.870 | 10 | -23,055.870 | 122,696 |
| rowband | 59,555.224 | -17,900.791 | 10 | -17,910.791 | 11,128 |
| joint_all_five | 275,169.247 | 9,050.962 | 40 | 9,010.962 | 1,561,144 |

The exact-frequency ideal length is **120,224.570521 B**; RC64 framing adds
**0.429479 B** to make the measured 120,225 B stream. Replacing full HPAC
probabilities by the base bucket oracle loses **25,345.193683 B**. That loss
explains the negative single-context rows: they do not establish that those
contexts lack information. The joint row is measured directly, not a sum.

The joint oracle has **195,143 occupied cells**. Its 9,010.962 B optimistic
net subtracts a prospective 35-weight/5-header-byte cost, not the cost of its
actual free cell parameters. A dense fp16 probability table alone would cost
**1,561,144 B**, before indices. This is why the MM row is not a counted-mixer
performance claim or a universal ceiling.

| Symbol class | Symbols | Actual ideal bits | Bits/symbol |
|---|---:|---:|---:|
| 0 | 27,406,232 | 374,480.278828 | 0.013664056 |
| 1 | 692,015 | 327,892.300862 | 0.473822534 |
| 2 | 58,412,953 | 107,876.305767 | 0.001846787 |
| 3 | 1,460,368 | 101,103.355697 | 0.069231424 |
| 4 | 29,993,232 | 50,444.323016 | 0.001681857 |

The hit/miss question costs **938,431.831700 bits (97.570720%)**:
285,425.692653 bits for correct predictions and 653,006.139048 for wrong predictions.
Choosing among wrong classes costs the remaining 23,364.732471 bits.
Every occupied per-context cell and its symbol denominator is retained in
`statistics/REALIZED_CONTEXT_BITS_0600.json` (7,015,531 B, SHA-256
`e1dcf1fd50ed099d97b07ff6543f33884217298ade25ab173aaa9f914e888417`).

The full-resolution numerical supporting-plane calculation gives **673.907583 B
net** for this specified 35-weight box. It includes **79.344547 B** of optimistic
credit from omitted, correct, nearly certain positions and subtracts 35 weight
bytes plus five header bytes. The numerical dual gap is **0.112140 B**. All
15,477,467 retained optimization positions include every wrong prediction; all
117,964,800 positions update the causal statistics. This clears the 500 B build
gate. Receipt: `predictor_bound_by_winner/BOUND_FINAL.json`. The smaller, earlier
unconditioned diagnostic is superseded, not a second built codec.

Exactly one mixer was built. Its 35 int8 coefficients (scale 1/32) are counted in
the archive, SHA-256
`35d56667911d1b593434e30c5334915ca516cbf8f547d54b32c091246ca75f3b`.
A deterministic rounding and single coordinate pass retains every trial vector.
The decoder uses integer Q10 logarithm features and a generic power table built
from correctly rounded square roots. Predictor counts update after a whole plane;
spatial reads are restricted to strictly earlier HPAC groups. The algorithm uses
no original field at decode time.

| Exact byte item | Base | TC1 | Saving |
|---|---:|---:|---:|
| RC64 stream | 120,225 | 119,636 | 589 |
| Counted mixer weights + header | 0 | 40 | -40 |
| Complete tail, including unchanged 96 B prefix | 120,321 | 119,772 | 549 |
| Complete archive | 181,373 | **180,824** | **549** |

Both independent n600 encodes produce stream SHA-256
`f3e86799e220bc93b6c8a83ef0a529878edb919b18d1c0f858ef6f7d900278ca`.
Their exact-frequency ideal length is **119,635.638846 B**. Each independently
resumed its complete frame-575 checkpoint after the mandatory 780-second watchdog;
neither consumed the other's state. Every actual symbol, including positions
omitted from fitting, was priced. The retained encodes also verify their features
against all 600 prepared frame traces.

The retained **40 complete archives** cover raw or Brotli q9/q10/q11 at windows
22/23/24, each with ZIP STORE or DEFLATE levels 1/6/9. Raw RC64 + ZIP STORE wins.
Every Brotli rider costs five additional bytes; ZIP DEFLATE adds 55 or 60 bytes
to the raw-rider winner. The complete winner is
`candidate_runtime/archive.zip`, SHA-256
`20f7e67e6ce2373552c22301a7a35891cf48e244d31d154dfff1ddd254ce0929`.
The source census preserves HPAC, semantic, carrier, and the 96 B residual prefix
byte-for-byte. Changes are the tail rider and its flag, receiver code, optional
receiver checkpoint helper, and archive hash/length pins. The live source tree
is untouched. Container census: `mixer/RESULT.json`.

At held distortion the exact rate projection is
`Delta S = -549 * 25 / 37,545,489 = -0.00036555656526407205`,
so the projected S is **0.13845770776790636**. This is arithmetic on archive bytes,
not an evaluated score; `score_claim=false` until MAIN obtains the exact T4 row.
The result converts 81.46% of the specified-family numerical ceiling, but only
6.09% of the optimistic joint MM table. The prior-law prediction that 40-60% of
that table would transfer to a tiny mixer is falsified on this field/formulation;
the shared-family capacity restriction is material. The 549 B candidate would close
1.942% of the current score gap to 0.12 at held distortion and leave a further
27,720.147 B to remove. This arm has not reached the mission target.

## Public receiver proof

The full proof invokes `runtime.f26_inflate.inflate_archive` with its required
CPU four-thread configuration, through real archive parsing, semantic/carrier
setup and `runtime.residual_archive.decode_production_tokens`. It preserves the
public token-stage checkpoint and stops there before rendering. No scorer runs.
The full n600 field equality **passed**: the actual public reader returned
117,964,800 B with the original field SHA-256
`a73289e0a30dd765215fbb615f7802804b298ebdc816aeb9e921f32b2971f4d4`.
Receipt: `public_identity/PUBLIC_FIELD_IDENTITY.json`; the decoded bytes are at
`public_identity/public_stage_checkpoint/tokens_cpu_stage_complete.u8`. This
is the public pipeline token phase, not a library-only decoder check. No full
render, scorer, or contest runtime timing was measured. The final stage resumed
from frame 350, so its logit/CDF digests explicitly describe that suffix; the
reported token digest covers all 600 pairs.

Crash-resume is exercised on the public path: frame 1 saved to disk, a fresh
process resumes to frame 2, and an independent uninterrupted public run reaches
frame 2. All **100 checkpoint arrays**, the RC64 interval, native corrector tables,
35-weight mixer state, and decoded field agree. The two NPZ checkpoint files are
also byte-identical, SHA-256
`98e57b5ae8b07a458f5bb982067d789b26f16554febc9e3ee89a33f05fe7ad7e`.
Receipt: `public_identity/RESUME_CONTROL.json`. This is an implementation control,
not a prefix-based research verdict. Full proof stages persist every 25 frames,
with archive/stream/source/build bindings checked before resumption.

Candidate and frontier public entry probes use an explicit observer at the
actual token-stage call, after semantic/carrier setup. They do not infer success
from a timeout. Separate `inflate.sh` executions must reach their real CUDA gate.
Both roles are bound to the live pc2 pointer. The seal validator's own
`_public_smoke_problems` checker accepted the block with zero problems; receipts
are `PUBLIC_SMOKE.json` and `PUBLIC_SMOKE_VALIDATOR.json`.

## Retention and recovery

Every materialized control stream, coding row, predictor trace and optimizer trial
is retained in the assigned SSD store with a hash/length receipt. Complete corrector
and encoder state is saved every ten pairs and at immutable stage boundaries.
Runs use one CPU worker and sequential bounded stages; at most one additional
statistics/optimization process runs beside the trace. Canonical detached launcher
receipts record best-effort nice status, and `safe_run.py` enforces a 780-second
process-group limit for the continuing stages. No bulk deletion is performed.

## Seal and handoff

**SEAL READY:**
`/Volumes/VertigoDataTier/pact/ddm_tc1_tail_shared_mixer/rebase_pc2/SEAL_ddm_tc1_tail_shared_mixer.json`.
The producer and independent consumer validation both return `SEAL_VALID`.
The sealed runtime has 46 files, 975,639 B, tree digest
`d643772ad04cab6f` (prefix; full digest in the seal). The archive, public reader,
public-smoke block and pc2 baseline are pinned. A pointer change refuses the seal.
The public parser independently confirms identical restored HPAC (16,061 B),
semantic (31,792 B), carrier (18,931 B), and residual-table content. These are
restored public-parser sizes, distinct from the counted component sizes above.
All 41 copied Python/C/shell source files match the live pc2 source. Initialization
excludes bytecode and native-library build products; the copied frontier's tree
digest therefore need not equal an older fire-time tree digest.

`FIRE_ORDER.json` is **QUEUED-WITH-A-FIRE-ORDER**, owner **MAIN**. Its consumer is
`/Volumes/APDataStore/pact/ddm_tc1_tail_shared_mixer_t4_20260909/MODAL_REMOTE_RESULT.json`.
Its exact argv uses `tools/fire_modal_auth_eval.py --seal`, a unique named lane
requiring MAIN's active claim, and the charter's contest-CUDA axis. Fire only
after revalidating the seal against the unchanged live pointer. This arm made
**no Modal dispatch**. The expected rate delta and report-8dp base bound are in
the sealed object; MAIN must measure the real T4 row before promotion.

The source release is retained in `SOURCE_RELEASE.json` and `source_release/`.
Eight owned Python files satisfy their two-review policies; Python compilation,
C syntax, whitespace, the C/NumPy differential bound check, native codec control,
public resume control, full twin encodes, all 40 containers, the independent
public parser census and full public n600 token identity passed. Validation
receipts are linked by `VALIDATION_SUMMARY.json`. No borrowed `ddm_tc1_seal_tr1_*`
source is included in this arm's file set. Serializer intent must include only
this arm's canonical-equation append, preserving sibling registry rows.

## LIVE-HYPOTHESES

* The T4 row should preserve distortion and realize the 549 B saving because the
  decoded field and all other model/carrier contents are identical. Full public
  T4 timing and the exact oracle result remain unmeasured; this is the live fire.
* Joint-context interactions may retain structure beyond this 35-weight family:
  its numerical ceiling is much smaller than the joint MM table estimate. This
  is an untested lead, not an admissible new codec or a promised saving. A second
  architecture is outside this charter and is folded into this memo for recall.

## DEAD-ENDS

* INSTANCE: the full-plane teacher-forced HPAC shortcut failed parity against the
  shipped sparse logits at pair 10. `trace20.log` retains the refusal. It supplied
  no admitted rows; the source-faithful sparse trace reproduced the full stream.
* INSTANCE: reconstructing the original float32 row from integer RC64 frequencies
  is not a neutral zero-weight operation. One real pair-1 row changed frequencies
  by up to 14. The retained counterexample led to forwarding the original row
  for zero banks; the corrected null control passes.
* FORMULATION: treating the corrected context table as a universal or attainable
  small-mixer ceiling is invalid. Coarse buckets discard predictor resolution,
  MM is a bias estimate, and the joint table has 195,143 free cells. The explicit
  35-weight numerical bound and actual archived bytes carry narrower claims.
* INSTANCE: every tested Brotli/DEFLATE container loses to raw RC64 + ZIP STORE
  on this exact stream. All 40 archives are retained; do not repeat that sweep
  without changed bytes or a different container mechanism.
* PRIOR FORMULATION NEGATIVES CONSUMED: motion compensation, label reordering,
  HPAC capacity changes, and per-cell parameter tables remain excluded on their
  documented evidence. This arm did not rerun or broaden those verdicts.

Live own-vehicle frontier: **S = 0.13882326433317044 @ 181,373 B
[contest-CUDA T4 n600]**, moved by pc2 and unchanged by this scorer-free arm.

## Equations leg (`tac.canonical_equations`) — MAIN landing note, 2026-09-09

Registered law: `token_tail_context_mixing_bound_v1` (registration receipt `.omx/research/ddm_tc1_context_mixing_bound_registration_20260909.json`; registry rows appended through `register_canonical_equation`). Anchor: the shared-predictor bound table above and the measured 180,824 B archive (−549 B vs 181,373 B; twin byte-identical; public-path n600 field identity). MEASURED, `score_claim=false` until the T4 row.
