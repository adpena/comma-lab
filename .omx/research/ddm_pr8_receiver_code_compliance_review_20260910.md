# ddm_pr8 — second-family receiver-code compliance review, moves 41 and 42-candidate

Tokens: `[no-triality] [p0-ledger-ok]`  
Status: **REVIEW COMPLETE — P0 RULE-118 BLOCKER; PR/SUBMISSION SWAP REFUSED**  
Scope: receiver-code review only; no scorer, encode, decode, dispatch, pointer edit, or upstream write.

## Verdict

Move 41's exact T4 score is a real measured row, but its receiver is not eligible for a PR or
submission swap in its present form. The selected decoder embeds video-selected semantic and spatial
parameters in free code: Lane class `1` and rows `128..319`. Those choices are traceable to the prior
full-public-video census rather than to a generic, content-independent receiver specification. This is
the charter's P0 falsifier: video-fitted content is reconstructive even when it is only a few scalar
constants. It must be counted in `archive.zip`, or replaced by an honestly predeclared generic rule,
before the receiver can be proposed for public use.

The unpromoted TC4 candidate inherits that defect and adds another one: Movable class `3` combined
with the same rows `128..319` is embedded in free receiver code. TC4's selection mask and all fitted
mixer weights are counted correctly, but counting the selector does not make the selected map's
video-derived operands free.

There are three independent swap blockers in addition to that P0:

1. Move 41 measured **1,336.668977565 s** of T4 inflate, so it fails the live 1,260 s seal-margin
   policy by 76.668977565 s. TC4 timed out at 1,800 s and produced no exact result.
2. The public entry point refuses CPU unless the non-contest advisory environment switch is set.
   No exact contest-CPU row exists for this lineage.
3. The selected move-41 geometry is the float64 `LaneGeometry` branch, not the integer
   `RunTrackingGeometry` branch. Cross-host token identity happened for the one archive, but generic
   bit-identical arithmetic across CPU/CUDA/NumPy versions has not been proved.

This review does **not** move or erase the live research pointer. Per MAIN's 2026-09-10 failure
directive, move 41 remains the internal own-vehicle pointer pending an explicit adjudication. The
finding withdraws the present runtime from PR/submission consideration; it does not rewrite the
already measured T4 receipt.

## Reviewed objects and complete diff census

The roots below are used in every line citation:

- `B40` = `/Volumes/VertigoDataTier/pact/ddm_tc3_lane_predictor_seal/rebase_move40/move40/source_runtime`
- `M41` = `/Volumes/VertigoDataTier/pact/ddm_tc3_lane_predictor_seal/rebase_move40/move40/candidate_runtime`
- `M42C` = `/Volumes/VertigoDataTier/pact/ddm_tc4_context_slate/move41/candidate_runtime`

The independently hashed trees contain 47, 49, and 51 files respectively. `B40 -> M41` changes
exactly `archive.zip`, `inflate.py`, `inflate.sh`, and `runtime/residual_archive.py`, and adds
`runtime/tc3_geometry.py` and `runtime/tc3_mixer.py`. `M41 -> M42C` changes exactly `archive.zip`,
`inflate.py`, and `runtime/residual_archive.py`, and adds `runtime/tc4_fast.py` and
`runtime/tc4_maps.py`. The changed-path census is closed: no other receiver file differs.

### File-by-file changed-line classification

| Runtime lines | What the changed or added lines do | Provenance and compliance classification |
|---|---|---|
| `M41/inflate.py:18-19,27-39` | Pin and verify the exact counted archive | Artifact identity metadata. The hash/length describe already-counted bytes and cannot reconstruct them. **Compliant.** |
| `M41/inflate.py:56-73` | Add an advisory CPU switch, otherwise require CUDA, pin four threads, and label advisory output | Generic execution policy, no content. **Compliant as code**, but `:57-61` makes the normal CPU contest entry point fail and therefore blocks the required CPU axis. |
| `M41/inflate.sh:33-40` | Reuse a declared RC64 shared library or build it | Generic dependency/build wiring. **Compliant.** No payload constant or new external package. |
| `M41/runtime/residual_archive.py:527-532,658-663,739-765` | Recognize `TC3M`, construct the mixer, observe decoded symbols, and report the selected codec | Generic framing and dispatch. The config is read from the counted rider. **Compliant.** |
| `M41/runtime/tc3_mixer.py:1-145` | Parse 40 counted int8 weights, maintain causal counts, combine the old mixer with one added map, and checkpoint it | Algorithm/state code. The archive version byte selects the geometry and all 40 weights come from the counted rider at `:20-31,37-48`; no fitted weight table is in code. **Compliant except for the selected map's operands described next.** |
| `M41/runtime/tc3_geometry.py:17-74` | Define the 384x512 five-class field, the inherited 190-group schedule, causal-prefix validation, and sentinel `K` | Fixed format and generic causal geometry. **Compliant.** |
| `M41/runtime/tc3_geometry.py:77-132` | Implement the shipped variant-A moment predictor | `previous == 1`, `known != 1`, `symbols == 1` at `:84,109,127` and the hard gate `yy < 128 or yy >= 320` at `:122` are video-selected semantic/spatial content left in free code. **P0 Rule-118 violation.** Float moment arithmetic and its remaining thresholds are classified below. |
| `M41/runtime/tc3_geometry.py:135-252` | Implement unshipped variant-B integer endpoint tracking | Generic causal algorithm predeclared before TC3 encoding; it is not selected by move 41. Its occurrences of class `1` and rows `128..319` at `:219,223-224` inherit the same content provenance and may not be activated for a public archive until counted or replaced. The other tracking constants are generic algorithm parameters. |
| `M42C/inflate.py:18-19` | Replace only the archive hash and byte pin | Artifact identity metadata for counted bytes. **Compliant.** |
| `M42C/runtime/residual_archive.py:527-534,660-669,745-771` | Recognize `TC4M`, dispatch `FastContextMixer`, keep TC3 compatibility, and report the codec | Generic framing/dispatch. Counted config remains the data source. **Compliant.** |
| `M42C/runtime/tc4_maps.py:1-187` | Define five causal maps, parse the selected-map mask and weights, maintain their online counts, and checkpoint | The mask and weights are read from the counted rider at `:97-105,111-120`. Maps 0-3 are generic causal transforms. Map 4 embeds Movable class `3` and rows `128..319` at `:84-86`; those operands came from the earlier public-video census. **P0 Rule-118 violation.** |
| `M42C/runtime/tc4_fast.py:1-90` | Incrementally compute the same five maps | Generic integer implementation except that it repeats Movable class `3` and rows `128..319` at `:56-60`. **P0 Rule-118 violation.** It also computes all maps even though only selected columns reach mixing at `:66-70`, which is a live timing optimization opportunity, not a correctness finding. |

### Constant, table, coefficient, threshold, and seed ledger

| Constant family | Runtime evidence | Provenance verdict |
|---|---|---|
| Archive SHA-256 and sizes `180154` / `180021` | `M41/inflate.py:18-19`; `M42C/inflate.py:18-19` | Integrity pins for counted artifacts; free non-reconstructive metadata. |
| Rider magics, versions, config lengths, masks, and byte offsets | `M41/runtime/tc3_mixer.py:17,20-31,37-48`; `M42C/runtime/tc4_maps.py:18,97-105,111-118` | Generic format framing. The TC4 selector mask itself is counted. |
| Move-41 old 35 plus new 5 fitted int8 mixer coefficients | `M41/runtime/tc3_mixer.py:24,42-48` | **Counted.** Direct archive parse found the 40 bytes, ending `29,23,6,10,5`, before the token stream. No coefficient is duplicated in code. |
| TC4 old 40 plus 15 fitted int8 coefficients and mask `19` (maps 0, 1, 4) | `M42C/runtime/tc4_maps.py:100-105,113-119` | **Counted.** Direct archive parse found mask `19` and 55 weight bytes; the last 15 end `23,20,-16,16,21,-12,20,11,-10,10,10,12,18,14,-32`. |
| Shape/alphabet `H=384,W=512,K=5`; inherited group moduli `64`, `190`; slot count `8`; sentinels `K`, `8`, `9`, `25`, `40` | `M41/runtime/tc3_geometry.py:17-22,41-52`; `M42C/runtime/tc4_maps.py:14-18,21-31,34-60`; `M42C/runtime/tc4_fast.py:27-60` | Public format and generic finite-state/cardinality constants. **Compliant.** |
| Online KT smoothing `0.5`, ratio clip `1/16..16`, fixed log quantization | `M41/runtime/tc3_mixer.py:62-67`; `M42C/runtime/tc4_maps.py:131-144` | Generic adaptive-coder arithmetic inherited as a receiver algorithm, not a stored fitted table. **Compliant**, subject to the float determinism limit below. |
| Variant-A history window `16`, prior weight `0.25`, validity thresholds `count>=4`, residual `<=36`, `abs(slope)<=2`, width floor/multiplier `0.5,3`, wide-span `24`, and distance bins `[0,1,2,4,8,16]` | `M41/runtime/tc3_geometry.py:25-27,95-121` | No per-frame table and no scored sweep was found. The implementation memo documents them, but the searched corpus did not furnish an a-priori preregistration for the exact values. **Provenance incomplete; do not call these proven-generic in a PR packet.** |
| Lane class `1` and row band `128..319` | `M41/runtime/tc3_geometry.py:84,109,122,127,219,223-224` | The prior n600 census says every Lane/Road selection lay in rows `128..319`; the Lane-focused arm was routed from that same-video attribution. **Video-selected and uncounted: P0.** |
| Variant-B quarter-pixel scale `4`, filter `3/4,1/4`, hit/gap gates `2,4`, endpoint gate `16+4*gap`, velocity clamp `8`, hit cap `16`, infinity `1<<20`, and the same distance bins | `M41/runtime/tc3_geometry.py:135-204,231-247` | Predeclared causal integer tracker mechanics; inactive in the shipped variant. **Generic algorithm constants**, except its repeated Lane/class/row selection. |
| TC4 map cardinalities `(41,26,26,10,18)`, 3x3 majority/tie rule, row-run bins `[1,2,4,8,16,32,63]`, 64-cell resets, vertical transition logic | `M42C/runtime/tc4_maps.py:17,21-60`; `M42C/runtime/tc4_fast.py:10-55` | Determined by alphabet/product cardinalities or generic powers-of-two causal bins. The five maps were preregistered before TC4 pricing and no post-encode map sweep was found. **Compliant.** |
| TC4 Movable class `3` and row band `128..319` | `M42C/runtime/tc4_maps.py:84-86`; `M42C/runtime/tc4_fast.py:56-60` | TC4's own charter calls this “the census's second class” and names the census band. Predeclaring it before the TC4 encode does not erase its earlier same-video origin. **Video-selected and uncounted: P0.** |
| Random seeds or stored generated tables | all changed/new files above | **None.** The runtime maps are generated causally from decoded state. The only evolving tables are receiver-online counts, not fitted payload. |

## Rule-118 packet adjudication

The clean part of the packet is important: the fitted weights are not hidden. `TC3M` stores one
version byte plus 40 signed int8 weights; `TC4M` stores its map mask plus the old 40 and five weights
for each selected map. Both decoders slice those bytes from the payload before the RC64 stream. The
selection mask is therefore honestly paid.

The defect is narrower and decisive. “Generic geometry algorithm” is not a blanket exemption for
parameters chosen after inspecting this video's 600 decoded token planes. The TC2 report records
117,964,800 symbols and says all 28,098,264 Lane/Road selections happen to lie in rows 128–319. The
TC4 charter cites that census and explicitly routes its fifth map to “horizon-band indicator ×
Movable-run state.” Those scalar choices are learned from the contest content just as surely as a
small fitted table would be. Their small byte price makes the cure easy; it does not make omission
legal.

Accordingly:

- **Move 41:** withdraw the current receiver runtime from PR/submission packets. Preserve the T4
  score receipt as an internal measured row and leave pointer adjudication to MAIN.
- **TC4 candidate:** no move 42 exists. It failed before scoring and is also Rule-118-ineligible.
- **Cure:** serialize every content-selected semantic ID and spatial boundary in the counted rider,
  parse it at the receiver, rebuild and twin-price the archive, then repeat public identity and the
  required exact axes. Merely adding comments that call the constants generic is not a cure.

## Determinism and public-600 identity

The public-600 proof establishes a useful but bounded fact. On one macOS arm64 CPU host, the actual
`inflate.sh -> inflate.py` path decoded all 117,964,800 labels to
`b50da438e65b62d5d6f4ca1e151463d097feafd102bbd11d3e0556f849fa4ab5` and emitted
3,662,409,600 raw bytes equal to that host's pinned source output, SHA
`c5a7986cf3f16360a0ef5f197ad4b5cc4cdf4c143f0c8a4319460476fc986ea5`. The child and safe-run wrappers
exited zero. The token stage resumed at frame 2, so its corrected-logit and CDF digests cover only the
suffix even though the stored full-field token digest covers n600.

The later fresh-start T4 receipt decoded the same token SHA and ended at the same RC64 bit position,
956,332. That is strong instance evidence that the selected arithmetic did not cross a coding bin on
these exact bytes between the two tested stacks. It is not a proof for every host or NumPy version.
The selected archive uses `LaneGeometry` (`M41/runtime/tc3_mixer.py:77-78`), whose division, square
root, rounding, and search-bin boundaries are float64 (`M41/runtime/tc3_geometry.py:97-120`). The prior
prediction that the shipped predictor is integer-only is false; the integer tracker is the unselected
variant B.

The raw-video SHA differs across the two axes: the T4 receipt emitted
`db6ad16bab2bbaf9b6883a3a90e129a79fbb4cf6de4e2d195e5eafd504c02c15`, not the macOS raw SHA above.
The same difference exists in the parent receiver and the exact score components agree, so this review
did not attribute that renderer-level host difference to TC3. It nevertheless means the broad claim
“same archive produces bit-identical raw output on every host” is not established by these receipts.

TC4 has only the same-host advisory public identity proof. Its T4 run timed out before `report.txt`,
`contest_auth_eval.json`, or a final token digest was retained, so it has no cross-host decode proof.

## Dependency closure and strict-scorer audit

Static import-root comparison found no new absolute external import from `B40 -> M41` or
`M41 -> M42C`. New imports are local `runtime` modules; `numpy` and conditional `brotli` already belong
to the declared closure. The changed shell logic reuses or builds the existing RC64 library. The exact
T4 runtime manifests report no recursive repo-local `tac` dependency, no unresolved declared root,
and the forbidden-reference guard passed with zero direct references to the contest source video or
upstream scorer weights.

No changed receiver file imports SegNet, PoseNet, the evaluator, scorer weights, cached logits, or
scorer-derived tables. The maps consume only the decoded current/previous token plane, counted rider,
and fixed shape metadata. **Strict-scorer dependency closure passes.** The guard's zero count does not
excuse the semantic and spatial scalars above; its forbidden-reference vocabulary is narrower than
Rule 118.

## Decode cost and CPU-axis risk

All timing numbers below are measured receipts or explicitly bounded estimates; none is a score.

| Comparison | Token/inflate evidence | Delta and interpretation |
|---|---:|---|
| Move 40 parent, fresh T4 | token 917.580790290 s; inflate 990.053829427 s | Baseline receipt used by the same receiver family. |
| Move 41, fresh T4 | token 1,258.585670749 s; inflate 1,336.668977565 s | +341.004880459 token s = **2.890734 us per one of 117,964,800 tokens**; +346.615148138 inflate s. The earlier prediction “<60 s” is falsified. |
| TC3 advisory macOS, resumed from frame 2 | token 918.843160152 s; total 1,230.413294334 s | Same-host identity apparatus, not a fresh contest-CPU row and not comparable as an authority runtime. |
| TC4 advisory macOS, resumed from frame 2 | token 1,029.905117035 s | +111.061956883 s over the same resumed TC3 setup = **0.944633 us per suffix token**. Advisory only. |
| TC4 T4 | hard timeout at 1,800 s | Timeout, not a crash. Relative to move 41's completed T4 inflate, the added path consumed **more than 463.331022435 s**, a lower bound of **3.927706 us/token** if normalized by n600. No exact score resulted. |

Move 41 already exceeds the current 1,260 s seal-margin threshold; there is no positive time budget
for an added context map until the inherited decode is accelerated. TC4's fast path computes temporal,
vertical, lane, and horizon state for every group before slicing to selected columns
(`M42C/runtime/tc4_fast.py:30-70`). Mask 19 selects only maps 0, 1, and 4, so lazy computation keyed by
the counted selection mask is a plausible generic-code speed repair, but it is unmeasured.

The CPU risk is stronger than “not yet timed.” Without `TC3_ADVISORY_CPU=1`, a host with no CUDA raises
immediately at `M41/inflate.py:56-61`; the TC4 entry point inherits those lines unchanged. Thus the
ordinary contest-CPU entry point is structurally unavailable. The macOS advisory numbers do not cure
that and cannot be promoted to contest CPU. A PR or submission swap requires a conforming CPU path and
an exact contest-CPU receipt on 1:1 hardware after the Rule-118 and time-budget repairs.

## Exact score arithmetic

For move 41, using the exact archive size and the evaluator's eight-decimal component outputs:

`S = 100(0.00010636) + sqrt(10(0.00000489)) + 25(180154 / 37545489)`

`= 0.010636 + 0.006992853494818835 + 0.119957153840771657`

`= 0.137586007335590492`, recorded canonically as **0.13758600733559048
[contest-CUDA T4 n600]**. This is the same exact archive and component receipt already under custody;
the review generated no new score.

TC4's 180,021-byte candidate did **not** score. If, counterfactually, move 41's distortion components
had held, its arithmetic projection would be:

`0.010636 + 0.006992853494818835 + 25(180021 / 37545489)`

`= 0.137497448094825244`.

That is a **projection only**. The 133-byte rate delta is `-0.000088559240765249 S`; no distortion
components or exact score were harvested because inflate timed out.

## Move-41 packet claim review

The packet's narrow claims remain true where the receipts support them: variant A is 79 bytes smaller
than move 40; variant B was only 37 bytes smaller and was not shipped; the 40 mixer coefficients are
counted; the fresh T4 run decoded the expected full token field and produced a real exact score; no
CPU-axis row was claimed.

Two statements need correction before the packet can be reused:

- “generic geometry code, no per-frame fitted table” is incomplete. There is no per-frame table, but
  class `1` plus rows `128..319` are same-video-selected scalar content. Absence of a large table is not
  the Rule-118 boundary.
- `passed=true` describes the exact evaluation wrapper. It does not mean PR-ready: the same retained
  evidence now fails the live 1,260 s seal-margin policy, and the packet has no usable CPU axis.

## RECALL EVIDENCE

Queries covered `.omx/research/` for `lane predictor`, `predictor context`, `public600`, `receiver
code`, `context slate`, Rule-118 language, class/row-band constants, timing receipts, and the move-41
archive SHA. I also searched the canonical research indices/DAGs and exported the canonical equation
registry, including `lane_boundary_context_map_bound_v1`. No prior review was treated as authority for
the current runtime; each conclusion above was checked against the exact runtime copies and retained
archive/remote receipts.

Beyond the charter seeds, three findings changed or bounded this review:

- `ddm_blp1_born_lane_predictor_20260831.md` measured a 60,191-byte minimal learned receiver dependency.
  It reinforces the correct boundary: a learned predictor's whole transitive dependency is counted;
  moving its effect into free code is not a legal rate shortcut.
- `wave_f_lane_tracking_coherent_fit_measured_20260702.md` found only a 0.5% lossless rate refinement on
  a different lane-trajectory object. It does not validate TC3's token-context geometry or its
  constants and was not transferred as evidence.
- The live `ddm_dwc1` T4 receipt audit measured move 41 at 1,336.668977565 s and TC4 at the 1,800 s
  timeout. This replaced the charter's optimistic timing prediction with a hard seal-margin refusal.

No prior compliant precedent was found in the searched corpus for leaving same-video-selected semantic
class IDs or exact row bounds in free receiver code.

## Follow-on dispositions

- **QUEUED-WITH-A-FIRE-ORDER** — owner `MAIN`; consumer store
  `.omx/research/ddm_pr8_20260910/`; fire on harvest of this P0 review. Withdraw the present move-41 and
  TC4 runtimes from PR/submission packets, while adjudicating the internal pointer separately.
- **QUEUED-WITH-A-FIRE-ORDER** — owner `receiver-rule118-cure`; consumer store
  `/Volumes/VertigoDataTier/pact/ddm_tc3_rule118_counted_config_cure/`; fire only after MAIN assigns the
  lane. Move each content-selected class ID and row bound into the counted rider (or replace it with an
  independently preregistered generic rule), then rebuild, twin-price, and repeat public identity.
- **FOLDED** — owner `ddm_dwc1`; consumer store `.omx/research/ddm_dwc1_20260910/`; fire when a
  Rule-118-clean receiver is byte-closed. Optimize only selected TC4 maps, then require a measured T4
  inflate at or below 1,260 s before any exact-eval fire.
- **QUEUED-WITH-A-FIRE-ORDER** — owner `MAIN`; consumer store
  `/Volumes/APDataStore/pact/ddm_tc3_cpu_axis_cure/`; fire only after Rule-118 and T4-margin gates pass.
  Remove the CPU refusal through a conforming deterministic implementation and obtain an exact
  contest-CPU n600 receipt on 1:1 hardware; advisory macOS identity is not a substitute.

## LIVE-HYPOTHESES

- Serializing the two semantic IDs and two row bounds should cost only a few counted bytes, so the
  Rule-118 cure is likely to preserve most of the 79-byte move-41 rate win; this is plausible because
  the fitted mixer weights are already stored and the decoder needs only scalar configuration.
- Replacing selected float64 geometry boundaries with an integer/fixed-point equivalent may close the
  remaining cross-host coding risk without losing the rate gain; this is plausible because variant B
  already demonstrates a causal integer geometry implementation, though it did not win as currently
  parameterized.
- TC4 may recover substantial wall time by computing only mask-selected maps; this is plausible because
  mask 19 uses three of five maps while `FastContextMixer` currently materializes all five on every
  token group. It remains unmeasured and cannot by itself overcome move 41's existing 76.669 s margin
  deficit.

## DEAD-ENDS

- Treating “no per-frame fitted table” as sufficient Rule-118 compliance is closed: same-video-selected
  scalar class IDs and row bounds are content too.
- Calling move 41's shipped predictor integer-deterministic is closed: the archive selects the float64
  `LaneGeometry` branch; the integer tracker is variant B and is not shipped.
- Promoting macOS public-output identity into a contest-CPU receipt is closed: it is advisory, resumed
  from frame 2, and the ordinary CPU entry point refuses to run.
- Treating TC4's 180,021 bytes as move 42 or `S=0.13749744809482524` as measured is closed: the T4 run
  timed out before evaluation and produced no `contest_auth_eval.json`.
- Adding more per-token context work before reducing the inherited receiver wall is closed under the
  current seal policy: move 41 already exceeds the 1,260 s threshold.

Own-vehicle frontier unchanged: **S 0.13758600733559048 @ 180,154 B [contest-CUDA T4 n600]**, move 41,
archive `299a8201662c8a407881a63214d944d0c8da25bf244ca4af034ecb730f5a7936`.

<!-- # FORMALIZATION_PENDING: second-family compliance review with no measured score row of its own; its findings (rule-118 content in receiver code; decode margin; float64 geometry) are gates and cures, not an equation — the cure lane rlc1 registers any equation its rebuild measures -->
