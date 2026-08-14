# DDM MT1 #978 receiver-native multi-token screen — 2026-08-14

## VERDICT

**LOCAL SIGN: NEGATIVE. FORMULATION CLOSED LOCALLY; T4 CONFIRMATION QUEUED.**

The corrected optimal-form candidate is a parsed, counted, receiver-native
probability simplex over CP135's five existing semantic embeddings. On a
fixed-seed, eight-stratum random `n32` heldout set it reduced the named
Road→Lane error count from `306` to `297`, but equal collateral on other
directed edges left total Seg errors unchanged at `1,529 / 6,291,456`.
Pose MSE increased by `0.00010290218779118732` against the same exact CP135
frame-0 carrier. It therefore failed both the total-Seg-improvement and
zero-pose-damage gates.

This is a `[macOS-CPU advisory]` component screen. It is not an exact score,
not a whole-container price, not an n600 result, and not authority to close
the entire #978 family. The sealed T4 sign gate is queued for MAIN and was not
fired by this arm.

## MEASURED RESULT

All three arms use the same 32 heldout pair IDs and the same exact retained
CP135 frame-0 carrier. Seg and Pose were evaluated through the frozen local
CPU-torch scorers after the camera lift and uint8 round trip.

| arm | Seg errors / 6,291,456 | d_seg | d_pose | GT Road→predicted Lane |
|---|---:|---:|---:|---:|
| CP135 hard tokens | 1,529 | 0.00024302800496419272 | 0.00011780139902839437 | 306 |
| HC1 direct C1 representative | 2,588 | 0.0004113515218098958 | 0.0057630776427686214 | 434 |
| MT1 parsed multi-token state | 1,529 | 0.00024302800496419272 | 0.0002207035868195817 | 297 |

The MT1 candidate beats HC1's direct representative on both measured
components, so jointly mixing existing semantic embeddings is materially
better than substituting C1 tokens directly. It does **not** beat CP135 at
matched carrier: total Seg is tied and Pose is worse. The named Road→Lane
surface moved by `-9 / 306 = -2.94%`, but its value was exactly cancelled by
other directed-edge errors.

The selected EMA module is `1,270 B`, SHA-256
`af9da3dcdba84b1b0a5705d0bcc67eedad227d5a785f91dbd770aab90de5be19`.
At the charter's `0.785 flips/B` anchor, a naive standalone-module projection
would require about `997` net flips before container overhead; measured net
heldout flips were `0`. This is only a projection because whole-container
pricing was correctly skipped after the negative sign.

## THE RECEIVER-NATIVE OBJECT

The receiver computes five one-hot fields from decoded CP135 tokens: center,
left, right, up, and down, with neighbor channels active only across decoded
token edges. A counted hidden-4 convolutional module emits:

1. a support mass bounded by `0.25`; and
2. a five-class support distribution with the center class masked out.

Their mixture is a simplex whose center weight is at least `0.75`. The simplex
mixes CP135's own `token_embed.weight`, then enters CP135's existing
`coord_mix`, frame embedding, four nonlinear TokenBlocks, and RGB head. The
module transmits no site IDs, changed-site list, mask, token plane, or new
carrier. Its video-derived content is the counted int8-weight/fp16-bias module.
Deterministic Brotli serialization repeats byte-identically and parse-back is
exact under the storage quantizers.

The screen used two 32-step stages, saved distinct live and warmup-EMA
checkpoints after every step and at both stage boundaries, and consumed the
serialized EMA parse-back at the endpoint. The corrected objective gives
fourfold weight to actual `GT Road → current predicted Lane` errors and an
additional twofold weight where those errors touch BG2's pre-registered top
five spatial cells.

## SELECTION

Seed: `20260814978`.

The 600 pairs were divided into eight strata using:

- BG2 introduced-error count quartile (`73`, `85`, `101` cuts); and
- above/below the median (`20`) count in BG2 cells `(5,8)`, `(5,9)`,
  `(6,12)`, `(5,7)`, `(6,3)`.

Four random pairs per stratum were assigned to training and four disjoint
random pairs per stratum to heldout, for `n32/n32`. No prefix sample was used.
The exact IDs and source hashes are retained in `inputs/SELECTION.json` and
`inputs/INPUT_MANIFEST.json`.

## RETAINED CUSTODY

Authoritative store:

`/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/multitoken_978/ddm_mt1_20260814/optimal_form_r2/retained/`

The store contains `3,622,029,852` retained bytes. Every train step and every
endpoint batch retains probability state where materialized, pre-R RGB,
camera uint8, scorer input, logits, argmax, target, frame-0 carrier, Pose input,
and Pose output. It also contains every live/EMA checkpoint, every stage model,
serializer repeats, selected fire inputs, and manifests with bytes and SHA-256.
No materialized candidate payload was discarded.

Key receipts:

- `FINAL_RESULT.json`: `95,996 B`, SHA-256
  `c753984338d2a039fba102d2a6ff65f553cd0bb7b13d53d174437bd39c2b9a35`.
- `COMPARISON.json`: `90,031 B`, SHA-256
  `924e49d2af1719f03bd3bebdf864d8a6dffdb3218f082af467439acfc54a8aa2`.
- `DIRECTED_ERROR_POSTMORTEM.json`: `1,428 B`, SHA-256
  `79d6e7c4f029fbc20286dde5f26c0005f43730d4f6819c56b855711019d5773f`.
- selected module: `stages/20_collateral_finish/ema.mt1.br`, `1,270 B`,
  SHA-256 `af9da3dcdba84b1b0a5705d0bcc67eedad227d5a785f91dbd770aab90de5be19`.

The earlier sibling `.../ddm_mt1_20260814/retained/` and its T4 seal revisions
are preserved but superseded: adversarial review found that their loss had the
Road→Lane direction reversed. They are not result authority and must not be
fired.

## SEALED T4 SIGN GATE

Disposition: **QUEUED-WITH-A-FIRE-ORDER**. Owner: **MAIN sole Modal scorer-lane
router**. Consumer store:

`/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/multitoken_978/ddm_mt1_20260814/optimal_form_r2/retained/t4_sign_gate_r1/`

The fire order is `SEALED_FIRE_ORDER.json`, `5,481 B`, SHA-256
`892cf1e0a6d43682dd448b6d58bbefb16fbec64506f8ab79e116e03620fab75c`.
The request is `SEALED_REQUEST.json`, `9,124 B`, SHA-256
`c9d6d62c8115f6c209576a57d4cbf7e40c2191c542473fa0df33bc82af91dffc`.
All nine fire-input records and all nine source/dependency records were
re-hashed after sealing. The worker is deterministic, resumes by immutable
four-pair batch receipts, retains every scorer payload on the Modal volume,
and self-installs pinned `pydantic` and `Brotli` into the locked upstream venv.

Hard cap: `960 s`. Cost assumption: `$0.60/T4-hour`. Hard-cap cost: `$0.16`.
The exact command is embedded in the sealed fire order. Fire only when MAIN
confirms the #978 T4 scorer lane is free, the local claim is terminal, and all
sealed hashes still match.

Because the local sign is negative, **no second train fire order exists** and
no whole-container pricing or n600 authority row was queued.

## IMPLICATION FOR #984

#984 must not consume this MT1 formulation as a positive CP135 Seg leg. The
measured object moved the named Road→Lane surface but did not move total Seg,
and its semantic frame made Pose materially worse even with the exact base
carrier held fixed. If the T4 sign gate confirms this sign, the current
hidden-4/max-mass-0.25 local-simplex formulation is closed, and the advertised
last open major CP135 Seg route is no longer available to #984. The composition
campaign then has to pivot the base/representation—such as the already queued
fresh ps135/HY1 probability-object path—rather than stacking another CP135
adapter and narrating mechanism progress as a score move.

A T4 sign reversal would reopen only this formulation long enough to seal the
charter's second, joint train. It would not itself establish an n600 row or
whole-container economics.

## RECALL EVIDENCE

Searched by content before design and adjudication:

- `.omx/research/`, arm final messages, canonical research index, the
  `sub015_DAG_*` FEED blocks, design/SPEC documents, `main_hot_state.md`, lane
  registry/claims, and canonical task-status/bridge stores with query families
  `#978|multi-token|support|representative|probability state`,
  `Road|Lane|directed edge|spatial introduced`,
  `BG2|HC1|RFO1|EC2|RE1T|JS1B|QS5`, and `#984|HY1|ps135`.
- Generated and filtered the canonical equations registry with
  `tools/list_canonical_equations.py --json` for representation, Road/Lane,
  parse-back, Pose, and gap-decomposition surfaces.
- Read BG2's memo and retained decomposition; HC1; RFO1; EC2 dispatcher and
  worker; RE1T; JS1B custody/field materialization; QS5 compensation; the exact
  CP135 receiver/evaluator surfaces; and the live frontier board.

Beyond the charter's seeds, the corpus showed that CP135 already combines
semantic tokens and learned frame state, so “add a latent” would be a fake
#978 implementation. It also showed the live #984 alternative base route in
TF1/HY1 and the GDL1 class-pair coder queue. The former changes the handoff:
#984 has a named pivot if T4 confirms the negative. The latter did not change
this build because a scorer-free edge coder does not satisfy #978's jointly
learned receiver-native object. No prior concrete #978 candidate store or
active same-lane owner was found in the searched task/status scopes.

## VERIFICATION AND BOUNDARIES

- Focused tests: `5 passed`.
- Ruff: clean on all new Python files.
- Python compilation: clean.
- `git diff --check`: clean.
- Payload-retention static census on all four production Python files:
  `0 findings / 4 files`.
- Worker dependency closure: passed and sealed.
- Two distinct `review_tracker` clean passes recorded for every Python file
  with trackable entities; the package-only `__init__.py` has no entities.
- Lane registry validates; the local claim is terminal and the paid T4 claim
  remains unclaimed until MAIN fires.

Not measured: T4 sign, whole-container archive bytes, exact Seg/Pose at n600,
contest-CPU, contest-CUDA exact evaluation, `upstream/evaluate.py`, or a score.
The exact pointer did not move, sub-0.15 was not achieved, and this unit did
not produce goal progress under the repository's definition.

Current effective frontier remains CP135
`S=0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`. Current
own-vehicle frontier remains LC2
`S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, adjudicated, n600]`.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN sole Modal scorer-lane router; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/multitoken_978/ddm_mt1_20260814/optimal_form_r2/retained/t4_sign_gate_r1/`; fire trigger: the #978 T4 scorer lane is free, the local build claim is terminal, and every sealed request/source hash matches; action: run the exact argv in `SEALED_FIRE_ORDER.json`, then harvest `FINAL_RESULT.json` and complete remote custody.
- **QUEUED-CONTINGENT** — owner: MAIN/#978 successor; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/multitoken_978/ddm_mt1_20260814/optimal_form_r2/short_joint_train/`; fire trigger: harvested T4 `FINAL_RESULT.json` has `positive_t4_sign=true`; action: seal, but do not fire until separately claimed, the resumable n120-train/n120-heldout joint train with zero-pose-damage and exact-integer-flip endpoint gates. If T4 is non-positive, fold this row without creating the train order.

## LIVE-HYPOTHESES

- **CPU/CUDA sign reversal remains possible.** LC2 already demonstrated large, sign-relevant CPU/CUDA component drift on identical bytes, so the two-flip-scale local Seg difference and Pose response require the sealed T4 reproduction before authority closure.
- **Joint pose-conditioned support may be a distinct formulation.** The simplex improved the named Road→Lane surface while Pose caused the decisive failure; conditioning or jointly solving the existing frame-0 carrier, in the QS5 in-compile manner, could preserve that local semantic gain without reviving an explicit site list. This is plausible but untested and must register as a new formulation rather than silently reopening this one.
- **A changed base may expose more useful support geometry.** The current CP135 tokens leave only a tiny Road→Lane gain; the queued ps135/HY1 probability-object path changes the base and may move the support/pose coupling enough for #984. No number from this screen transfers to that base.

## DEAD-ENDS

- **BG1 bilinear gate:** closed by BG2's heldout incremental `R²=-0.018448`, `p=0.89781`; do not retry it as #978.
- **HC1 direct C1 substitution:** closed on the prior exact row and again here (`2,588` heldout errors and `d_pose=0.0057630776`); direct token replacement is not the receiver-native multi-token object.
- **Explicit changed-site transmission:** excluded by the charter and not present here; renaming a transmitted mask/list as “support” remains fake.
- **Pre-correction MT1 screen and T4 seals under the sibling `retained/` store:** superseded because the Road→Lane loss direction was reversed; preserved only for custody and must never be fired.
- **Hidden-4, max-support-mass-0.25, two-stage local-simplex formulation on CP135:** locally closed at `n32` because Road→Lane improved by 9 but total Seg improved by 0 and Pose MSE worsened by `0.0001029022`; verdict scope is this formulation and this advisory axis pending its already-sealed T4 confirmation.
