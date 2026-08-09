# DDM-DT1 ANS decode wall-clock gate receipt

Date: 2026-08-09  
Owner: codex arm `ddm_dt1`  
Axis: `[macOS-CPU advisory, scorer-free]`  
`score_claim=false`

## Outcome

**AFFORDABLE on the measured Mac inflate-only surface.** The retained ANS token
payload is real, lossless, and 2,120 bytes smaller than the shipped Range token
payload. The clean Range decode plus CPU render control took 792.773501 s, so
the measured inflate-only headroom was 1,007.226499 s. The later retained-object
n600 verification observed ANS 62.923605 s slower than Range under a different,
drifted shared-work regime. That is still only 6.25% of the measured headroom.
The entropy coder itself was 0.110952 s faster under ANS.

| frames | Range whole decode s | ANS whole decode s | ANS - Range s | ANS / Range | timing surface |
|---:|---:|---:|---:|---:|---|
| 2 | 1.793940 | 1.820909 | +0.026969 | 1.015033 | isolated probe |
| 8 | 7.049708 | 7.080155 | +0.030448 | 1.004319 | isolated probe |
| 32 | 28.558119 | 28.054172 | -0.503947 | 0.982354 | isolated probe |
| 120 | 106.579315 | 105.154962 | -1.424352 | 0.986636 | isolated probe |
| 600 | 741.952889 | 804.876494 | +62.923605 | 1.084808 | later sequential retention verification; order-confounded shared-work drift |

Every table row reconstructs all `393,216` tokens per measured frame. The n600
rows each reconstruct `235,929,600 / 235,929,600` tokens exactly. Both decoded
objects have SHA-256
`c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece`.
The ANS terminal state is empty.

The sub-n600 rows are declared **causal-prefix throughput scale points**. A
Range/ANS decoder cannot start from a random frame without the preceding coder
and autoregressive state. These rows make no population-quality, Seg, or Pose
claim, so the known prefix-quality bias is outside their verdict scope.

## Clean pre-retention scaling gate

The declared selection rule was linear unless a power law reduced RMSE by at
least 10%. Linear was selected for Range, ANS, and signed delta over
`n={2,8,32,120}`.

| quantity | intercept s | slope s/frame | RMSE s | selected n600 prediction s |
|---|---:|---:|---:|---:|
| Range | 0.024915820 | 0.888156903 | 0.070194682 | 532.919058 |
| ANS | 0.060464973 | 0.875730488 | 0.017316357 | 525.498758 |
| ANS - Range | 0.035549154 | -0.012426415 | 0.086974953 | **-7.420300** |

| frames | Range residual s | ANS residual s | delta residual s |
|---:|---:|---:|---:|
| 2 | -0.007289 | +0.008983 | +0.016273 |
| 8 | -0.080463 | +0.013846 | +0.094310 |
| 32 | +0.112182 | -0.029669 | -0.141851 |
| 120 | -0.024430 | +0.006839 | +0.031268 |

This was the pre-registered retention gate. It predicted no ANS time penalty,
so retention fired before the later drifted n600 cross-check existed.

## Five-point fit and the n600 regime break

Adding the sequential n600 verification still selects a linear form, but its
large residuals show that the n600 pass was not on the same absolute-time
surface as the clean probe.

| quantity | intercept s | slope s/frame | RMSE s | fitted n600 s |
|---|---:|---:|---:|---:|
| Range | -13.145600 | 1.248900 | 15.515865 | 736.194536 |
| ANS | -17.542183 | 1.357871 | 20.737088 | 797.180344 |
| ANS - Range | -4.396583 | 0.108971 | 5.221924 | +60.985808 |

| frames | Range residual s | ANS residual s | delta residual s |
|---:|---:|---:|---:|
| 2 | +12.441740 | +16.647351 | +4.205611 |
| 8 | +10.204106 | +13.759372 | +3.555265 |
| 32 | +1.738912 | +2.144487 | +0.405575 |
| 120 | -30.143112 | -40.247360 | -10.104248 |
| 600 | +5.758353 | +7.696150 | +1.937797 |

The observed n600 whole-pass delta is real for that sequential run, but it is
not a pure coder cost. Shared selected-logit work increased from 691.143654 s
in the Range pass to 754.636273 s in the later ANS pass, a 63.492619 s shift.
The coder component moved in the opposite direction: Range 1.563080 s versus
ANS 1.452127 s, delta -0.110952 s. The affordable verdict does not require a
claim about the sign of a sub-minute, order-confounded whole-pass delta because
both +62.923605 s observed and -7.420300 s projected are far below 1,007.226499 s.

## Component profile and optimal-form decision

At the clean n120 point:

| component | Range s | ANS s | Range share | ANS share |
|---|---:|---:|---:|---:|
| shared neural forward | 102.871461 | 101.537191 | 96.5210% | 96.5596% |
| probability table | 1.582533 | 1.564038 | 1.4848% | 1.4874% |
| entropy coder | 0.250717 | 0.222355 | 0.2352% | 0.2115% |
| state update | 1.838075 | 1.789862 | 1.7246% | 1.7021% |

The source dependency structure was verified against the read-only CPR1 intake:

- frames are serial because frame `f` consumes frame `f-1`'s decoded output;
- groups are serial because group `g+1` consumes the current frame state filled by group `g`;
- positions inside a group are already one vectorized coder call.

An n8 CPU-thread control measured 16.3191 s at 1 thread, 10.1547 s at 2,
7.13190 s at 4, and 7.42954 s at 6. Four threads were selected for both
coders. The attempted 8-thread shell observation produced no valid timing row
and is excluded.

Interleaved multi-state rANS and a Rust/native coder port were not built. On
this receiver they target roughly 0.2% of decode time, while the identical
neural forward is roughly 96%. This is a FORMULATION/current-receiver routing
decision, not a family-wide claim about interleaved rANS.

## End-to-end Range control and headroom

| quantity | measured seconds |
|---|---:|
| exact n600 Range decode | 536.960713 |
| CPU render compute | 255.812788 |
| decode plus render | **792.773501** |
| complete control process | 797.539986 |
| inflate-only headroom against 1,800 s | **1,007.226499** |

The literal shipping `inflate.sh` was not executed because its main path
requires `torch.cuda.is_available()` and this Mac has no CUDA. The control
calls the same committed receiver, semantic renderer, carrier renderer, resize,
round, and uint8 operations on explicit CPU. Headroom excludes checkout, LFS,
dependency fetch, evaluator data loading, and the scorer pass. It is not a
contest-host wall-clock claim.

The completed render produced 3,662,409,600 bytes with SHA-256
`a18eb42a8da9399bcc03e795e17597bfbd459412dbb37990117665f48c4c0353`
before success-only cleanup. A later interrupted restart scratch at the same
preallocated size was separately certified as
`3204b6f126665758795928c4abd00f7bff6a2c29e48dabdf8c65a473bd45378c`
and removed. Its machine-readable cleanup record is
`/Volumes/VertigoDataTier/pact/ddm_dt1_20260809/range_e2e/interrupted_restart_cleanup.json`.

## Retained payload and custody

Materialization completed 28 atomic int16-code chunks in 586.413475 s. The
storage preflight found 85,839,392,768 free bytes and required 8,589,934,592.

| object | bytes | SHA-256 | status |
|---|---:|---|---|
| shipped/re-encoded Range token stream | 116,980 | `948379872ff81a4e5d948ec301c143be00ebd0033544c8abdfb4af0f4c4a15eb` | byte-identical control passed |
| retained ANS token stream | **114,860** | `a0b18dc0803ef541d3eb265bba5380f7aa067593f6af584b0891ded5bdd74488` | retained and n600-decoded exactly |
| retained result JSON | 15,828 | `5c15f38ab68df68c09a5859d17d19e4247f90e76457282edccbc8a34d060916c` | complete |
| chunk manifest JSON | 16,813 | `23089d6f627e1da56a3f947900727e94ee4a99d1a2ce30fd582aeeac3130caea` | 28/28 complete |

Durable roots:

- `/Volumes/VertigoDataTier/pact/ddm_dt1_20260809/probe_result.json`
  (17,311 B, SHA-256 `093a1e03e187e7d722d32d2d3108ae1eba3c39ca541ae418edd0ed120addeaa2`)
- `/Volumes/VertigoDataTier/pact/ddm_dt1_20260809/range_e2e/range_e2e_result.json`
  (5,909 B, SHA-256 `79cd506ef42ffa6b5a0f5770203cf96a9c507ebb62e0bc00f83f5b11e1bf1129`)
- `/Volumes/VertigoDataTier/pact/ddm_dt1_20260809/retained/retained_n600_result.json`
- `/Volumes/VertigoDataTier/pact/ddm_dt1_20260809/retained/ans_n600.bin`
- `/Volumes/VertigoDataTier/pact/ddm_dt1_20260809/retained/chunk_manifest.json`
- `/Volumes/VertigoDataTier/pact/ddm_dt1_20260809/retained/coder_checkpoints.json`

## Provenance and boundaries

- archive: 191,052 B, SHA-256
  `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`;
- receiver commit pin: `5de03569ad`; receiver SHA-256
  `d9689091430b31b37f5f12d2eaa8025187f7f08899ae1b99ba43a30480b7ac4f`;
- decoded model SHA-256
  `62dd72dfa0858a25ca32bdee1e536627a17883b6fc7efd7cd5b2de7b13b84517`;
- measurement HEAD: `9a1b483b220eb67c560521cc65ad1efe93cffbab`;
- Python 3.11.15, Torch 2.10.0, NumPy 2.3.4, constriction 0.5.0,
  macOS 26.4 arm64, four Torch threads, one interop thread;
- recorded encoder argv was reused from
  `/Volumes/VertigoDataTier/pact/ddm_pr130_encode_tokens_metal_20260809/run/launch_manifest.json`;
- `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/` remained read-only.

Not measured: literal CUDA `inflate.sh`, Linux x86_64/T4 timing, whole-CI
checkout/LFS/dependency time, scorer time, `upstream/evaluate.py`, an assembled
ANS archive, or any exact score. Absolute Mac seconds do not transfer to the
contest host. The ratio is more portable than absolute time, but even the n600
ratio is order-confounded and is not promoted without a contest-host run.

The PR130 baseline remains **S=0.172141297491896447 at 191,052 B
`[contest-CUDA, DALI GT, n600]`**. This arm did not move it. The 2,120-byte
token saving is a receiver-valid retained component, not an archive or score.

## RECALL EVIDENCE

Sources and exact query families:

- `rg -l -i 'ANS|RangeDecoder|range coder|entropy coder|decode wall.clock|constriction' .omx/research --glob '*.md' --glob '*.json' --glob '*.jsonl'`
  searched the full research corpus (7,773 matching files in scope), followed
  by bounded content reads of the relevant CPR1/RC1 receipts;
- the same content query was run over `CANONICAL_RESEARCH_INDEX*`,
  `sub015_DAG_*`, `harness_tasklist_bridge_20260803.jsonl`, and
  `canonical_task_status.jsonl`;
- `.venv/bin/python tools/list_canonical_equations.py --json` was filtered for
  `ans|range|entropy|decode` and surfaced, among others,
  `pr95_family_l30_range_arithmetic_coding_categorical_v1` and
  `decode_determinism_integer_arithmetic_v1`;
- source reads covered `codec_hpac_integer.py`, the committed receiver, RC1's
  receipt, `ANS_REAL_TABLE_MEASUREMENT.md`, and the wall-clock instrumentation
  precedent in `wallclock_burndown_build_20260715.md`.

Beyond the charter seeds, recall found that RC1 had already corrected the
"805 seconds of ANS" wording as shared-work over-attribution; the prior n600
real-table run retained lengths but discarded both word streams; and the
repository already had deterministic integer-decode and `perf_counter`
instrumentation precedents. Those findings changed the work from an ANS-only
timer into a component-profiled paired receiver, added the byte-identical Range
control, reused the exact encoder argv, and made payload retention plus restart
custody part of this arm. No earlier completed `ddm_dt1` wall-clock result was
found in the searched scopes.

## Validation

- `ruff check` passed for the harness and focused tests;
- `pytest -q experiments/tests/test_ddm_dt1_ans_decode_wallclock.py` passed
  `5/5`;
- the integration run passed archive, receiver-ancestor, constriction-version,
  Range byte-identity, ANS length, exact-token, and ANS empty-state guards;
- both Python files receive two review-tracker passes before serializer commit.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER**. Disposition: assemble the split-Brotli plus retained-ANS PR130 archive, parse it back, and measure the literal runtime before any score claim. Owner: MAIN's PR130 archive-assembly successor. Consumer store: `.omx/state/main_hot_state.md` `NEXT_BOUNDARIES`, with task identity resolved through `.omx/research/harness_tasklist_bridge_20260803.jsonl`. Fire trigger: consume retained result SHA-256 `5c15f38a...916c` and ANS payload SHA-256 `a0b18dc0...4488`, claim the Linux/T4 lane, then require exact receiver output and total runtime below 1,800 s before `upstream/evaluate.py`.

## LIVE-HYPOTHESES

- A split-Brotli plus ANS archive will materialize at about 188,029 bytes without changing decoded video. This is plausible because the model and token sections are disjoint, the selector-explicit receiver is built, and the retained ANS payload decoded all n600 tokens exactly; archive framing and parse-back are still untested.
- The ANS/Range ratio will remain close to one on Linux/T4 even if absolute time changes. This is plausible because roughly 96% of work is the identical Torch model forward; the Mac n600 ratio remains order-confounded until tested there.
- Neural-forward intra-op tuning can reclaim materially more runtime than entropy-coder work. This is plausible because shared selected-logit/context work is about 96% of decode time while the coder is about 0.2%.

## DEAD-ENDS

- The "805 seconds of ANS overhead" premise is closed. It extrapolated whole shared decode and mislabeled it as coder cost; measured coder work is about 0.2% and ANS coder work was slightly faster.
- Frame-parallel and group-parallel decode are closed for this receiver formulation because they violate causal AR and checkerboard refinement dependencies. Positions within a group are already vectorized.
- Interleaved rANS and a Rust/native coder port are closed as the next optimization rung on this receiver. They target about 0.2% of wall clock and cannot materially improve the 1,007-second affordability margin; this does not kill either technique on a coder-dominated receiver.
- A two-point or n2-only linear extrapolation is closed for this gate. Five sizes were measured, both declared fit residuals and the n600 regime break are preserved.
