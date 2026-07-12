# FreSh and corpus-gated meta-init for fewer witness epochs (2026-07-12)

Status: **FreSh build/test/triality complete; governed GPU slice commands dry-compiled, but execution is BLOCKED because this managed session has no Metal device, both native-app routes were refused before launch, and the current governor admission check refuses coexistence under the live-arm memory load. No epochs-reduction number exists. Meta-init is intentionally not started while the FreSh measurement gate remains open.**

Axis and authority: all initialization selection and faithful-slice training rows in this memo are `[macOS-MLX research-signal]` or `[macOS-CPU advisory]`, NON-PROMOTABLE. They make no score claim and do not move `reports/latest.md`. A score claim still requires the untouched `upstream/evaluate.py` on exact archive bytes on a contest-equivalent CPU/CUDA axis.

## Operator objective and formulation scope

The frozen SegNet forward/backward is the measured dominant epoch cost (about 95% after the already-landed 16.9x grouped-backward speedup). The useful clock variable is therefore epochs to a fixed realized-through-R `d_seg`, not more acceleration of the already-cheap coordinate trunk.

This build tests one still-open formulation only:

- **OPEN:** from-scratch, init-time FreSh selection of the self-orient along-tangent frequency and first-layer periodic bias, before the structured prefit/training trajectory.
- **SETTLED; do not reopen:** bounded warm-start `self-orient ON, freq_along=26` from mod32cap ep650 through ep700. That arm was marginally worse than `self-orient OFF` at every trained cell; the ep700 delta was `+3.2e-05 d_seg` with a single-seed noise floor explicitly unmeasured. Its verdict scope is `.omx/research/owed16v2_verdict_20260710.json`.

The warm-start result refutes a fixed along-heavy fine-tune allocation. It does not adjudicate whether a spectrally selected directional basis changes *early partition formation*. FreSh may retain the baseline frequency; the measured 3.2x endpoint is a candidate, not a presumed optimum.

## STORES CONSULTED

- `CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`
- `.omx/research/fast_witness_training_oss_survey_20260712.md`
- `~/.claude/projects/-Users-adpena-Projects-pact/memory/max_throughput_over_bit_identity_operator_override_20260712.md`
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md`; `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`
- `.omx/research/owed16v2_verdict_20260710.json`; `.omx/research/curriculum_candidate_pool_p0_20260710.md`; `.omx/research/default_off_decision_table_20260710.jsonl`
- `src/tac/canonical_equations/anisotropic_basis_two_regime_allocation_20260707.py`
- `experiments/train_levelset_witness_realized_through_R_mlx.py`; `experiments/train_witness_realized_through_R_mlx.py`
- `src/tac/witness_dsl/curriculum_dsl.py`; `src/tac/witness_dsl/spec_v9_cgauge.py`
- FreSh, arXiv:2410.05050 and its released implementation; Fourier-reparameterized training, arXiv:2401.07402

## FreSh executable law

For a channel-first target or initialized boundary signal `A`, use the unshifted two-dimensional DFT and omit DC:

`S_d(A) = sum_c sum_{i+j=d} |FFT2(A_c)[i,j]|, d=1,...,n`, with `n=64`, followed by one L1 normalization.

For normalized spectra `p` and `q`, the exact discrete unit-bin Wasserstein distance is:

`W1(p,q) = sum_d |CDF_p(d) - CDF_q(d)|`.

The selected initialization is conditioned on the measured thin/dashed class-1 residual rather than the dominant all-class edge mass. For fixed target-derived weights `omega_t`:

`j* = argmin_j mean_t W1(S(omega_t * boundary(L*_t)), S(omega_t * boundary(argmax SegNet(R(witness_init_j)))))`.

The unweighted global-boundary `W1` is retained as a diagnostic but cannot select the candidate. A regression with a dominant unrelated normal edge and a weak variable dash comb proves that outside mass cannot move the residual-conditioned choice.

The target samples are deterministic evenly spaced cached `L*` maps. The candidate is the cold witness's frame-1 output rendered through the actual `R` operator and reread by the frozen MLX SegNet; its 4-connected argmax boundary map is transformed immediately, so no image-sized candidate corpus is retained. The matched slice is explicitly a **bare-witness V9-core** surface: the pair-dependent seed/island composer and witness-alone seed loss are disabled in both DSL arms, residual mode fails closed, and an epoch-0 lane-band composer fails closed. This makes the single shared cold output exact for the declared slice rather than assuming all V9 composers are pair-independent. Exact ties retain the incumbent configuration.

The ordered along-tangent candidate set keeps the incumbent first, then the measured-deficit ladder:

`{f_current, 8, 8*sqrt(3.2), 8*3.2}` with stable de-duplication.

The bias candidates are `k=0.0,0.1,...,3.0`. One dedicated standardized RNG vector is scaled by every `k`, matching the existing FINER stream, so `k` is the only bias confound. `k=0` is the SIREN zero-bias baseline. The selector is init-only: it seeds the directional columns and bias before the structured prefit; the training loop itself is unchanged. The matched control executes the identical one-pass self-orientation/re-render path but fixes `(freq_along=8,bias_k=0)`; early reorientation is therefore not a treatment confound.

Default treatment cost is explicit, not called free: the selection sweep uses `3 frequencies * 31 bias widths = 93` frozen-SegNet forwards/pair-equivalents versus `1` for the matched control, and both arms then perform one mandatory committed epoch-zero scorer forward. End-to-end init accounting is therefore `94` versus `2`. The FFT and selection are backprop-free, but the scorer forwards have real wall cost. Fixed-quality receipts use `N_total=N_init+P*E`, so a one-epoch training win at `P=8` is correctly a net call loss (`18` control vs `102` treatment pair-equivalents) when all 93 candidates are scored.

## FreSh triality

| Leg | Durable surface | Contract |
|---|---|---|
| DSL | `FreShInitControl`, `FreshFrequencyShift`, `FreShFixedQualitySlice` in `tac.witness_dsl.curriculum_dsl` | all default OFF; matched basis/composer surface; typed per-epoch eval/checkpoint measurement cadence |
| DAG | `FEED-fresh-init-20260712` in the canonical witness DAG | from-scratch init-selection formulation; exact execution blocker; warm-start along26 remains closed |
| Equation | registered `fresh_frequency_shift_init_v1` | residual-weighted spectrum/W1, candidate law, `N_total=N_init+P*E`, and wall-to-first-crossing identity |
| Runtime/provenance | `tac.witness_init.fresh_frequency_shift`, runtime receipt, checkpoint cfg | bounded-memory selection; candidate/target hashes; selected frequency/bias; resume inheritance; no `/tmp` evidence |
| Launcher | typed `CrucibleV7LaunchConfig.with_dsl_lever_factories` | fixes the pre-existing typed-v7/V9 `--dsl-lever` crash; regenerates typed hash + flag manifest after composition |

## Matched faithful-slice protocol (pre-registered before either arm)

1. Compile both arms from the same typed witness DSL program. The only treatment delta is the default-off `FreshFrequencyShift` Lever. Do not add naked trainer flags.
2. Use the real cached frozen-scorer targets and real through-R MLX SegNet forward. Run an `n8` integration/trajectory screen first; if the device/storage/governor gates remain green, run `n64` as the primary faithful spectral/per-pair slice. Never use the live `v9_cgauge_432_coherent_arm_20260711` directory or process.
3. The fixed-quality threshold is **pre-registered as `q* = 0.90 * d_seg_baseline(epoch 0)`**, where epoch 0 is the baseline arm's post-init/pre-training through-R verdict. Freeze that scalar before inspecting the FreSh training trajectory. A FreSh epoch-0 value already at or below `q*` counts as zero training epochs.
4. `E_base` and `E_fresh` are the first emitted verdict epochs with `d_seg <= q*`. Interpolation is forbidden. If an arm never crosses within the fixed budget, report right-censoring rather than manufacture an epoch reduction.
5. Report `(E_base-E_fresh)/E_base`, training scorer pair-calls `P*(E_base-E_fresh)`, **total** scorer pair-equivalents including the `2` vs `94` end-to-end init calls, and the explicitly bounded accounted time: FreSh sweep start through the committed epoch-zero state plus epoch-zero-verdict-to-first-crossing time. This excludes and labels common pre-sweep/post-handoff trainer setup rather than calling it full launch wall. Also report seed, the exact matched parsed-config payload/hash, active-GT authority hash, git/upstream hashes, and both linked receipt SHAs. Actual one-time FreSh overhead is never silently zeroed.
6. The faithful slice is advisory and cannot promote the vehicle. A governed, resumable, per-stage-checkpointed `n600` A/B is explicitly owed to the operator-fired real arm; this build does not launch it.

## FreSh build and measurement disposition

Build evidence:

- Portable FreSh frequency/bias law, residual-weighted bounded runtime, exact receipt writers, resume registry/controller state, typed DSL factories, live trainer wiring, typed-launcher composition, fixed-quality parser/receipt tool, canonical equation, and DAG node are implemented.
- The final consolidated FreSh/trainer/equation/typed-launcher/measurement suite is **103/103 green**; the surrounding curriculum-DSL, autoconfig, launcher, and resume-registry suites are **225/225 green**. Focused Ruff checks over the new FreSh and measurement surfaces are clean.
- Governed GPU dry-runs compile and flag-validate: control `204/204` real flags and treatment `211/211`; both pass the DSL-manifest, schedule-provenance, per-run memory-preflight, and safe-compile gates. The current system-admission gate correctly **REFUSES** both dry arms: projected system use is `100.0 GiB`, above the adaptive `82.4 GiB` ceiling, while the live V9 arm remains in scope. The intended fixed-quality commands additionally compose `FreShFixedQualitySlice`, giving per-epoch realized verdicts and preserved per-epoch checkpoints when admitted.

Execution blocker (verdict scope: **environment/measurement**, not lever family):

- The managed command sandbox cannot initialize MLX at import: `RuntimeError: [metal::load_device] No Metal device available`.
- The governed launcher independently refuses a concurrent launch at current memory pressure (`100.0 GiB` projected versus `82.4 GiB` adaptive ceiling). A governor refusal is evidence; it is not bypassed.
- The computer-use runtime refuses both `com.apple.Terminal` and `com.openai.codex` for safety; no UI action occurred.
- The canonical native-Terminal delegation harness then failed before launch with `osascript ... Connection Invalid error for service com.apple.hiservices-xpcservice`; its returned state was `"launched": false`.
- No control or treatment trainer process started. The live `experiments/results/v9_cgauge_432_coherent_arm_20260711` process/directory was untouched.
- Durable fail-closed receipt: `experiments/results/fresh_init_n8_fixed_quality_20260712/measurement_blocker.json`, SHA-256 `2e6305c33c5d616f4c7b0ede0d9f96e26390841ddfa5557ee3d6fb19421f62d6`, schema `tac.witness_init.fixed_quality_blocker.v1`, claim scope `measurement_blocker_no_epochs_reduction_claim`.

Therefore **epochs reduction = UNMEASURED**, **wall-clock reduction = UNMEASURED**, and **no paper-derived or spectrum-derived convergence number is reported**. Reactivation requires a native Metal-capable shell **and a green governed admission**, then running the already-compiled control followed by treatment and passing their run dirs to `tools/measure_witness_fixed_quality.py`. `n64` multi-seed noise estimation and governed `n600` validation remain owed after the n8 gate.

## Meta-init task #211

Not started. The operator required a completed FreSh slice measurement and serializer commit first. The build is sealable, but the measurement gate above is still blocked; starting meta-init would violate the directed sequence. Once reactivated, the next section will inventory compatible, real `levelset_witness_ema_BEST.npz` and preserved stage checkpoints, identify the true independent-task count, and choose Reptile/MAML/hypernetwork scope without treating checkpoints from one contest clip as a fictitious multi-video corpus.
