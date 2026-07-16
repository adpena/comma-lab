# Curvelet optimal-form crux — harvest and completion (2026-07-16)

`lane_id=curvelet_optimal_form_crux_20260715`; `research_only=false`;
`training_launched=false`; `live_c2_touched=false`; `pointer_delta=ZERO`.

## Verdict

`BLOCKED_FAIL_CLOSED / FORMULATION-COMPOSITION IMPLEMENTATION GAP`.

The literal finite curvelet kernel, strict semantic identity, same-width native placement,
train/resume/deploy path, isolated generated receiver, deterministic exact-byte rate matcher,
two-phase measurement receipt, DSL leg, equation leg, and DAG leg are now executable. The complete
frozen form is not honestly fire-ready: arbitrary ground-chart coordinates and post-render
supersampling are not yet implemented op-for-op in the generated receiver. Both combinations fail
closed. The curvelet family remains `OPEN`; no launch command is emitted for a program the receiver
cannot reproduce.

## What the predecessor got right

- The over-broad composition guard was not a curvelet-family conflict. The genuine conflict is the
  scalar global-plane-wave IPE formula; output-space supersampling remains mathematically valid.
- The selected dictionary is a literal finite polar radial/angular wedge with a rotated parabolic
  translation lattice, not the invalidated spatial-Gabor `windowed_curvelet` token.
- Boundary tangents are vectors (`Jt`) while normals are covectors (`J^-T n`).
- Decoder-native orientation should gate the existing 80 columns, not append another counted Fourier
  bank; taper should fold into the first learned layer at deploy.
- Equal learned values are insufficient. The decisive A/B must use the exact final ZIP byte count and
  prove the padding is receiver-inert.
- The six draft files were substantive: literal construction/proof, placement algebra, exact taper
  fold, and transactional equal-ZIP publication were already well tested.

## What was incomplete or unsafe

- Commit `5c0ffb7382` froze only the spec. The six implementations were uncommitted and no runtime
  consumer selected them.
- No versioned `BasisProgramConfig` bound family/version/atom/chart/orientation/taper/AA semantics;
  shape-compatible resume drift could pass.
- Placement had pure operators but no GT-free decoder-native fixed point or receipt.
- `generated_inflate_source()` existed, but packet assembly never selected it and the LVLS1 manifest
  had no literal program. An initial recovery hunk also called generated-only `_io_unpack` from the
  host assembler; the adversarial receiver suite caught this and the host now uses `_read_blob_bytes`.
- Equal-ZIP code had no governed post-inflate/measurement finalizer and no executable transfer law.
- The historical trainer built nonlegacy MLX features before taper and hardcoded the legacy Fourier
  builder for its supersampled grid. Those silent selection bugs are fixed.

## Completed implementation

- Recovered finite dictionary: `literal-polar-curvelet-finite-v1`, exactly 80 columns (4 scaling,
  76 directional), atom-spec SHA-256
  `48df53b84660396adc522fe966cb8e7c631c108332a3529eefe17ee9aaa44f6e`.
- Current recovered source SHA-256 before landing:
  `90bb014e033ebe572c81b87aa6e9e6fd3e5977cc1e697e71762a46799c343c48`; it explicitly does not
  claim the lost historical source hash.
- Structural NumPy proof is `passed=true`: polar factorization and Hermitian error `0`, endpoint
  error `0`, direct/FFT max error `2.384185791015625e-07`, measured scale aspects
  `(1.0, 2.001275827188583, 4.003279530749405)`. Scope is this finite period-two dictionary only.
- Strict `BasisProgramConfig` rejects unknown hashes/versions, uncounted chart dependencies,
  scalar-IPE semantics, inconsistent fold state, and exact-field drift. The isolated typed treatment
  (`native=true`, `kappa=2`, cap `6`) hashes to
  `22d429118981a30fc93dc5902a347483901ce5537e03a28547bd4911b11be891`.
- Trainer paths select NumPy/MLX literal features, persist config/hash and unfolded taper custody,
  refuse appended self-orientation, run same-width native reorientation, use the selected basis for
  supersampled coordinates, and rebuild selected MLX features after taper.
- Byte-close validates the strict checkpoint, folds taper once into `in_proj.weight` with a
  content-bound receipt, embeds literal kernel plus placement source in the generated receiver, and
  applies the same native fixed point in both NumPy oracle and receiver.
- `equal_archive_budget_v2` publishes two deterministic ZIP copies transactionally, hashes original
  members, fixes metadata, proves exact byte equality, and cleans temporary snapshots on success or
  failure. `curvelet_equal_byte_ab_receipt.py` separates `match` from `finalize`, so finalization can
  occur only after both exact matched archives have been inflated and measured.
- `curvelet_equal_archive_transfer_v1` requires n600, scorer batch 32, actual through-R, official
  evaluator parse-back, exact archive hashes/bytes, output-tree identity, and the treatment program
  hash. Equal bytes cancel the rate term and the law evaluates
  `100*delta_d_seg + sqrt(10*d_pose_t) - sqrt(10*d_pose_c)`. Its verdict is formulation-instance only
  and `pointer_authorized=false`.

## Evidence and non-evidence

- `MEASURED_ADVISORY / retained only`: saved-OFF deploy-RGB n600/batch32 finite-truncation row
  `d_seg=0.5048239560` for the earlier literal curvelet versus Fourier `0.4097223155`. It is equal
  values/support, not equal bytes, not fresh training, non-promotable, and not a family verdict.
- `MEASURED LOCAL STRUCTURAL`: proof values above and deterministic unit/integration tests.
- `UNMEASURED_SOFT_UNAVAILABLE`: MLX parity could not execute in this headless session because no
  Metal device is available. NumPy is the reference; no MLX equivalence is claimed by this run.
- `NOT MEASURED`: no new n600 treatment/control `d_seg`, `d_pose`, archive bytes, contest score,
  CUDA parity, or pointer result.

## Exact remaining closure

1. Make the per-pair chart a counted receiver program (pose-carrier `xi` or explicit chart payload),
   and prove the trainer and generated receiver evaluate identical charted coordinates without an
   n600-prohibitive direct sparse transform.
2. Implement `A_s` in generated inflate after the nonlinear renderer and prove `s=1` identity plus
   nontrivial selected-basis NumPy/receiver uint8 equality. Seal fine-grid native-gate semantics.
3. Re-run MLX/NumPy parity on an available Metal host.
4. Only then compile the two governed, resumable, per-stage-checkpointed n600 arms. Independently
   byte-close them, use the receipt driver's `match`, inflate the matched archives, run the official
   scorer at batch 32 on the same contest axis, and use `finalize` to obtain the only admissible
   transfer verdict.

No exact ready-to-fire trainer command is recorded because steps 1–2 are code blockers, not operator
authorization blockers. Firing the stripped isolated kernel would answer a narrower formulation and
could not close the preregistered optimal-form A/B.

## Verification seal

- Pre-seal focused integration: `130 passed, 4 skipped, 1 deselected` (the deselected test requires a
  Metal device); the suite includes the predecessor kernel/rate tests, strict config, DSL/equation,
  generated receiver hardening, pose-carrier byte-close, and trainer source gates.
- Changed standalone Python files pass Ruff. Hot legacy trainer/byte-close files pass `py_compile`,
  Ruff `F/E9`, and `git diff --check`; broad historical lint debt was not mechanically rewritten.
- Final landing requires three identical no-edit passes of the same 130-test suite and serializer
  commit with post-edit content hashes.

## Stores consulted

`CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`; v7.5/v8 specs;
`reports/latest.md`; lane registry; predecessor authority/original prompt/log tail/worktree; Task
#497/#502 proof, invalidation, advisory, DSL/equation, launch-ticket, and DAG artifacts; live per-arm
inbox and broadcast inbox through `2026-07-15T16:56:17Z`.
