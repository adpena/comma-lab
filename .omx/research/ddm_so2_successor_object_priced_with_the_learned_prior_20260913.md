# SO2 — learn the price of a reversible multiscale residual field

Owner/checkpoint: `ddm_so2`. Charter: `charters/ddm_so2_successor_object_priced_with_the_learned_prior_charter_20260913.md`.
Contract: `../tmp/codex_runs/_common_contract.md`. `research_only=true`, `score_claim=false`,
`promotion_eligible=false`, `pointer_moved=false`. Design and ONE rung specification only.

**Recommendation: one construction, block-local modulo-five lifting, followed by one HPAC refit.**
Keep the exact move-49 field, including its renderer-specific pre-distortion, but describe each
2×2 block as an anchor plus three exact categorical differences. Pack the four planes into each
existing 64×64 HPAC patch and train the existing five-class prior on that representation. This
changes the symbols and neighborhoods the learned model must predict without removing any render
freedom. It is an untested rate hypothesis, not an established successor or an originality claim.

**There is no measured learned-prior byte estimate for this representation.** The charter both
requires pricing by a prior trained on the representation and forbids this arm from training or
building it. The honest design output is therefore a precise conditional price expression and a
falsifiable numerical target. Assigning a 100–120 KB forecast before the fit would repeat SO1.
The expected *measurement* is the retained packed-prior-plus-RC64-stream length; its expected
numerical value is UNKNOWN. No generic-coder ratio, field-area ratio, or ancestor fit is substituted.

## Current arithmetic and custody

Labels: **M** means measured in the cited receipt; **V** means independently verified here by
reading/hash/stat of existing bytes; **D** means derived; **H** means untested hypothesis. Historical
M observations keep their original object and axis. Source and input hashes are in
`ddm_so2_20260913/source_pins.json` and `input_pins.json`.

| Quantity | Value | Authority |
|---|---:|---|
| Move-49 archive | V 179,153 B | Existing archive stat/hash; scorer-free custody |
| HPAC member | V 11,629 B | RX1M header in that archive |
| RC64 stream | V 118,938 B | Tail minus 96 B prefix and 64 B RLC1 rider; SJ1 subset7 encode receipt |
| Actual current prior + stream | D 130,567 B | 11,629 + 118,938 |
| Everything outside prior + stream | D 48,586 B | Renderer 29,862 + carrier 18,450 + framing/riders 274 |
| Charter comparison | 130,525 B | Move-48 prior + stream, 42 B stale on move 49 |
| Charter's 5% pass ceiling, retained unchanged | D **123,998 B** | floor(0.95 × 130,525); includes ALL new side information |
| Sub-0.12 subsystem ceiling at held distortion | D **106,052 B** | Whole archive must be at most 154,638 B |

The stale comparison is corrected in the accounting, **not relaxed in the acceptance test**.
At the 5% pass ceiling the saving against actual move 49 is D 6,569 B, only D 26.8% of the
24,514.2 B rate demand. Passing this rung does not solve the cross.

The inherited measured components are `d_seg=0.00010287`, `d_pose=0.00000455`, n600,
`[contest-CUDA T4]`, from
`experiments/results/modal_auth_eval_mirror/contest_auth_eval_ddm_sj1_pass7_cuda_20260912.json`.
Using the source evaluator's expression and denominator:

```text
D49 = 100 × 0.00010287 + sqrt(10 × 0.00000455) = 0.01703236878161602
r = 25 / 37,545,489 = 6.658589531221714e-7 S/B
S(J), conditional on unchanged realized frames = D49 + 25 × ((48,586 + J)/37,545,489)
J = 123,998 -> S approximately 0.13194897034725284
J <= 106,052 -> S < 0.12 at held components
```

These are DERIVED conditional scores, never new evaluator rows. The final bits of floating-point
arithmetic depend on preserving evaluate.py's division-before-multiplication order. The stored
pointer remains `0.13632299781031237`; it is not replaced by a reassociated calculation.

Pinned real inputs:

* Archive `/Volumes/VertigoDataTier/pact/ddm_sj1_pass7/candidate/candidate_runtime/archive.zip`,
  SHA `73e41a6620bd4ea3aaf236eff9de46391857907527358e8eb40ded0925a1c214`.
* Current field `/Volumes/APDataStore/pact/ddm_sj1_pass7/rlc1/fields/subset7.u8`,
  V 117,964,800 B, SHA `fdf2255f60364dcd1e67fb7de107c0640f5fe3a39a7b575c76c86636660efd1d`.
* The same field's NPZ `/Volumes/VertigoDataTier/pact/ddm_sj1_pass7/admission_pass7/field_admitted.npz`,
  SHA `6056c90585763078e2bea8dea32b14abf57462d39eb56c17d46fd21bd702ae0f`.
* Full-state warm initializer `/Volumes/VertigoDataTier/pact/ddm_dpi1/train_inputs/init_depths.pt`,
  SHA `f05bae5b2696b1e96817c214d9b1d94ecb64ca18f1c6c30b6e398cd5d4b22072`.
  DPI1's `INIT_DEPTHS.json` identifies the nine depth buffers and the actual CL2 epoch-60
  ancestor. This is an initializer, not a move-49 trained model or a credited saving. Its
  +37 B negative was on the untransformed move-48 field. We reuse its complete state to avoid
  silently resetting learned depths; we do not rerun that already-closed depth-restoration bet.

## RECALL EVIDENCE

The charter and common contract were read in full, as were PROGRAM, the craft manual, SO1's design
and GS3 Addenda 47–59. Governing CLAUDE/AGENTS sections included NO FAKE, retention, checkpoints,
source immutability, CLI verification, axis separation and serializer discipline. The live board
and canonical pointer both identify move 49. The contract's August frontier paragraph is historical
and superseded by these current stores and the explicit SO2 charter.

Own recall, beyond the charter's named seeds:

* Content queries across `.omx/research/` Markdown, including arm receipts:
  `polyphase|space.to.depth|modulo.?5|mod.?5|learned.{0,20}residual|lane.{0,15}separat|two.alphabet|quotient.{0,20}occupancy`.
  Full matched-path list: `content_recall_paths.txt`; focused September results and extracts are
  retained beside it. No measured n600 learned-prior price for this exact lifting map was found
  in that scope. Most polyphase hits concern YUV packing, which is a different object.
* Ran `.venv/bin/python tools/list_canonical_equations.py --json`; complete output and a 19-row
  relevant subset retained. `hpac_prior_capacity_slope_v1`,
  `coder_strength_substitutes_for_capacity_v1`, `generator_form_fit_error_entanglement_v1`,
  `model_section_edit_container_break_fee_v1`, and `hpr1_counted_section_refit_debt_v1` were
  found. They rule out transferring a generic entropy, generator fit, or model-size saving.
* Queried `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED surfaces,
  `.omx/state/canonical_task_status.jsonl`, and `docs/` for
  `polyphase|learned.prior|hpac|quotient|successor.object`; retained `index_dag_tasks_recall.txt`.
  Queried the lane registry for SO1/SO2/CL2/HPR1; the hot board already assigns this design to SO2.
  No training/scorer slot is claimed by this design-only arm.
* **D3B**, `ddm_d3b_lossless_lane_factorization_20260826.md`: nine real n600 exact decodes;
  its best conditional Lane packet costs M 64,276 B and complete subsystem M 127,499 B,
  207 B worse than its matched joint coder `[macOS-CPU advisory, exact rate]`. This removed a
  naive Lane-separated proposal from this memo. Lane's 33.5% attributed surprise is not the cost
  of a binary occupancy stream; non-Lane confirmations and geometry also cost bytes.
* **HPR1**, `ddm_hpr1_hpac_receptive_field_shape_rung_20260911.md`: patch 64, five labels,
  23 causal current-plane taps, nine past-plane taps, patch temporal summary, 20,416 stored
  values. Its temporal dilation rung lost M 812 B to its refit control despite favorable
  conditional-information ranking. Spatial mixed-spacing's −6,740 ranking bytes are NOT
  borrowed as SO2's expected saving. This changed the proposal from plain spatial rearrangement
  to exact categorical lifting: changing tap geometry alone already has adverse evidence.
* **DPI1 source and receipt**, plus the actual trainer cache validator: the cache must be
  CPU uint8, shape `(600,384,512)`, values 0–4, with a raw-content SHA matching both metadata
  and CLI. Neither a `(600,192,256)` cache nor a new binary/six-label head is admitted by the
  existing profile. This eliminated a superficially simple coarse-plane or two-prior CLI sketch.
* **Current packer source**: CL2's older direct Brotli IHS1 representation is not today's
  RC3H→CK2→Brotli HPAC container. HPR1 `prepare` and SJ1's RLC1 rail supply the current path.
  This changed the encoding specification to use those exact layers, including counted mixer
  state. The trainer has no whole-field pack/encode CLI; claiming it does would invent a tool.
* **NO1 row 2**, `ddm_no1_new_object_derivation_20260826.md`, surfaced through the task ledger:
  the lossless learned-probability lead was already queued, and the old model's bytes must be
  reclaimed in its replacement budget. This memo is one concrete representation experiment
  beneath that lead, not a new duplicate broad-prior project. Its old 30–43% recovery scenarios
  are not forecasts on move 49; the live thresholds above replace them for this rung.

The lightweight Codex memory-registry query found no relevant SO2 precedent; no numerical claim
in this memo is taken from unverified memory. The recall files record bounded searches, not a
claim of exhaustive absence across the entire project.

## The one construction

**Definition, fully specified.** For each frame and each aligned 64×64 patch of F, divide it
into 2×2 cells `(a,b;c,d)`. Put anchors `a` into the patch's upper-left 32×32 quadrant; put
`(b-a) mod 5`, `(c-a) mod 5`, `(d-a) mod 5` into its upper-right, lower-left and lower-right
quadrants respectively. Call this full-size five-symbol field Z. At the receiver read those four
quadrants and reconstruct `(a,(a+u) mod 5;(a+v) mod 5,(a+w) mod 5)`.

The dimensions are D from the real trainer: 64 is its patch size, 32 is 64/2; 384 and 512 are
divisible by 64. There are D 28,800 patches, 29,491,200 anchor symbols and 88,473,600 detail
symbols. **No area-proportional byte credit is taken.** The map is an integer bijection over
all five-label fields, not a lossy coarse field, a fitted pattern table, or a generator fit.
No sixth escape symbol, side support bitmap, topology table, or training-only context is needed.

**Learned price, not a paper coder.** Let theta_Z be the terminal EMA at epoch 60 from the
CL2 profile at lambda 1 trained on exactly Z. Let P(theta_Z) be the current real HPAC pack,
and C(Z;P) the actual RLC1/RC64 stream under that DESERIALIZED prior and the retained counted
correctors. The only price admitted is

```text
J_Z = len(P(theta_Z)) + len(C(Z;P(theta_Z))) + 8
```

The 8 B are the declared format prefix `SO2L` plus version/tile/stride/modulus bytes
`[1,64,2,5]`, physically placed before the RX1M member in the research archive's sole STORED
`p` member. ZIP overhead stays 100 B. Any additional implementation side information increases
J_Z; it may never be silently omitted. A two-prior implementation is a different construction
and is not authorized by this rung.

The trained prior's own cross-entropy `-sum log2 p_thetaZ(Z_i | decoded history)/8` is useful
telemetry, but cannot decide the gate: integer CDFs, adaptive correctors and the actual pack
change its price. DPI1's surrogate-to-real reversal was M 809 B on its own object. All quadrants,
including every zero/nonzero detail decision, go through the real model and coder.

**Why plausible, H.** In uniform 2×2 interiors, all three differences become zero regardless of
semantic class. The learned model can share that event across all five classes, while anchors
retain the partition and exceptions retain jagged pre-distortion. Quadrant packing groups those
events within the prior's existing patch geometry. Unlike SO1, no separately generic-coded
coarse stream must pay 169,100 B. Unlike HG1, no approximation error is hidden in a residual bill.

**Why uncertain.** An invertible transform does not reduce true information entropy. It can only
reduce the finite learned coder's modeling cost enough to repay any new model/framing cost.
The inherited RLC1 Lane geometry treats symbol 1 specially; symbol 1 in a detail quadrant now
means a difference, not Lane. Its counted parameters remain present and the entire penalty is
priced. Quadrant seams and anchor/detail separation may defeat the gain. There is no measured
lower or upper rate band for Z. No confidence interval or positive expected saving is claimed.

**Distortion band, D and object-bound:** delta d_seg in `[0,0]` and delta d_pose in `[0,0]`,
conditional on exact inverse field equality and the unchanged renderer/carrier receiving that
inverse. Anchored absolute components are move 49's measured values above. This is a constructive
zero-debt statement, not a new render/scorer measurement. RGB/raw equality and timed public
receiver closure remain separate success-branch gates. Feeding Z directly to the semantic renderer
is an implementation error; it would not realize this construction.

**Pre-image freedom survives.** SJ1 proposals remain in original F coordinates. For a proposed
`F[t,y,x]=k`, lift the affected 2×2 block and reprice Z. A detail change alters one coded symbol;
an anchor change can alter four. Reconstruct F before the existing realized composite verify
and pose re-solve. Thus every single-token and multi-token proposal is expressible; no smoothness
constraint or born-object distortion has been imported. The current field's single-token family
remains closed: a lossless recode does not invent new scorer repairs. Only its byte admission can
change. Later edits must pay the whole recoded support, not assume one F edit costs one Z edit.

No lossy second stage is queued. RQ1's 116–2,075 cells/B coarsening debts versus D 0.7855 payable
cells/B, its 40.39% best-pass/45.8% cumulative repair observations, and REN1's 2.825× worse true
partition are adverse same-lineage evidence against dropping details. Pass 8's 0.115 admitted
fraction is not a discount that can be applied to an arbitrary new error field.

**Single scientific falsifier:** after a passing original-field control, terminal-EMA pack,
independent twin encodes, and full inverse decode equality, **J_Z > 123,998 B** closes this
specified lifting + one-prior + 60-epoch law at INSTANCE/FORMULATION scope. A failed control,
cache pin, resume, or inverse is INSTRUMENT-REFUSED, not a scientific family negative.

## ONE $0 rung — exact tool commands and the adapter specification

Disposition **QUEUED-WITH-A-FIRE-ORDER**. Owner **MAIN / SO2 byte-rung owner**. Consumer:
`/Volumes/VertigoDataTier/pact/ddm_so2_first_rung/RESULT.json`.
Fire at harvest after MAIN binds this specification to an owned reviewed adapter, verifies source,
field and pointer pins, admits SSD/system resources, and claims the single Metal training slot.
There is one new training run, no parameter sweep, no Modal and no scorer. No command here was
launched by SO2. The adapter is deliberately not installed by this design-only arm.

The exact command sketch consists of the following sequential stages. There is no existing
one-command stock tool that accepts both a new field AND a new prior on the current RLC1 rail;
the small composition adapter below is part of MAIN's one rung, not a fictitious CLI capability.

1. **Prepare and bind.** Verify the source field and NPZ hashes above. Materialize Z in chunks
   of at most 120 frames, preserving stage files plus SHA/bytes; atomically publish `Z.u8`,
   `Z.npz` (600 decimal-string keys), and `cache.pt` containing
   `{'seg': torch.from_numpy(Z), 'spatial_token_sha256': sha256(Z.tobytes(order='C'))}`.
   CPU uint8 and shape/range are mandatory. Reconstruct F using the independent inverse and
   compare all 117,964,800 bytes, not a sample. Store the derived raw Z SHA in
   `TRAIN_INPUTS.json.expected_cache_content_sha256`; also pin the file SHA of cache.pt.
   This new hash cannot be honestly precomputed here without building the forbidden representation.
   Preserve complete initializer `init_depths.pt`, source pins, argv and source archive/runtime
   hashes in this manifest. Never re-use `--cache` pointing to the original F.

2. **Control before interpreting the rung.** Use the existing pricer verbatim, with its real
   environment overrides, against move 49's own field. Its independent processes must reproduce
   the live archive exactly. MAIN places each long encode behind the governed launcher.

```bash
export SJ1_RLC1_ROOT=/Volumes/VertigoDataTier/pact/ddm_so2_first_rung/control
export SJ1_RLC1_BULK=/Volumes/VertigoDataTier/pact/ddm_so2_first_rung/control_bulk
export SJ1_RLC1_LIVE=/Volumes/VertigoDataTier/pact/ddm_sj1_pass7/candidate/candidate_runtime
export SJ1_RLC1_CONTROL_NPZ=/Volumes/VertigoDataTier/pact/ddm_sj1_pass7/admission_pass7/field_admitted.npz
.venv/bin/python experiments/ddm_sj1_rlc1_price.py init
.venv/bin/python experiments/ddm_sj1_rlc1_price.py encode --field control --tag primary
.venv/bin/python experiments/ddm_sj1_rlc1_price.py encode --field control --tag repeat
```

3. **Actual trainer, exact flags.** This is the exact governed command generator, after preparation.
   It reads the newly computed cache-content SHA instead of inventing a value. The launcher flags
   and every training flag were verified against current argparse and the profile table. The
   complete 60-epoch terminal EMA is the preregistered selection, never `best.pt` chosen by entropy.
   Run this Python command from the repo root; the script exits after launching, so MAIN harvests
   the named done receipt before stage 4. The first invocation uses the fresh launcher root;
   subsequent invocations use a new `launch_train_NNN` root and the same causal training outputs.

```python
# SO2_TRAIN_COMMAND_BEGIN
import json, subprocess
from pathlib import Path
r = Path('/Volumes/VertigoDataTier/pact/ddm_so2_first_rung')
m = json.loads((r/'TRAIN_INPUTS.json').read_text())
sha = m['expected_cache_content_sha256']
assert len(sha) == 64 and all(c in '0123456789abcdef' for c in sha)
trainer = ['.venv/bin/python', 'tools/train_ddm_cl1_hpac_capacity.py',
 '--profile', 'cl2_shipped_ladder', '--cache', str(r/'cache.pt'),
 '--expected-cache-content-sha256', sha,
 '--init', '/Volumes/VertigoDataTier/pact/ddm_dpi1/train_inputs/init_depths.pt',
 '--expected-init-sha256', 'f05bae5b2696b1e96817c214d9b1d94ecb64ca18f1c6c30b6e398cd5d4b22072',
 '--epochs', '60', '--batch-size', '8', '--eval-batch-size', '4', '--eval-every', '2',
 '--lr', '0.003', '--lr-exponent', '0.0002', '--lr-bits', '0.01', '--bit-eps', '0.000001',
 '--rate-lambda', '1.0', '--qat-fraction', '0.5', '--init-bits', '8.0',
 '--channels', '64', '--patch', '64', '--delta', '2', '--frame-dim', '8',
 '--past-dilation', '1', '--conv-a-dilation', '1', '--norm-mode', 'none',
 '--activation', 'relu', '--frame-scale', '--weight-bound', '127', '--activation-bound', '127',
 '--weight-scales', '--weight-exponent-min', '-6', '--spm', '--target-mode', 'raw',
 '--seed', '20260716', '--ema-target-seed-fraction', '0.01', '--device', 'mps',
 '--save', str(r/'train/terminal.pt'), '--out', str(r/'train/RESULT.json'),
 '--min-free-bytes', str(40 << 30)]
latest = r/'train/terminal.checkpoints/latest.pt'
if latest.exists():
    trainer += ['--resume-from', str(latest)]
terminal = r/'train/terminal.checkpoints/qat_stage_end_epoch_0060.pt'
assert not terminal.exists(), 'Training already complete; harvest and pack, do not retrain.'
n = 0
while (r/f'launch_train_{n:03d}').exists(): n += 1
command = ['.venv/bin/python', 'tools/launch_detached_process.py',
 '--output-dir', str(r/f'launch_train_{n:03d}'), '--cwd', str(Path.cwd()),
 '--purpose', 'SO2 one lossless lifting HPAC rung',
 '--authority', 'local Metal training; no score authority', '--derive-resource-budgets',
 '--walltime-cap-s', '7200', '--done-receipt', f'ddm_so2_train_{n:03d}',
 '--arm-watchers', '--', *trainer]
subprocess.run(command, check=True)
# SO2_TRAIN_COMMAND_END
```

Exact extraction command (this launches training only after the preparation/claim gates above):

```bash
sed -n '/^# SO2_TRAIN_COMMAND_BEGIN$/,/^# SO2_TRAIN_COMMAND_END$/p' \
  .omx/research/ddm_so2_successor_object_priced_with_the_learned_prior_20260913.md \
  | .venv/bin/python -
```

The 7,200 s cap is an ASSUMED two-hour operational ceiling, not a runtime prediction. CL2's
measured twin took 3,172 s on its own field `[macOS-MPS research-signal]`; the charter's ~1 h
is planning context only. The trainer records full RNG/optimizer/EMA state, periodic checkpoints,
continuous-stage end at epoch 29 and QAT-stage end at 60. Resume uses the same cache/init/source
pins and `--resume-from`, with no trainer-drift waiver. Resource refusal is not bypassed. CL2's
Metal smoke measured a 15.557 GiB system-availability delta while RSS was only 1,697 MiB; the
launcher/governor must account for Metal memory and fleet SUM, not substitute RSS as total use.

4. **Pack and encode through the real current rail.** The owned adapter must use these actual
   callable APIs; none is a promised nonexistent command-line switch:
   `ddm_rx2_mc36_identity_race._pack_terminal_ihs1(checkpoint_path, output)` on the epoch-60
   checkpoint's **state_dict = EMA shadow**; verify checkpoint causal hash and cache/init/profile
   identity first. Preserve its raw IHS1, XZ and pack receipt. The topology helper matches the
   default C64/P64/delta2/D8/SPM profile; changing alphabet/shape would invalidate this reuse.
   Do NOT use CL2's older `pack_model` Brotli-only result as today's member price.

   Reuse the **HPR1 `prepare` body-packing sequence**, re-bound to the current archive:
   `rx.materialize_ihs1(parts.hpac_blob, renderer)` and
   `ihs2.layout_from_runtime(renderer)` establish the current layout; read RC3 header, counted
   24 weights and learning byte from `parts.hpac_blob`. Then
   `rc3.encode(new_ihs1, counts, header[2], weights, learning)` → `ck2_interleave(rider)` →
   `brotli.compress(outer, quality=10, lgwin=22)`. Retain rider, range payload, parameters,
   CK2 bytes and compressed member for both deterministic pack twins. Assert
   `rc3.restore_hpac(rider, counts) == new_ihs1`. Price this packed member, not raw IHS1 length.
   Use the same process on the original body as an additional member-identity control.

   Use the **SJ1 RLC1 known-symbol loop**, but bind BOTH the new packed prior and Z as input.
   The stock SJ1 CLI changes only the field and cannot be invoked with `--checkpoint` or
   `--model`; those flags do not exist. Its stock `init` also refuses a tree whose archive
   differs from the live pointer. Therefore do not point it at a candidate and falsify its
   control, patch its global guard, or silently change `runtime_copy`. The adapter owns a new
   input record with distinct source and candidate identities and composes HPR1's altered-parts
   input with SJ1's known-symbol injection. Preserve the real LaneMixer group order, native
   FreeCorrector, frame_even consequences of the packed model, fixed residual prefix, all counted
   RC3/RLC1 state, and both independent RC64 processes. An unrounded newly trained frame embedding
   is priced as it actually packs; no unmeasured even-rounding credit or second variant is taken.

   At every receiver frame checkpoint, retain encoder interval snapshots plus COMPLETE corrector,
   mixer and previous-plane state using the existing ReceiverCheckpoint/SJ1 mechanism. Bind the
   transformed field, candidate HPAC and producer/source hashes into every checkpoint. On resume,
   assert those pins; preserve every stage, including partial losing candidates. Rebuild RX1M
   lengths using `jg2.RX1_HEADER`, keep semantic/carrier/tail-prefix/rider bytes unchanged, and
   retain twin standard RX1M archives for the inner decode. The SO2 research archive wraps the
   RX1M member in the 8 B prefix above; `jg2.pack_archive(member, path)` keeps the single STORED
   member and 100 B ZIP overhead. This archive is not accepted by the public inflate runtime yet.

5. **Independent decode and harvest.** In fresh processes, without known-symbol injection or
   encoder caches, unwrap the research member, run the actual `rx.decode_production_tokens`
   under the DESERIALIZED candidate prior, and apply the integer inverse. Retain decoded Z and
   inverse F with SHA/bytes; all 600 planes must match the source F hash. Require independent
   process twins to match prior, stream and final archive bytes. A teacher-forced echo of the
   supplied Z is not independent decode. `RESULT.json` records actual P, C, 8 B overhead, J,
   whole archive, input/source hashes, control/twin/inverse outcomes and the thresholds above.
   Report trainer entropy separately. J > 123,998 fires the specified instance falsifier;
   J <= 123,998 queues receiver closure only; J <= 106,052 additionally reaches the conditional
   rate target. None of these outcomes is a contest score or authority to dispatch Modal.

The missing adapter is an explicit MAIN implementation obligation **inside this single rung**.
Only its interfaces and behavior are specified here; neither a complete end-to-end executable
nor its tested resumability is claimed. MAIN must review it under the Python review gate before
launching it. This is the material readiness boundary the current tools impose, not permission
to label an untrained model or an unbuilt encoder as a measured design.

## Retention, integration, and review boundaries

All materialization in the rung goes under its owned Vertigo root, with APDataStore fallback only
after explicit rebinding of the manifest and admission. No local bulk opt-in or symlink workaround.
Fail closed if the tier cannot retain payloads plus the measured resource reservation. The
specification uses a 40 GiB free-space floor, consistent with HPR1/DPI1; MAIN's storage-waterfall
decision remains authoritative. Keep per-stage fields, caches, model variants, both streams,
archives, inverse decodes, checkpoints and logs. Atomic temporary writes and deterministic restart
are part of the adapter, not optional future cleanup. No payload is deleted; a future move or
cleanup requires original path, bytes, SHA, command/config, reproducibility and destination records.
Control snapshots and byte-coder build products likewise remain retained until certified cleanup.

No upstream/PR/sealed tree, contract implementation, shared staged index or protected file was
edited. No training, rendering, scorer, GPU job, Modal call, or materialized candidate was run.
Only existing byte custody, source compatibility and arithmetic were checked. The reconstruction
identity is a mathematical derivation; it has not been exercised on the real field by this arm.
This expressly follows the charter's OPTIMAL_FORM boundary:

`# OPTIMAL_FORM_NA: design memo + one rung specification; nothing is built or trained by this arm; reference forms are the incumbent's own trainer and coder, cited by path and sha in the memo.`

Borrowed-substrate accounting: HPAC architecture/training and renderer are the existing PR130
lineage; CL2's resumable lift, RC3/CK2, RLC1, and SJ1 verification machinery are reused project
substrate. The proposed change is the explicit reversible categorical representation and its
newly trained prior. No claim that classical lifting is a newly invented mathematical method,
that these borrowed weights are an original payload, or that this is the born witness capstone.

Six integration hooks: sensitivity map, Pareto constraint, allocator, autopilot dispatch,
posterior update, and probe are all N/A as production hooks because no new empirical object exists.
The conditional budget is consumed by the queued rung; its RESULT is MAIN's measurement consumer.
The sole probe is that rung; no second sweep is deferred. The equations registry is recalled,
not appended with an untested law. Review and verification receipts are beside this memo.

Serializer is attempted LAST, once, with post-edit hashes and `[no-triality] [p0-ledger-ok]`,
`REVIEW_GATE_OVERRIDE=1` only for these non-Python documents. Actual rc and any fallback bundle
are recorded in `ddm_so2_20260913/LANDING.json`. Git-object denial does not erase or stop the handoff.

## NEXT_IF_RESUMED

* **QUEUED-WITH-A-FIRE-ORDER** — owner **MAIN / SO2 byte-rung owner**; consumer store
  `/Volumes/VertigoDataTier/pact/ddm_so2_first_rung/RESULT.json`; trigger **memo harvest, verified
  move-49/source pins, owned reviewed composition adapter, admitted storage/resources and free
  Metal slot**. Build the specified lossless cache and execute the ONE CL2-law training/real-price
  rung, including original control, twins and inverse decode. A pointer move requires rebinding
  and re-deriving the budget before fire; no automatic rerun on the next field.
* **FOLDED into the rung's success branch** — owner **MAIN / receiver owner**; consumer store
  `/Volumes/VertigoDataTier/pact/ddm_so2_first_rung/receiver/`; trigger **J <= 123,998 B with
  independent n600 inverse equality and twins**. Implement the format inverse before rendering,
  verify raw equality, timing and immutable contract readiness; rebind archive pins and manifest
  together. Route through the existing candidate process only after these proofs. SO2 authorizes
  no evaluator dispatch and claims no inherited timing or distortion authority for an unbuilt tree.
* **QUEUED-WITH-A-FIRE-ORDER, custody only** — owner **MAIN / serializer landing owner**;
  consumer store `.omx/research/ddm_so2_20260913/LANDING.json`; trigger **serializer rc 17 and
  verified fallback bundle**. Land exactly the declared files/hashes, preserving unrelated work.

composition S 0.13632299781031237 @ 179,153 B [contest-CUDA T4 n600] (move 49), unchanged.

## LIVE-HYPOTHESES

* The lifted field may cost at least 5% less under a prior trained on it. Interior differences
  share a class-independent zero event, and every exceptional token remains exactly representable.
  Its plausibility is structural; its byte price is unknown until the real fit/pack/encode.
* If that price passes, the cheap inverse may fit the existing receiver budget. It consists of
  integer rearrangement/addition on the already-required token tensor, but no timing transfer is
  claimed; the receiver success branch owns the measurement.

## DEAD-ENDS

* SO1's generic-coded coarse plane as a competitive price: its measured 169,100 B already exceeds
  the field gate before residuals. FORMULATION closure; not all coarse learned representations.
* A plain context-spacing prediction or another lambda/depth/mixer refit on unchanged F: HPR1's
  ranking reversed at real bytes, and CL2/DPI1/TMX1 already priced these specific rungs. Their
  losses do not price a genuinely transformed field, but their positive forecasts cannot transfer.
* A naive separate Lane stream credited with one third of surprise: D3B's real exact occupancy
  bill exceeded the joint coder; changing the probability factorization changes all symbols' costs.
* Dropping the lifted details and relying on repair: REN1/RQ1 and the exhausted single-token
  passes give no demand-sized payable repair band on this object. No such stage is queued.
* Treating a 5% byte pass as sub-0.12, or a syntax-checked command as a completed rung: the former
  still projects S approximately 0.131949 and the latter leaves training, real coding and receiver
  proof unmeasured. Neither moves the exact pointer.
