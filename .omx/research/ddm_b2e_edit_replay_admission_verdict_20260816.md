# B2E edit-replay admission — VERDICT: REGIME_THESIS_INSTANCE_REFUTED

**Date:** 2026-08-16 · **Arm:** `ddm_b2e_admission_measure` (Opus, under MAIN supervision)
**Axis:** `[macOS-CPU advisory n600 — env-mismatch advisory instrument, same as the mp2
calibration rows; NEVER a score]` · `score_claim: false` · `promotion_eligible: false`

## The verdict

The burn-2 "train-for-editability" F2-alone window did **not** make the three MP2 semantic
weight edits cheaper. The pre-registered bar asked each edit's pose-damage excess to collapse
**≥50×** versus the pinned MP2 calibration. Measured collapse is **0.75× to 1.06×** — no
collapse at all. Two of the three edits are **worse** on the burn-2 model than on the shipped
hv1 model.

Harness output, verbatim (`experiments/ddm_b2e_edit_replay_admission.py admit`):

| edit | verdict | collapse_factor | required |
|---|---|---:|---:|
| `mixed_q3q4` | **REFUSED** | 0.9450984170613949 | 50.0 |
| `film_row_prune_keep87` | **REFUSED** | 1.0589531636556442 | 50.0 |
| `film_row_prune_keep75_minus_keep87` | **REFUSED** | 0.7478913404712103 | 50.0 |

`overall: REGIME_THESIS_INSTANCE_REFUTED` · `measured_count: 3` · `admitted_count: 0`

## The measured table

All rows are n600 (600/600 samples), same instrument, same decode path, same receiver.
`S_adv` is recomputed from components — never the evaluator's rounded `final_score` field.

| row | d_seg | d_pose | archive bytes | S_adv | ΔS_adv vs burn-2 base |
|---|---:|---:|---:|---:|---:|
| hv1 base (calibration) | 0.00042714 | 0.00014747 | 182,759 | 0.202807539 | — |
| **burn-2 base q4** | 0.00042728 | 0.00014826 | 183,089 | 0.203143995 | 0.000000000 |
| `mixed_q3q4` | 0.00042839 | 0.00076924 | 182,121 | 0.251812227 | +0.048668232 |
| `film_row_prune_keep87` | 0.00042715 | 0.00065754 | 182,844 | 0.245552155 | +0.042408160 |
| `film_row_prune_keep75_minus_keep87` | 0.00042801 | 0.00069677 | 182,939 | 0.248085321 | +0.044941326 |

### Pose excess against each model's OWN base

| edit | calibration base→edited | calib excess | burn-2 base→edited | burn-2 excess | collapse |
|---|---|---:|---|---:|---:|
| `mixed_q3q4` | 0.00014747 → 0.00073123 | 3.958500 | 0.00014826 → 0.00076924 | 4.188453 | 0.945098 |
| `film_row_prune_keep87` | 0.00014747 → 0.00068390 | 3.637553 | 0.00014826 → 0.00065754 | 3.435047 | 1.058953 |
| `film_row_prune_keep75_minus_keep87` | 0.00014747 → 0.00055551 | 2.766936 | 0.00014826 → 0.00069677 | 3.699649 | 0.747891 |

I re-derived the three calibration excesses from the raw MP2 result JSONs rather than trusting
the harness constants: 3.958500 / 3.637553 / 2.766936 — they match the pinned `CALIBRATION`
block exactly.

### Seg deltas (context, not the bar)

The bar is a pose bar. Seg moved by almost nothing on every edit, so the pose damage is not
being bought with seg improvement:

| edit | Δd_seg vs burn-2 base | seg score leg |
|---|---:|---:|
| `mixed_q3q4` | +1.11e-6 | +0.000111 |
| `film_row_prune_keep87` | −1.3e-7 | −0.000013 |
| `film_row_prune_keep75_minus_keep87` | +7.3e-7 | +0.000073 |

### Real archive byte deltas (measured on the built `archive.zip`, not projected)

| row | bytes | Δ vs burn-2 base | Δ vs hv1 frontier |
|---|---:|---:|---:|
| burn-2 base q4 | 183,089 | 0 | +330 |
| `mixed_q3q4` | 182,121 | −968 | −638 |
| `film_row_prune_keep87` | 182,844 | −245 | +85 |
| `film_row_prune_keep75_minus_keep87` | 182,939 | −150 | +180 |

Every edit's byte saving is dwarfed by its pose cost. `mixed_q3q4` saves the most (−968 B vs
its own base = −0.000645 rate leg) and pays **+0.049 S** in pose. That ratio is ~75× against.

## The burn-2 window itself barely moved anything

Burn-2 base vs hv1 base: Δd_seg **+1.4e-7**, Δd_pose **+7.9e-7**, Δbytes **+330**,
ΔS_adv **+0.000336456**. So 3,000 steps of F2-alone training left both scored axes
statistically flat and cost bytes.

I decomposed the +330 B honestly, because it conflates two different changes:

| component | bytes | what it is |
|---|---:|---|
| carrier swap | **+321** | hv1 weights re-packed through the SD1M q4 carrier instead of hv1's native WANS1 entropy coding |
| weight change | **+9** | burn-2 weights vs hv1 weights, both through the SD1M q4 carrier |

Measured by packing the decoded hv1 state through the identical SD1M path: hv1/WANS1 = 34,763 B,
hv1/SD1M = 35,084 B, burn-2/SD1M = 35,093 B (Brotli q11 streams). The burn-2 weights are 9 bytes
less compressible than hv1's — the training window did not change the weight entropy either.
**The carrier cost cancels entirely in the edit-vs-base ratio**, since all four archives ride the
same SD1M carrier; it only affects the burn-2-base-vs-hv1 comparison.

## Instrument notes

**The base is the q4-packed deployment state, not the float EMA.** The edits are edits OF the
deployed object, so the base has to be that object. `final.pt` carries
`deployment_weights: "ema_shadow"` and its `state_dict` is the EMA shadow (38 tensors; checkpoint
sha256 `464d086dad62720f9a9a32a7deb5a823d7f415a648f3345ff5b59999a8bf32db`, pinned and enforced by
the splicer).

**The SD1M carrier is a re-header of the trainer-validated export, not a different quantizer.**
The trainer's `deployed_argmax_parity` gate validated `pack_semantic_pose.pack_semantic`, the flat
legacy-int4 layout. `sd1.pack_semantic_state(..., legacy_int4=True)` is that byte format (40,252 B —
matches the sd1 pin), and `legacy_int4=False` is the identical payload behind a 14-byte SD1M header
(40,266 B). The splicer **asserts** the two dequantize to a byte-identical realized state before
shipping either. The SD1M header buys a receiver the staged runtime can actually parse; the WANS1
path cannot carry arbitrary retrained weights without re-deriving the record schema.

**Byte-identity control ran FIRST and passed.** Re-splicing the UNEDITED hv1 semantic section
through the same `rx1.pack_rx1_model` + `deterministic_zip` path reproduces the pinned frontier
archive exactly: 182,759 B, sha256 `80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e`.
The RX1M member also re-packs byte-identically. A later byte delta is therefore attributable to the
semantic section and nothing else.

**Receiver-closed, all four.** Each generation's staged receiver decodes the archive's semantic
blob back to the packer's exact 38-tensor state (`semantic_state_exact_to_packer: true`).

**Advisory, never a score.** Every row carries `score_axis: cpu_env_mismatch_advisory` /
`evidence_grade: auth-eval env mismatch advisory`. No row here is a contest-CPU or contest-CUDA
score, and none is promotion-eligible. What the instrument *does* support is the **ratio**: base
and edits ride one instrument, one decode path, one receiver, and the calibration used the
identical instrument, so instrument bias divides out of `pose_edited / pose_base`.

**Not precision-limited.** Pose is reported at 8 decimals; the relative rounding error on
0.00014826 is ~3.4e-5. To reach a collapse of 50, `mixed_q3q4` would need d_pose ≤ 0.000159998;
it measured 0.00076924 — **4.81× too high**. The verdict has four orders of magnitude of margin
over the reporting precision.

**Not a prefix.** Every row is the full 600-pair population, so the m96 prefix-bias law
(pose prefixes measure 2.5–4.2× harder) does not apply and cannot be blamed. This negative is
drawn on the population it generalizes to.

## verdict_scope on the negative

`verdict_scope: formulation` — this refutes **this burn-2 window**, not the train-for-editability
family.

Refuted: the claim that the F2-alone editability window as run — `--weight-qat-q3q4`, 3,000 steps,
lr 2e-7, ce_fraction 0.5 / softplus_fraction 0.85, `film_critical_multiplier` 9.68, seeded from the
PR130 `w96_b4_qat4_fixedtau05_tail6k_lr2e7` init, EMA-shadow deployment, q4, semantic section only —
makes the three MP2 post-hoc semantic edits materially cheaper in pose.

NOT refuted, and explicitly out of scope:
- train-for-editability as a family, at other budgets, other levers, or other lr;
- editability of the `hpac` token model (a different learned object in the same archive — this
  window never touched it);
- any claim about the exact contest evaluator, which was never run here;
- the MP2 family verdict itself, which already stood REFUSED and is unchanged by this row.

One caution against over-reading: the burn-2 window moved the base by ~1e-7 on seg and ~8e-7 on
pose. A window that changed the model this little was arguably never in a position to change the
model's edit response. The honest reading is that **this window did not train, so it did not
train for editability** — which is a weaker and more specific finding than "training for
editability does not work."

## Retained payloads (ALWAYS KEEP THE PAYLOAD)

Root: `/Volumes/APDataStore/pact/ddm_b2e_f2_alone_run/`

- `admission_archives/B2E_ADMISSION_ARCHIVES.json` — build receipt, control, all four archives
  with path/bytes/sha256 and both byte-delta columns.
- `admission_archives/generations/<candidate>/archive.zip` — the four real archives:
  - `burn2_base_q4` 183,089 B sha256 `b38dacdf60556fceb93d19929ccb7dd0c45e63b180fd2e9ca5cdbeed534f86f4`
  - `mixed_q3q4` 182,121 B sha256 `b30284b59b4e2e5a65fd346a8620ea85d6bd5e4687ab48588deecae7cfd88f03`
  - `film_row_prune_keep87` 182,844 B sha256 `a9999526462244e34bcf80af9306a2a7f53c00eecd6dda4d5bb993fbaf864f8a`
  - `film_row_prune_keep75_minus_keep87` 182,939 B sha256 `f3d580d181b1fa3fa35b8eaac5fe63fc26e22d0050c76e6a1e8a91bf5c598aa6`
- per generation: `retained/semantic.raw.bin`, `retained/semantic.br`, `retained/p`,
  `retained/models.rx1m`, `retained/archive.repeat.zip` (determinism repeat),
  `retained/receiver_decode/semantic_state/*.npy` (the realized 38-tensor state, receiver-decoded),
  candidate-bound runtime tree, `GENERATION_RECEIPT.json`, `RECEIVER_PARSEBACK.json`.
- `admission_advisory/<candidate>/attempt_0000/` — `contest_auth_eval.json`, `safe_run_status.json`
  (exit 0, peak RSS 10.3–10.7 GiB under a 16 GiB cap), `launcher/`, and the kept `work/` tree with
  the inflated raw frames.
- `edit_replay/B2E_MEASURED_POSE_ROWS.json` — the measured input to `admit`
  (sha256 `32d3045a8533d73e37bdc379d1dfd763375798054fa894a476933da967a76769`).
- `edit_replay/B2E_ADMIT_RESULT.json` — the adjudication above.

Wall clock: 906 / 906 / 834 / 831 s per row (wc1 cached-token fast path; the token cache hit for
all four, since the `hpac`/carrier/tail sections are byte-identical across them).

## STORES CONSULTED

- `CLAUDE.md` — NO-FAKE supreme rule, ALWAYS KEEP THE PAYLOAD, axis-label honesty,
  never-invent-flags, upstream read-only, "recompute the score from components".
- `experiments/ddm_b2e_edit_replay_admission.py` — harness docstring, pinned `CALIBRATION`,
  `REQUIRED_COLLAPSE_FACTOR = 50`, `build_edit` / `build_mixed_q3q4` / `_prune_rows`.
- `experiments/ddm_mp2_mixed_precision_receiver_close.py` — `build_generation`, `split_member`,
  `retain_receiver_decode`, `build_control` (the splice mechanics reused verbatim).
- `experiments/ddm_mp2_advisory_queue.py` — the canonical advisory n600 invocation shape.
- `/Volumes/APDataStore/pact/ddm_mp2_mixed_precision_receiver_close_20260815/MP2_ADVISORY_ADJUDICATION.json`
  and the three `advisory_n600_cpu/*/attempt_0000/contest_auth_eval.json` rows (calibration,
  re-derived not trusted).
- `/Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/contest_auth_eval.json` — hv1 base row.
- `/Volumes/APDataStore/pact/ddm_wc1_advisory_decode_wallclock_20260815/receipts/ADMISSION_GATE.json`
  — fast decode path; all four `consumer_code` hashes re-verified live before use.
- `src/tac/pr130_lift/train_semantic_quantized_resumable.py::deployed_argmax_parity` and
  `src/tac/pr130_lift/pose/lifted/pack_semantic_pose.py::pack_semantic` — the export path the
  trainer's parity gate validated.
- `$RUN/final.pt`, `$RUN/result.json` (verdict PASS, parity 600/600),
  `$RUN/edit_replay/B2E_REPLAY_RESULT.json`, `$RUN/edit_replay/B2E_PAIRS_RESULT.json`.
- Memory: `m96` (prefix-bias sign inverts between seg and pose), `m04` (own-vehicle frontier),
  the ns1/mp2 post-hoc-edit refusal lineage.

## NEXT_IF_RESUMED

1. **Do not re-run this window.** The bar is pre-registered and the margin is ~50×; a repeat at the
   same settings cannot change the verdict. Any reopening needs a *different* mechanism, not a
   longer run of the same one.
2. **The prior question is "did the window train at all?"** ΔS_adv +0.000336 and a 9-byte entropy
   change say it barely did. Before spending another editability window, measure the *training*
   response: does any lr / step budget move the burn-2 base d_seg or d_pose beyond the noise floor
   of this instrument? If nothing moves the base, editability training on this leg is untestable,
   not refuted.
3. **If a window is re-tried, put the edit in the loop, not after it.** Every collapse factor here
   is ≈1, which is the signature of an edit the training never saw. The b2e thesis needs the edit
   operator applied *inside* the forward pass (fake-quant to q3 on the selected set, FiLM rows
   masked) so the loss actually pays for the damage — the same "only joint descent crosses the wall"
   lesson the pose leg already measured.
4. **The rate side is not where the loss is.** The best edit saves 968 B (−0.000645 rate leg) and
   costs +0.049 S in pose. Even a perfect editability window would have to collapse pose damage by
   ~75× before these edits pay. That ratio, not the training recipe, is the real bar.
5. **Owned handoff.** All four archives and their receiver decodes are retained and byte-pinned;
   any successor can re-adjudicate without rebuilding. The exact-eval leg was never fired and is
   not owed — no row here is close enough to the frontier to justify contest hardware.
