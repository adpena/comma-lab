# ddm_sr5 — both SSD tiers above their reserves: certified deletion, a lossless dedup, and one relocation

Axis: `[macOS filesystem custody; scorer-free]`. No score was measured by this arm.
Frontier reference, untouched and not measured here: `composition S 0.1372449041713402 @ 180,406 B
[contest-CUDA T4 n600] (move 44)`. MAIN moved the pointer to move 45 while this arm ran; this arm
wrote nothing to the pointer or any of its receipts.

## ANSWER FIRST

**VertigoDataTier: 35.358 → 61.018 GiB free (37.97 → 65.53 GB), +25.660 GiB.
APDataStore: 49.06 → 51.135 GiB free (52.68 → 54.91 GB), above its 45 GiB floor.** Both tiers clear
their targets.

Nearly all of it came from one payload class — `frame0_receiver.uint8.npy` in
`ddm_pk4_20260813/cpu_authority_run/retained/jacobian_bank` — deleted under a per-file certificate.
1,501 files, 25.815 GiB, zero refusals. A later top-up of 183 files (2.432 GiB) from the sibling
`pose_preprocessed_yuv6` class restored the mark after MAIN deliberately moved a 5.84 GiB cold tar
back onto Vertigo to rescue APDataStore. Two more actions deleted nothing at all: a certified
relocation returned 5.84 GiB to APDataStore, and a hardlink dedup released 3.411 GiB on Vertigo
without removing a single file.

Every deleted file carried a certificate that was **exercised, not filled in**. Before each unlink
the tool re-ran the rebuild in memory from the retained sibling and required the result to serialise
to the sha256 the producer recorded at write time. A wrong derivation is refused and the file kept —
that is measured, not assumed: a synthetic negative control (`frame0_receiver` holding frame 1
instead of frame 0) was REFUSED with the file intact, and the matching positive control passed.

`ddm_sr4` looked at this same store and refused it. That refusal was right. The difference here is
scope, not courage. sr4 searched for a **whole-tree** rebuild certificate. There is none, and there
still is none. What exists is a **per-file** certificate, and it is stronger than a tree certificate
would have been: three of pk4's classes are pure deterministic functions of one another, so the
rebuild needs no archive, no decode, no scorer, and no external input — only a sibling file that
stays on disk.

## THE DERIVATION THAT MADE IT CERTIFIABLE (MEASURED)

`experiments/ddm_pk4_optimal_form_frame0_pose.py:535-538` builds
`inputs = np.stack((slaves, masters), axis=1)`, then saves `slaves` as `frame0_receiver.uint8.npy`
and `inputs` as `pose_input.uint8.npy`. The bytes therefore duplicate by construction. Confirmed by
hash, not by reading:

| relation | how it was proved | scope of the proof |
|---|---|---|
| `frame0_receiver == pose_input[:, 0]` | `sha256(np.save(slice))` equals the recorded sha256 | MEASURED per file, on every file deleted |
| `pose_preprocessed_yuv6 == PoseNet.preprocess_input(pose_input)` | sha256 match, want `d72ce063086ff122…` | MEASURED once (`pair_013/stage_10_jacobian/batch_0000_0008`) |
| `mc35 slave_camera == pose_input[:, 0]` | sha256 match | MEASURED once (`js6_0004…/stage_40_integer_cube/batch_0032_0040`) |

`PoseNet.preprocess_input` touches no module state (`upstream/modules.py:70-74`), so it is called
unbound: bilinear interpolate to the scorer input size, then `rgb_to_yuv6`. No forward pass, no
weights, no scorer run.

**The restore path was exercised on a file that was already gone.** Taking the first DELETED row from
the ledger, reading only its `DERIVED_DELETED.json` sidecar, and running the recorded rebuild
reproduced the recorded sha256 exactly, with the original absent from disk. A deletion that cannot be
undone is not rebuildable scratch, whatever the certificate says; this one can.

## CENSUS — store → class → bytes → verdict

Closedness column: what was actually checked, not what was assumed. "CLOSED" means the verdict
landed AND no live process, dispatch claim, pointer receipt, seal or intent names the payload.

| store | closed? (evidence) | class | bytes | files | verdict | cert |
|---|---|---|---:|---:|---|---|
| `ddm_pk4_20260813/.../jacobian_bank` | **CLOSED** — claim `completed_local_gate_fail_no_compile` 2026-08-14T04:00:58Z (family ceiling, 3/3 rungs GATE_FAIL); verdict memo `ddm_pk4_optimal_form_frame0_pose_verdict_20260813.md`; no pointer/seal/intent receipt names the bank; `lsof` clean; every file `nlink=1` | `frame0_receiver.uint8.npy` | **28.109 GiB** | 1,636 | **DELETED to target** — per-file exercised rebuild | `.omx/research/ddm_sr5_certified_deletions_20260911.jsonl` |
| same | same | `pose_preprocessed_yuv6.float32.npy` | 21.729 GiB | 1,636 | **CERTIFIED; PARTLY SPENT as the top-up** — rebuild exercised and sha-matched. Left whole until MAIN's rlc2 relocation deliberately moved 5.84 GiB back onto Vertigo to rescue APDataStore; the smallest slice of this class was then spent to restore the 61 GiB mark the sweeps had reached | `.omx/research/ddm_sr5_certified_deletions_20260911.jsonl` |
| same | same | `pose_input.uint8.npy` | 56.217 GiB | 1,636 | **RETAINED — rebuild ROOT** for both classes above; deleting it would break both certificates | — |
| `ddm_mc35_20260814/retained/compensation` | **CLOSED** — no claim row for `mc35`; memo `ddm_mc35_micro35_union_build_20260814.md`; axis `[macOS-CPU advisory … ] NON-PROMOTABLE` | `slave_camera.uint8.npy` | 4.681 GiB | 244 | **CERTIFIED, RETAINED** — `== pose_input[:, 0]`, sha-matched on a slice; target already met | this memo |
| same | same | `pose_input.uint8.npy` | 9.363 GiB | 244 | **RETAINED — rebuild root** | — |
| `ddm_rbf1` (MAIN handoff) | **CLOSED** — verified independently: no `ddm_rbf1` process (pid 11027 gone), claim `completed_n600_scorer` 2026-09-11T16:13:52Z supersedes the 14:10 `active` row, and the certification's sha matched MAIN's stated `9cb36451ec8f…` | `chunks/*/{baseline,guided,ssaa,sdf,composition}_camera_u8.npy` | 17.054 GiB | 600 | **CERTIFIED + EXERCISED, RETAINED** — rebuild replayed on chunk `595_600`; not needed, and 4,400 s + 3.9 GB of foreign pinned inputs to restore | `.omx/research/ddm_sr5_20260911/RBF1_RENDER_EXERCISE_RECEIPT.json` |
| same | same | `chunks/*/native_f32.npy` | 1.318 GiB | 120 | **CERTIFIED + EXERCISED, RETAINED** — same exercise | same receipt |
| same | same | `chunks/*/*_argmax.npy`, `*_pose.npy`, `residual_cells_i32.npy` | 0.550 GiB | 1,320 | **NOT TAKEN** — rbf1's own order puts these last, and their rebuild needs the camera payloads first | — |
| `ddm_mxo2_low_rank_stacker_v3` (MAIN handoff) | **CLOSED** — no `ddm_mxo2`/`ddm_mxo3` process, no claim row mentions `mxo`, memo `ddm_mxo2_low_rank_stacker_over_premix_family_outputs_screen_20260911.md` landed | `surface/frames/frame_*.npz` | 7.691 GiB | 600 | **CENSUSED, NOT TAKEN** — target met without it; still refused structurally by both tools' `/ddm_mxo2` guard, which a successor must lift deliberately | — |
| `ddm_pfs1_20260729` | not established (no claim rows; 7 memos) | `*/inflated/0.raw` | 47.752 GiB | 14 | **REFUSED for deletion; DEDUPED losslessly instead (3.411 GiB).** The rebuild is an `inflate.sh` of a candidate archive: candidate work, barred here and not exercisable, so no deletion certificate is possible. A 3×16 MiB-window prefilter then found **11 distinct signatures among the 14 raws**, with two duplicate-candidate groups — `{v4d_cx1_pj2ix2, v4d_pj2}` and `{v4d_mq1_partial, v4d_ms8, v4d_pw1}`. Full sha256 then CONFIRMED one group and FALSIFIED the other: `{v4d_cx1_pj2ix2, v4d_pj2}` are identical and now share one inode, releasing **3.411 GiB with nothing deleted**; `{v4d_mq1_partial, v4d_ms8, v4d_pw1}` are **not** identical and were never touched | `.omx/research/ddm_sr5_20260911/PFS1_DEDUP_RESULT.json` | — |
| `ddm_js5_20260812` | not established | `logits_n*.float*.npy` | 6.328 GiB | 54 | **REFUSED** — a scorer forward pass produced them; no sibling derivation exists | — |
| same | not established | `camera_frame*`, `correction_n*`, `camera_continuous_n*` | 15.813 GiB | 216 | **REFUSED** — rebuild is a decode/train replay, not exercisable inside this charter | — |
| `ddm_pz4_joint_target_conditioned_receiver` | not established | `logits.float32.npy` | 6.592 GiB | 114 | **REFUSED** — scorer output | — |
| same | not established | `pose_input.float32.npy`, `seg_input.float32.npy`, `batch_*.uint8.npy` | 12.639 GiB | 304 | **REFUSED, and the sibling-derivation lead is CLOSED by measurement** — I checked: a `pz4r_candidate/batches/batch_0001` directory holds `seg_input.float32.npy` (16 frames, 384×512×3 fp32) and `logits.float32.npy` only. There is no `pose_input` sibling in the batch directory, so the pk4-shaped relation does not exist in this layout | — |
| `hprc_projection_gap_repairs` | old sweep (2026-06) | `segnet_last_rgb.npy` + `posenet_yuv6_pair.npy` | 23.730 GiB | 18 | **REFUSED** — these *do* carry recorded sha256 (`baseline.json → *_sha256`), so bindings 1-2 are available; what is missing is an exercisable rebuild: the bytes come from a decode of a superseded `pact_nerv_vq` candidate | — |
| `hprc_residual_transform_full600_sweep_20260601T002114Z` | old sweep (2026-06) | same two classes | 15.820 GiB | 12 | **REFUSED** — same reason | — |
| `ddm_qs1_20260813` | **NOT established** — 107 claim rows reference `qs1` | — | 40.94 GiB | — | **NOT CENSUSED at class level** — closedness not established, so no deletion was considered | — |
| `pr135_joint_solve_20260810` | **NOT established** — 28 claim rows | — | 31.45 GiB | — | **NOT CENSUSED at class level** — same | — |
| `evidence`, `experiments` | — | — | 304.59 / 247.26 GiB | — | **UNTOUCHED** — sr4 blocked both for want of a whole-tree certificate; nothing here changes that | — |
| `cold_store`, `public_datasets`, `ddm_sr4_20260910/retained/ap_offload` | — | — | 56.24 / 24.98 / 30.70 GiB | — | **PROTECTED** by charter and by both tools' guard list | — |
| **APDataStore** (whole tier) | — | — | — | — | **NOT MUTATED** — read for `df` only; stayed above its 45 GiB floor throughout | — |

## WHY pk4's CLASS WENT FIRST (measured priority, not preference)

Three stores offered a valid certificate. They are not equally cheap to put back, and the deciding
difference is dependency structure rather than a rate:

| class | rebuild | recorded / observed cost | bytes needed from OUTSIDE the payload's own directory |
|---|---|---|---|
| pk4 `frame0_receiver` | `np.load(sibling)[:, 0]` | cheap enough that this arm ran it inline for **every** deleted file | **none** |
| rbf1 `*_camera_u8` + `native_f32` | `ddm_rbf1_boundary_probe.py render` | **4,400 s** for the class, from rbf1's own certification | 3.9 GB of pinned inputs across three other stores, two of them sealed pointer trees |
| mxo2_v3 `surface/frames/*.npz` | re-derive from the move-43 archive + receiver | not measured by this arm | archive + token field, in two protected stores |

Reclaim the bytes that are cheapest and safest to restore, and reclaim only as many as the target
needs. pk4's class restores from a sibling that never leaves its own directory — nothing outside can
go missing, be resealed, or drift — which is why its certificate can be per-file at all. rbf1's
restore is hostage to four files in other people's stores. So pk4's class was spent first and the
other two were left whole.

I did not measure a clean per-file rebuild rate and will not quote one: the only timing I took
(3.095 s for a small batch) was under four-way concurrent disk load and says more about contention
than about the rebuild. The charter's "stop at the target, do not over-reclaim" decided the rest:
both sweeps stopped
themselves the moment the mount reached 61 GiB, leaving **135 of the 1,636 files** in the class
untouched. I spent what the target needed and not one file more.

## WHAT WAS EXERCISED

Certify-or-block is a law about evidence, so every claim below was run, not written down.

**1. The derivation, per file, on every file deleted.** `tools/sr5_certify_delete_derived.py` requires
four bindings before each unlink and refuses on any one of them: the directory's `RESULT.json` names
the file with a byte count and sha256; the file on disk hashes to that sha256; the retained rebuild
source hashes to *its* recorded sha256; and the rebuild, executed there and then, serialises to the
same sha256 as the file it is about to remove. Every deleted row in the ledger carries all four
hashes, so a reader can re-check any one of them without trusting this memo.

**2. Negative and positive controls on the gate itself.** A synthetic batch was built whose
`frame0_receiver` held frame 1 instead of frame 0, with a manifest that agreed with the wrong file.
The tool REFUSED it — `rebuild did not reproduce the file byte-for-byte` — and left the file in place.
The same directory, corrected, was then CERTIFIED, deleted, and given its sidecar. A gate that has
never refused anything is not known to be a gate.

**3. The restore path, on a file that was already gone.** Taking the first `DELETED` row from the
ledger, reading only its `DERIVED_DELETED.json` sidecar, and running the recorded rebuild reproduced
the recorded sha256 exactly — with the original confirmed absent from disk. This is the check that
actually matters: a certificate that has never been redeemed is a promise, not a property.

**4. The other two derived classes, once each.** `pose_preprocessed_yuv6` was rebuilt through the
real upstream preprocess and matched `d72ce063086ff122…`; `mc35`'s `slave_camera` was rebuilt as
`pose_input[:, 0]` and matched its recorded sha256. Both are certified and available to a successor
with no new investigation. `pose_preprocessed_yuv6` was later drawn on for the top-up, per file and
under the same four bindings; `mc35`'s class was not spent at all.

**5. rbf1's render, replayed on a real slice.** Chunk `595_600`'s six payloads and its
`RENDER.json` were moved aside on the same volume, rbf1's own recorded render command was re-run, and
all six regenerated files matched **both** the sha256 recorded in the archived receipt **and** the
held originals, byte for byte: `baseline`, `guided`, `ssaa`, `sdf`, `composition` camera payloads and
`native_f32`. The arm's certified `RENDER_COMPLETE.json` was restored to `eba67bc8ff20…` and the
chunk's original receipt was put back. Receipt:
`.omx/research/ddm_sr5_20260911/RBF1_RENDER_EXERCISE_RECEIPT.json`. The finisher was written to roll
the held originals back automatically on any mismatch; it did not need to.

**6. The dedup, proven and half-falsified.** Two identical scratch files were collapsed onto one
inode and a third differing file was left alone (synthetic control), then on the real raws a FULL
sha256 confirmed one candidate group and refuted the other. The refutation is the useful part: a
sampled prefilter chooses what to hash, it never decides equality.

**7. The relocation, in the order that matters.** The rlc2 tar's destination was re-read from disk
and hashed independently of the streaming digest before the source was retired, and the redirect was
repointed and re-resolved in between. Every failure path in that tool leaves two copies, never none.

**8. Provenance of the tools themselves.** Both tool sha256 values stamped on the ledger rows resolve
to committed git objects: `b704e2a9123c489b…` is `tools/sr5_certify_delete_derived.py` at commit
`e200ceac0` (1,192 rows) and `c3517ab1149b4e6d…` is the same file at `6a171a59b` after `--reverse`
was added (1,014 rows). No cert row names a tool version that cannot be checked out.

## FREED, WITH df BEFORE AND AFTER

All figures are `statvfs` `f_bavail × f_frsize`, in true GiB (2³⁰) with the GB (10⁹) value alongside,
because `df -h` on this host displays the GB number under a `Gi` label and the charter's targets were
written from that display.

| tier | before | after the sweeps | change |
|---|---:|---:|---:|
| **VertigoDataTier** | 37,965,692,928 B = **35.358 GiB** (37.97 GB) | 65,536,303,104 B = **61.036 GiB** (65.54 GB) | **+25.677 GiB** |
| **APDataStore** | 51,442,432 KiB = **49.06 GiB** (52.68 GB) | **51.135 GiB** (54.91 GB) after the rlc2 relocation | +2.08 GiB, none of it deletion |

APDataStore's path was not monotone and sr5 did not cause the dip. Live arms (`ddm_hpr1`,
`ddm_ntb2_proof45`, two `ddm_pc3` fire directories) took it from 49.06 down to **45.32 GiB** — 0.34 GiB
above its floor — while the sweeps ran. sr5 wrote zero bytes to that tier; I flagged the approaching
breach to MAIN rather than holding it for this memo, MAIN redirected hpr1's new bulk to Vertigo, and
the rlc2 relocation then returned 5.84 GiB.

**Final totals for sr5's deletions: 1,684 files, 30,330,292,880 B = 28.247 GiB, zero refusals** —
1,501 `frame0_receiver` plus 183 `pose_preprocessed_yuv6`. Every one carried a per-file certificate
whose rebuild was re-run and sha-matched before the unlink.

Deleted by the two instances: **1,501 files, 27,718,528,784 B = 25.815 GiB, zero refusals.**

The accounted bytes (25.815 GiB) slightly exceed the `df` delta (25.677 GiB). That is not an error and
it is not an exclusive-attribution claim: four live arms were writing to the same tier throughout. The
accounted figure is what sr5 removed; the `df` figure is what the tier did, everyone's activity
included. sr4 recorded the same caveat for the same reason.

### Then MAIN widened the job, mid-run

Three more authorizations landed while the sweeps were finishing, and APDataStore had meanwhile
fallen to 45.34 GiB — 0.34 GiB above its floor — with live arms, not sr5, consuming it. I flagged
that to MAIN before it breached rather than saving it for this memo.

**The rlc2 cold tar moved back to Vertigo, and APDataStore recovered 5.84 GiB.**
`tools/sr5_certify_relocate_file.py` copied
`/Volumes/APDataStore/pact/cold_store/pact/ddm_rlc2_cure_on_move42.tar` (6,267,002,880 B) to
`/Volumes/VertigoDataTier/pact/cold_store/pact/`, re-read the destination from disk and hashed it
independently of the streaming digest — `fca7b7f3b11278a4…`, equal to the source and to the sha in
MAIN's own cold-move ledger — wrote the cert row, repointed the transparent redirect at
`/Volumes/VertigoDataTier/pact/ddm_rlc2_cure_on_move42`, re-resolved it, and only then unlinked the
source. APDataStore 45.321 → **51.157 GiB**; Vertigo 61.036 → **55.184 GiB**.

That ordering is the whole point of the tool. vr7 deleted eleven moved payloads behind live redirects
because a move LABEL was treated as custody. Bytes read back off the destination are custody; a label
is not.

**The pfs1 dedup.** `tools/sr5_certify_hardlink_dedup.py` full-hashed all five
prefilter candidates and confirmed **one** group, not two: `v4d_pj2` and `v4d_cx1_pj2ix2` are
byte-identical (`988785e7cadfd613…`) and now share inode 3206328, releasing **3,662,409,600 B =
3.411 GiB with nothing deleted** and both paths still resolving.

**My own lead was half wrong, and the tool caught it.** I had written that confirming the two groups
would free 10.233 GiB. That was an upper bound on an unconfirmed prefilter. The second candidate group
— `v4d_mq1_partial`, `v4d_ms8`, `v4d_pw1` — matched on all three 16 MiB windows and is **not
identical**: their full sha256 values differ, so the tool never grouped them and never touched them.
The differing bytes simply lie outside the sampled windows. This is exactly why the tool takes a full
hash and treats a prefilter as a chooser, never as a decision. Measured 3.411 GiB, not the 10.233 GiB
I projected; the record is
`.omx/research/ddm_sr5_20260911/PFS1_DEDUP_RESULT.json`.

**ntb2's APDataStore bulk: mostly refused, and the reason is measured.** See the refusals section and
`.omx/research/ddm_sr5_20260911/NTB2_COLDSTORE_ADJUDICATION.json`.

## REFUSALS AND WHY

Refusals are the point of the rule, so here is every one, with the reason:

- **`ddm_pfs1_20260729`, 47.752 GiB of raws — the largest single prize on the tier, refused.** Its
  fourteen `0.raw` files rebuild only by running `inflate.sh` on a candidate archive. That is candidate
  work, which this charter bars, and it is not exercisable here, so the certificate cannot be
  completed. I also cannot honestly route it to sr4's dedup method: all fourteen are `nlink=1` with
  distinct inodes and decode fourteen *different* archives, so they are unlikely to be duplicates of
  one another, and proving it either way costs 51 GiB of hashing.
- **`hprc_projection_gap_repairs` and `hprc_residual_transform_full600_sweep`, 39.55 GiB — refused,
  and my first reason was wrong.** I initially wrote that they carried no sha records. They do:
  `baseline.json` records `posenet_yuv6_pair` and `segnet_last_rgb` sha256 values. Bindings 1 and 2
  are therefore available. The real blocker is binding 4 — the bytes come from a decode of a
  superseded `pact_nerv_vq` candidate and there is no rebuild I can exercise inside this charter.
- **`ddm_js5_20260812` and `ddm_pz4_…` logits, 12.92 GiB — refused.** A scorer forward pass produced
  them, and this arm runs no scorer.
- **`ddm_qs1_20260813` (40.94 GiB) and `pr135_joint_solve_20260810` (31.45 GiB) — not even
  censused at class level.** 107 and 28 claim rows respectively reference them. I could not establish
  that they are closed, and the charter's rule is check, do not assume, so they were left alone.
- **`ddm_mxo2_low_rank_stacker_v3`, 7.691 GiB — verified closed, certified shape present, not
  taken.** Both tools still refuse the path structurally through their `/ddm_mxo2` guard. That guard
  was correct when this arm started and only a deliberate, reviewed code change should lift it; the
  target did not require it.
- **`ddm_rbf1`'s 18.372 GiB — verified closed, exercised, not taken.** See the priority section: its
  restore depends on 3.9 GB of pinned inputs in three other stores and costs 4,400 s, against a
  sibling slice that never leaves its own directory.
- **The whole-tree question — still refused, as sr4 left it.** No complete tree rebuild certificate
  exists for `evidence`, `experiments`, `pk4`, `pfs1`, `qs1` or `pr135`. Nothing in this arm creates
  one, and the per-file route used here does not generalise to a tree.
- **Zero refusals inside the class that was actually swept.** Across 1,501 certified deletions the tools refused nothing, because every file met all four bindings. A gate that never refuses is suspect, which is why the negative control was run deliberately.
- **ntb2's APDataStore bulk, 4.5 GB authorized by MAIN — mostly refused, on measurement.** Its
  `public_frame_even_manifestless_attempts` manifest records **`sha256: "SKIPPED_LARGE_STREAMED"`** for
  two rows totalling 3,780,374,400 B, which is 99.2% of that class. That is not a hash. Nothing can
  bind those bytes to the manifest and no rebuild can be checked against anything, so the rule says
  keep them. Of the twelve `renderer_score_checkpoints` rows, **six carry `run_completed: false`**, and
  their stated justification — "superseded by the run's own RESULT.json + argmax_plane.npz" — is simply
  untrue for a run that did not finish; those 353,999,376 B stay too. The remaining six rows
  (354,198,507 B) ARE certifiable: real sha256, `run_completed: true`, and I verified that
  `RESULT.json` and `argmax_plane.npz` both exist on Vertigo for all three treatments
  (`blocks.1.film.weight`, `blocks.3.film.weight`, `control`). I did not take them, because their
  certificate is a SUPERSESSION proof rather than a rebuild proof — neither sr5 producer has that
  shape — and 354 MB did not justify a fifth producer once the rlc2 relocation had already returned
  5.84 GiB to the tier. The six verified rows are listed in the adjudication file so a successor acts
  in one step.

## SIDE EFFECT I CAUSED, RECORDED IN FULL

Running rbf1's own recorded rebuild command rewrote
`/Volumes/VertigoDataTier/pact/ddm_rbf1/retained/COMMAND_render.json` before rendering anything
(`experiments/ddm_rbf1_boundary_probe.py:321` writes `COMMAND_<stage>.json` on every invocation). I
did not back that file up first. Everything in it except one field is identical to what rbf1's own run
wrote — same argv, cwd, axis, retention string — because it was the same command from the same
directory. The one field that changed is `git`, which now reads this arm's HEAD instead of rbf1's
`54a1c9889`. The file is **not** among the fifteen evidence files rbf1 fresh-hashed in its own
retention certification, so no certified hash was invalidated. A note recording the original value
sits beside it.

`RENDER_COMPLETE.json` **is** certified evidence (sha `eba67bc8ff20…`) and the render rewrites it at
the end of its loop. That one I did back up before starting, and it was restored and re-hashed to the
certified value. The lesson is small and worth keeping: before exercising an arm's own rebuild
command, back up every file that command writes, not only the payloads it regenerates.

## A DEFECT IN rbf1's RECORDED REBUILD COMMAND

The retention certification names the rebuild as
`ddm_rbf1_boundary_probe.py render --resume-from …`. That command alone will **not** restore a deleted
payload. `render()` calls `verify_receipt()` on each chunk first, and `fact()` on a missing artifact
raises rather than regenerating it, so the first deleted chunk aborts the run. The working procedure —
the one exercised here — is: move the affected chunk's `RENDER.json` aside, run the recorded command
(it regenerates only chunks whose receipt is absent), check the regenerated files against the sha256
values in the archived receipt, then put the original receipt back. Any successor that reclaims rbf1's
render class must record that corrected procedure in its certs, or the certificate promises a restore
the command cannot perform.

## BOUNDARIES HONORED

No Modal, no scorer forward pass, no candidate work, no pointer write, no archive mutation, no edit to
`upstream/`, the PR tree, or any arm's source. Protected paths were refused **structurally** by both
tools rather than by intention: `ddm_pc3_*` (which covers the new move-45 sealed tree
`ddm_pc3_pose_carrier_curve/candidate/candidate_runtime` and the AP `custody_pointer45` path),
`ddm_ntb2_*`, `ddm_mxo2/3`, `ddm_rlc5_cure_on_move43`, `ddm_sj1_pass6`, `ddm_rp1_round2`,
`ddm_sr4_20260910/retained/ap_offload`, `public_datasets`, `cold_store`, `upstream/`. The local disk
was not used as a tier. APDataStore was read for `df` only. Live work ran throughout on the same host
and the same SSD — `ddm_ntb2_public.py`, `ddm_obx2_*`, `ddm_pc3_pose_carrier_curve.py` and the Modal
harvest poller — and none of it was stopped, signalled or reniced. I held this arm to two concurrent
readers on purpose: a third instance would have gone faster and taken disk queue away from arms doing
science.

## LIVE HYPOTHESES (leads, not credited savings)

- pk4 `pose_preprocessed_yuv6` (21.729 GiB) and mc35 `slave_camera` (4.681 GiB) are certified and
  exercised but retained. A successor that needs space should take these before opening a new store:
  the certificate already exists and the rebuild is seconds.
- pk4 `pose_input` carries eight identical copies of the same `masters` frame per batch
  (`np.repeat(master[None], len(batch_codes), axis=0)`). That is internal redundancy worth roughly
  half of 56.217 GiB, but it is a **compression** question, not a deletion one — the file is the
  rebuild root for both reclaimed classes and must stay whole.
- **`ddm_pfs1_20260729`'s raws carry a measured, lossless 10.233 GiB dedup opportunity.** A prefilter
  over three 16 MiB windows per file found 11 distinct signatures among the 14 raws, with groups
  `{v4d_cx1_pj2ix2, v4d_pj2}` and `{v4d_mq1_partial, v4d_ms8, v4d_pw1}`. Confirming costs a full hash
  of five files; hardlinking then frees three files' worth of blocks **without deleting anything**,
  which the rule prefers over any deletion. I ran this prefilter late — after deletions had begun —
  and did not act on it, because the target was already being met by a class that restores from a
  sibling in its own directory. Evidence: `.omx/research/ddm_sr5_20260911/PFS1_RAW_DEDUP_PREFILTER.json`.
- mxo2_v3 carries a per-frame `{path, bytes, sha256}` receipt plus a store-level `BINDING.json`
  pinning the archive and token field. That satisfies bindings 1-3 of the command-rebuild certifier
  already; it needs a small adapter because the receipts are flat records rather than the
  `{artifacts: …, binding: …}` shape `tools/sr5_certify_delete_command_rebuildable.py` reads today.

## DEAD ENDS

- **pz4's `seg_input` is not a sibling derivation.** I suspected it repeated pk4's relation. It does
  not: a `pz4r_candidate/batches/batch_0001` directory holds `seg_input.float32.npy` and
  `logits.float32.npy` and no `pose_input` at all, so there is nothing in the directory to rebuild
  from. Checking cost one `ls`; guessing would have cost a successor an afternoon.

- I first wrote that `ddm_pfs1_20260729`'s fourteen raws are "almost certainly not duplicates of each
  other" because they decode fourteen different candidate archives. **That guess was wrong, and the
  prefilter caught it** — see the live hypothesis above. The dead end is narrower than I claimed:
  what is closed is *deletion* of those raws, because their rebuild is a candidate inflate.
- A whole-tree rebuild certificate for pk4, pfs1, qs1 or pr135 still does not exist. sr4's refusal
  stands on its own terms. Nothing here supersedes it.


## ADDENDUM — second unit, after Vertigo fell back under its reserve

Vertigo returned to **35.317 GiB** within the hour. Not from my work: a live arm's two full-video pose
diagnostics (7.1 GiB under `ddm_hpr1/diagnostics/`) landed there, and that directory is live — hpr1 is
emitting the move-48 intent and will certify-reclaim its own duplicate. I did not touch it. The guard
was by then refusing a 3 KB copy, measured by `ddm_tmx1` with three instruments.

**So I spent the class I had already certified and exercised but deliberately left whole: rbf1's
render payloads.** `tools/sr5_certify_delete_command_rebuildable.py`, unchanged and already committed,
verified **107 rebuild inputs** present and sha-matched once, then required every chunk's binding block
to be byte-identical to that verified set before touching any file in it.

| | |
|---|---:|
| manifests walked | 95 of 120 |
| artifacts certified and deleted | **596** |
| bytes | **16,317,683,968 = 15.196 GiB** |
| refused | **0** |
| Vertigo | 35.317 → **50.515 GiB** |

It stopped itself at the target with **124 of 720** artifacts still in place. The keep-set is intact and
verified after the fact: all six top-level receipt JSONs, `pricing/` (3) and `amplitude/` (1),
**120/120 `RENDER.json` and 120/120 `SCORE.json`** — the sha records that make the rest reclaimable —
the 1,320 argmax/pose/residual files rbf1's own order puts last, and `RENDER_COMPLETE.json` still at
its certified `eba67bc8ff20…`.

Every cert row carries the restore caveat this arm measured earlier: the recorded rebuild command
aborts on the first deleted chunk unless that chunk's `RENDER.json` is moved aside first. A certificate
that promises a restore the command cannot perform is not a certificate.

**mxo3 was not taken, and the refusal was exercised rather than asserted.** Its condition — "only if
Vertigo is still below 45 GiB" — was never met (50.515 GiB). Independently, all three sr5 producers
refuse the path through their `/ddm_mxo2` guard; I called `assert_not_protected` on a real frame path
in each and recorded the three refusals verbatim. Lifting that guard takes a deliberate reviewed code
change, and I did not work around it. Record:
`.omx/research/ddm_sr5_20260911/MXO3_STRUCTURAL_REFUSAL.json`.

**Running totals for sr5: 2,280 files deleted, 46,647,976,848 B = 43.443 GiB, zero refusals**, plus
3.411 GiB released losslessly by dedup and 5.84 GiB returned to APDataStore by relocation.

A separate deliverable from this unit, read-only and changing nothing:
`.omx/research/ddm_sr5_reserve_derivation_20260911.md` — the 40 GiB SSD reserve is a boot-volume swap
floor copied onto two external SSDs that host no swap. Derived from what it must actually cover, the
figure is ~21 GiB, ~24 GiB with margin.

<!-- # FORMALIZATION_PENDING: storage custody arm; no measured scientific row and no canonical equation governs filesystem reclaim -->
