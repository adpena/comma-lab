# ddm_ri1 — RC1 is receiver-closed and folded: the 113,006 B byte win scores 17.306291367262396 on the completed local n600 full-RGB advisory

**Disposition:** `FOLDED`, `verdict_scope=INSTANCE`. The exact K=2,048/i3 RC1 archive is
byte-feasible but catastrophically distortion-infeasible after the actual copied DX2 shipping
receiver renders full RGB. MAIN must not fire this object. The sealed closure is
`/Volumes/APDataStore/pact/ddm_ri1_rc1_full_rgb_receiver/SEALED_MAIN_DISPOSITION.json`, 4,657 B,
SHA-256 `e3537d5843aa669deed75c4ed1a4d937364d0a576f814c0c8130ca4ddc8513cf`.

## Result first

The canonical local firer completed the exact 113,006 B archive over all 600 pairs on CPU. The
harness labeled the row `[env-mismatch advisory]` because the mirrored upstream `uv.lock` declares
no comparison group; GT followed upstream's CPU fork, `PYAV_YUV420_TO_RGB`. This is a local
advisory diagnostic, not a contest-CPU/CUDA authority row and not promotion evidence.

| exact component | measured value | score contribution |
|---|---:|---:|
| archive | 113,006 B, SHA-256 `6756ae8f39116907828ee27b8f9686b9935eaae94c61f68c3eb02de16d45e87a` | `25*113006/37545489 = 0.0752460568565241` |
| `d_seg` | 0.01605413 | 1.605413 |
| `d_pose` | 24.41603851 | 15.625632310405873 |
| **S, recomputed from the captured components** | **17.306291367262396** | **17.306291367262396** |

The score receipt is
`/Volumes/APDataStore/pact/ddm_ri1_rc1_full_rgb_receiver/advisory_r1/contest_auth_eval.json`,
37,333 B, SHA-256 `9d08795f9101a38c03f5b90e4081ced5fd112b15796af345a76789c168ed6425`.
It records `n_samples=600`, `dispatch_attempted=false`, `gpu_launched=false`,
`promotable=false`, and total/inflate/evaluate elapsed times of 815.762/386.415/428.096 seconds.
No Modal, Metal, contest-CPU, or contest-CUDA run occurred.

The archive retains its 24,980 B rate headroom below the fixed-DX2-distortion strict ceiling, worth
only 0.01663315664899184 score units. Lane alone contributes 0.509110344780816 score units,
30.61 times that entire headroom. The overall Seg term is 1.605413 and the Pose term is 15.6256323.
This is not close enough to justify an authority fire.

## Full-RGB shipping receiver integration

Commits `41a8583b16` and `96f6ee03be` land the receiver builder, runtime adapter, exact retained
diagnostic, and the advisory harness retention hook:

- `experiments/ddm_ri1_rc1_full_rgb_receiver.py` verifies every charter pin, the RC1 payload
  `eab66bad9d113ed79475a810f4002ec821deb335c3e87fc1b1e90ef2b8e61164`, and the sealed DX2
  runtime before building.
- `experiments/ddm_ri1_runtime_receiver.py` is copied additively into the staged runtime as
  `runtime/ri1_rc1_receiver.py`. It strictly parses the RC1A terminal-program payload with the
  exact copied `rc1_terminal_program_vq.py`, then supplies its decoded token tensor to the
  unchanged shipping semantic/carrier inverse and full-RGB renderer.
- The integration replaces only the terminal token decoder. The copied DX2 semantic weights,
  carrier, residual, frame-0 selector, compensation, `SemanticTokenRenderer`, and
  `render_video` remain the actual shipping implementation in `cpr1/inflate.py`.
- The source DX2 runtime is 39 files / 685,975 B / tree SHA-256
  `7799b291a99027c705b42f094cf0533459399f3ea711ec34d754f81c1fde5f1d`. The built RI1
  runtime is 41 files / 656,798 B / tree SHA-256
  `f80c7720b2ec2a7c00c4fa0185167fa355ded5974ad05956d7c1ad25001c2645`.

The build receipt is
`/Volumes/APDataStore/pact/ddm_ri1_rc1_full_rgb_receiver/build_r2/RESULT.json`, 15,575 B,
SHA-256 `7b19b12a6eb13a9cf2ee848ce4bc03ec069c6bb078fd6d68859b1d3ca391bf7e`.
The scored full-RGB raw is retained at
`advisory_r1/work/inflated/0.raw`, 3,662,409,600 B, SHA-256
`d424febe33d90fcc93ba6e618b17585f16d2e0eb26ebcdc8d4770f990bd3756c`.

## Per-class SegNet retention

The advisory harness's pre-existing `src/tac/pose_per_pair_retention.py` hook failed after scoring
because it calls the current upstream `DistortionNet` with obsolete constructor arguments:
`TypeError: DistortionNet.__init__() takes 1 positional argument but 3 were given`. The scored row
was unaffected. RI1 cured the retention debt with its own resumable post-pass over the exact retained
raw using the current upstream `DistortionNet`, AV/Tensor datasets, weights, seed, and batch geometry.

Before reduction, each of 38 chunks fsyncs the GT and candidate SegNet argmax fields, per-pair Seg
and Pose distortions, and both six-value PoseNet outputs. The 600 retained Seg values reproduce
`d_seg=0.01605413` with relative error `1.250977066741702e-7`; the retained Pose values reproduce
`d_pose=24.41603851` with relative error `4.785300905965781e-8`. The argmax confusion reduction and
per-pair Seg vectors agree to relative error `1.2085709337787791e-9`.

| class | GT pixels / area | conditional disagreement | contribution to total `d_seg` | IoU |
|---|---:|---:|---:|---:|
| Road (0) | 27,407,043 / 0.232332 | 0.0159922 | 0.00371550 | 0.952917 |
| **Lane (1)** | **690,639 / 0.00585462** | **0.869587** | **0.00509110** | **0.127481** |
| Undrivable (2) | 58,413,282 / 0.495176 | 0.00285072 | 0.00141161 | 0.983603 |
| Movable (3) | 1,460,325 / 0.0123793 | 0.430390 | 0.00532794 | 0.527204 |
| MyCar (4) | 29,993,511 / 0.254258 | 0.00199790 | 0.000507982 | 0.995950 |

Lane loses 600,571 of 690,639 GT pixels and retains IoU 0.1274806340345608. It is a named severe
failure, not hidden by the aggregate. Movable contributes slightly more total error because its area
is larger, while Pose collapse dominates the complete score.

The diagnostic receipt is
`/Volumes/APDataStore/pact/ddm_ri1_rc1_full_rgb_receiver/per_class_r1/RESULT.json`, 13,151 B,
SHA-256 `849d64e093fcdcc3b3240c637c1cf4aa0e51f15fa3eb155bde3b93012db363a1`.
It validates all retained file hashes and reproduces the full aggregate on `--resume-from` without
re-running the scorer.

## Repeats and paid-section controls

The builder retains two independently decoded 117,964,800 B token tensors. Both have SHA-256
`2c85d29698782b2b12f75a897665f80c59a40a9549f0697e18db16feaca93168`. Independently built
`archive.zip` and `archive.repeat.zip` are both 113,006 B with SHA-256
`6756ae8f39116907828ee27b8f9686b9935eaae94c61f68c3eb02de16d45e87a`.

Five complete 113,006 B mutation archives are retained under
`build_r2/retained/mutation_controls/`. One-bit mutations of semantic, carrier, residual,
assignment, and codebook sections are all refused by the strict receiver with section-specific CRC
errors. These controls prove receiver consumption and integrity, not score quality.

## Prior-law verdict

**SUPPORTED, `verdict_scope=INSTANCE`.** The prediction was that this exact K=2,048/i3 temporal
program would be byte-feasible and distortion-infeasible after the shipping receiver. Its measured
`d_seg=0.01605413` is 43.66 times the fixed-DX2-pose `d_seg` ceiling
`0.0003677271516778194`, and its Pose output collapses. This closes this exact RC1 candidate.

It does not kill terminal programs, quotient representations, or scorer-aware dictionaries as
families. The evidence is one completed representation/receiver instance on one advisory CPU axis.
The newer CB2 audit also changes the diagnosis: RC1 already gives Lane-valued slots 8.97 times
Lane's area share, yet Lane agreement remains 14.7%. Simple class-area reweighting is therefore a
closed formulation; the remaining plausible problem is temporal/topological allocation or an
objective that optimizes scorer cells rather than Hamming agreement.

## RECALL EVIDENCE

The pre-build recall searched `.omx/research/` by content for `RC1`, `terminal program`,
`token agreement`, `shipping receiver`, `full-RGB receiver`, `K2048`, and `Lane`; filtered
`tools/list_canonical_equations.py --json` for `receiver`, `token`, `representation`, `archive`, and
`rate`; and searched `CANONICAL_RESEARCH_INDEX*` plus `sub015_DAG_*` for the same surfaces.

Beyond the charter seeds, the search found:

- DX2's candidate seal and retained runtime prove the current object and exact shipping tree that
  RI1 had to copy rather than approximate.
- JO6's receiver/container receipts reinforce strict single-member parse-back and mutation refusal;
  they changed no mechanism but confirmed additive staging was required.
- The canonical receiver-forward parity equation requires real receiver consumption and offers no
  rule that turns token agreement into evaluator credit. No current-DX2 equation or indexed receipt
  supplied RC1 distortion, so the n600 scorer leg remained mandatory.
- NI1 has since produced a distinct K32 byte-closed receiver object at 122,250 B but explicitly has
  no scorer result. It is not evidence for or against this K=2,048 row.
- CB2 measured that RC1's Lane capacity is already 8.97 times its area share. This changed the
  interpretation of the Lane collapse from simple underallocation to a topology/objective mismatch;
  it did not change RI1's receiver or scorer plan.
- The existing per-pair retention helper had drifted from upstream's current `DistortionNet` API.
  That discovery changed the plan after the canonical score completed: RI1 added and ran a separate
  exact, resumable, payload-retaining DistortionNet post-pass rather than inventing per-class values.

Within those bounded corpus/index/equation scopes, RI1 did not find any prior current-DX2 n600
full-RGB score for the exact RC1 K=2,048/i3 archive.

## Verification and boundaries

- `ruff check`, `py_compile`, deterministic archive/token repeats, strict mutation controls, completed
  canonical local advisory, and completed diagnostic resume all passed.
- `src/tac/tests/test_pose_per_pair_retention.py` and `src/tac/tests/test_contest_auth_eval.py` ran
  together: 83 passed and two existing `contest_auth_eval` tests failed because they monkeypatch
  `subprocess.run` while the implementation now calls `run_in_process_group`; neither failure touches
  the RI1 retention-env addition. The real canonical advisory completed through that process-group path.
- Repo-wide `tac.preflight --scope dev` remained red on unrelated shared-worktree/state/custody gates;
  both serializer commits passed their staged pre-commit gates. No unrelated dirty file was staged.
- `upstream/` stayed read-only. NR1/OS1 sources and retained trees were not edited. No payload was
  deleted or discarded; scalar reports always point to retained source bytes.
- The authority boundary is strict: this row is `[env-mismatch advisory]`, not contest-CUDA/CPU,
  and cannot move the canonical pointer. The sealed MAIN disposition is `FOLDED` with
  `dispatch_argv=null` and no fire trigger.

Own-vehicle frontier remains **DX2 S = 0.14821987563243377 @ 180,368 B `[contest-CUDA T4, n600]`**, archive SHA-256 `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`; RI1 did not move it.
