# DDM PZ4R full-n600 evaluation — direct-v6 is closed

## Result

PZ4R direct-v6 is **FOLDED at INSTANCE scope**. The receiver-closed `183,137 B` archive preserved SegNet almost exactly relative to the matched CP135 raw (`50,412` versus `50,394` flips, `+18`), but its PoseNet distortion collapsed from `0.00014746535453014076` to `0.6310142278671265`. On one matched full-population instrument, the complete recomputed advisory total moved from `0.20513830286735124` to `2.676677850377788`, a **worsening of `+2.471539547510437 S`**.

Axis: **MEASURED `[macOS-CPU advisory, frozen CPU-torch SegNet+PoseNet, n600, batch=16] NON-PROMOTABLE`**. This is not a contest-CPU or contest-CUDA score, does not update the canonical pointer, and earns no T4 confirmation.

Verdict scope: **INSTANCE — exact PZ4R direct-v6 archive SHA-256 `c408adf9101bb19a363039a5e0f7185aabce8f31edb6787e2deaf6d0fe6738f4` through its shipped public receiver**. This verdict does not kill a different jointly learned pose representation or a different archive.

## Matched n600 row

The arithmetic uses `S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37,545,489`. Candidate minus base:

| Component | CP135 matched base | PZ4R direct-v6 | Delta |
|---|---:|---:|---:|
| Archive bytes | 186,252 | 183,137 | **-3,115** |
| Seg flips / 117,964,800 | 50,394 | 50,412 | **+18** |
| `d_seg` | 0.00042719523111979166 | 0.0004273478190104167 | +1.52587890625003e-7 |
| `100*d_seg` | 0.042719523111979164 | 0.04273478190104167 | **+0.00001525878906250694** |
| `d_pose` | 0.00014746535453014076 | 0.6310142278671265 | +0.6308667625125964 |
| `sqrt(10*d_pose)` | 0.03840121801846144 | 2.5119996573788113 | **+2.47359843936035** |
| Rate term | 0.12401756173691066 | 0.12194341109793509 | **-0.0020741506389755637** |
| Complete advisory total | **0.20513830286735124** | **2.676677850377788** | **+2.471539547510437** |

The real CP135-relative rate credit is `0.0020741506389755637 S`, or `17.349449%` of the `0.01195513827824177` gap-to-sub-0.15 reference. The realized candidate instead worsens by `206.7345011` times that gap. The charter's `-4,089 B` and approximately `-0.00272 S` figures are the **LC2** comparison (`187,226 B`), not the stated CP135 comparator. HV1 already contained both correct byte comparisons: `-4,089 B` versus LC2 and `-3,115 B` versus CP135.

## Axis boundary

The contest-CUDA T4 CP135 reference is `34,970` flips, `d_pose=6.885642960696714e-6`, `186,252 B`, `S=0.16195513827824176`. It is recorded only as cross-axis context and was **not used** in any PZ4R delta. The matched macOS-CPU CP135 values are materially different (`50,394` flips and `d_pose=0.00014746535453014076`), which demonstrates why mixing instruments would be invalid.

No Modal job, contest-CUDA scorer, contest-CPU scorer, MPS scorer, or pointer promotion ran. The exact shipped receiver front door did run; only the subsequent scorer axis is advisory.

## Receiver, custody, and retention

Before decode, the candidate archive was pinned at:

- `/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/direct_v6/submission/archive.zip`
- `183,137 B`
- SHA-256 `c408adf9101bb19a363039a5e0f7185aabce8f31edb6787e2deaf6d0fe6738f4`

The unchanged shipped `inflate.sh` was the receiver entry point. It completed in `800.1684911251068 s` and retained the exact `3,662,409,600 B` raw at SHA-256 `7810185baf2033d3b256130e60b173ff4410dffc496a8df2e668a81c2f6e15f3`.

Core retained payloads under `/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/direct_v6/full_n600_eval/`:

| Payload | Bytes | SHA-256 |
|---|---:|---|
| Candidate raw `retained/raw/pz4r_candidate/0.raw` | 3,662,409,600 | `7810185baf2033d3b256130e60b173ff4410dffc496a8df2e668a81c2f6e15f3` |
| GT argmax | 117,964,928 | `fee51ccfb9c18405667d2d9b6f38eb0a68462e114477f85ad00b349744bdd6ee` |
| CP135 argmax | 117,964,928 | `b8f063eb53891ca89d02c80adc7fca7d8c5f638f80cecf364a7f7a89bc68647a` |
| PZ4R argmax | 117,964,928 | `24c40a805f0c4a62a8432cce15ce5c4b4e2d21a4c0ff3b8708d5f77d9784e784` |
| GT Pose first-six vectors | 14,528 | `82ed61ce6a11a6612502527fbb6864a22fe6c6099312e637d971214ab660fb27` |
| CP135 Pose first-six vectors | 14,528 | `e64e8bd36c1a603da30c15fa581cdaeda409e8939cefe61c3d01d09ac0850386` |
| PZ4R Pose first-six vectors | 14,528 | `fd965ef6bf635042ea67628e9f0bbae6dfc6471842c92752f3803f5a5a2899bb` |
| PZ4R Pose repeat vectors | 14,528 | `fd965ef6bf635042ea67628e9f0bbae6dfc6471842c92752f3803f5a5a2899bb` |
| Final result JSON | 26,049 | `99f3361767145299221389843b8a435f8581442855e3e1da62a5b094544faa0c` |

The deterministic Pose repeat is byte-identical and has MSE `0`. The receiver also retained `tokens.npz` (`117,967,988 B`, SHA `f11bf4a5996cd7a416821e2b8c48968fd90bef5c1a5c9b4f92b2c765d99b0952`), its progress checkpoint, every GT RGB batch, all Seg/Pose inputs, all full logits/outputs, and every batch receipt. No generated payload was deleted or moved. The run root occupies approximately `23 GiB`.

Independent verification rehashed **619 unique file records totaling 28,463,997,213 referenced bytes**, checked `38/38` batches and `600/600` pairs for each of three Seg and four Pose sources, recomputed every metric from the retained arrays, and reproduced the complete delta exactly. Receipt: `EXECUTION_PROVENANCE_FINAL.json`, SHA-256 `d9a81f71ce2935060067ad4a2bddb8fe659c178aebaec59aa57041e8ae5e753f`.

## Resumability and runtime closure

The fleet scorer lock and per-run `RUN.lock` were both held. The governed lane `ddm_pz4r_full_n600_eval` was claimed before launch and closed terminally as `completed_folded_instance`. Immutable checkpoints exist for inputs, launch contract, receiver, runtime closure, all three Seg sources, all four Pose sources, and the final result.

Two attempts failed before any decoded raw or scorer output existed:

1. The public receiver's dependency smoke accepted the pinned Brotli CLI, but `pose_target_receiver.py` imports the Python `brotli` module unconditionally.
2. A fresh public dependency bootstrap then failed because `uv` tried to mutate the sandbox-read-only home cache.

Both failure logs and the extracted archive member remain retained. The successful attempt used the exact cached Brotli `1.2.0` source (238 files, 22,253,972 B, tree SHA `7e31b62d04b0669366c0addd26f1103837ebdfbe92cbbe1ff83c04aeb01cac25`) compiled for the upstream CPython 3.11 interpreter and combined with constriction `0.5.0`; source identity, full build tree, installed closure, log, round-trip proof, and hashes remain in `RUNTIME_BOOTSTRAP_PLAN.json` and `RUNTIME_BOOTSTRAP_RECEIPT.json`. The shipped receiver source and archive were not edited.

Stage 0 froze runner SHA `188317517afb725444378c57378a9b634b9fb1353ea7a328fd3265f8998a570d`; the successful resume-only generation is SHA `93dddd7dc43b8f4790c0852281b34706efc4c3cdac1b038779be782c406927de`. The metric worker remained SHA `03dc9e81a21409f5881cff642d5dc334a8f04deae5b008f31cd2719bba4a14fb`. The sidecar above makes this historical/current distinction explicit.

## RECALL EVIDENCE

Searched the full corpus before adjudication, not only the charter seeds:

- `.omx/research/` by content with `PZ4R|direct_v6|c408adf9|PGQ1|pose gauge|183,137`, including HV1, PZ4R/PGQ1, PZ4P, capstone/5W, CN4, NA6, PO1, HY1, and their final messages/charters.
- `.omx/research/CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks, `.omx/state/main_hot_state.md`, and SPEC/design/task-ledger surfaces with `PZ4R|direct_v6|PGQ1|pose gauge`.
- The full canonical equation registry via `.venv/bin/python tools/list_canonical_equations.py --json`, filtered for contest score, archive bytes, pose, and score marginal. The relevant law remained the exact contest formula plus `score_marginal_lagrange_multipliers_v1`; no PZ4R-specific alternative score law was found.

Beyond the seeds, recall found the already-recorded correction that `-4,089 B` is versus LC2 while CP135 gives `-3,115 B`; the retained local CP135 raw needed for a same-instrument comparison; and the strong known T4-versus-local base drift that forbids delta-adding across axes. These findings changed the plan from “compare PZ4R to the T4 literals” to “rerun GT, CP135, and PZ4R on one macOS-CPU instrument and keep the T4 row reference-only.” No cheaper PZ4R-specific full-n600 distortion receipt was found in the searched index/DAG/SPEC/task-ledger scope.

## Verification and commit boundary

- Focused PZ4R tests: `3 passed`.
- Payload-retention AST gate on the worker, runner, and tests: `0` findings.
- Ruff and CPython 3.11 compile checks: passed.
- Two explicit review-tracker passes were recorded for each changed Python file, with two further resume-specific passes for the runner.
- Two older JS1B tests failed against unrelated, already-modified stage-0 custody logic in the shared worktree; they do not exercise the PZ4R runner or changed worker paths and were not altered.
- The serializer was invoked with post-edit hashes, required tags, explicit files, and no co-author. Git staging failed before index mutation with `unable to create temporary file: Operation not permitted`; the managed sandbox exposes `.git` read-only. The artifacts are therefore honestly **uncommitted**, with unrelated dirty work preserved.

## Follow-on disposition

- **PZ4R direct-v6 candidate: FOLDED** at INSTANCE scope.
- **T4 confirmation / Modal dispatch: FOLDED**; the matched advisory sign is catastrophically negative, so the charter's confirmation trigger is not met.
- **Lane/result consumption: FIRED** through the terminal lane row and this retained result/report.
- **Exact pointer: UNMOVED** at CP135 composed `S=0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`.
- **Own-vehicle frontier: UNMOVED** at LC2 `S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]`.

## NEXT_IF_RESUMED

- **Disposition: QUEUED-WITH-A-FIRE-ORDER; owner: MAIN/operator with a Git-writable checkout; consumer store: repository Git history for the runner, worker adaptation, tests, and this report; fire trigger: `.git` object/index writes are permitted.** Re-run the serializer with the final post-edit SHA for each explicit file, no co-author, and `[no-triality] [p0-ledger-ok]`; do not restage unrelated work.

## LIVE-HYPOTHESES

- A different jointly learned pose payload may reuse PZ4R's low-rate semantic portion. This remains plausible because Seg changed by only 18 flips while the failure is overwhelmingly pose-only, but it requires a new counted archive and is not a retry of direct-v6.
- The pose-gauge proxy/parse-back checks are structurally blind to PoseNet semantics. A future pose recode should gate on retained randomized or full-population PoseNet vectors before any remote fire; the current receiver was repeat-deterministic yet semantically wrong.

## DEAD-ENDS

- Exact direct-v6 archive `c408adf9…38f4` is closed: `d_pose=0.6310142278671265` overwhelms every byte saving.
- Rate-only extrapolation from either `-4,089 B` versus LC2 or `-3,115 B` versus CP135 is closed for this object; receiver equality and deterministic repeat did not preserve distortion.
- T4 confirmation of this archive is closed by the charter's own fire rule; no paid/remote dispatch is justified.
- Mixing the T4 CP135 components with the macOS PZ4R components is closed because the matched local base materially differs.
- The CLI-only Brotli closure and the default home `uv` cache are closed in this managed sandbox: the receiver imports Python Brotli, and the home cache is not writable.
