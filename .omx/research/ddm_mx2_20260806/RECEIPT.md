# ddm_mx2 PR130 Pose-Leg Lift Receipt

## Verdict

Status: **PARTIAL, honest landing**.

Measured this turn:

- PR130 pose-leg source was vendored under `src/tac/pr130_lift/pose/lifted/` with borrowed-substrate headers and a source manifest.
- CPR1 symbol codec round-tripped PR130 legacy-shape carrier symbols in the mx2 unit test.
- Lossless byte-only repack race ran over 17 current banked pose-adjacent sections; CPR1 applied to 0/17 because none matched the PR130 legacy carrier byte layout.
- Local mx2 Python slice passed: `PYTHONPATH=src .venv/bin/python -m pytest src/tac/pr130_lift/tests/test_mx2_pose_lift.py` -> 5 passed.

Not measured:

- No n600 scorer sweep.
- No `upstream/evaluate.py` exact score.
- No PR130 pose fit against our tq1c/mx1 surface.
- No real-frame MLX/PoseNet parity pass; local MLX execution is blocked by Metal device access.

Score/frontier claim: **none**. Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]` from tq1c; contest borrowed pointer remains separate.

## Source Custody

- source repo: `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo`
- source head: `2f94596bb0136d342254022a5c9584756eae0468`
- manifest: `src/tac/pr130_lift/pose/vendor_manifest.json`
- manifest sha256: `02c7f52a5f294912dcd77dc602de63acefa76ea3177c3157d2def966d6628fad`

Vendored PR130 pose files:

- `code/carrier_codec.py` -> `d2f14402374b4e622b7f981d736389fb04f0ca0165180e4c75f3a32ffe996bed`
- `code/learned_pose_carrier_oracle.py` -> `59a574bf2ec52953ccdcf50edab94fff5d9c62158b993bd4da2e245f5fe310a4`
- `code/pack_semantic_pose.py` -> `151413b0696a23e26fbc246dd2cbf7f42c3d64f67290e7c61f21ccc1d0a9bfbc`
- `code/pose_basis_oracle.py` -> `90909bdd600cf8f518b8fc951b19fd6b66796a1ed963ece4c7638cb1b14057cb`
- `code/refine_pose_coeff_codes.py` -> `8912464ba22f6bde09d4d53e4e16d1c6880b48d2358752219073fa2d4710a403`
- `code/repack_carrier.py` -> `df6bdaa23d0bd1f717af588931ebdb5ed3777517af8b046e287835aa16a7a72e`
- `code/search_pose_coeff_cpu.py` -> `42b0f57cad5170b6da2d7147ba90af147cda7baecc74b848b9cd2c4bed7c9285`
- `code/train_pose_carrier_full.py` -> `684a4906edecb7653572db77c11a03a4e445eb256a8dc7b665e8fa0f78cab649`

Borrowed-substrate accounting: the PR130 mechanism, tensor shapes, optimizer/search semantics, and CPR1 carrier codec are theirs. The mx2 additions are custody/import wrappers, lossless applicability gates, local MLX wrapper shape, tests, and receipts. No originality or score claim is made from the borrowed source.

## Implemented Artifacts

- `src/tac/pr130_lift/pose/source_loader.py` sha256 `85d9133138b8947d989f83c3746a1842f9acf08de67b8067bb1b0a88bbaf8001`
- `src/tac/pr130_lift/pose/repack_race.py` sha256 `820392995f78f1843fd2eb1e4a63fb1cf88e63b84ad161447652ddb396367245`
- `src/tac/pr130_lift/pose/mlx_pose_carrier.py` sha256 `806eac3c6ddb95923b042ead60a222b420bd15d316a56428fe908035d057251e`
- `src/tac/pr130_lift/tests/test_mx2_pose_lift.py` sha256 `14bfaa92d27ea1f857900b489d5ddf8daaf41582b715133029048499edb377e5`
- `/Volumes/VertigoDataTier/pact/ddm_mx2_20260806/repack_race/repack_race.json` sha256 `268e83faa8871c750fc8d61ee60a2572d0384f48bc82e4d2f149f39b97893459`

## PR130 Artifact Context

- local PR130 base archive: `artifacts/base/int5_delta_archive.zip`, 197,228 B, sha256 `e16abacf3a83062f96139ef980fe95d9fd2061a5ce89d1d31c80dcfe52d65051`
- PR130 derived self-compressed predecessor named in `recipe/artifacts.json`: 194,380 B, sha256 `f4457de09a6e69c8cd29e886a84705462a8c77dc6978020b11dff52e661a1451`
- PR130 final archive: `artifacts/final/archive.zip`, 191,052 B, sha256 `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`

The vendored repacker refused the local base archive as the frozen canonical source and also refused it under noncanonical mode because its LZMA source-model stream did not reproduce with PR130's recorded settings. I did not count that as an mx2 failure; the unit-level CPR1 symbol round-trip is the admitted local codec control.

## Stores Consulted

- `PROGRAM.md`
- `CLAUDE.md` / `AGENTS.md`
- `docs/operating_manual_craft_handoff.md`
- `.omx/state/main_hot_state.md`
- `.omx/research/ddm_mx2_20260806/CHARTER_AMENDMENT_MAIN.md`
- PR130 `recipe/TRAINING.md`
- PR130 `recipe/artifacts.json`
- PR130 `evidence/cpr1_verification.json` path noted through artifact listing, not used as a local score
- `.omx/research/ddm_mx1_20260806/LAUNCH_TICKET.md`
- `/Volumes/VertigoDataTier/pact/ddm_mx1_20260806/launch_ticket.json`

Recall/search scope: `mx2`, `common contract`, `lane`, `charter`, `codex_runs`, PR130 pose files, current IX2/PFS1/SC1 pose-adjacent sections, and existing mx1 launch evidence. Absence statements here are bounded to those scopes.

## Review And Commit Gate

Two review-tracker passes were run for every non-empty mx2 Python file touched. The tracker marked all wrapper, test, and vendored source entities as `reviewed`. Empty `__init__.py` files have no tracked entities and reported "not yet ingested"; there was nothing to mark in those files.

Serializer commit was attempted twice with explicit file list and post-edit sha guards. Both attempts were blocked by the pre-commit CI-blind MLX test step: the hook selected unrelated MLX-gated tests from broad staged tokens and the local runtime failed with `Fatal Python error: Bus error` after `[metal::load_device] No Metal device available`. I did not set `PREFLIGHT_SKIP_CI_BLIND_TESTS=1`.

Commit status: **unlanded, blocked by local no-Metal CI-blind hook**. This receipt itself makes no score or frontier claim.
