# EK1 Receipt — Einstein-Kolmogorov Worktree Residue (2026-08-05)

## Answer First

EK1 did **not** land code. The presumed head module, `src/tac/canonical_equations/einstein_kolmogorov_crux_20260719.py`, is already byte-identical on current main and superseded by `57e4c4e52b` / current `HEAD` (`5c7c3b03f8`).

The remaining code/docs residue is real signal, but it is **HOLD_BLOCKED_CURRENT_MAIN_PREFLIGHT_RED**: a clean three-way merge of the residue bundle compiled and passed diff checks, then focused pytest failed with 7 failures in the Catalog #351 suite. The expanded Catalog #351 scanner reports 188 live current-main violations, so landing the residue as a strict gate would break the authority preflight. I backed the attempted merge out. No worktree bytes were deleted.

Inventory result: `git status --porcelain --untracked-files=all` shows **45 status-visible files**, not the charter's stale 17-file count. Branch delta `git log main..codexwt/einstein_kolmogorov_crux_20260719T212159Z --oneline` is empty; committed branch content is already behind/currently contained by main.

Machine manifest: `ek1_residue_manifest.json` (SHA-256 `7dc3458e702203580b15e9688b6e7c66328820fba4eedc19442393feb796240e`).

## Recall Evidence

Sources searched beyond the charter seeds:

- Memory registry query: `rg -n "ek1|common_contract|lane|charter|#899|required-component|required_component" /Users/adpena/.codex/memories/MEMORY.md`; relevant result was the #876 worktree-custody group and porcelain-based audit method.
- Repo recall query: `rg -n "Einstein|Kolmogorov|U2|u2|entropy|lower[-_ ]bound|xi|crux" src/tac/canonical_equations .omx/state/canonical_equations_registry.jsonl .omx/research/CANONICAL_RESEARCH_INDEX* .omx/research docs`.
- Repo recall query: `rg -n "einstein_kolmogorov|kolmogorov|u2|lower_bound|xi_bridge" .omx/research .omx/state src/tac docs experiments`.
- Canonical equation registry command attempted: `.venv/bin/python tools/list_canonical_equations.py --json`; it produced a very large registry dump, so follow-up recall used scoped text searches and direct path checks.

What changed the plan: the canonical module was not missing from main. Current main already tracks it, and the residue module bytes are identical to current main. The plan changed from direct harvest to a supersession/blocker classification plus an attempted three-way merge of the non-identical guard bundle.

## Inventory Commands

- `git -C <worktree> status --porcelain=v1 --untracked-files=all`: 12 modified tracked paths + 33 untracked paths.
- `git -C <worktree> log --oneline -5`: `6a78ee8209 Einstein crux: close recursive-review custody classes; 03ae758880 Einstein-Kolmogorov: close recursive review authority classes (323 tests); d30b243c80 Einstein-Kolmogorov: close review custody classes (264 tests); 6941d625d7 Einstein-Kolmogorov: seal v3 banked rate-wall custody (124 tests); 5758418c0e Fix V10 receiver cleanup suffix custody`.
- `git -C <worktree> log --oneline main..codexwt/einstein_kolmogorov_crux_20260719T212159Z`: empty.
- `git diff --no-index` confirmed the residue canonical module is byte-identical to main; `uv.lock` is also byte-identical.

## Triage Table

| Path | Status | Bytes | SHA-256 | Class | Disposition |
|---|---:|---:|---|---|---|
| `.omx/research/einstein_kolmogorov_crux_20260719T214408Z_codex.md` | modified | 54729 | `b1c27ebb4a862828e60969a512f77b98db23bd85e07f19b37abb70ea76861c50` | SIGNAL | HOLD_BLOCKED_CURRENT_MAIN_PREFLIGHT_RED |
| `.omx/research/einstein_kolmogorov_crux_DAG_FEED_20260719.md` | modified | 17323 | `32b785bf745b506ad04c7fdfa38010c48471988ebf62d32b7e7fe3fe1d59fba6` | SIGNAL | HOLD_BLOCKED_CURRENT_MAIN_PREFLIGHT_RED |
| `.omx/research/einstein_kolmogorov_review_bugclass_scope_extensions_20260720_codex.md` | modified | 9430 | `9835f8d7c19db043b2b85dc1d2e52bc860b9820a834820e36d200434a8727812` | SIGNAL | HOLD_BLOCKED_CURRENT_MAIN_PREFLIGHT_RED |
| `CLAUDE.md` | modified | 420347 | `8acb86dbb65a40f988dd62c26cca8af96e2c1e0b3c137603e2afd9280757077e` | SIGNAL | HOLD_BLOCKED_CURRENT_MAIN_PREFLIGHT_RED |
| `docs/meta_bug_class_catalog.md` | modified | 820809 | `d39999d2beea8d60adf4d70aded27b714b0ac30695cd2f1704e3de8952d709db` | SIGNAL | HOLD_BLOCKED_CURRENT_MAIN_PREFLIGHT_RED |
| `src/tac/canonical_equations/einstein_kolmogorov_crux_20260719.py` | modified | 68061 | `d1b213feb8b80fb6128e6120befa274ebff3c21185c238624997b0bebd3298e6` | SUPERSEDED | SUPERSEDED_ALREADY_ON_MAIN |
| `src/tac/canonical_equations/registry.py` | modified | 50152 | `f1d9c54b12f1a8efeb9038466c8dc736da27d06fe0b79defe23e57b2fc52bb29` | SIGNAL | HOLD_BLOCKED_CURRENT_MAIN_PREFLIGHT_RED |
| `src/tac/preflight.py` | modified | 3967210 | `79461e23f1680823420fa67638af1aae2a99be5ca18c0c92973073089329d74b` | SIGNAL | HOLD_BLOCKED_CURRENT_MAIN_PREFLIGHT_RED |
| `src/tac/tests/test_check_154_manifestless_cleanup_identity_scope_extension.py` | modified | 17061 | `6d780108749f0c4c59b5345d97be3d0b9c23326481f21fa84cb51555a220f926` | SIGNAL | HOLD_BLOCKED_CURRENT_MAIN_PREFLIGHT_RED |
| `src/tac/tests/test_check_344_anchor_roundtrip_scope_extension.py` | modified | 26058 | `592d5365730eb14f8cbd8e192a6fdda421fbb4a0b78b29a9ec19f78fea7559db` | SIGNAL | HOLD_BLOCKED_CURRENT_MAIN_PREFLIGHT_RED |
| `src/tac/tests/test_check_351_canonical_producer_identity_scope_extension.py` | modified | 25421 | `36115ed13791680c5c382c044521e666c7c5ae0a74f5e7664b0015dd5906f6a7` | SIGNAL | HOLD_BLOCKED_CURRENT_MAIN_PREFLIGHT_RED |
| `uv.lock` | modified | 991240 | `c37ac551ab5797af2d6200a8613b392c522cd4cf53b8c4fe78f8195f9c7daa90` | SUPERSEDED | SUPERSEDED_ALREADY_ON_MAIN |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719/baseline/checkpoint.json` | untracked | 12775 | `74bc563ffaae877d0d3b3135d342e5eb4db33e60765ea0ab584912a55702f92b` | SUPERSEDED | CERTIFIED_HISTORICAL_V1_RUN_SCRATCH_SUPERSEDED |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719/baseline/receipt.json` | untracked | 13214 | `e9cebad25f16b9c4e1a0cc7375e163e8bef4a96c92d0f0f9c424462c5d93877c` | SUPERSEDED | CERTIFIED_HISTORICAL_V1_RUN_SCRATCH_SUPERSEDED |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719/coordinate/checkpoint.json` | untracked | 13669 | `48800777234c0435faaea76823dc3ee44d24dd6f3c86b2ba8181ed81221503af` | SUPERSEDED | CERTIFIED_HISTORICAL_V1_RUN_SCRATCH_SUPERSEDED |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719/coordinate/receipt.json` | untracked | 14160 | `bb3437c9a0b5193284e709db1f00cc03ad5922ec700abd6a83a2d0acc70b602d` | SUPERSEDED | CERTIFIED_HISTORICAL_V1_RUN_SCRATCH_SUPERSEDED |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719/coordinate_deep/checkpoint.json` | untracked | 13715 | `edee710310b2dbf999fd0f7aeb41716ecde1ed8b6cfb42f7a96eade878c056ea` | SUPERSEDED | CERTIFIED_HISTORICAL_V1_RUN_SCRATCH_SUPERSEDED |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719/coordinate_deep/receipt.json` | untracked | 14329 | `57d5816953ab0438224dc76af530eec14c9129982182ab3df84d1be6bc748674` | SUPERSEDED | CERTIFIED_HISTORICAL_V1_RUN_SCRATCH_SUPERSEDED |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719/dspsa/checkpoint.json` | untracked | 15556 | `ab023b99801c0d10bfb2ac61f71475f253af2bc92550de91c385ef835ddc1f08` | SUPERSEDED | CERTIFIED_HISTORICAL_V1_RUN_SCRATCH_SUPERSEDED |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719/dspsa/receipt.json` | untracked | 74299 | `cf3ab6f1d7c8e8b4a948817d45f9b70a86a8f03ec833fc2ee411d93b639a0315` | SUPERSEDED | CERTIFIED_HISTORICAL_V1_RUN_SCRATCH_SUPERSEDED |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719/dspsa_deep/checkpoint.json` | untracked | 15562 | `ea61fea6a88c4bcfc64058a76a1902c8203c179261ccccb5e6d9aad4126f0e99` | SUPERSEDED | CERTIFIED_HISTORICAL_V1_RUN_SCRATCH_SUPERSEDED |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719/dspsa_deep/receipt.json` | untracked | 253371 | `78e6e6e8311bf1172acfb0b80a57bd8eb3895cd745242513ae71ca8514c3c54e` | SUPERSEDED | CERTIFIED_HISTORICAL_V1_RUN_SCRATCH_SUPERSEDED |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719/dspsa_repeat/checkpoint.json` | untracked | 15563 | `e9c9c40f74d05a9f2c930dd6d98e1fe96aa076b4a3605cbf4f3f7085c20c0401` | SUPERSEDED | CERTIFIED_HISTORICAL_V1_RUN_SCRATCH_SUPERSEDED |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719/dspsa_repeat/receipt.json` | untracked | 75825 | `9e65fe7af0cddc7c5671f4e9717fb291a6bf1b287b8ac6c65594e53ed3d1b1b9` | SUPERSEDED | CERTIFIED_HISTORICAL_V1_RUN_SCRATCH_SUPERSEDED |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719/global/checkpoint.json` | untracked | 12775 | `73b0201d4ea111925258b57d7045bd971b8808f0432592805165a09763764f0c` | SUPERSEDED | CERTIFIED_HISTORICAL_V1_RUN_SCRATCH_SUPERSEDED |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719/global/receipt.json` | untracked | 13164 | `d2637908bb5271851b05601571c0e4d77333b1fee7d73893ce946c83e928a59d` | SUPERSEDED | CERTIFIED_HISTORICAL_V1_RUN_SCRATCH_SUPERSEDED |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719/hybrid_dspsa32_coordinate12/checkpoint.json` | untracked | 13820 | `741fc6c933d92bee542537bae81cd3d45c828d35840ee8b7c6dface73fce64f6` | SUPERSEDED | CERTIFIED_HISTORICAL_V1_RUN_SCRATCH_SUPERSEDED |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719/hybrid_dspsa32_coordinate12/receipt.json` | untracked | 14647 | `6b7a2b28be481e7bf768e035e6f0ba715b16e351a5c5941580145b3b6ea05a25` | SUPERSEDED | CERTIFIED_HISTORICAL_V1_RUN_SCRATCH_SUPERSEDED |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719/validate_global_final/checkpoint.json` | untracked | 12880 | `9f47357f1d82ee42a333b2627d13ec10c45bf6aa9f499a3913694c57219bb08c` | SUPERSEDED | CERTIFIED_HISTORICAL_V1_RUN_SCRATCH_SUPERSEDED |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719/validate_global_final/receipt.json` | untracked | 15979 | `487272b485c5b2d966456052e68edf83a7332fd7a716f9bf75bd7b77284aa5c2` | SUPERSEDED | CERTIFIED_HISTORICAL_V1_RUN_SCRATCH_SUPERSEDED |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719/validate_hybrid_final/checkpoint.json` | untracked | 12888 | `79d48d36bbbfbbe0f1f4cc0eeb86f6cea0b4588e7c62a2b5ee2f43d7495cc484` | SUPERSEDED | CERTIFIED_HISTORICAL_V1_RUN_SCRATCH_SUPERSEDED |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719/validate_hybrid_final/receipt.json` | untracked | 15993 | `1726eab500f20c4cb0a43b799b885381c3a1852f74e7001b561bd08bb2a3dc61` | SUPERSEDED | CERTIFIED_HISTORICAL_V1_RUN_SCRATCH_SUPERSEDED |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719/validate_source_final/checkpoint.json` | untracked | 12788 | `35656c6853b227a511c8220be7c50ec5d01e80720bad756f6b0971f30e0b9cde` | SUPERSEDED | CERTIFIED_HISTORICAL_V1_RUN_SCRATCH_SUPERSEDED |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719/validate_source_final/receipt.json` | untracked | 15887 | `1cba64be5708072cf1e7300279b717f5bae73a1790c1114bce8b780f613a48c6` | SUPERSEDED | CERTIFIED_HISTORICAL_V1_RUN_SCRATCH_SUPERSEDED |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719/zero/checkpoint.json` | untracked | 12421 | `5660a667bf0a384fe87055962073264dbdd356472899004eaedee9846ea71709` | SUPERSEDED | CERTIFIED_HISTORICAL_V1_RUN_SCRATCH_SUPERSEDED |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719/zero/receipt.json` | untracked | 13096 | `2f3aac1ff732505d4bbb41bf1f442c7698d01203f29f242e2b4c4ea7101ada87` | SUPERSEDED | CERTIFIED_HISTORICAL_V1_RUN_SCRATCH_SUPERSEDED |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719_pre_review_invalid/baseline/checkpoint.json` | untracked | 11018 | `aa065e9c945c34f6444ba490c183fa732eefe680478859133d685643ad38cc62` | SCRATCH | CERTIFIED_PRE_REVIEW_INVALID_SCRATCH |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719_pre_review_invalid/baseline/receipt.json` | untracked | 13177 | `a4eea7b8599787120162b1ef84044d559e2d1a50458003755b4ff6373955d22f` | SCRATCH | CERTIFIED_PRE_REVIEW_INVALID_SCRATCH |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719_pre_review_invalid/global/checkpoint.json` | untracked | 11022 | `5dd296f8d4e18922448d89e2b425160a0a049d1bb66fea2b59f0aab3d6bd3493` | SCRATCH | CERTIFIED_PRE_REVIEW_INVALID_SCRATCH |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719_pre_review_invalid/global/receipt.json` | untracked | 13127 | `97d9de9f5b9ae4893e2257e4eb5fa3645b4a23c7cb157d242b664d687f57ca89` | SCRATCH | CERTIFIED_PRE_REVIEW_INVALID_SCRATCH |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719_pre_review_invalid/zero/checkpoint.json` | untracked | 10672 | `438ee805405c27c1cda2c36afd2621ec413a5a8e7af41fa9f7c31351723d216d` | SCRATCH | CERTIFIED_PRE_REVIEW_INVALID_SCRATCH |
| `.omx/research/einstein_kolmogorov_crux_runs_20260719_pre_review_invalid/zero/receipt.json` | untracked | 13059 | `97633201c88692b583c6613fbe3a7b6122c55df22fa81ecbf26f696d046e45ea` | SCRATCH | CERTIFIED_PRE_REVIEW_INVALID_SCRATCH |
| `.omx/research/einstein_kolmogorov_xi_bridge_blocker_20260719.json` | untracked | 2166 | `cb59f52aface149c94dc59bbfa05bed640dff9b9c2d099efbd9794792d6403c8` | SUPERSEDED | CERTIFIED_XI_BRIDGE_BLOCKER_SUPERSEDED |
| `.omx/research/einstein_kolmogorov_xi_bridge_full_preflight_blocker_20260719.json` | untracked | 2723 | `0a851a72ad6788cfc1d76fefade2e2db5428bcbfcde5e2f349df28cd95cfab96` | SUPERSEDED | CERTIFIED_XI_BRIDGE_V2_BLOCKER_SUPERSEDED |
| `.omx/research/einstein_kolmogorov_xi_bridge_hash_receipt_20260719.json` | untracked | 4624 | `855ef64cab70865f97744bd1105c068ed3c84c281c194be650226dac5848ac8f` | SUPERSEDED | CERTIFIED_XI_BRIDGE_V1_SUPERSEDED |

## Verification

- Compile check passed: `.venv/bin/python -m py_compile src/tac/preflight.py src/tac/canonical_equations/registry.py src/tac/tests/test_check_154_manifestless_cleanup_identity_scope_extension.py src/tac/tests/test_check_344_anchor_roundtrip_scope_extension.py src/tac/tests/test_check_351_canonical_producer_identity_scope_extension.py`.
- Whitespace/conflict check passed before test: `git diff --check` on the attempted merge set.
- Focused tests failed after attempted harvest: `.venv/bin/python -m pytest src/tac/tests/test_check_154_manifestless_cleanup_identity_scope_extension.py src/tac/tests/test_check_344_anchor_roundtrip_scope_extension.py src/tac/tests/test_check_351_canonical_producer_identity_scope_extension.py` => 121 collected, 114 passed, 7 failed. Failure class: expanded Catalog #351 strict scanner is red against current main, including 188 live current-main violations.
- Attempted merge was backed out; the only durable new files are this receipt and the JSON manifest.

## NEXT_IF_RESUMED

1. Do not whole-file copy the residue preflight bundle. Rebase or split the guard family against current main.
2. Treat Catalog #351 expanded recursive scanning as blocked until current-main violations are either fixed, substantively waived, or the scope is narrowed with a positive-control proof. No strict flip while live count is 188.
3. Catalog #154/#344 parts passed in the attempted bundle; they can be split into a smaller landing if isolated from the red #351 gate.
4. Before any code landing: run the three focused pytest files, run `git diff --check`, run two `tools/review_tracker.py mark-file ... --status reviewed` passes for every changed `.py`, and commit only through `tools/subagent_commit_serializer.py` with post-edit `--expected-content-sha256`.

No scorer job, no `evaluate.py`, no archive claim, no deletion, no pointer movement.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
