# ddm_fm2 Next If Resumed

1. Fire order: apply `fmtools_patches/0001-fmtools-full-sdk-surfaces.patch` from a Git-writable shell in `/Users/adpena/Projects/fmtools`, then run the fmtools test and Ruff commands in `fmtools_patches/APPLY.md`.
2. Fire order: refresh the external fmtools venv to `apple-fm-sdk==0.2.1` once shell DNS/PyPI access is available. Then rerun the live plain and structured generation probes; only mark the on-device leg VERIFIED if an actual response object/string returns.
3. If generation still fails under `0.2.1`, classify the blocker as host Apple Foundation Models runtime/model-manager, preserving the exact `GenerationError status 255` / `ModelManagerServices.ModelManagerError Code=1008` evidence.
4. Keep all Pact fmtools consumers advisory-only. Do not let FM labels rescue/refuse queue lint, control actuation, scorer routing, exact-row selection, or frontier claims.
5. The next Pact consumer to consider is a WARN-only `ty1`/negative-language audit reader of `mechanism_reduction_language`, using the same fail-open shape as `tools/codex_arm_queue.py`.
6. Remove or cold-store `/Volumes/VertigoDataTier/pact/ddm_fm2_20260806/fmtools_patch_verify` only after recording the patch SHA, base commit, test commands, and clone size; it is retained now as reproducible patch evidence.
