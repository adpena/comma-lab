# Codex Session Summary - V8 Local Implementation

**UTC:** 2026-05-20T03:44:13Z  
**Owner:** codex  
**Score claim:** none  
**Promotion eligible:** false

## Completed

- Integrated the latest V8 design memo into code instead of leaving the lane as a dead scaffold.
- Preserved Worker A's fixed-header runtime archive contract and Worker B's torch-backed learned-export smoke, then connected both through the V8 trainer.
- Added tests for archive parsing/inflate, malformed-input fail-closed behavior, file-list path safety, non-promotional custody flags, full-local trainer mode, categorical codewords, scale-hyperprior metadata, and recipe refusal state.
- Hardened the macOS Faiss/Torch OpenMP collision by setting `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1` before the torch-backed V8 smoke module imports.
- Added file-level Tier-1 research-only waivers and recipe Tier-2 hardware-routing metadata so local pre-deploy now refuses on the real blockers only: missing auth-eval path and disabled recipe/promotion evidence.

## Current Authority Boundary

- Local implementation and focused Faiss/probe/V8 tests: green.
- Provider dispatch: blocked by recipe and missing auth-eval path.
- Score claim: blocked.
- Promotion: blocked until real contest-video scorer training, exact CUDA auth eval, and Catalog #324 Tier-C validation exist.

## Recommended Next Step

If V8 remains priority, move from local fixture export to a claimed, operator-authorized scorer-training smoke. Otherwise, do not keep adding V8 scaffold polish; pivot to the next exact-evaluable frontier artifact.
