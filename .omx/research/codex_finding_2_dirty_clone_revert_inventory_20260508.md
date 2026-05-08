# Codex Finding 2 — Dirty Public PR Intake Clone Revert Inventory

**Date:** 2026-05-08
**Trigger:** Codex adversarial review HIGH finding — public PR intake clones contain in-place `KL_BATCHMEAN_OK` waivers that corrupt source provenance.
**Resolution:** Inline waivers are STALE. The KL_BATCHMEAN scanner's `_VENDORED_PATH_MARKERS` (`src/tac/preflight.py:12032-12041`) already excludes `_intake_` paths; running the scanner on current state produces 0 violations. The waivers were added before the exclusion landed and are no longer demanded. Reverting all clones to pristine upstream state.

## Reverted edits (all comment-only `KL_BATCHMEAN_OK` waivers on `F.kl_div(..., reduction="batchmean")`)

| Clone | File | Line | Waiver content |
|---|---|---:|---|
| public_pr100_intake_20260504_codex/source | submissions/fp4_mask_gen/compress.py | ? | `    return F.kl_div(log_p, q, reduction="batchmean") * (temperature**2)  # KL_BATCHMEAN_OK:public-PR-intake-external-cod` |
| public_pr100_intake_20260504_codex/source | submissions/neural_inflate/train_ren.py | ? | `    loss_seg = F.kl_div(  # KL_BATCHMEAN_OK:public-PR-intake-external-code` |
| public_pr100_intake_20260504_codex/source | submissions/ph4ntom_drv/compress.py | ? | `    return F.kl_div(log_p, q, reduction="batchmean") * (temperature ** 2)  # KL_BATCHMEAN_OK:public-PR-intake-external-c` |
| public_pr100_intake_20260504_codex/source | submissions/quantizr/compress.py | ? | `    return F.kl_div(log_p, q, reduction="batchmean") * (temperature ** 2)  # KL_BATCHMEAN_OK:public-PR-intake-external-c` |
| public_pr100_intake_20260504_codex/source | submissions/svtav1_dilated_ren/svtav1_dilated_ren_training.py | ? | `        bs += F.kl_div(F.log_softmax(s,dim=1), gt_seg[i].to(DEVICE), reduction='batchmean').item()  # KL_BATCHMEAN_OK:pu` |
| public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source | submissions/fp4_mask_gen/compress.py | ? | `    return F.kl_div(log_p, q, reduction="batchmean") * (temperature**2)  # KL_BATCHMEAN_OK:public-PR-intake-external-cod` |
| public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source | submissions/neural_inflate/train_ren.py | ? | `    loss_seg = F.kl_div(  # KL_BATCHMEAN_OK:public-PR-intake-external-code` |
| public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source | submissions/ph4ntom_drv/compress.py | ? | `    return F.kl_div(log_p, q, reduction="batchmean") * (temperature ** 2)  # KL_BATCHMEAN_OK:public-PR-intake-external-c` |
| public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source | submissions/quantizr/compress.py | ? | `    return F.kl_div(log_p, q, reduction="batchmean") * (temperature ** 2)  # KL_BATCHMEAN_OK:public-PR-intake-external-c` |
| public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source | submissions/svtav1_dilated_ren/svtav1_dilated_ren_training.py | ? | `        bs += F.kl_div(F.log_softmax(s,dim=1), gt_seg[i].to(DEVICE), reduction='batchmean').item()  # KL_BATCHMEAN_OK:pu` |
| public_pr103_intake_20260504_codex/source | submissions/fp4_mask_gen/compress.py | ? | `    return F.kl_div(log_p, q, reduction="batchmean") * (temperature**2)  # KL_BATCHMEAN_OK:public-PR-intake-external-cod` |
| public_pr103_intake_20260504_codex/source | submissions/neural_inflate/train_ren.py | ? | `    loss_seg = F.kl_div(  # KL_BATCHMEAN_OK:public-PR-intake-external-code` |
| public_pr103_intake_20260504_codex/source | submissions/ph4ntom_drv/compress.py | ? | `    return F.kl_div(log_p, q, reduction="batchmean") * (temperature ** 2)  # KL_BATCHMEAN_OK:public-PR-intake-external-c` |
| public_pr103_intake_20260504_codex/source | submissions/quantizr/compress.py | ? | `    return F.kl_div(log_p, q, reduction="batchmean") * (temperature ** 2)  # KL_BATCHMEAN_OK:public-PR-intake-external-c` |
| public_pr103_intake_20260504_codex/source | submissions/svtav1_dilated_ren/svtav1_dilated_ren_training.py | ? | `        bs += F.kl_div(F.log_softmax(s,dim=1), gt_seg[i].to(DEVICE), reduction='batchmean').item()  # KL_BATCHMEAN_OK:pu` |
| public_pr105_kitchen_sink_intake_20260504_codex/source | submissions/fp4_mask_gen/compress.py | ? | `    return F.kl_div(log_p, q, reduction="batchmean") * (temperature**2)  # KL_BATCHMEAN_OK:public-PR-intake-external-cod` |
| public_pr105_kitchen_sink_intake_20260504_codex/source | submissions/neural_inflate/train_ren.py | ? | `    loss_seg = F.kl_div(  # KL_BATCHMEAN_OK:public-PR-intake-external-code` |
| public_pr105_kitchen_sink_intake_20260504_codex/source | submissions/ph4ntom_drv/compress.py | ? | `    return F.kl_div(log_p, q, reduction="batchmean") * (temperature ** 2)  # KL_BATCHMEAN_OK:public-PR-intake-external-c` |
| public_pr105_kitchen_sink_intake_20260504_codex/source | submissions/quantizr/compress.py | ? | `    return F.kl_div(log_p, q, reduction="batchmean") * (temperature ** 2)  # KL_BATCHMEAN_OK:public-PR-intake-external-c` |
| public_pr105_kitchen_sink_intake_20260504_codex/source | submissions/svtav1_dilated_ren/svtav1_dilated_ren_training.py | ? | `        bs += F.kl_div(F.log_softmax(s,dim=1), gt_seg[i].to(DEVICE), reduction='batchmean').item()  # KL_BATCHMEAN_OK:pu` |
| public_pr106_belt_and_suspenders_intake_20260504_codex/source | submissions/fp4_mask_gen/compress.py | ? | `    return F.kl_div(log_p, q, reduction="batchmean") * (temperature**2)  # KL_BATCHMEAN_OK:public-PR-intake-external-cod` |
| public_pr106_belt_and_suspenders_intake_20260504_codex/source | submissions/neural_inflate/train_ren.py | ? | `    loss_seg = F.kl_div(  # KL_BATCHMEAN_OK:public-PR-intake-external-code` |
| public_pr106_belt_and_suspenders_intake_20260504_codex/source | submissions/ph4ntom_drv/compress.py | ? | `    return F.kl_div(log_p, q, reduction="batchmean") * (temperature ** 2)  # KL_BATCHMEAN_OK:public-PR-intake-external-c` |
| public_pr106_belt_and_suspenders_intake_20260504_codex/source | submissions/quantizr/compress.py | ? | `    return F.kl_div(log_p, q, reduction="batchmean") * (temperature ** 2)  # KL_BATCHMEAN_OK:public-PR-intake-external-c` |
| public_pr106_belt_and_suspenders_intake_20260504_codex/source | submissions/svtav1_dilated_ren/svtav1_dilated_ren_training.py | ? | `        bs += F.kl_div(F.log_softmax(s,dim=1), gt_seg[i].to(DEVICE), reduction='batchmean').item()  # KL_BATCHMEAN_OK:pu` |
| public_pr81_qzs3_range_mask_intake_20260503_codex/repo | submissions/fp4_mask_gen/compress.py | ? | `    return F.kl_div(log_p, q, reduction="batchmean") * (temperature**2)  # KL_BATCHMEAN_OK:public-PR-intake-external-cod` |
| public_pr81_qzs3_range_mask_intake_20260503_codex/repo | submissions/neural_inflate/train_ren.py | ? | `    loss_seg = F.kl_div(  # KL_BATCHMEAN_OK:public-PR-intake-external-code` |
| public_pr81_qzs3_range_mask_intake_20260503_codex/repo | submissions/ph4ntom_drv/compress.py | ? | `    return F.kl_div(log_p, q, reduction="batchmean") * (temperature ** 2)  # KL_BATCHMEAN_OK:public-PR-intake-external-c` |
| public_pr81_qzs3_range_mask_intake_20260503_codex/repo | submissions/quantizr/compress.py | ? | `    return F.kl_div(log_p, q, reduction="batchmean") * (temperature ** 2)  # KL_BATCHMEAN_OK:public-PR-intake-external-c` |
| public_pr81_qzs3_range_mask_intake_20260503_codex/repo | submissions/svtav1_dilated_ren/svtav1_dilated_ren_training.py | ? | `        bs += F.kl_div(F.log_softmax(s,dim=1), gt_seg[i].to(DEVICE), reduction='batchmean').item()  # KL_BATCHMEAN_OK:pu` |
| public_pr82_henosis_frontier_intake_20260503_codex/repo | submissions/fp4_mask_gen/compress.py | ? | `    return F.kl_div(log_p, q, reduction="batchmean") * (temperature**2)  # KL_BATCHMEAN_OK:public-PR-intake-external-cod` |
| public_pr82_henosis_frontier_intake_20260503_codex/repo | submissions/neural_inflate/train_ren.py | ? | `    loss_seg = F.kl_div(  # KL_BATCHMEAN_OK:public-PR-intake-external-code` |
| public_pr82_henosis_frontier_intake_20260503_codex/repo | submissions/quantizr/compress.py | ? | `    return F.kl_div(log_p, q, reduction="batchmean") * (temperature ** 2)  # KL_BATCHMEAN_OK:public-PR-intake-external-c` |
| public_pr82_henosis_frontier_intake_20260503_codex/repo | submissions/svtav1_dilated_ren/svtav1_dilated_ren_training.py | ? | `        bs += F.kl_div(F.log_softmax(s,dim=1), gt_seg[i].to(DEVICE), reduction='batchmean').item()  # KL_BATCHMEAN_OK:pu` |
| public_pr91_intake_20260504_worker/pr91_src/repo | submissions/fp4_mask_gen/compress.py | ? | `    return F.kl_div(log_p, q, reduction="batchmean") * (temperature**2)  # KL_BATCHMEAN_OK:public-PR-intake-external-cod` |
| public_pr91_intake_20260504_worker/pr91_src/repo | submissions/neural_inflate/train_ren.py | ? | `    loss_seg = F.kl_div(  # KL_BATCHMEAN_OK:public-PR-intake-external-code` |
| public_pr91_intake_20260504_worker/pr91_src/repo | submissions/quantizr/compress.py | ? | `    return F.kl_div(log_p, q, reduction="batchmean") * (temperature ** 2)  # KL_BATCHMEAN_OK:public-PR-intake-external-c` |
| public_pr91_intake_20260504_worker/pr91_src/repo | submissions/svtav1_dilated_ren/svtav1_dilated_ren_training.py | ? | `        bs += F.kl_div(F.log_softmax(s,dim=1), gt_seg[i].to(DEVICE), reduction='batchmean').item()  # KL_BATCHMEAN_OK:pu` |

## Why these waivers were stale

`src/tac/preflight.py::_scan_python_for_kl_div_batchmean` already excludes any path containing `_intake_` substring (line 12039). Running `check_kl_div_reduction_correct(strict=False)` on the current repo produces **0 violations**. The waivers were added before the path-exclusion fix landed and have been dead code since.

## Replacement mechanism

A waiver manifest at `reverse_engineering/public_pr_waiver_manifest.json` provides a structured location for future cross-clone waivers (if a scanner were ever to demand one). The manifest is consulted by `_consult_waiver_manifest()` in `src/tac/preflight.py` as a future-proofing hook. Current state: empty `waivers: []` array — the path-exclusion logic in each scanner is the actual mechanism.

## STRICT preflight gate

`check_public_pr_intake_clones_pristine` (Check 109) wired into `preflight_all()`. Discovers all clones under `experiments/results/public_pr*_intake_*/{source,repo,pr*_src}/` and refuses any clone with non-empty `git status --short`. Forbidden-pattern entry added to CLAUDE.md.
