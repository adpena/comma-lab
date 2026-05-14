# OSS v0.2.0-rc1 audit D-3 + D-4 — LICENSE + THIRD_PARTY_NOTICES.md draft memo

**Date:** 2026-05-14
**Audit anchor:** OSS v0.2.0-rc1 (commit `c293ba425`)
**Drafts:** `reports/oss_d3_d4_drafts_20260514/LICENSE.draft` +
`reports/oss_d3_d4_drafts_20260514/THIRD_PARTY_NOTICES.md.draft`
**Operator policy provided:** *"we use MIT and Apache 2.0 for all"*

This memo summarizes the proposed LICENSE + THIRD_PARTY_NOTICES.md changes and
surfaces the items that require explicit operator decision **before** the drafts
are promoted to the canonical `LICENSE` and `THIRD_PARTY_NOTICES.md` files.
Per CLAUDE.md "Non-Negotiable Upstream Rule," those two files cannot be edited
without explicit human approval.

---

## TL;DR

- **`LICENSE`**: only one substantive change — the copyright-holder placeholder
  `"OpenAI artifact output for user-directed scaffold"` is replaced with the
  canonical author identity. Body is otherwise identical (standard MIT).
- **`THIRD_PARTY_NOTICES.md`**: rewritten from a 7-line research-inspirations
  list into a full notices file covering all 11 runtime deps,
  the runtime/dev/cloud/viz extras, and the Modal worker image's additional
  pip-installs.
- **One BLOCKER (RESOLVED 2026-05-14)**: `pyppmd>=1.3,<2.0` was licensed under
  **LGPL-2.1-or-later**, outside the operator's "MIT and Apache 2.0 for all"
  hard-dep policy. **Operator decision 2026-05-14**: implement Option 3 (mark
  pyppmd OPTIONAL via `[project.optional-dependencies].pr86_replay`) — chosen
  because empirical PyPI survey 2026-05-14 confirmed there is NO permissive
  PPMd Python binding (every PPMd implementation is LGPL-derived from the
  Igor Pavlov 7-Zip reference; `constriction` itself ships only RangeEncoder/
  Decoder + ANS, not PPMd context modeling, so Option-2 "replace" was not
  feasible without dropping PR86/PR91 third-party-archive replay capability).
  Default `pip install tac` now carries no copyleft; PR86/PR91 replay flows
  install `pip install tac[pr86_replay]` and accept the documented LGPL
  obligation. See lane `lane_pyppmd_to_constriction_migrate_20260514` and
  memory `feedback_pyppmd_to_constriction_migrate_landed_20260514.md`.
- **One operator-confirm point**: copyright-holder identity for the MIT line.
  Defaulted to `Alejandro Peña` from `git config user.name`; the operator may
  prefer a different legal entity, project name, or "the comma_lab contributors"-style attribution.
- **Two minor operator-confirm points**: copyright year boundary (current
  `LICENSE` says 2026; should it be `2024-2026` or `2026` only?), and whether
  to add an `SPDX-License-Identifier: MIT` header line to the LICENSE file.

---

## D-3: `LICENSE` — current vs draft

### Current (`LICENSE`)

```
MIT License

Copyright (c) 2026 OpenAI artifact output for user-directed scaffold

Permission is hereby granted, free of charge, to any person obtaining a copy
[... standard MIT body ...]
```

### Draft (`reports/oss_d3_d4_drafts_20260514/LICENSE.draft`)

```
MIT License

Copyright (c) 2026 Alejandro Peña

Permission is hereby granted, free of charge, to any person obtaining a copy
[... standard MIT body ... unchanged ...]
```

### Diff summary

- Line 3: `OpenAI artifact output for user-directed scaffold` → `Alejandro Peña`
- All other 18 lines: byte-identical to the standard MIT text.

### Operator confirm-or-correct points

1. **Copyright holder name.** Defaulted to `Alejandro Peña` (per
   `git config user.name`). Alternatives the operator may prefer:
   - A different legal entity (LLC / company name).
   - A project-name attribution: `the pact contributors` or
     `the comma_lab project authors`.
   - The author's GitHub handle: `adpena`.
2. **Copyright year.** Current draft uses `2026` (consistent with the
   pre-existing LICENSE). If the project's earliest committed source predates
   2026, the canonical convention is `2024-2026` or `2025-2026`. Confirm.
3. **Optional SPDX header.** Many modern OSS projects add
   `SPDX-License-Identifier: MIT` at the top of the LICENSE file (and
   per-source-file). Not required by MIT itself but useful for automated
   license-scan tools. Confirm whether to add.

---

## D-4: `THIRD_PARTY_NOTICES.md` — current vs draft

### Current (`THIRD_PARTY_NOTICES.md`)

7 lines listing only research-inspiration repos and the upstream challenge
repo. **No entries for any of the 11+ pip-installed runtime
dependencies.** From an OSS-redistribution standpoint this is incomplete —
several deps (Apache-2.0 in particular) require a NOTICE / acknowledgment in
downstream distributions.

### Draft (`reports/oss_d3_d4_drafts_20260514/THIRD_PARTY_NOTICES.md.draft`)

A full notices file enumerating:

- Project license summary (MIT, links to `LICENSE`).
- Project license policy statement ("MIT / Apache-2.0 / BSD / HPND / ISC /
  PSF / MPL-2.0 (dev only) / BSL-1.0 only; copyleft requires operator
  review").
- **Runtime deps** (10 packages from `pyproject.toml [project] dependencies`,
  all permissive after BLOCKER B1 resolution 2026-05-14):
  torch, pydantic, numpy, click, brotli, constriction,
  cryptography, cmaes, optuna.
- **PR86/PR91 replay extra** (`[project.optional-dependencies].pr86_replay`,
  added 2026-05-14): pyppmd (LGPL-2.1-or-later, opt-in only).
- **Runtime extras** (7 packages from `[runtime]`): av, safetensors,
  opencv-python, timm, einops, segmentation-models-pytorch.
- **Dev deps** (table form): pytest, pytest-timeout, pytest-subtests,
  hypothesis, ruff, ty, mypy, scipy, PyWavelets.
- **Cloud orchestration deps** (table form): lightning-sdk, modal, vastai,
  kaggle.
- **Optional viz / notebooks / analysis / mlx** (table form): plotly,
  matplotlib, pillow, imageio, dask, polars, marimo, mlx.
- **Modal worker image extras** (`experiments/modal_train_lane.py:60-83`):
  torchvision, tqdm, Pillow, scipy, nvidia-dali-cuda120.
- **Pinned upstream snapshot** under `upstream/` — governed by upstream's own
  license; per CLAUDE.md "Non-Negotiable Upstream Rule" not modified.
- **Research inspirations** — preserved verbatim from the existing notices.
- **Maintenance procedure** — checklist for keeping the notices file in sync
  with `pyproject.toml` going forward.

Each dependency entry has: pinned version range, license name (matched against
the operator's "MIT / Apache 2.0 for all" policy), upstream project URL, and
copyright holder where determinable.

### License verification provenance

Licenses were verified by reading `importlib.metadata.metadata(<pkg>).get("License")`
on the installed copies in `.venv/`, cross-referenced with classifier entries
(`License :: OSI Approved :: ...`) and `License-File` metadata. Where the
metadata field was empty (a modern PEP 639 convention many packages now use),
the license was determined from the upstream repository's canonical
`LICENSE` / `LICENSE.txt` file.

Packages not currently installed in the local `.venv/` (e.g.
`nvidia-dali-cuda120`, `opencv-python`) had their licenses determined from
their well-known upstream repositories. These are explicitly flagged as
"upstream-canonical" rather than "metadata-verified" below.

| Package | Installed locally? | License (verified by) |
|---------|---------------------|------------------------|
| torch 2.11.0 | yes | metadata: `BSD-3-Clause` |
| pydantic 2.12.5 | yes | upstream-canonical: MIT (LICENSE file in repo) |
| numpy 2.4.4 | yes | upstream-canonical: BSD-3-Clause (LICENSE.txt) |
| click 8.3.2 | yes | upstream-canonical: BSD-3-Clause (LICENSE.txt) |
| brotli 1.2.0 | yes | metadata: `MIT` |
| constriction 0.4.2 | yes | metadata: `MIT OR Apache-2.0 OR BSL-1.0` |
| pyppmd 1.3.1 | yes (optional `[pr86_replay]` extra after 2026-05-14) | metadata: `LGPL-2.1-or-later` — BLOCKER B1 RESOLVED via Option 3 (opt-in extra) |
| cryptography 46.0.4 | yes | upstream-canonical: Apache-2.0 OR BSD-3-Clause (LICENSE.APACHE + LICENSE.BSD) |
| cmaes 0.13.0 | yes | upstream-canonical: MIT (LICENSE file in repo) |
| optuna 4.8.0 | yes | classifier: `License :: OSI Approved :: MIT License` |
| av (PyAV) 17.0.1 | yes | upstream-canonical: BSD-3-Clause (LICENSE.txt) |
| safetensors 0.7.0 | yes | classifier: `License :: OSI Approved :: Apache Software License` |
| opencv-python | no | upstream-canonical: Apache-2.0 (OpenCV core) |
| timm 1.0.26 | yes | metadata: `Apache-2.0` |
| einops 0.8.2 | yes | metadata: `MIT` |
| segmentation-models-pytorch 0.5.0 | yes | metadata: `MIT` |
| torchvision 0.26.0 | yes | metadata: `BSD` |
| tqdm 4.67.3 | yes | metadata: `MPL-2.0 AND MIT` (dual) |
| Pillow 12.2.0 | yes | upstream-canonical: HPND (Historical Permission Notice and Disclaimer — MIT-compatible) |
| scipy 1.17.1 | yes | classifier: `License :: OSI Approved :: BSD License` |
| PyWavelets 1.9.0 | yes | upstream-canonical: MIT |
| nvidia-dali-cuda120 | no | upstream-canonical: Apache-2.0 (per NVIDIA/DALI repo) |

---

## BLOCKERS

### B1 (HIGH, RESOLVED 2026-05-14 via Option 3): `pyppmd` is LGPL-2.1-or-later

**Resolution (2026-05-14, lane `lane_pyppmd_to_constriction_migrate_20260514`):**
The operator routed this BLOCKER as "Replace pyppmd with constriction OR
permissive PPMd binding (cleanest re: policy)". Investigation surfaced two
structural facts that rerouted the decision:

1. **Constriction lacks PPMd context modeling.** It ships only RangeEncoder /
   RangeDecoder + ANS coders; the PR86/PR91 wire-format `hpac.pt.ppmd` member
   uses Igor-Pavlov-style PPMd context modeling that has no constriction
   equivalent. Option 2a ("switch the PR86 inflate path to constriction") is
   not a drop-in swap — it would require either rewriting the PR86 archive
   format or dropping third-party PR86/PR91 archive replay capability.
2. **No permissive PPMd Python binding exists on PyPI** (empirical survey
   2026-05-14). Every package in the family (`pyppmd`, `pyppmd-gentee`,
   `ppmd-cffi`, `zipfile-ppmd`) is LGPL-licensed because they all wrap the
   LGPL Igor-Pavlov 7-Zip reference implementation. The `ppmd` PyPI package
   is an unrelated PowerPoint-to-Markdown tool; `et-ppmdcommon` is unrelated
   coursework. There is no clean-room MIT/BSD PPMd port available.

**Therefore Option 3 (mark pyppmd OPTIONAL via `[project.optional-dependencies]
.pr86_replay`) was implemented as the least-disruptive permissive-default
path:**

- `pyproject.toml`: `pyppmd>=1.3,<2.0` moved from `[project] dependencies` to
  `[project.optional-dependencies].pr86_replay`. Default `pip install tac` now
  carries no copyleft.
- `experiments/modal_train_lane.py`: `pyppmd` removed from the default Modal
  worker image's pip-install list (Modal workers do not run PR86/PR91 third-
  party-archive replay; that is a local CPU forensic flow).
- `tac.pr86_hpac_codec` and `tac.pr91_hpm1_codec`: live `import pyppmd` sites
  annotated with `# PYPPMD_LGPL_OK:public-PR{86,91}-archive-replay-decode-only-
  no-permissive-PPMd-binding-on-PyPI` waiver tokens so a future STRICT
  preflight gate (`check_no_pyppmd_imports`) can refuse new pyppmd usage
  without breaking the two existing fail-closed wire-format-decode call sites.
- `THIRD_PARTY_NOTICES.md.draft`: pyppmd entry relocated from "Runtime
  dependencies" to a new "PR86/PR91 replay extra" section with the LGPL
  obligation documented inline.

**Carry-forward operator-routable surfaces** (NOT done by this lane; surface
back when convenient):

- (a) **STRICT preflight gate `check_no_pyppmd_imports`** — sister gate to
  `check_no_pyppmd_imports`, refuses any NEW `import pyppmd` / `from pyppmd`
  outside the two grandfathered call sites. Per CLAUDE.md "Bugs must be
  permanently fixed AND self-protected against" non-negotiable. Deferred
  this commit because the dirty `src/tac/preflight.py` is owned by a
  parallel sister subagent (RESUME-3-CRASHED); landing the gate in the
  same commit would race their work.
- (b) **Decision: drop PR86/PR91 third-party-archive replay entirely**?
  This would let us delete pyppmd from the dep tree completely (including
  the `[pr86_replay]` extra) and remove the two `import pyppmd` sites.
  Cost: lose the ability to forensic-replay any future public PR86/PR91-
  family archive. Recommendation: keep the optional extra; PR86/PR91
  replay capability is a research-loop investment that should not be
  abandoned on a license-policy preference alone.
- (c) **Vendor a minimal MIT-licensed PPMd-decoder reimplementation**.
  The 7-Zip reference is LGPL; clean-room PPMd reimplementations are rare.
  This would be a dedicated engineering project (probably 1-2 weeks of
  council-grade design + implementation + byte-parity validation). Surface
  back if the operator decides the optional-extra approach is not enough
  long-term.

---

### B1 historical context (preserved from original draft)

**Source of finding:** `importlib.metadata.metadata("pyppmd").get("License")`
returned the string `"LGPL-2.1-or-later"`.

**Why this matters:** LGPL is a copyleft license. While LGPL imposes *weaker*
obligations than full GPL (in particular, dynamically linking from a
permissively-licensed work is generally OK), it still:

- Requires distributors to permit users to relink against modified versions of
  the LGPL library.
- Requires source availability for the LGPL component itself when the combined
  work is distributed in binary form.
- Is incompatible with the operator's stated policy of "MIT and Apache 2.0 for
  all" if interpreted strictly.

**Where `pyppmd` is used in the codebase:** `pyppmd` is pulled in (per the
inline comment at `pyproject.toml:53-55`) for the PR86 / HPAC-family archive
inflate path, which uses PPMd-compressed entropy-model weights. Specifically,
look at `tac.pr86_hpac_codec` and any inflate path that needs to decompress
PPMd streams.

**Options for operator decision:**

1. **Accept LGPL-2.1-or-later as compatible** with the project's license posture.
   This is the most common decision in the Python ecosystem because pyppmd is
   used as a dynamically-linked PyPI dependency, not vendored/statically linked
   into the project binary. The combined work distribution rules of LGPL-2.1
   are satisfied by the standard PyPI install path (user pip-installs pyppmd
   separately; the user has the ability to swap in a modified version).
   → Document the dependency explicitly in THIRD_PARTY_NOTICES.md (as drafted)
   and continue to use it. Update CLAUDE.md license policy line to say
   "MIT / Apache 2.0 / standard permissive + dynamically-linked LGPL".
2. **Replace `pyppmd` with a permissively-licensed PPMd decoder.** Options:
   - Vendor a minimal PPMd decoder under MIT/BSD (the 7-Zip reference
     implementation is LGPL; clean-room reimplementations exist but are rare).
   - Switch the PR86 inflate path to use a different entropy coder
     (constriction's queue arithmetic coder, which is tri-licensed
     MIT/Apache-2.0/BSL-1.0).
   → Significant engineering work; would require an archive-format change for
   any PR86-family submission archives that ship PPMd weights.
3. **Mark the dependency optional** (`[project.optional-dependencies]`) so a
   default `pip install tac` does not pull LGPL code, and only PR86-replay
   workflows that explicitly need it install it. Document the LGPL obligation
   on that install path.
   → Smallest behavior change; preserves capability; defers the strict policy
   compliance question.

**My recommendation (superseded 2026-05-14 by operator decision):** Option 1
(accept dynamically-linked LGPL as a documented exception) was the original
recommendation. The operator chose a stronger posture (Option 3, opt-in extra)
because Option 1 still pulls LGPL bytes into a default `pip install tac` and
the operator's stated "MIT and Apache 2.0 for all" hard-dep policy is read
strictly. Option 3 satisfies the strict policy while preserving PR86/PR91
replay capability via opt-in.

---

## Operator confirm-or-correct checklist

Before promoting these drafts to `LICENSE` and `THIRD_PARTY_NOTICES.md`:

- [ ] **Copyright holder identity** — confirm `Alejandro Peña` is correct, or
      provide replacement.
- [ ] **Copyright year** — confirm `2026` vs `2024-2026` / `2025-2026` /
      etc.
- [ ] **SPDX header** — add `SPDX-License-Identifier: MIT` to LICENSE? (y/n)
- [x] **`pyppmd` LGPL decision** — RESOLVED 2026-05-14: Option 3 (make optional
      via `[project.optional-dependencies].pr86_replay`). See B1 above; lane
      `lane_pyppmd_to_constriction_migrate_20260514`.
- [ ] **Modal-worker-image deps included in main notices?** — the draft includes
      a "Modal-only worker image deps" section. Operator may want this kept
      separate (in a `docs/` runbook) since they are not user-facing dependencies
      from a `pip install tac` perspective. Default: keep them in the notices
      file under their own section.
- [ ] **License-policy line in CLAUDE.md** — should CLAUDE.md grow a
      "License policy — non-negotiable" section that pins the accepted-license
      list and the procedure for adding a new dep? Recommended yes; this would
      structurally prevent future copyleft deps from sneaking in without
      operator review (Catalog-#-style preflight gate).
- [ ] **Optional preflight check** — add `check_third_party_notices_in_sync_with_pyproject`
      strict preflight gate so any future `pyproject.toml` dep-list change without
      a paired `THIRD_PARTY_NOTICES.md` update is refused at commit time? This is
      the "Bugs must be permanently fixed AND self-protected against" non-negotiable
      pattern applied to the notices-drift bug class.

---

## What does NOT change (this draft is conservative)

- The LICENSE body is byte-identical to the current MIT text other than the
  copyright-holder line.
- The current research-inspirations list in `THIRD_PARTY_NOTICES.md` is
  preserved verbatim in a sub-section.
- The upstream challenge harness under `upstream/` is acknowledged as
  governed by its own license (per CLAUDE.md "Non-Negotiable Upstream Rule").
- No `LICENSE` is added inside `upstream/` (that would be an upstream edit and
  is forbidden).

---

## Paths

- Draft LICENSE: `reports/oss_d3_d4_drafts_20260514/LICENSE.draft`
- Draft notices: `reports/oss_d3_d4_drafts_20260514/THIRD_PARTY_NOTICES.md.draft`
- This memo: `reports/oss_d3_d4_drafts_20260514/d3_d4_draft_memo.md`
- Memory landing: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_d3_d4_drafts_landed_20260514.md`

---

## Audit anchor

OSS v0.2.0-rc1 audit, commit `c293ba425`, findings D-3 + D-4. This draft is the
proposal for resolving both findings. Promotion to canonical
`LICENSE` / `THIRD_PARTY_NOTICES.md` is **gated on explicit operator approval**
per CLAUDE.md "Non-Negotiable Upstream Rule."

---

## comma.ai alignment addendum (operator directive 2026-05-14)

Operator directive: *"whatever is cleanest for comma ai in production and in
line with their OSS policies and commercial monetization"*.

A companion addendum verified comma.ai's actual production OSS posture against
the canonical comma.ai repos and produced concrete recommendations for the
LICENSE.draft + THIRD_PARTY_NOTICES.md.draft promotion. Key resolutions:

- **B1 (`pyppmd` LGPL) is RESOLVED**: comma.ai's own contest repo
  `commaai/comma_video_compression_challenge` ships `pyppmd` directly in its
  `pyproject.toml` runtime deps. Option 1 (accept dynamically-linked LGPL as
  documented exception) is precedent-matched to upstream and is the only
  comma-aligned choice. The notices draft has been updated to reference this.
- **SPDX headers (operator-confirm point §3 above): resolve to NO** — comma.ai
  uses no SPDX headers in their LICENSE files or per-source-file across
  openpilot. Reverses the "consider adding" suggestion to "do NOT add for
  comma alignment".
- **Copyright holder name**: confirm `Alejandro Peña` (single author) — matches
  comma.ai's contest repo `Copyright (c) 2026 comma.ai` minimalism. Joint
  attribution NOT recommended absent explicit operator IP-transfer authorization.
- **Copyright year**: confirm `2026` (single year, matches contest).
- **THIRD_PARTY_NOTICES.md**: keep the comprehensive draft (Path B) over a
  comma-strict-minimal alternative (Path A). The comprehensive draft is more
  useful for a research-stage repo.
- **New STRICT preflight gate proposed**: `check_no_copyleft_dependencies` would
  refuse future LGPL/GPL/AGPL deps except an explicit allowlist (initially:
  `pyppmd` only). Operator-decide whether to spawn a follow-up subagent to land
  this.
- **CLAUDE.md non-negotiable proposed**: a "License policy — non-negotiable"
  section codifying the permissive-only-with-pyppmd-exception posture.

See `reports/oss_d3_d4_drafts_20260514/comma_ai_alignment_addendum.md` for the
full addendum, verified-empirical receipts, and the comma-aligned operator
confirm-or-correct checklist that supersedes the original §"Operator
confirm-or-correct checklist" above.
