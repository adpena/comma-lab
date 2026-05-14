# comma.ai OSS posture alignment addendum — D-3 + D-4 drafts

**Date:** 2026-05-14
**Subagent:** `claude:comma_ai_oss_alignment_research_20260514`
**Lane:** `lane_comma_ai_oss_alignment_research_20260514` (Phase 2, L0→L1)
**Operator directive:** *"whatever is cleanest for comma ai in production and in line with their OSS policies and commercial monetization"*
**Companion memo:** `reports/oss_d3_d4_drafts_20260514/d3_d4_draft_memo.md`
**Anchor commit:** OSS v0.2.0-rc1, head `c293ba425`

This addendum extends the D-3 + D-4 draft memo by aligning the LICENSE +
THIRD_PARTY_NOTICES.md drafts with comma.ai's actual production OSS posture and
commercial monetization model, verified against the canonical comma.ai
repositories on 2026-05-14.

---

## TL;DR

- **comma.ai's posture is unambiguous and uniform**: MIT across every
  first-party public repo we checked (openpilot, comma_video_compression_challenge,
  tinygrad). No copyleft. No CLA. No SPDX headers in source.
- **The contest repo (`commaai/comma_video_compression_challenge`) ships
  `pyppmd` directly in its own `pyproject.toml` runtime deps**. The LGPL blocker
  flagged in the D-3+D-4 memo (B1) is therefore PRECEDENT-MATCHED to comma.ai's
  own scaffold — Option 1 (accept dynamically-linked LGPL as documented exception)
  is the only choice that stays aligned with comma.ai's posture.
- **Recommended pact-repo defaults (operator confirm-or-correct):**
  1. LICENSE: `Copyright (c) 2026 Alejandro Peña` (MIT body unchanged) —
     mirrors the contest repo's `Copyright (c) 2026 comma.ai` style (lowercase,
     no Inc., year matches). Single-author convention; no joint attribution
     unless operator specifies otherwise.
  2. No SPDX headers in source files (matches openpilot's pattern — comma does
     not use SPDX headers anywhere we sampled).
  3. THIRD_PARTY_NOTICES.md: keep the openpilot-style minimalism (the existing
     pact draft is more comprehensive than openpilot's posture demands; the
     addendum proposes a "comma-aligned" minimal variant the operator may
     prefer for the canonical promotion).
  4. Permissive-only dep policy stays the project posture; pyppmd LGPL is the
     ONE documented exception that traces back to upstream contest scaffold.
- **One new STRICT preflight gate proposed (operator-decide whether to land):**
  `check_no_copyleft_dependencies` — refuses pyproject.toml deps with
  GPL/LGPL/AGPL except an explicit allowlist (currently: `pyppmd` only).

---

## 1. comma.ai's verified OSS posture (as of 2026-05-14)

### 1.1 Repositories sampled

| Repo | License (GitHub API) | Copyright line in LICENSE | pyproject `license` field |
|------|----------------------|----------------------------|----------------------------|
| `commaai/openpilot` | MIT | `Copyright (c) 2018, Comma.ai, Inc.` | `license = {text = "MIT License"}` |
| `commaai/comma_video_compression_challenge` | MIT | `Copyright (c) 2026 comma.ai` | (no `license` field declared) |
| `tinygrad/tinygrad` (the tiny corp; spun out of comma) | MIT | `Copyright (c) 2024, the tiny corp` | (not checked) |

Sources (verified 2026-05-14 via `gh repo view` + `curl raw.githubusercontent.com`):
- https://github.com/commaai/openpilot/blob/master/LICENSE
- https://github.com/commaai/comma_video_compression_challenge/blob/master/LICENSE
- https://github.com/commaai/openpilot/blob/master/pyproject.toml
- https://github.com/commaai/comma_video_compression_challenge/blob/master/pyproject.toml

### 1.2 Stylistic conventions

- **License:** MIT, every time. No Apache 2.0, no dual-licensing, no CLA.
- **Copyright holder name:** they vary the form per repo — older openpilot
  uses formal `Comma.ai, Inc.` (with comma between year and name); the recent
  contest uses lowercase `comma.ai` (no Inc., no comma). The pattern signals
  comma.ai treats the copyright-holder line as informal author identity, not
  load-bearing legal entity declaration.
- **Year:** single year (`2018`, `2026`). No `2018-2026` ranges in the LICENSE
  file itself. Year set to the year the LICENSE was first added.
- **No SPDX headers** in source files (sampled openpilot via `gh search code`).
- **No root-level THIRD_PARTY_NOTICES.md** in openpilot. They rely on
  per-submodule LICENSE files (vendored deps live under `git+...` URLs in
  pyproject + each upstream repo carries its own LICENSE). Their
  `release/release_files.py` is a release-bundle filter, not a notices file.
- **No CONTRIBUTING.md-style copyright assignment** in openpilot. Contributors
  hold their own copyright; the repo aggregates under MIT.

### 1.3 Commercial monetization implications

comma.ai's commercial model is the canonical "open-source software, sell-hardware"
pattern:

- They sell the **comma 3X** dashcam-driver-assistance hardware.
- openpilot (the software) is MIT — anyone can fork, modify, redistribute.
- The hardware is the moat; the software is the marketing flywheel.

**Why MIT, not Apache 2.0:** MIT has zero patent-license obligations. Apache 2.0
includes a patent grant clause (§3) that some hardware-bundling lawyers worry
adds friction to commercial deployment of the bundled binary. MIT is the
"frictionless" choice for hardware OEMs that want to bundle the software with
zero downstream contributor-patent-claim risk.

**Why permissive-only deps:** copyleft (GPL/LGPL static-linked) deps would
contaminate the binary distributable with source-availability obligations on the
hardware bundle. Dynamic-linked LGPL deps via pip-install (where the user
re-downloads the dep separately from the comma binary) is the standard
practical workaround — comma.ai's contest pyproject demonstrates this with
their own use of pyppmd.

---

## 2. Pact repo alignment recommendations

### 2.1 LICENSE (comma-aligned)

The current draft `Copyright (c) 2026 Alejandro Peña` already satisfies the
comma.ai posture (MIT body, single year, single author). Two style refinements
to consider matching the contest repo's exact format:

**Recommended canonical form:**
```
MIT License

Copyright (c) 2026 Alejandro Peña

[standard MIT body unchanged]
```

This matches `Copyright (c) 2026 comma.ai` (no comma between year and name —
the contest repo's style; openpilot's older `2018, Comma.ai, Inc.` had the
comma but that's the older formal style). Recommend the operator NOT use
"Inc." unless they have a registered legal entity, since the contest repo
itself dropped the "Inc." in 2026.

**Operator confirm-or-correct points (comma.ai-context):**
1. **Copyright holder name:** `Alejandro Peña` (default) — mirrors contest
   repo's single-name pattern. Joint with comma.ai (e.g.,
   `Alejandro Peña and comma.ai`) would imply IP transfer/grant that the
   operator has not authorized. **Default to `Alejandro Peña` only** unless
   the operator explicitly requests joint attribution.
2. **Copyright year:** `2026` — matches contest repo. No range.
3. **SPDX header in LICENSE:** **NO** — comma.ai does not use SPDX in their
   LICENSE files. Reverses the D-3+D-4 memo's "consider adding" to a
   "do NOT add for comma alignment". (The operator may still override on
   automated-scan-tool grounds.)
4. **SPDX header per-source-file:** **NO** — matches openpilot.

### 2.2 THIRD_PARTY_NOTICES.md (comma-aligned)

**The operator's canonical comma.ai precedent is openpilot, which has NO
root-level THIRD_PARTY_NOTICES.md.** openpilot relies on per-submodule LICENSE
files (vendored deps under `git+commaai/dependencies` each carry their own
LICENSE) plus pip-installed PyPI deps that carry their own LICENSE in the
installed wheel.

Two paths the operator may choose between:

**Path A — comma-strict-minimal (matches openpilot exactly):**
- Replace the proposed comprehensive notices file with a 7-line file pointing
  at: (a) project LICENSE, (b) `pyproject.toml` for the dep list, (c) the
  pip-installed package's own LICENSE in the wheel.
- **Risk:** a downstream redistributor of a binary archive (e.g., a frozen
  pyinstaller bundle of pact for offline use) needs to bundle individual dep
  LICENSE files themselves. The current pact use-cases ship as PyPI packages
  + Modal/Vast.ai workers (which pip-install at remote setup time), so the
  binary-bundle risk is low. Path A is the cleanest match to comma.ai.

**Path B — comprehensive-notices (current draft):**
- The 240-line draft already in
  `reports/oss_d3_d4_drafts_20260514/THIRD_PARTY_NOTICES.md.draft` is more
  exhaustive than openpilot's posture demands. It is also genuinely more
  useful for a research-stage academic codebase where the dep list is in
  flux. **Recommend keeping Path B as the default** (the draft is good as-is)
  but offer Path A as a "comma-strict" alternative if the operator prioritizes
  visual parity with openpilot over thoroughness.

**Recommended operator decision: Path B (keep the comprehensive draft)** with
a one-line front-matter note explaining the divergence from openpilot's
minimalism is intentional ("research-stage repo with rapidly-evolving dep set;
explicit notices are easier to audit than tracing each `pip install` package's
internal LICENSE").

### 2.3 pyppmd LGPL — RESOLVED via comma.ai precedent

**The original D-3+D-4 memo's BLOCKER B1 is resolved:** the contest's own
`pyproject.toml` includes `pyppmd` as a runtime dep (line 16, alongside other
permissively-licensed deps). This means comma.ai's own contest scaffold
adopts the dynamic-linked-LGPL exception — accepting pyppmd in the pact repo
is therefore PRECEDENT-MATCHED and aligned with comma.ai's posture.

**Action:** the addendum recommends Option 1 from the D-3+D-4 memo (accept
LGPL as documented exception) and explicitly cites the contest's pyproject.toml
as the precedent. The THIRD_PARTY_NOTICES.md.draft already documents pyppmd
transparently; no further changes needed there.

**Cross-ref to PYPPMD-MIGRATE sister subagent (`a34d2ad63c1c68c94`):** that
subagent owns pyproject.toml + experiments/modal_train_lane.py + pyppmd-importing
files. The PYPPMD-MIGRATE work proceeds independently; this addendum's role is
to document why pyppmd is acceptable to keep, in case the migrate-away decision
ever needs to be revisited from a license-compliance angle.

### 2.4 Permissive-only dep policy as CLAUDE.md non-negotiable

The current ad-hoc operator policy ("we use MIT and Apache 2.0 for all") is
not yet codified in CLAUDE.md as a non-negotiable. Codifying it would:

- Make the policy machine-checkable via a STRICT preflight gate (proposed
  Catalog # — see §3 below).
- Document the pyppmd dynamic-link exception so future agents don't try to
  retire it as a bug.
- Match comma.ai's de facto posture so that any future code-share / upstream
  contribution lands cleanly.

**Recommended CLAUDE.md insertion point:** after the existing "Public Disclosure
Hygiene" section (which is the closest analog). Suggested text in §4 below.

### 2.5 Commercial-bundling-friendliness checklist

For comma.ai (or any commercial entity) to ingest pact code into a hardware
bundle without legal friction, the project must satisfy:

- [x] **MIT or Apache-2.0 license on first-party code** — pact draft is MIT.
- [x] **No copyleft deps statically linked** — pact has zero. pyppmd is
      dynamically loaded via Python `import pyppmd` inside the `tac.pr86_hpac_codec`
      decode path; comma.ai's own contest scaffold adopts the same pattern.
- [x] **No copyright assignment requirement on contributors** — pact has no
      CLA; comma.ai has none either.
- [x] **No hidden patent claims** — MIT explicitly. Apache 2.0 §3 patent grant
      not invoked.
- [x] **No trademark restrictions on derivative work** — MIT silent on
      trademark; pact does not assert any trademark.
- [x] **Notices file or per-dep LICENSE bundling possible** — pact draft
      ships a comprehensive notices file (Path B above). Path A (comma-strict
      minimal) would require downstream binary-bundlers to assemble dep
      LICENSEs themselves.

**Verdict: pact is commercially-bundling-friendly under MIT, matching
comma.ai's posture, and the only nuance (pyppmd LGPL) is precedent-matched
to upstream contest scaffold.**

---

## 3. New STRICT preflight gate proposal (operator-decide)

The operator may want to land a new strict preflight gate to structurally
prevent future copyleft deps from sneaking into `pyproject.toml`:

### Proposal: `check_no_copyleft_dependencies`

**Scope:** scans every `pyproject.toml` in the repo for runtime-dep entries
matching a known-copyleft license token (`GPL-`, `LGPL-`, `AGPL-` per SPDX
identifiers; same scan also catches `(GPL` / `(LGPL` / `(AGPL` parenthetical
forms) AND refuses unless the dep name appears in an explicit
`_COPYLEFT_DEP_ALLOWLIST` or carries a same-line `# COPYLEFT_DEP_OK:<reason>`
waiver.

**Initial allowlist:**
- `pyppmd` — LGPL-2.1-or-later, dynamically-linked PyPI dep, precedent-matched
  to upstream contest scaffold (`commaai/comma_video_compression_challenge`
  pyproject.toml line 16).

**Bug class extincted:** the D-3+D-4 audit caught pyppmd as a documented LGPL
dep. Without this gate, a future subagent could add another LGPL/GPL/AGPL dep
without operator review, silently breaking the comma-aligned permissive-only
policy.

**Sister of:** Catalog #205 (presumably exists; not in scope to verify).

**Deferred:** This gate is NOT landed by this subagent. The decision to land
it (or defer to a follow-up subagent) is operator-routable. Recommend landing
in the same wave as the LICENSE/THIRD_PARTY_NOTICES.md promotion since both
codify the same permissive-only policy.

---

## 4. Recommended CLAUDE.md non-negotiable text

Insert after the "Public Disclosure Hygiene" section:

> ## License policy — non-negotiable
>
> All first-party pact code is MIT-licensed, mirroring comma.ai's canonical
> posture across openpilot, comma_video_compression_challenge, and tinygrad.
> Runtime dependencies must be under permissive licenses compatible with MIT
> redistribution: **MIT, Apache-2.0, BSD-2/3-Clause, HPND, MPL-2.0 (dev only),
> ISC, PSF, Python-2.0, BSL-1.0**.
>
> **Copyleft licenses (GPL family, LGPL static-linked, AGPL) are FORBIDDEN as
> runtime deps** without operator review. The single documented exception is
> `pyppmd` (LGPL-2.1-or-later, dynamically-linked PyPI dep), which is
> precedent-matched to the contest's own `pyproject.toml`.
>
> When adding a new runtime dep:
> 1. Verify the dep's license via `importlib.metadata.metadata(<pkg>).get("License")`
>    or its upstream `LICENSE` file.
> 2. If the dep is anything other than the permissive list above, surface as a
>    BLOCKER for operator review BEFORE merging.
> 3. Add the dep to `THIRD_PARTY_NOTICES.md` in the same commit batch as the
>    `pyproject.toml` change.
> 4. STRICT preflight gate `check_no_copyleft_dependencies` (Catalog #TBD)
>    enforces this at commit time.
>
> Why MIT and not Apache-2.0: comma.ai uses MIT exclusively, and MIT has zero
> patent-license clauses (Apache 2.0 §3) that some hardware-OEM legal reviews
> flag as bundling friction. MIT is the "frictionless" choice for hardware
> bundling, which is comma.ai's commercial moat (sell-hardware,
> open-source-software model).

---

## 5. Operator confirm-or-correct checklist (comma-aligned, supersedes D-3+D-4 §)

Before promoting `LICENSE.draft` and `THIRD_PARTY_NOTICES.md.draft` to canonical:

- [ ] **Copyright holder name:** `Alejandro Peña` (default, single-author,
      matches contest repo's `comma.ai` minimalism). Confirm or override.
- [ ] **Copyright year:** `2026` (matches contest repo). No range.
- [ ] **SPDX header in LICENSE:** NO (matches openpilot). Confirm.
- [ ] **SPDX headers per-source-file:** NO (matches openpilot). Confirm.
- [ ] **THIRD_PARTY_NOTICES.md path:** Path B (comprehensive draft) /
      Path A (comma-strict minimal). Default: Path B. Confirm.
- [ ] **pyppmd LGPL decision:** Option 1 (accept + document, precedent-matched
      to contest pyproject). Default per this addendum: Option 1.
- [ ] **License policy in CLAUDE.md:** add the proposed non-negotiable text in
      §4? Default: yes.
- [ ] **Land `check_no_copyleft_dependencies` STRICT gate:** spawn follow-up
      subagent? Default: yes (in the same wave as LICENSE promotion).
- [ ] **Joint attribution with comma.ai:** NO unless operator explicitly
      requests an IP transfer arrangement. Default: solo.

---

## 6. Paths

- This addendum: `reports/oss_d3_d4_drafts_20260514/comma_ai_alignment_addendum.md`
- Original D-3+D-4 memo: `reports/oss_d3_d4_drafts_20260514/d3_d4_draft_memo.md`
- Draft LICENSE: `reports/oss_d3_d4_drafts_20260514/LICENSE.draft`
- Draft notices: `reports/oss_d3_d4_drafts_20260514/THIRD_PARTY_NOTICES.md.draft`
- Memory landing: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_comma_ai_oss_alignment_addendum_landed_20260514.md`
- Lane registry: `lane_comma_ai_oss_alignment_research_20260514` (Phase 2, L0→L1)

---

## 7. Wire-in declaration (Catalog #125 compliance)

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable:

1. **Sensitivity-map contribution:** N/A — research/policy memo, no per-tensor
   sensitivity.
2. **Pareto constraint:** N/A — no archive bytes / score axis touched.
3. **Bit-allocator hook:** N/A — same.
4. **Cathedral autopilot dispatch hook:** N/A — research-only memo.
5. **Continual-learning posterior update:** N/A — no empirical anchor produced.
6. **Probe-disambiguator:** **MAYBE.** If the operator's comma-strict-minimal
   (Path A) vs comprehensive (Path B) THIRD_PARTY_NOTICES.md decision needs
   structural arbitration rather than operator preference, a tiny probe under
   `tools/probe_third_party_notices_path_a_vs_b.py` could compare the two
   formats against (a) GitHub License-Detection API, (b) PyPI metadata
   completeness, (c) a hypothetical downstream binary-bundler workflow.
   Recommended only if the operator wants empirical arbitration; default is
   operator-preference Path B.

`research_only=true` per the absence of any solver-stack actuator. This
addendum's outputs (draft refinements + addendum file + CLAUDE.md text
proposal + STRICT gate proposal) are operator-routable decisions, not
auto-applied changes.

---

## 8. Empirical receipts (verification commands)

Anyone re-verifying this addendum's comma.ai-posture findings can re-run:

```bash
# 1. openpilot LICENSE + license metadata
gh repo view commaai/openpilot --json name,description,licenseInfo,url
curl -sL https://raw.githubusercontent.com/commaai/openpilot/master/LICENSE

# 2. contest LICENSE + license metadata
gh repo view commaai/comma_video_compression_challenge --json name,description,licenseInfo,url
curl -sL https://raw.githubusercontent.com/commaai/comma_video_compression_challenge/master/LICENSE

# 3. tinygrad cross-reference
gh repo view tinygrad/tinygrad --json name,licenseInfo,description
curl -sL https://raw.githubusercontent.com/tinygrad/tinygrad/master/LICENSE | head -3

# 4. contest pyproject pyppmd dep (PRECEDENT for B1 resolution)
curl -sL https://raw.githubusercontent.com/commaai/comma_video_compression_challenge/master/pyproject.toml | grep pyppmd

# 5. openpilot SPDX-header search (returns empty — confirms no SPDX usage)
gh search code --repo commaai/openpilot "SPDX-License-Identifier" --limit 5

# 6. openpilot root-file inventory (confirms no THIRD_PARTY_NOTICES.md)
gh api repos/commaai/openpilot/contents/ --jq '.[] | .name'
```

Verified outputs (2026-05-14):
- openpilot LICENSE: MIT, `Copyright (c) 2018, Comma.ai, Inc.`
- contest LICENSE: MIT, `Copyright (c) 2026 comma.ai`
- tinygrad LICENSE: MIT, `Copyright (c) 2024, the tiny corp`
- contest pyproject.toml line 16: `pyppmd` (no version pin)
- openpilot SPDX search: 0 results
- openpilot root: 32 files, none named THIRD_PARTY*, NOTICE*, COPYING, etc.
