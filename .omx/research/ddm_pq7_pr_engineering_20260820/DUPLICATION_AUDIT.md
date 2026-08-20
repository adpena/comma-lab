# Duplication audit — gen5 packet vs the merged state of the art

`date_utc: 2026-08-20` · `owner: ddm_pq7` · `score_claim: false` · `frontier_moved: false`

**Own-vehicle frontier: S = 0.14839100138338618 @ 180,625 B `[contest-CUDA T4, n600]`.** Unmoved here.

Driven by the binding merge rule (`YOUSFI_REVIEW_CHECKLIST.md` item 1): *"the code has too much
overlap with code that is already merged"* — #125/#127/#128, 2026-07-12.

**⛔ SEAL RESPECTED.** I read the two trees and hashed them. I edited nothing in
`gen5_jg5_waterfill/` and nothing in `submissions/robust_current/jg5_sub015_runtime/`.

---

## 0. The finding that reframes the whole question

**"Code that is already merged" does not include anything we inherited.**

I listed the live `submissions/` directory on `master` (38 entries). The merged neural lineage is:
`hnerv_muon` (#95), `hnerv_muon_finetuned_from_pr95` (#98), `hnerv_lc_v2_scale095_rplus1` (#102),
`hnerv_fec6_fixed_huffman_k16` (#110), `rhnerv_comma` (#112), `roro_perframe_hnerv`,
`belt_and_suspenders` (#106), `kitchen_sink` (#105), `quantizr` (#55),
`jas0xf_adversarial_neural_representation` (#86).

**No CPR1 / semantic-pose-HPAC code is merged.** PRs #130 (0.172), #133 (0.166) and #135 (0.162)
are all `CLOSED` — leaderboard-only. Our body descends from #130's CPR1.

Two consequences, and they pull in opposite directions:

1. **The duplication rule does not bite our `cpr1/` directory.** You cannot duplicate merged code by
   vendoring an unmerged PR's code. The #125/#127/#128 rejections were about re-implementing the
   *merged* HNeRV/`rhnerv_comma` stack alongside itself.
2. **But that vendoring is still an attribution obligation**, and a heavier one — we are shipping a
   modified derivative of another contestant's unmerged public work. `BORROWED_SUBSTRATE_ACCOUNTING.md`
   is what discharges it. This audit checks that it is complete.

---

## 1. Per-file classification — 38 files in `gen5_jg5_waterfill/`

Method: sha256 of every file in the packet vs sha256 of every file in
`experiments/results/public_pr130_intake_20260725_fable/source/submissions/semantic-pose-HPAC_CPR1/`
(13 files), plus a live listing of every merged submission on `master`, plus a static-AST import
closure rooted at `inflate.py` for reachability.

### Counts

| Class | Count | Bytes |
|---|---:|---:|
| OURS-ORIGINAL | 25 | ~343 KB |
| BORROWED-REQUIRED (modified, attributed) | 5 | 52,779 |
| BORROWED-REQUIRED (contest-mandated interface) | 2 | 4,484 |
| PACKET-LAYER (docs/archive, outside runtime seal) | 5 | 227,177 |
| BORROWED-GRATUITOUS | **0** | 0 |
| DEAD (unreachable) | **0** | 0 |
| **Byte-identical copies of ANY public PR file** | **0** | — |

### BORROWED-REQUIRED — vendored from PR #130's CPR1, all modified

Contest submissions are self-contained directories (`upstream/README.md` §submission format), so the
receiver code must ship. None is a verbatim copy.

| File | Ours (B) | PR130 (B) | Δ | Attributed in `BORROWED_SUBSTRATE_ACCOUNTING.md`? |
|---|---:|---:|---:|---|
| `cpr1/hpac_integer.py` | 17,471 | 16,236 | +1,235 | yes |
| `cpr1/inflate.py` | 13,792 | 28,304 | −14,512 | yes |
| `cpr1/hpac_integer_sparse.py` | 8,282 | 8,012 | +270 | yes |
| `cpr1/carrier_codec.py` | 7,353 | 13,770 | −6,417 | yes |
| `cpr1/integer_model_io.py` | 5,881 | 5,840 | +41 | yes |

`cpr1/ddm_mp2_semantic_receiver.py` (12,648 B) has **no PR130 counterpart** and is ours.

### BORROWED-REQUIRED — contest-mandated interface

`inflate.py` (2,281 B) and `inflate.sh` (2,203 B) are the names `evaluate.sh` invokes. Both are
ours by content: our `inflate.py` shares no body with PR130's 28,304 B `inflate.py`, and our
`inflate.sh` is 1,867 B larger than PR130's 336 B stub.

### The one real duplication-vs-MERGED risk — checked and cleared

`runtime/frame0_selector.py` (4,964 B) implements the frame-0 selector concept whose ancestor is
#110's FEC6, and #110 **is merged**; the closest merged file is
`rhnerv_comma/frame_selector.py` (8,011 B). Measured:

- **Zero shared function or class names.** Ours: `decode_selector`, `apply_pixel_mode`,
  `_combination_unrank`, `_unpack_labels`, `SelectorMode`, `Frame0SelectorError`.
  Merged: `apply_frame0_mode`, `apply_selector_to_frames`, `unpack_selector_indices`,
  `_bits_per_selector`, `_blue_tile`. Intersection: **empty**.
- Different container magic (`F0E1` vs `FES1`), different mode grammar, **numpy vs torch**.

**Verdict: independent reimplementation of a published idea, not duplicated code.** The idea is
credited; the bytes are ours.

### DEAD-file scan — nothing dead

Static AST closure from `inflate.py` reaches **29 of 30** `.py` files. The one not reached by any
import is `runtime/entropy/__init__.py` (54 B) — a package marker, structurally required.
The six `cpr1/` modules are reached **dynamically**: `inflate.py:60` passes `renderer_dir=here /
"cpr1"` into `inflate_archive`. Legitimate dispatch, not dead code.

Two files that *look* redundant are not:
- `free_corrector.py` (14,709 B) and `rr4_free_corrector.py` (15,421 B) — a real chain, not a
  duplicate: `residual_archive.py` imports `free_corrector`, which imports `rr4_free_corrector`,
  which is also imported by `fx1_…` and `fx2_…`.
- `f26_hpac_native.c` (40,911 B) + `f26_hpac_native.py` (26,781 B) — C fast path plus Python
  fallback, both reachable.

---

## 2. What PR #130 shipped that we do not — the OSS-posture gap

We deliberately did not take 6 of PR130's 13 files. Four of them are *standards* artifacts, and
their absence is a gap, not a saving:

| PR130 file | B | We ship it? |
|---|---:|---|
| `LINEAGE_AND_CITATIONS.md` | 15,393 | equivalent: `BORROWED_SUBSTRATE_ACCOUNTING.md` (33,077 B) |
| `MANIFEST.sha256` | 995 | **no** — see `OSS_STANDARDS_GAPS.md` |
| `verification.json` | 9,820 | partial: `archive_manifest.json` (664 B) |
| `test_carrier_codec.py` | 10,537 | **no test ships at all** |
| `compress.sh` | 5,798 | **no** — T3 exists to close this |
| `repack_carrier.py` | 11,290 | not needed; ours is `runtime/carrier_repack.py` |

Merged submissions go further still: `rhnerv_comma` and `hnerv_fec6_fixed_huffman_k16` both ship
`LICENSE`, `THIRD_PARTY_NOTICES.md` and `expected_output.sha256`; `hnerv_fec6` ships a `tests/`
directory. **Our packet ships none of those four.** Every merged neural submission ships
`compress.sh`. So does PR130.

---

## 3. Rebuild-boundary queue — runtime-internal findings, NOT applied

The 33-file runtime manifest and `archive.zip` are byte-frozen; touching any of them moves
`runtime_tree_sha256` (`2103073d…`) and voids the 0.14839100138338618 row. These are queued for the
rr2-native-port rebuild, which re-seals and buys a new T4 row anyway.

**First, a correction to my own framing.** Not all of these need the rebuild. The manifest keeps a
file only if its suffix is in `_RUNTIME_DEPENDENCY_SUFFIXES` (`contest_auth_eval.py:79-91`), so
**`LICENSE` (no suffix), `.md` and `.sha256` files are hash-neutral and can be staged for free.**
Only `.py`, `.sh`, `.c`, `.txt`, `.json`, `.toml`, `.env`, `.h/.hpp/.cc/.cpp` move the hash.

### FREE now — no rebuild, no re-seal, no new T4 row

| # | Finding | Why free |
|---|---|---|
| R1 | No `LICENSE`; every merged neural submission has one. Repo root already has MIT `LICENSE` (1,079 B) | no suffix → not in manifest |
| R2 | No `THIRD_PARTY_NOTICES.md`; we vendor **modified** PR #130 code, which makes this heavier for us than for them. Repo root has one (7,485 B) | `.md` → not in manifest |
| R3 | No `MANIFEST.sha256` / `expected_output.sha256`; PR130 ships both a manifest and `verification.json` | `.sha256` → not in manifest |

### Deferred to the rr2-native-port rebuild boundary

| # | Finding | Class | Cost at rebuild |
|---|---|---|---|
| R4 | No test ships; PR130 ships `test_carrier_codec.py` and #110 a `tests/` dir | quality | `.py` → moves hash |
| R5 | `inflate.py:56` auto-selects device; live README now says the submitter **picks** the runtime | routing | ties to `FREEZE_CHECKLIST` (b) |
| R6 | Per-file provenance headers absent from the 5 vendored `cpr1/` modules | attribution | `.py` → moves hash |

**R6 is the highest-value one for the duplication rule.** A maintainer opening
`cpr1/hpac_integer.py` sees no indication it is a modified derivative of PR #130. The accounting
document says so; the file does not. PR130 itself solved this with `LINEAGE_AND_CITATIONS.md` **and**
SPDX headers (`rhnerv_comma/frame_selector.py:1` is `# SPDX-License-Identifier: MIT`).

### The manifest is a suffix-filtered directory walk, not an allowlist

Worth stating because it is easy to assume otherwise, and the assumption is load-bearing:
`_runtime_root_file_manifest` (`contest_auth_eval.py:203-226`) does `root.rglob("*")` and keeps
every file whose suffix matches. **Adding any `.py` under the submission directory changes a
re-derived `runtime_tree_sha256`.** It does not change `archive.zip`, so it cannot change the score.

Corollary already true today: `report.txt` and `archive_manifest.json` **are** suffix-eligible yet
absent from the 33 rows, because the 33 were measured on the Modal host's `submission_dir`, which
never held them. **The staged packet therefore already differs from the evaluated tree by two
manifest-eligible files**, and pq3's identity proof was sound because it re-derived from the 33
enumerated rows rather than from a fresh walk. That subset rule is operative and undocumented; see
`OSS_STANDARDS_GAPS.md` S4.

## 4. Packet-layer cleanups — what I actually removed

**Nothing. I removed nothing, because there was nothing gratuitous to remove.** The packet-layer
files are `README.md`, `report.txt`, `archive_manifest.json`, `BORROWED_SUBSTRATE_ACCOUNTING.md`,
`archive.zip` — all five required by the template or by our own disclosure duty. The census guard
had already purged 51 AppleDouble sidecars (`FREEZE_CHECKLIST`). I re-ran a walk for `._*`,
`.DS_Store` and `__pycache__` across both trees and found **zero**.

A null result is the honest outcome here: the tree was already clean, and reporting a fabricated
cleanup would be worse than reporting none.

---

## 5. Verdict against the merge rule

| Question | Answer |
|---|---|
| Do we duplicate already-merged code? | **No.** 0 byte-identical copies of any merged file; the one conceptual overlap (`frame0_selector.py`) shares zero symbols with the merged equivalent. |
| Do we duplicate unmerged public PR code? | **Yes, 5 files, all modified, all attributed** — and that is structurally required by the self-contained-submission rule. |
| Is anything dead or gratuitous? | **No.** |
| Is the attribution complete? | **At document level yes; at file level no** (R6). |

**The exposure is not duplication. It is that a reviewer opening a vendored file finds no lineage
marker in it.** That is cheap to fix and it should be fixed at the rebuild boundary.
