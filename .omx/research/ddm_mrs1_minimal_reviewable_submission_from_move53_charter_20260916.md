# ddm_mrs1 — a MINIMAL, REVIEWABLE submission packet from the sealed move-53 tree (charter, MAIN 2026-09-16; codex astra xhigh; scorer-free)

## Why (operator 2026-09-16, verbatim: "It's terrible and not reviewable and way too complex and over engineered")
PR #140 was closed by the maintainer on 2026-09-15 with no eval and no comment. What we asked him to review: 64 files, 34
Python modules (12,680 LOC), 24 committed .pyc, 3 C sources compiled at inflate time behind a C-compiler + Brotli==1.2.0
gate, modules named after internal arm ids, a 7,303-character PR body. The contest's own winners were 2 files and 241–605
LOC. A maintainer reviews BEFORE triggering eval; an unreviewable PR never gets a score. The score chain is the lab; the
submission is a separate artifact written FOR a stranger. Memory `pr140_closed_unreviewable_ship_a_submission_not_the_lab_20260916`.

The sealed move-53 tree (`/Volumes/APDataStore/pact/ddm_pd6/candidate2/candidate_runtime`, read-only; archive 179,286 B
sha aab908d3b32c65582b7de1a3855b67a4f6fdb9373b0dc81e3049ee8164ec3957; S 0.1361014714463198 [contest-CUDA T4 n600],
decode 1,185.9 s of the 1,800 s budget) has 43 .py (15,767 LOC) + 4 .c (2,550 LOC); the STATIC closure of inflate.py is
35 modules / 11,148 LOC, and much of that is dynamically dead: `ddm_wc1_advisory_runtime.py`, four alternative correctors
(fx1/fx2/rr4/rr5/native_free), three mixers (tc1/rc3/rlc1), gates (ihs2/ihs2_gate_a), checkpoints. Nobody can tell
from the tree which lines run.

## Deliverable — `submissions/mrs1/` = exactly the files a reviewer sees (REVIEWER BUDGET, pre-registered, measured)
- `inflate.sh` ≤ 5 lines. `inflate.py` ONE file. At most ONE vendored `.c` file, only if the measured decode without it
  exceeds 1,500 s on T4 (report the measurement; the rc64 range-coder backend is the expected survivor), compiled by one
  `cc` line in inflate.sh with a one-sentence explanation in the README. `README.md` ≤ 40 lines: what the archive holds
  (one paragraph), how decode works (one paragraph, in plain words: a small token renderer + a learned prior + a pose
  carrier), the exact decode command, GPU requirement + measured time, the archive sha/size. NO .pyc, NO internal arm
  names (no ddm_/fx1/rr4/ihs2/tc1/rc3/rlc1/f26/sm1 identifiers in any file or symbol — rename to what the thing IS),
  NO advisory/instrumentation/checkpoint/resume code, NO environment switches (F26_TOKEN_DECODER, RLC1_ADVISORY_CPU,
  RLC1_PROOF_BLAS_THREADS…), NO dependency pins the upstream evaluator's environment does not already satisfy (verify
  against `upstream/` requirements; if brotli is needed, import it and fail with one clear line). Target inflate.py
  ≤ 1,500 LOC (report the number; the budget is the gate, the target is the aim).
- The archive is UNCHANGED: `submissions/mrs1/archive.zip` byte-identical to the sealed one (sha above).
- Proof, in this order: (1) DYNAMIC line coverage of the real decode on the host's CPU advisory path for ≥ 24 seeded
  pairs (`coverage` or `sys.settrace`), retained — the live line set is the ONLY code that goes into inflate.py; (2) the
  new packet's cold n600 public parse-back on the host (advisory device) produces raws BYTE-IDENTICAL to move 53's
  retained raws (shas in the pd6 memo / seal) — raw identity is the whole correctness proof; (3) public entrypoint smoke
  with the upstream `inflate.sh <archive-dir> <output-dir> <file-list>` signature in a bare venv (prove the bootstrap);
  (4) a FRESH-READER test: write `.omx/research/ddm_mrs1_20260916/FRESH_READER_PROMPT.md` — MAIN spawns a second arm
  with no context that reads only the 3–4 files and must explain in ≤ 10 sentences what inflate.py does and name every
  external dependency; record its answer verbatim; (5) the PR body draft ≤ 20 lines from the upstream template: name,
  archive link placeholder + sha + size, report.txt VERBATIM from the move-53 harvest receipt
  (`/Volumes/APDataStore/pact/ddm_pd6_fire/run2/MODAL_REMOTE_RESULT.json` → contest_auth_eval.json), "GPU: yes, 1,186 s on
  T4", compression script: "no" (honest; the encoder is the lab), NO @-mentions of anyone, no arm ids.
- MAIN's part afterwards (not yours): T4 exact eval + decode leg on the NEW receiver (a receiver change needs its own
  measured leg and its own exact row — pr19 / tc4), then the operator sees the files and decides; nothing is published
  without the one-line confirm.

## Boundaries
No Modal, fire, packet, PR, push, authorize_*; never edit upstream/, the closed PR tree `submissions/semantic_joint_ctxmix/`,
sealed trees, contract code; the sealed tree is read-only (copy). Work under `/Volumes/APDataStore/pact/ddm_mrs1/`
(report free space; retain ≤ 2 GiB with sha256; keep the n600 raws only as shas + a certificate, not bytes, once
identity is proven — they are rebuildable from the archive). Heavy steps via `tools/launch_detached_process.py
--done-receipt`; background receipt waits; retained-bytes accounting skips `.pending`/`._`. Serializer commits with
post-edit shas; two visible review passes per .py; `[no-triality] [p0-ledger-ok]`; never a co-author trailer or AI
attribution; rc 17 is NOT a stop — continue, bundle, MAIN lands, commit LAST. Checkpoint `ddm_mrs1`; lane
`ddm_mrs1_minimal_reviewable_submission_20260916`.

## OPTIMAL FORM
Reference form: the contest winners' packet shape (PR101: codec.py + model.py + inflate.py, 605 LOC; PR103: 241 LOC) and
the sealed move-53 decode (the behaviour, byte-for-byte). Declared deltas: NONE in behaviour (raw identity is the gate);
the only change is what a reader sees. Provenance pins: move 53 packet a91a7dde3; sealed tree archive sha above; pd6
memo sha 57d14f671f24e2e4; the closed PR #140 tree at HEAD (record its sha) as the negative reference.

## Prior negatives accounted (operator 2026-08-15)
PR #140 itself (this charter's reason); rih1 (91/93 compliance count measured rules, not readers); the 7,303-char body
with @-mentions (pr_packet_rediscovery_fatigue: the body is not the place for the lab's story); tc4 (receiver change
needs a measured decode wall-clock — MAIN's T4 row); pd6's rider-drop law (the archive is copied, never re-staged).

Final message: the file list with LOC per file, the coverage numbers (live lines / static lines), the raw-identity
result (600/600 or the count), the bare-venv smoke result, the fresh-reader prompt path, the PR body draft path, the
measured decode time on the host (advisory) and the projected T4 time, every boundary, serializer rc, ending with
`composition S 0.1361014714463198 @ 179,286 B [contest-CUDA T4 n600] (move 53)` unchanged.

<!-- # FORMALIZATION_PENDING: charter, not a finding -->
