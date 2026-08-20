# ddm_sd2 — Build the runner that RETAINS the argmax, so the seg term can finally be decomposed.

**Owner:** codex arm · **Base:** PR130 CPR1 · **scorer-free BUILD** (MAIN fires the run) ·
`score_claim=false`

## OPTIMAL FORM (read first)

Reference form: a resumable paired-candidate n600 runner that produces, for the PR130 base AND a
candidate, the FULL retained evidence the seg decomposition needs — decoded camera frames, SegNet
argmax, PoseNet outputs, all chunked with bytes+sha256 — and from them the **directed
target→prediction edge matrix**, per FRAME and per EDGE. Reference pin, verify at source: base
archive 191,052 B sha `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`;
the retention contract is `.omx/research/ddm_sg2_20260810/SG2_SCORER_QUEUE.json`
(`required_retention`, chunk_pair_limit 120, resume_required, do_not_launch_if_any_payload_writer_is_absent).
Declared reductions: SCOPE only — chunked execution and resume are REQUIRED, not a reduction.
MECHANISM reductions are TOY-BRACKET: an argmax from a proxy instead of the frozen CPU-torch SegNet;
a per-CLASS table presented as the edge decomposition; a run that computes the matrix but discards
the argmax it came from (that is the exact defect this arm exists to end).

**Authority:** full online research, full OSS, full internal leverage — our own code, docs, and
unwired modules are off the shelf to use, adapt, refactor, extend, or fork. PR130 repro code is
off-the-shelf authorized. Cite path + commit for anything reused.

## WHY — the operator asked, and the answer is currently UNMEASURABLE

Operator 2026-08-09: *"Full recursive analysis and decomposition of PR130's archive and all of the...
especially on the segment side, where specifically the errors are, like, which frames and which
classes and edges and boundaries... could inform our optimization."*

`ddm_sg2` tried and returned an honest BLOCKED with the mechanism:
- `0.00028609 × 117,964,800 = 33,748.549632` mismatches — **NON-INTEGER**. The published seg scalar
  is a ROUNDED DISPLAY VALUE and cannot be inverted to a mismatch count.
- **No candidate argmax payload was ever retained.** So the matrix cannot be recomputed either.

That is #1001 (ALWAYS KEEP THE PAYLOAD) striking the seg axis. The cure is not cleverness — it is a
run that KEEPS what it measures. Build that runner.

MEASURED context you must not re-derive:
- real AV-vs-DALI control `20,671/117,964,800 = 0.000175230238` = 61.25% of the seg scalar, Road in
  **89.65%** of differing pixels (sg2, `[macOS-CPU advisory]`)
- `ddm_pc2` on a DIFFERENT vehicle: Road in **87.8%** of flips, Road↔Lane alone **49.2%** — the
  ~88–90% Road participation is cross-vehicle, so decompose per EDGE, never per CLASS
- #906 is WRONG-OBJECT (hypothetical chroma perturbation, not PR130 error) — do not build on it
- mixed q3/q4 candidate: −848 B, +1.40720e-6 d_seg, semantic leg −0.000423928 S, pose UNMEASURED

## WHAT TO BUILD

1. **The paired runner.** Base + candidate through the REAL decode path, n600, chunked at ≤120
   pairs, resumable from disk, atomic progress receipts. It must REFUSE to start if any payload
   writer is absent (the queue says so explicitly — honor it).
2. **Full retention, per the queue's `required_retention` list.** Every chunk with bytes + sha256.
   This is P0 DEF CON 1000: a run that measures and discards is FORBIDDEN at the typing moment.
   Run `tac.payload_retention_gate` on what you write. Known detector limitation: it tests
   PERSISTENCE, not reachability-to-a-writer, so bytes consumed only by `sha256()` still count as
   LOST (`.omx/research/ddm_main_p0_triage_20260810.md`).
3. **The decomposition, computed FROM the retained argmax:**
   - directed target→prediction 5×5 edge matrix, with its own denominator stated
   - per-FRAME error counts → which frames carry the mass (heavy-tail or uniform?)
   - per-EDGE, and boundary-vs-interior split (pc2 measured interiors ≈0.058% on its vehicle —
     does that hold here?)
   - the class-index order is the comma10k CANONICAL order (0 Road · 1 Lane · 2 Undrivable ·
     3 Movable · 4 MyCar) — CLAUDE.md is explicit that luma-sorting gives the WRONG order and has
     bitten us 3×. Self-detect by spatial/static signature; do not hardcode blindly.
4. **DISK — CORRECTED 2026-08-09 (operator: two SSDs connected, both usable).**
   TWO tiers are live: `/Volumes/VertigoDataTier` (tier-1, **40 GiB free, 98% FULL — do NOT write
   bulk here**) and `/Volumes/APDataStore` (tier-2, **997 GiB free** — USE THIS). Write your bulk to
   `/Volumes/APDataStore/pact/ddm_sd2_20260810/` (already created). The earlier "largest scope that
   fits" concession is WITHDRAWN: **full n600 paired retention now fits with ~100x headroom.** Do
   the full n600. Still measure free space and declare your footprint; still never delete anything
   (certify-or-block is MAIN's call).

## HARD RULES

- Bulk → `/Volumes/APDataStore/pact/ddm_sd2_20260810/` (tier-2, 997 GiB free). No `/tmp`.
- Commits via `tools/subagent_commit_serializer.py`, POST-EDIT `--expected-content-sha256` per file,
  tags `[no-triality] [p0-ledger-ok]`, **no attribution trailer of any kind**.
- `.py`: 2 × `tools/review_tracker.py mark-file <f> --status reviewed`; never `REVIEW_GATE_OVERRIDE=1`
  with a `.py`.
- Write ONLY under `ddm_sd2` paths. Sister arms' landed artifacts are APPEND-ONLY inputs — sg2's
  audit tool and queue are INPUTS, never edits.
- **DO NOT RUN THE SCORER.** Build it, smoke it at n≤4 pairs if you must prove the writer works, and
  hand MAIN the fire command. MAIN has Metal and fires it. A codex arm cannot reach Metal.
- `upstream/` IMMUTABLE. Intake clones READ-ONLY. Every number carries its axis; macOS is
  `[macOS-CPU advisory]`, never `[contest-CPU]`. No Modal without operator GO.

## DELIVERABLE

The runner + its retention proof + the exact MAIN fire command with projected footprint and runtime.
If the full n600 paired retention does not fit in the available disk, say so with the arithmetic and
propose the largest scope that DOES fit — a decomposition on a stratified-random n≥120 (NEVER a
prefix, per m88/m96; note seg prefix bias is the mild ≈0.96× axis) is a legitimate SCOPE reduction
and still answers the operator's question. What is NOT acceptable is another run that measures the
seg term and throws away the argmax.

## RESPAWN NOTE (MAIN, 2026-08-10) — two blockers, both diagnosed at source

**Blocker 1 CURED (was yours, not a capacity problem).** Your `PermissionError:
[Errno 1] Operation not permitted` on `/Volumes/APDataStore` was the codex sandbox,
not the disk — the keeper granted `--add-dir` for tier-1 ONLY. Fixed at the spawn
site in `tools/codex_arm_queue.py` (commit `002208ddf6`): every MOUNTED tier is now
granted, both-direction controls executed. You can write to APDataStore now.

**Blocker 2 STILL OPEN — your runner asks the PR130 base receiver for a capability
it does not have.** MAIN fired your exact command under `safe_run` and it failed
closed in 2.57 s at the first real decode:

```
inflate.py:737  ValueError: periodic token progress checkpoints require the ANS stack
```

Guard: `if progress_cache_path is not None and token_codec != "ans": raise`.
Your resumability design assumes ANS token checkpointing; the PR130 base ships the
**Range** coder. This is a real coupling, not plumbing: Range is sequential with no
cheap resumable state, ANS's stack discipline gives restartable positions. The ANS
receiver is `ddm_ai1`'s live work and is not landed.

FIX IT AT THE RIGHT LEVEL — make the checkpoint request CONDITIONAL on the decoded
archive's actual `token_codec`, not unconditional. Range decodes lose intra-decode
checkpointing (~1,011 s to redo, acceptable); your 60-pair CHUNK resume is
unaffected and still gives crash-safety. Do NOT weaken the receiver guard — it is
correct. Do NOT wait for ANS.

Retained failure evidence (yours, already persisted correctly):
`/Volumes/APDataStore/pact/ddm_sd2_20260810/matched_local_n600/retained/decode/pr130_q4_control/attempts/attempt_0001/decode.log`
sha256 `8bb9bc56f4bd8ce86d99104b493dbeb5fa1f0bc2bd8bf7d4aa464dbd1d84e54f`.
MAIN's plan-only receipt (admitted, 27.86 GB required / 997 GiB free):
`SD2_RETENTION_PREFLIGHT.json` sha `583a15ff048122e554b002867fe089874e27b0ce257a73d27e5d2b561d136dbc`.
Lane `lane_ddm_sd2_seg_decomposition_20260810` is CLAIMED by MAIN — reuse it.
