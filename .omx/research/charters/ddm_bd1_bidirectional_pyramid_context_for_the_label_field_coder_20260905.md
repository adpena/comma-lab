# ddm_bd1 — BIDIRECTIONAL (B-pyramid) temporal context for the label-field coder (charter, 2026-09-05)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: Opus arm · Spawned by MAIN 2026-09-05 ~17:05Z under the operator's standing GO ("big structural,
possibly new paradigms"). Axis of every byte number: `[exact local byte arithmetic, scorer-free]`; `score_claim=false` until a T4 row.

## The object and why this door is open (recall, MEASURED)
The frontier archive (cl2 repack, 179,982 B, S 0.14781744131049854) spends **113,419 B (63%)** on the RC64 token stream = the GT SegNet argmax
label field (600 pairs × 512×384 × 5 classes) coded by the HPAC integer context-mixing model. hc1 decomposed it: 97.8% is the binary
"is the receiver's predicted class right?" — the "no" branch (boundary jitter the predictor cannot foresee) is **76,601 B**, the "yes"
(confirmation) branch 34,674 B. The coder's temporal context is CAUSAL ONLY: `hpac_integer.py::prepare_frame_context(idx, previous_raw)`
one-hots the PREVIOUS pair's field into `conv_past`; the trainer sets `previous[1:] = raw_tokens[:-1]` (tools/train_ddm_cl1_hpac_capacity.py
:1083-1084, :1181, :1348). The receiver decodes pairs in index order. **No arm has ever given the coder the NEXT pair's field.** Corpus grep
(bidirectional / B-frame / future field / decode order) returns nothing on this coder; mc1 (motion-compensated PREVIOUS plane, +160 B) and
mi1 (model axis ≤211 B held-out) both stayed inside the causal family. B-pyramid prediction is video coding's largest single lever
(20–35% over P-only on natural video) and the evaluator does not care in what order tokens are decoded — only that the 1,200 frames come out
in order within the 1,800 s inflate budget (current inflate 532 s on T4).

## PRIOR-LAW PREDICTION (owed before any measurement; m38)
Pyramid: level-8 keyframes (pairs ≡ 0 mod 8; 75 pairs) coded P-only from the previous keyframe (distance 8); level-4 (75 pairs) coded
bidirectionally from ±4; level-2 (150) from ±2; level-1 (300) from ±1. Predictions on the stream (113,419 B):
- **Bidirectional at distance 1 cuts the "no" branch 30–45% and the "yes" branch 15–25%** on those pairs (a boundary pixel whose class agrees
  in both neighbours is almost never a flip; the residual flips are the sweeps).
- Distance 2 / 4: 20–30% / 10–20% on the "no" branch.
- Keyframes at distance 8 P-only: **+40…+80% worse** than the shipped distance-1 causal cost on those 75 pairs.
- **Net pyramid prediction: −15…−30% of the stream = −17…−34 KB**, model bytes +≤1,500 B for the `conv_future` branch + level embedding.
- **FALSIFIER F1 (screen, $0):** if the counting-model screen (step B) predicts a pyramid net saving < 8% of the stream, CLOSE the door at
  family scope (bidirectional context on this label field) — do NOT train. **F2 (trained):** if the trained bidirectional mixer's exact
  stream at matched model size (+≤1,500 B) does not beat 113,419 B by ≥ 5,000 B (mc1's bar for a new context input), CLOSE at formulation scope.
Write the measured numbers beside these lines in the memo; residuals go to a new law `bidirectional_pyramid_context_gain_v1` (register it).

## What to do
A. RECALL (read, do not re-derive): the receiver copy `/Volumes/VertigoDataTier/pact/ddm_cl2_hpac_prior_capacity_ladder/rungs/lambda_1p0/retained/
   receiver_copy_runtime/cpr1/hpac_integer.py` (`prepare_frame_context` :315, `forward` :490, the decode loop / `decode_production_tokens`),
   `tools/train_ddm_cl1_hpac_capacity.py` (context construction, profile `cl2_shipped_ladder`), `experiments/ddm_cl2_hpac_prior_capacity_ladder.py`
   (the exact pack/stage/encode/verify path — `encode_tail` mirrors the receiver line for line), memos `ddm_hc1_hpac_calibration_reliability_20260824.md`
   (the decomposition), `ddm_mc1_*` (why the MC previous plane failed: field change ≠ rigid motion), `ddm_mi1_*`, `ddm_cl2_*`. Inputs (fields, cache,
   ep634 EMA init) are pinned by sha under `…/ddm_cl2_hpac_prior_capacity_ladder/inputs/`. `tools/subagent_checkpoint.py read --subagent-id ddm_bd1` first.
B. **$0 SCREEN on CPU (minutes, decisive, no training):** over the exact 600 decoded fields, an adaptive-count (KT) conditional-entropy model in ONE
   model class for both arms: context = 6 causal spatial taps (W, NW, N, NE, WW, NN) + the previous-plane 3×3 taps at distance d [+ the NEXT-plane
   3×3 taps at distance d]. Report, for d ∈ {1, 2, 4, 8}: total bits and indicator/"no"-branch bits with and without the next plane; the RELATIVE
   saving is the statistic (the absolute counting-model bytes will be far above the mixer's 113 KB — dc1's 21-tap oracle coded at 144 KB — so
   transfer RATIOS, never absolutes). Assemble the pyramid net from the per-level ratios × per-level pair counts. Apply F1.
C. If F1 passes — IMPLEMENT (the shipped path, nothing re-implemented): (i) model: add `conv_future` (twin of `conv_past`, same quantized integer
   form) and a 4-way level embedding riding the existing `frame_codes`/`frame_shift` path; keyframes feed zeros to `conv_future`; keep every other
   layer identical so the warm start from the ep634 EMA is exact for shared weights; (ii) trainer: build (previous, next, level) per pair under the
   pyramid schedule (level-8: previous = field idx−8, next = zeros; level-4: idx∓4; …); 60-epoch cosine, seed 20260716, batch 8, QAT 0.5, λ = 1.0
   (cl2's reproducing law), profile `bd1_pyramid`; (iii) receiver: decode in pyramid order (all level-8, then 4, 2, 1), buffer fields, render in index
   order; (iv) `encode_tail` mirror in the same order; (v) price exactly via cl2's pack → stage → encode ×2 (byte-identical) → receiver decode identity
   → wall-clock vs shipped 1,494.5 s; (vi) twin the winner; (vii) `tools/make_candidate_seal.py` for contest-CUDA. **Never dispatch Modal; MAIN fires.**
D. Memo `.omx/research/ddm_bd1_bidirectional_pyramid_context_20260905.md` (screen table · prediction residuals · ladder row · verdict_scope on any
   negative · frontier line last) · law registered · lane `lane_ddm_bd1_bidirectional_pyramid_context_20260905` · owed items as `## ITEM n — …`
   registered with `tools/extract_canonical_tasks_from_directive.py --directive <memo> --register-all --owner ddm_bd1`.

## OPTIMAL FORM
Reference form = the shipped HPAC integer mixer trained by cl2's reproducing law on the full n600 field, priced through the exact shipped path.
Mechanism delta = the `conv_future` branch + level embedding + pyramid order (the object under test). SCOPE deltas allowed: none — the screen is a
prior-transfer instrument, not a verdict on the mixer. A trained rung with fewer epochs / a subset field / a non-integer branch is a TOY: refuse.

## Compute, memory, disk, resumability (binding)
- Step B is CPU (≤ 8 threads; another arm shares the CPUs). **Training is Metal: the HPAC trainer measures 34.8–38.6 GiB of system availability
  (cl3, 2026-09-05) and md3's 49.6 GiB cell holds Metal until its receipt `.omx/tmp/codex_runs/md3_different_init_DONE.json.done` (~18:50Z).
  ONE Metal occupant at a time. Do NOT launch training until MAIN messages "Metal is free — fire"; MAIN sequences bd1 BEFORE cl3's rungs.**
  Launch through `tools/launch_detached_process.py` with a distinct `--done-receipt` (`.omx/tmp/codex_runs/ddm_bd1_<stage>.done`); poll with a
  background until-loop (foreground > ~3 min dies rc=144).
- Disk: trees/checkpoints on APFS ONLY (`/Volumes/VertigoDataTier/pact/ddm_bd1_bidirectional_pyramid_context/`, 29 GiB free; the boot volume has
  ~200 GiB free for small trees under `experiments/results/`); APDataStore is ExFAT — AppleDouble `._*` companions corrupt staged-tree census/sha
  (cl3 measured 68→85 files) — payload blobs only. KEEP THE PAYLOAD (every checkpoint, both streams, archive, decoded field; sha256 + bytes).
- Resumable from disk; `tools/subagent_checkpoint.py` every ~10 tool uses. Commits ONLY via `tools/subagent_commit_serializer.py --message "…
  [no-triality] [p0-ledger-ok]" --files … --expected-content-sha256 <file>=<post-edit sha>`; `.py` files two visible review passes
  (`tools/review_tracker.py mark-file`); NO co-author trailer; no `/tmp`; grep argparse before any flag. Read CLAUDE.md +
  `docs/operating_manual_craft_handoff.md` first. Label every number MEASURED / DERIVED / PREDICTED. End with the frontier line
  `cl2 S 0.14781744131049854 @ 179,982 B [contest-CUDA T4 n600]` plus any advisory candidate line labeled advisory.
