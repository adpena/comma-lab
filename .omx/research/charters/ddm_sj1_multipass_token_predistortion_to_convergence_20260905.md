# ddm_sj1 — multi-pass token PRE-DISTORTION to convergence with joint admission + carrier re-solve (charter, 2026-09-05)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: Opus arm · Spawned by MAIN 2026-09-05 under the operator's standing GO. Axes: d_seg
`[macOS-CPU advisory, jg1 instrument, DALI GT lineage]`; d_pose `[cpu_torch fp32 authority, n600]`; bytes exact; `score_claim=false` until a T4 row.

## The object and the measured law (recall — do not re-derive)
jg1 (`ddm_jg1_joint_solve_20260819.md`): the stored tokens are 99.9985% identical to the DALI GT argmax — **95.9% of the seg debt is
render→re-segment loss**, so the actuator's job is PRE-DISTORTION of the stored labels, not better labels. Single-cell coordinate moves with
REALIZED acceptance (proposal → re-render frame 2p+1 through the receiver's own `SemanticTokenRenderer` → re-segment with the frozen SegNet →
accept only if flips fall) repair **1.50–1.55 cells per changed token**; block/dilation moves LOSE at every radius. One repaired cell = 1.273 B
= 10.18 bits → break-even **15.3 bits per changed token** under the shipped HPAC model; 95.6% of first-pass moves were under budget.
Token edits destroy pose ×387 because frame 2p is a photometric probe solved against the ORIGINAL frame 1 — and the carrier re-solve
(`ddm_jg5.refine_pair`, damped GN + ±2 polish) recovers d_pose to ~1.07× at ~0 bytes: **the two actuators COMPOSE**. jg5 ran ONE sparse pass over
n600 (573 pairs edited, 455 admitted by the Lagrange sweep) and produced the sub-0.15 crossing (−0.0078 S). The frontier body (cl2 repack,
179,982 B, S 0.14781744131049854) still carries **d_seg 0.00020139 = 23,756 flipped cells = 30,245 B-equivalent = 0.0201 S**. No second pass, no
richer proposal family, no convergence loop has ever run. GT lineage: T4 is DALI; the jg1 instrument reproduces the T4 leg to five figures on DALI
(0.00030307 vs 0.00030309) and PyAV is 1.43× off — the gate `ddm_up2.verify_gt_lineage` fails closed; use it.

## PRIOR-LAW PREDICTION (m38)
- **Passes 2–3 with the richer family** (the flipped cell itself + its 8 neighbours × the 4 alternative classes, then accepted-move-adjacent
  re-proposals; realized acceptance; pass until a pass repairs < 1% of remaining flips) repair **20–35% of the 23,756 remaining flips
  (4,750–8,300 cells)** at **1.3–1.6 cells per changed token**, median edit cost 5–9 bits under the shipped HPAC model → rate +3.5…+6.5 KB,
  seg −6.0…−10.6 KB-eq → **net −2,500…−5,500 B-eq = −0.0017…−0.0037 S** before pose; carrier re-solve leaves Δd_pose ≤ +5e-7 (+3e-4 S).
- **The shipped vehicle's PERSISTENT partition** (cells no single-cell 3×3×4-class move repairs) is predicted at **55–70%** of the 23,756 (the
  born vehicle measured 62–63%; md1/md2/md3) — measure it as a by-product; it is the ceiling of this formulation.
- **FALSIFIER:** if a full pass 2 over all 600 pairs repairs < 8% of the remaining flips (< 1,900 cells) → single-cell token pre-distortion is at its
  floor on this vehicle (formulation scope) — stop; if the admitted net ΔS ≥ −2e-5 → no candidate; record and stop.
Write the measured numbers beside these lines. Anchors go to jg1's registered law (find it under `tac.canonical_equations`; register
`token_predistortion_multipass_yield_v1` if none fits).

## What to do
A. RECALL first: `experiments/ddm_jg1_seg_solve.py` (the instrument, byte-exact forward model, DALI lineage gate), `experiments/ddm_jg5_pose_resolve_on_edited_renders.py`
   (Lagrange admission sweep over pose damage + the carrier re-solve), `experiments/ddm_b2e_edit_replay_admission.py`, memos jg1/jg5/afr1/fs2/hc2/md1,
   `ddm_up2` (verify_gt_lineage, render_frame0). Current field + renders + receiver copy: `/Volumes/VertigoDataTier/pact/ddm_cl2_hpac_prior_capacity_ladder/`
   (`inputs/`, `rungs/lambda_1p0/retained/receiver_copy_runtime/`, `parseback/`). `tools/subagent_checkpoint.py read --subagent-id ddm_sj1` first.
B. Reproduce the instrument on the frontier body: d_seg 0.00020139 on DALI at n600 (five figures) BEFORE proposing anything.
C. Pass 2: for every currently flipped cell, propose the richer family; realized acceptance through the receiver's own render + frozen SegNet
   (`cpu_torch` argmax is the authority; `coreml_cpu_fp32` bit-exact 3.28× may drive the search); price each accepted move's exact bit cost under the
   shipped HPAC model (the jg1 admission primitive) → keep the per-move ledger (pair, cell, old→new class, cells repaired, bits). Pass 3+: re-propose
   around accepted moves; stop on the < 1% rule. All 600 pairs every pass — scope by convergence, never by n.
D. Joint admission: jg5's Lagrange sweep over pose damage on the per-pair edit sets, then the carrier re-solve on every edited pair (fs2/jg5 solver
   verbatim), d_pose n600 cpu_torch. Exact bytes: re-encode the edited field through cl2's pack/stage/encode×2 path (the pricer is
   `experiments/ddm_cl2_hpac_prior_capacity_ladder.py`; the field changes, the model does not) + carrier Rice re-encode → ΔS exact-priced.
E. Candidate: archive through the shipped container path, receiver decode identity, twin (a second run of the deterministic passes reproduces the
   ledger), `tools/make_candidate_seal.py` contest-CUDA. **Never dispatch Modal; MAIN fires.**
F. Memo `.omx/research/ddm_sj1_multipass_token_predistortion_20260905.md` (pass table: proposals, accepted, cells repaired, bits, persistent partition;
   predictions vs measured; verdict_scope per negative; frontier line last); lane `lane_ddm_sj1_multipass_token_predistortion_20260905`; owed items as
   `## ITEM n — …` registered with `tools/extract_canonical_tasks_from_directive.py --directive <memo> --register-all --owner ddm_sj1`.

## OPTIMAL FORM
Reference form = jg1's instrument + jg5's admission + the carrier re-solve, all n600, DALI lineage, cpu_torch authority, exact bytes. Mechanism
delta = the proposal family (3×3 × 4 classes + adjacency re-proposal) and the convergence loop. SCOPE deltas allowed: a timing run on ≤ 10 pairs
to size the pass. A verdict from a subset of pairs, a PyAV-lineage d_seg, a modelled rate, or a truncated GN is a TOY: refuse.

## Compute, memory, disk, resumability (binding)
- CPU only (Metal is md3's cell, then bd1's trainer, then cl3). Start with ≤ 6 processes; pc1 (≤ 8) and bd1's screen share the machine; raise to
  ≤ 12 when `pgrep -f ddm_pc1` shows pc1 idle. Per-pass checkpoints of the move ledger (resume mid-pass).
- Trees on APFS only (`/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion/` — 29 GiB free — or `experiments/results/`); APDataStore is
  ExFAT — payload blobs only. KEEP THE PAYLOAD (edited fields per pass, per-move ledger, coefficients, streams, archive; sha256 + bytes).
- Detached launches via `tools/launch_detached_process.py` with distinct `--done-receipt`s (`.omx/tmp/codex_runs/ddm_sj1_<stage>.done`); waits as background
  until-loops. `tools/subagent_checkpoint.py` every ~10 tool uses. Commits ONLY via the serializer (`[no-triality] [p0-ledger-ok]`, post-edit shas);
  `.py` two review passes; NO co-author trailer; no `/tmp`; grep argparse first. Read CLAUDE.md + `docs/operating_manual_craft_handoff.md`. Label every
  number MEASURED / DERIVED / PREDICTED. End with `cl2 S 0.14781744131049854 @ 179,982 B [contest-CUDA T4 n600]` + any advisory candidate line.
