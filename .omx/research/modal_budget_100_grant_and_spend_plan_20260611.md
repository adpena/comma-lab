# Modal budget: $100 authorized — grant + disciplined spend plan (2026-06-11)

**Operator grant (2026-06-11):** "You have Modal authorization up to $100." This SUPERSEDES the prior
Modal $30/mo cap in the CLAUDE.md cost-caps table per "Older $ caps … are superseded by a newer explicit
operator directive to fund or launch a named campaign." Claim lifecycle, provider import probes, artifact
custody, contest compliance, and CPU/CUDA axis separation remain mandatory (the cap rises; the discipline
does not relax).

**Purpose (GOAL §"Spend the Modal budget to BUY exact rows"):** spend it to MEASURE real byte-closed
candidates on contest hardware (Linux-x86_64-CPU + CUDA-T4). A fail-closed paired exact eval on a candidate
that beats the advisory bar is the RIGHT default (decide-don't-defer). Do NOT hoard it while the pointer
sits unmoved; do NOT blindly spend on un-de-risked substrates.

## The spend plan (priority order, each gated on a MEASURED-ready candidate)

1. **~$0.3 — R1+R2+R3 lossless entropy recode paired exact eval (IMMINENT, the first bank).** Gated on the
   running build+measure (a0adabcba) showing a REAL local byte-shave vs 177,169 (lossless → d_seg/d_pose
   unchanged → only rate moves). If it shaves bytes → fire the paired Linux-x86_64-CPU + CUDA-T4 exact eval
   → bank the lower frontier (a defensive lossless bank, GOAL-approved; 0.19110 → ~0.190xx). If no shave →
   floor confirmed, no spend. **This is the cheapest, highest-readiness exact row.**
2. **~$20–30 — the PR95-scale frontier-class / capstone training run (THE distortion-axis pointer-mover),
   GATED on a clean local de-risk.** Per MVP-first phasing + "Substrate MUST be at OPTIMAL FORM before paid
   dispatch" + the dispatch optimization protocol (Catalog #270) + the per-substrate symposium (#325): do
   NOT fire until a $0 local advisory confirms the recipe reaches the corrected bar (d_seg ~0.001 + pose
   collapsed + sub-frontier bytes). The de-risk is in flight: (a) the latent-heavy capacity test (Cool-Chic
   allocation); (b) the negatives-audit-recommended EMA-fix re-measure of base_ch=20 (vindicated as
   basin-reaching, under-trained not capacity-bound). When de-risked → fire the paid n600 PR95-scale run
   (base_ch=20 / V2 scorer-shaped renderer + the hinge default seg loss + Cool-Chic-efficient coding +
   stored pose) → byte-close → paired CPU+CUDA exact eval.
3. **Headroom (~$60–70) — subsequent paired CPU+CUDA evals** of the V2 candidates + any recode/composition
   that beats the advisory bar. Keep the dual-axis discipline (Linux-x86_64-CPU = leaderboard authority;
   CUDA-T4 = separate axis; never infer one from the other).

## Discipline (binding before ANY paid dispatch)

- Local FREE advisory first (MVP-first); the candidate must beat the advisory bar (local sub-0.19 is a
  high-confidence contest-sub-0.19 per the conservative-upper-bound memo, ε≈6e-6).
- `tools/claim_lane_dispatch.py claim` before dispatch (cross-agent coordination); terminal row after.
- The dispatch optimization protocol (Tier 1/2/3) + the canonical NVML env block for substrate trainers.
- For any contest PR: `scripts/pre_submission_compliance_check.py --contest-final --strict` + hosted
  archive URL + report.txt + the auth-eval JSON (per "Submission auth eval — BOTH CPU AND CUDA").
- Harvest-or-lose (Modal `.spawn()` result cache ~24h TTL) + the call_id ledger auto-refresh of the
  frontier pointer.

## Status

Pointer UNMOVED 0.19110. Two de-risk measures in flight (R1+R2+R3 recode → the $0.3 bank gate; latent-heavy
capacity → the PR95-run gate). No paid dispatch fired yet — the $100 unblocks the IMMINENT firing the moment
a candidate is measured-ready. First likely spend: the ~$0.3 R1+R2+R3 paired eval when the recode confirms
a byte-shave.
