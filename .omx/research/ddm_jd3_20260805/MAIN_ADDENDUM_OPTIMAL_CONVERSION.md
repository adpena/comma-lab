# jd3 mid-flight addendum from MAIN — OPTIMAL-CONVERSION steer (operator, 2026-08-05)

If your receipt dir exists before you finish, consume this; else MAIN folds it at harvest.

Operator (during your flight): "we need to make sure that all convert as much as possible to
optimal." Applied to your build — two BANKED-BUT-UNCONVERTED assets sit exactly on your surface,
plus one discipline warning:

1. **SL2 solved-frame TEACHER surface (banked, never consumed).** The trainer's distill path
   (`--distill-*` flags, `distill_mm` per-pair logit memmap, `kd_logits` form in pair_loss) is
   BUILT and default-off. SL2 persisted 32 seg-SOLVED frame_1 NPZs (d_seg 0.0010) with full SHA
   custody at `/Volumes/VertigoDataTier/pact/ddm_sl2_20260805/sq2_persisted_frames/`, and its
   NEXT_IF_RESUMED names exactly this consumer: "use the persisted SQ2 frame bytes … as the
   teacher/custody surface for a constrained/joint descent carrier." A solved-frame distill term
   during pose descent is a REALIZED-space seg anchor — potentially stronger than the hinge,
   complementary to your realized hold. IF cheap to wire (the machinery exists; the work is
   materializing teacher logits for the 32→n600 coverage question), include as a SEPARATE
   default-off arm/flag with its own provenance — do NOT fold it into the primary v3 config
   (see 3). If not cheap, name it in NEXT_IF_RESUMED as the v4 rider with what's missing.
2. **PE3 conditioning (built, never-fired).** `--pe3-conditioning-mode conditioning_only` is in
   the trainer (74,408 B geometry prior, #941: conditioning-only for TR1). NAMED v4 rider —
   do NOT enable in v3.
3. **Single-variable discipline (binding).** v3 already changes TWO controller elements
   (realized hold + stage-scoped EMA) vs v2 — both CURES of measured defects, jointly justified
   because they repair the instrument/basis, not the physics. Any TREATMENT addition (distill
   teacher, PE3, margin-weight) beyond the cures must be a separately-flagged arm or a named
   v4 rider, never silently stacked. The en1 margin-weight A/B fire-order ("next clean window
   boundary") is explicitly DEFERRED past v3 by MAIN — record it in NEXT_IF_RESUMED, don't fire.

Also consume if useful: MAIN is running a $0 both-bases (live vs EMA) seg sweep over the w4m
chain's preserved checkpoints — receipt will land at
`/Volumes/VertigoDataTier/pact/ddm_jd1_20260805/chain_both_bases_sweep.json`. If it lands before
your resume-start adjudication, use it (it may reveal a better live-basis start than the two
charter candidates); if not, proceed with the charter's two.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
