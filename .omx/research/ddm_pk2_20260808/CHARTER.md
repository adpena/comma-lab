# ddm_pk2 — Candidate-A DECISIVE surface-fit: PR130-style CPR1 pose carrier on OUR ep854 surface

**Fire-order source:** PK1_FINDINGS.md "QUEUED-WITH-FIRE-ORDER — run the A surface-fit measurement
before any #984 promotion or rejection based on PR130-style pose." pk1 already proved custody:
CPR1 release archive SHA 0491d5df84fc70b6... matched; vendored carrier decodes/re-encodes
losslessly (600x12 coeffs + 12x3x24x32 basis fields, 23,054 B inside the 191,052 B archive).
Candidate B (terminal 6-eq GN on ep854) is a FOLDED formulation-scoped negative (d_pose
160.5->116.5, far outside the 0.0025 tube) — do NOT re-run B.

**The task (pk1's named remaining step, verbatim intent):** fit/train the PR130 neutral-gray
semantic-pose carrier against the ep854 cell_drop50 surface frames + PoseNet target cache on
seeded stratified n>=120 (seed 20260728 lineage; stratified-random per m88/m96 — NEVER prefix),
parse back the counted CPR1-style carrier (byte-close the counted section), then score d_pose
AND d_seg through CPU-torch REAL scorers on the same n. Report: realized d_pose vs the 0.0025
tube + PR130's external 2.33e-5 reference · d_seg collateral vs the ep854 base (frame1
byte-identity check like pk1-B did) · counted carrier bytes · joint composed dS on the banked
ep854 x cell_drop50 surface (#827 arithmetic). Off-the-shelf grant applies (PR130 repro-repo
code directly reusable, memory pr130_code_off_the_shelf_authorized_20260806) — honesty half
UNCHANGED: borrowed_substrate_accounting section REQUIRED.

**OPTIMAL FORM:** reference = PR130's own CPR1 training recipe from the repro repo
(/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo) — use THEIR fit
procedure at THEIR hyperparameters where documented; n>=120 stratified = SCOPE reduction
(legal); any mechanism delta from the PR130 recipe must be declared TOY-BRACKET. Provenance
pins (path+sha) for the carrier codec + init + caches.

**Boundaries:** CPU-only (CPU-torch scorers + CPU/MLX-cpu fit), NO Metal (ARM-VEH owns it),
scorer passes ARE in scope (this IS the decisive scorer leg). Read-only toward live run dirs.
Artifacts: .omx/research/ddm_pk2_20260808/PK2_FINDINGS.md + receipts.jsonl + fitted carrier
+ byte-close receipt. verdict_scope on every negative; score_claim=false, [macOS-CPU advisory].

**Discipline:** serializer + POST-EDIT --expected-content-sha256 per file; tags
[no-triality] [p0-ledger-ok]; review_tracker x2 per .py; NO Claude/AI attribution or
Co-Authored-By trailer. If serializer hits sandbox git-perms, write artifacts + say so.
