# Composition algebra — the coherence law for multi-action admission (V3-binding)

UTC 2026-06-10 · claude · operator trigger: "might need a little more math to ensure coherence and
synergy." The lanes now emit composable candidates (Class-3 repair atoms ∘ selector switches ∘
decoder reallocation ∘ S12 preimage ∘ S8 compose). THE LAW admits on SOLO ΔS; composition needs
its own algebra or the greedy waterfill silently double-counts.

## 1. The domain rule (the most important line)
**Compose in the DISTORTION domain (d_seg, d_pose, bytes), never in the score-delta domain.**
The score map S(d_seg, d_pose, B) = 100·d_seg + √(10·d_pose) + 25·B/N is nonlinear in d_pose:
score-deltas do NOT add. For any composed candidate, recompute S from the composed (d_seg, d_pose,
B) — never sum ΔS_i. Corollary (the pose-concavity surprise): ∂S/∂d_pose = 5/√(10·d_pose) GROWS as
d_pose falls, so successive pose improvements are worth MORE per unit, not less — summing solo ΔS
UNDERestimates joint pose value (super-additive benefit), while seg saturation (below) OVERestimates
joint seg value. The two biases do not cancel; only domain-composition is correct.

## 2. The seg additivity theorem (and its boundary)
d_seg is a mean of per-pixel disagreement indicators. Two repairs fixing DISJOINT pixel sets compose
EXACTLY additively in d_seg — *up to network coupling*: EfficientNet's receptive field means a
perturbation at x moves logits at y. So the practical law:
- **Disjoint supports separated by > the SegNet receptive-field radius** (measure once: perturb at x,
  find the logit-change support radius — a $0 one-off) ⇒ additive in d_seg, admit compositionally.
- **Supports closer than that, or overlapping** ⇒ the pair needs a measured joint row (the Class-4
  commutator: comm(a,b) = ΔS(a∘b) − ΔS(a) − ΔS(b)) before joint admission.
- Same-class-region atoms can SATURATE (both fix the same flips): never admit two atoms claiming the
  same flipped pixels on solo rows.

## 3. Shared-budget ledgers (the cone is a currency, spend it once)
Per-pixel joint_cone_radius is a BUDGET: two atoms each inside the radius can jointly exceed it.
The planner must DEBIT the cone per admitted atom (remaining_radius -= |δ|) and re-screen later
atoms against the residual. Same for: per-region margin (Class-3), selector/codebook bytes (shared
coding contexts make Δbytes sub-additive — price the JOINT stream), and the #49 zero-weight set.

## 4. Pipeline ordering (non-commuting operators have ONE canonical order)
render → apply pixel atoms (supports on the VISIBLE set only — atoms in the zero-weight set are
no-ops by theorem) → S12 preimage postprocess (fills the invisible set) → encode → archive.
Preimage-before-atoms destroys atom pixels; atoms-in-null-space waste bytes. The order is canonical
and enforced (a composition that reorders these must re-prove).

## 5. The conservative greedy (what V3's live loop runs)
Submodular-style sequential admission: admit the best candidate by measured solo row → MATERIALIZE
→ exact re-measure the composed base → all remaining candidates' rows are STALE vs the new base
(stale_for_base=true) → re-screen cheaply (advisory) → next admission. Never batch-admit solo rows.
Exception: candidate sets PROVEN independent by §2's separation law + §3's disjoint budgets may
batch within one exact re-measure.

## 6. Cross-vehicle composition (S8/SNeRV/frontier)
When a carrier swap changes the BASE (S8 compose, decoder reallocation), every atom row minted on
the old base is invalidated (base_archive_sha256 mismatch ⇒ stale). Atoms are re-screened on the
new base before ranking. This is why #51 (Class-3 host ratification) correctly waits for Branch-B's
byte verdict.

## Enforcement
- V3 admission: composed candidates carry composed-(d,B)-recomputed scores; solo-sum rows REJECTED
  at ingest (schema: `composition_method ∈ {measured_joint, domain_recompute, proven_independent}`).
- The receptive-field radius measurement = a one-off $0 tool run; until measured, NO compositional
  admission of same-frame atoms without a joint row.
- The cone-ledger debit lives in the #46 waterfiller (it already tracks budgets; add the debit).
Consumers: #51 (Class-3 ratification), #30 (the live waterfill loop), Branch-B round-3, the
decoder-axis candidates, any S8 follow-on.
