# ddm_rr4 T4 verdict — FOURTH micro-campaign pointer move (2026-08-17)

**PASS.** Call fc-01M08HHS64QJNDV7M34E6AG96T, Tesla T4, n600, rc=0, 326 s poller-side (~$0.16).
All three pre-registered falsifiers hit EXACTLY (memo ddm_rr4_cuda_prob_reencode_20260817.md §6):

- seg 0.00029611 (== base, decode-identical) · pose 6.88e-06 (== base)
- S recomputed **0.15853325034789678** @ **181,161 B** [contest-CUDA T4, n600] — the projection
  realized to all 17 digits. Archive sha 35ac2b9beb7e6fa81075c7d84b5247d8d24c056fe49ce1cbd22a334bc9618956.

**Pointer:** effective_frontier 0.15959729295498598 → **0.15853325034789678** (ΔS −0.0010640426070892,
−1,598 B vs hv1 ep0634). Anchor appended via `posterior_update_locked` (accepted, [contest-CUDA],
linux_x86_64_t4; upstream_snapshot_sha256 null per hv1-precedent — the hasher refuses on the
`libSvtAv1Enc.so.2` symlink, standing #836 drift); pointer refreshed. Sub-0.15 gap: **0.00853**.

**What this proves (the campaign story, corrected twice):** the rr2 free-decode win (−1,598 B at
identical distortion, HPAC-context token recode) was REAL from the start. rr2's S 27.83 refusal was
STAGING INFIDELITY (hand-assembled tree fired instead of the proved receiver tree), not device-scoped
probabilities — rr4 falsified my device memo at source (both Modal receipts carry identical
corrected_quantized_logit_sha256 across T4-CUDA and macOS-CPU; the HPAC student is an integer lattice,
device-exact). The cure that landed the row was firing `candidate_runtime` UNMODIFIED — now the
deterministic default via tools/fire_modal_auth_eval.py (65e15db4e9).

**Composability (CLAIM, review-round-1 object):** the coder win is a post-hoc RECODE operator on the
token stream — any future hv1-lineage checkpoint should admit the same re-encode for ~−1,598 B.
UNVERIFIED on any second checkpoint; the adversarial review arm adjudicates. Sister hazard: the banked
micro-edit offsets (qs2 −4.375e-6 @ +34 B, re1 −1.207e-6 @ 0 B) were compiled against the OLD coder —
per the cross-regime constant-transfer genus they need RECOMPILE against the new stream before any
union fire.

**Endpoint closure:** claim terminal-closed (completed_endpoint_harvested), ledger harvested, closure
manifest REFUSED_MANIFEST on payload-manifest absence (the fire predated the canonical fire tool's
manifest stage; payload custody intact at /Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/).
Modal spend ≈ $7.0/$20.

STORES CONSULTED: MODAL_REMOTE_RESULT.json (this row) · ddm_rr4_cuda_prob_reencode_20260817.md ·
ddm_rr2_t4_refusal_device_scoped_decode_identity_20260817.md (superseded mechanism) ·
continual_learning posterior hv1 anchor row (promotion-path precedent) · er1 error ledger.
