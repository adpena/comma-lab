# ddm_mt1 #978 multi-token T4 sign gate — NEGATIVE at formulation scope (verdict salvaged from failed-rc custody)

**Axis:** `[contest-CUDA T4 frozen SegNet/PoseNet; stratified-random n32 heldout]
COMPONENT-ONLY NON-PROMOTABLE` · score_claim=false · pointer_moved=false.

**Disposition:** the Modal call `fc-01M01CFQAS0HZTCJZS739HBH6X` (label
`ddm_mt1_978_t4_sign_gate`, ledger status `failed`, rc=1) CRASHED on a torch
deterministic-algorithms refusal (`RuntimeError: Deterministic behavior was
enabled ... but this operation is not deterministic`) — AFTER the measurement
completed and `FINAL_RESULT.json` was retained. ALWAYS-KEEP-THE-PAYLOAD paid:
all 3 arms × 8 batch receipts + verdict fields survive at
`/Volumes/APDataStore/pact/ddm_mt1_t4_sign_gate_20260814_custody/ddm_mt1_t4_sign_gate_20260814/`
(FINAL_RESULT.json + worker.log + STORAGE_PREFLIGHT.json). No re-fire needed
for the primary question; the rc=1 is an infra footnote, not signal loss.

## The measured verdict

`positive_t4_sign: false`. Gate decomposition (n32 heldout, n32 train,
seed 20260814978, Tesla T4, deterministic algorithms ON, tf32 OFF):

| gate | value |
|---|---|
| improves_cp135_seg | **false** |
| stronger_than_direct_c1_seg | true |
| zero_pose_damage | **false** (candidate−base pose MSE **+1.020e-4** ≈ 15× base d_pose 6.886e-6) |
| same_retained_exact_cp135_frame0_carrier | true |
| parsed_counted_model_consumed | true |
| no_changed_site_list | true |

**verdict_scope (verbatim from the receipt):** "single fixed-seed
stratified-random n32 heldout screen of the hidden-4,
max-support-mass-0.25 local simplex formulation" — FORMULATION-instance
negative, NOT a family kill. The family showed real signal
(stronger_than_direct_c1_seg=true) but does not transfer to heldout seg
improvement and carries material pose damage at this formulation.

## Routing consequence

The js1 joint line's re-route menu after js1c rho-fail + js8 REFUSED was:
trained-receiver (#982) OR coupled multi-token (#978). This screen closes the
#978 branch at its current formulation. **The remaining active route is #982 —
which is exactly the LIVE rx2 MC36-label HPAC line** (r5 CPU run + the wc2
MPS race). Convergence: all major seg/pose routes now flow through the
trained-receiver/HPAC program.

## Infra footnote (determinization queue)

The crash genus: `torch.use_deterministic_algorithms(True)` refusing a
nondeterministic op late in the worker (post-measurement). Future T4 workers
on this lane should either exempt the offending op
(`warn_only=True` on the summary stage only, never the measurement stages) or
route the op to a deterministic equivalent. Feed to the dt1 determinizer
program (#1047) as a named repeated-lesson candidate.
