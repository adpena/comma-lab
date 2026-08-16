# ddm_b2e SEALED LAUNCH TICKET — burn-2 train-for-editability

Status: **BLOCKED_ON_OBJECT_REPIN** (not READY_TO_FIRE_UNDER_STANDING_GO)
Prepared by: ddm_b2e arm · Owner of the FIRE: **MAIN** (governed Metal slot)
Landing memo: `.omx/research/ddm_b2e_landing_and_charter_repin_20260816.md` · Commit `ec83c44223`

The blocker is **not** budget, apparatus, or capacity. It is that two of the charter's four launch
pins name objects that cannot perform the job. A burn fired on the charter's pins as written would
train the wrong object and could not make the mp2 edits free. MAIN owns the re-pin; this arm did not
make it unilaterally.

---

## GATE CHECKLIST

| # | Gate | State | Evidence / blocker |
|---|---|---|---|
| 1 | Levers built at real form, default-off, byte-identical when off | **PASS** | `src/tac/pr130_lift/editability_levers.py`; 39 tests; zero-RNG-when-off pinned |
| 2 | F2 trains on the DEPLOYED grid | **PASS** | parity vs `sd1.quantized_tensor` at rtol=0/atol=0 |
| 3 | Admission instrument exists with a PRE-REGISTERED bar | **PASS** | `experiments/ddm_b2e_edit_replay_admission.py`; bar=50× pinned in code + test |
| 4 | Edit constructions are the shipped ones | **PASS** | replay reproduces ns1 §A to 4 decimals on all 3 edits + global L2 |
| 5 | Subset-bias law applied (m96) | **PASS** | seeded stratified n32 (ids 4→598), bias-tagged, cannot grant admission alone |
| 6 | Warm-start object identified and verified | **BLOCKED** | ep0634 verified (sha + optimizer state present) but it is the **cl1 token** object; the edits target the **semantic** object. Semantic warm-start artifact **NOT located**. |
| 7 | Host trainer identified for the levers | **BLOCKED** | charter pinned a non-trainer. Correct host is `src/tac/pr130_lift/train_semantic_quantized_resumable.py`; **MAIN must ratify** before wiring. |
| 8 | Resume/config-identity interaction verified | **NOT VERIFIED** | new flags change the config hash; `_assert_preregistered_config` + resume guards may refuse. Untested. |
| 9 | Derived schedule (~60–120 ep, per-stage checkpoints P0) | **NOT SET** | depends on gate 6/7 (steps-based vs epochs-based trainers differ) |
| 10 | Watcher configs + memory preflight at the REAL config | **NOT SET** | depends on gate 6/7 |
| 11 | Phase-B uncapped pose-solve harness (#850 cap-lift) | **NOT BUILT** | requires #850 + qs5 at source; this arm held them only via ns1 summaries. Declared unbuilt rather than guessed. |
| 12 | Governor admission | **NOT REQUESTED** | correctly not requested while 6–10 are open |

---

## WHAT MAIN MUST DECIDE (the re-pin)

1. **Object.** Confirm burn-2 trains the **`SemanticTokenRenderer`** (38-tensor semantic section), not
   the cl1 HPAC token model. All ns1 §A evidence, the FiLM protection list, and the mp2 refusals are
   measurements on the semantic object.
2. **Host trainer.** Ratify `src/tac/pr130_lift/train_semantic_quantized_resumable.py` as the lever
   host. Its real flags are already greppable; note `--bits` is hard-refused at ≠4, so F2 must enter
   as its **own** flag rather than by relaxing `--bits` (relaxing it would silently change the
   deployed packer contract).
3. **Warm-start artifact.** Locate the semantic checkpoint whose weights ship in the frontier archive,
   and confirm whether its optimizer state was retained (the wd3 3× pose-carry law). If the renderer
   is inherited PR130 intake rather than ours, "warm-start" means starting from the intake object
   under the off-the-shelf grant — a different plan with different provenance obligations.
4. **Whether the burn is scoped to the semantic section alone.** The archive has three sections
   (`hpac`, `semantic`, `carrier`). ns1's §A screen covers `semantic` only. Whether `carrier` (F4's
   22,032 B target) shares the ~94× anisotropy is **unmeasured** — F4 may be aimed at an untested
   regime.

## RECOMMENDED FIRST ROW (cheapest thing that decides the most)

Before any long burn: a **short warm-start run with F2 alone**, then `replay` + `admit` on the result.
F2 is the single lever whose mechanism is fully derived (train the grid you deploy), it is the one the
−823 B q3/q4 pool depends on, and the admission bar is pre-registered. If F2 alone does not move the
collapse factor materially off 1×, the regime thesis is in trouble early and cheaply — which is the
point of building the instrument before the burn.

## RETENTION / SAFETY

- All levers default-off ⇒ a lever-free run is byte-identical to the pre-lever trainer.
- `--out-dir` retention is opt-in and **not yet guarded against the SSD tiers**; add the storage-tier
  guard before any retention run (ALWAYS KEEP THE PAYLOAD + disk policy).
- Nothing in this ticket has been fired. No training, no Modal, no n600 scorer pass; the scorer slot
  was free at start and was never used.
