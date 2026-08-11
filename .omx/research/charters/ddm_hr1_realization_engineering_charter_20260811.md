# ddm_hr1 — REALIZATION ENGINEERING: the learned-prior + round-trip-in-loop stage SPEC (operator 08-11)

## Mission (operator verbatim: "We might need to incorporate a learned prior into our hybrid or
do other clever engineering techniques for realization. Remember the importance of doing
upstream and r during training and such in training... keeping the round trip in all transforms
and everything in mind.")

hy1 measured the hybrid head OPEN: the C1 solved partition rides the F26 wire at +11 B with
100% grammar representability — the ONE unknown is REALIZATION (learned renderer → R → uint8 →
SegNet survival), gated at 82.8236457% of the C1 seg gain for sub-0.15. Your job: design that
realization stage at OPTIMAL FORM (per the charter-time optimal-form law) so js1's fire-order 3
executes a derived config, not a naive first pass. DESIGN + $0 precedent derivation ONLY — no
training launches, no scorer, no Modal in this arm; the stage itself fires at the ps135
terminal under the js1 chain.

## THE BINDING LAW (operator, encoded in js1 AMENDMENT 2 — restate and design to it)

Every learned adaptation trains through the EXACT composite round trip IN-LOOP: render →
bicubic↑874 → uint8 (STE) → bilinear↓384 → scorer preprocess, differentiable yuv6 wherever pose
gradients flow (tac.differentiable_eval_roundtrip = canonical), eval_roundtrip=True semantics.
Optimizing pre-round-trip fields = the measured 2-11× proxy-auth trap = REFUSED. Sisters: fd2's
uint8-STE+R-in-loop (SPEC_tr1), [[realization_gap_is_fixable_through_actual_S_R_GT_20260806]],
#149 pre-R placement, #855 (MLX conv adapter flips 76 argmax px — any in-loop scorer surrogate
needs bit-identity or declared-advisory discipline), #903 (upsample-VJP × Adam sign(g) class).

## Ordered work

1. **RECALL FIRST — derive the precedent base from receipts, never memory:** (a) dw1
   (`.omx/research/ddm_dw1_qa75_distill_window_20260730.md`) — the solve-field distill window
   that used THESE C1 frames as teacher on TR1: what did it MEASURE (direction, magnitude,
   config, failure modes)? That result is the learned-prior route's in-repo precedent — extract
   its lessons and its verdict scope honestly. (b) QA75/KD-#74 lineage + rv1's reopening.
   (c) v14's per-stage realization ladder (WHERE realization died: paint/AA/uint8-amplitude vs
   grid) — the diagnostic instrument the stage must re-run on the PR135 renderer. (d) jd-line
   joint-descent machinery (compute_pose in-loop, seg-hold floors, EMA law). (e) fd135/eh1/pi1:
   the PR135 learned renderer's architecture, training recipe, and how tightly its decoded
   argmax follows its own token plane (any retained receipts on renderer-vs-token fidelity).
   (f) hb1/hb2 HPAC-on-our-labels (the compression learned-prior half, machinery + configs).
2. **DESIGN the 4-arm realization race at optimal form** (js1 Amendment-2 item 2): frozen-decode
   baseline · their-renderer fine-tune on the C1 plane (scorer-in-loop, round trip in-loop,
   counted Δbytes of adapted weights priced per rule-118) · light adapter/LoRA-class head
   (counted) · joint token+renderer descent. Per arm: exact objective (margin/CE form per the
   #63 lesson), optimizer + EMA (derived via the ema_decay_run_geometry_v1 LawRef, never flat
   0.997), schedule (event-driven per #686, no PR95 cargo), memory-preflight projection at the
   REAL config, resumability P0 (per-stage checkpoints), DSL-compiled config stubs, seg-hold +
   pose-guard floors, verdict cadence n600 chunked. Include the v14-ladder DIAGNOSTIC PASS as
   stage-0 of the race (localize WHERE the direct decode loses the solve before training
   anything).
3. **The SECOND learned-prior reading:** a learned entropy prior over generator/level-set
   coordinates for the conditional fallback stream (hy1 direction C) — design its
   equal-parameter control per sr1's surviving route; note hb1's machinery reuse.
4. **PRE-STAGING ASSESSMENT:** what is safely preparable BEFORE the ps135 terminal (harness
   code, DSL levers, configs, memory preflights) vs what must wait (anything scorer-lane or
   compute-heavy beside the live solve — the governor/admission rules bind; name the
   projected RSS of each pre-stage item and refuse anything non-trivial).
5. **Deliverable:** `.omx/research/ddm_hr1_realization_engineering_20260811.md` — the stage
   SPEC js1 Amendment-2 item 3 consumes, w/ NEXT_IF_RESUMED; serializer commits (post-edit
   --expected-content-sha256, [no-triality] [p0-ledger-ok], --no-co-author).

## Boundaries

Scorer-FREE, no Modal, no training launches, read-only on the live solve store. Public-PR
intake clones READ-ONLY. Honest borrowed-vs-ours accounting (their renderer/grammar vs our
solve/metrics/round-trip machinery). Any adapted-weight byte delta is COUNTED (rule-118).

## OPTIMAL FORM

Pins: hy1 memo commit 3a3825ad56 + solved tokens sha
2b0bdfc38a131ab1ebc3a2c2153a79b1ba23be0037adda66d01ab56f29f4fed5 + stream sha
9def0a4ba849757d473ba2a23cb0fd5370f2566355e5a5cfd398f847349636e8 · js1 charter (AMENDMENT 2)
`.omx/research/charters/ddm_js1_global_joint_solve_charter_20260810.md` · dw1 memo
`.omx/research/ddm_dw1_qa75_distill_window_20260730.md` · SPEC_tr1 (fd2 lesson encoded) ·
`src/tac/differentiable_eval_roundtrip.py` · v14 memo (realization ladder). SCOPE = the full
4-arm race designed at n600; no toy-scale configs presented as the stage. PRIOR-LAW PREDICTION
(derived fresh, verifiable by the recall): (a) dw1's retained receipts show solve-field
distillation moved d_seg TOWARD the teacher on the TR1 vehicle (direction check) — if dw1
measured NO transfer even in-vehicle, the learned-prior route inherits a negative precedent
that must be re-scoped BEFORE the stage fires, and the spec must say so; (b) the
round-trip-in-loop requirement changes the naive fine-tune design in ≥3 named places
(uint8-STE, differentiable yuv6, R-in-loop placement) — if the naive design already satisfies
all three, the law was already structurally embedded and the spec documents where. FALSIFIER
for the race design itself: if recall shows the PR135 renderer's decoded argmax follows its
token plane at ≥99.99% (retained receipts), direct survival is near-certain and the race
collapses to baseline+verification — say so and shrink the stage honestly.
