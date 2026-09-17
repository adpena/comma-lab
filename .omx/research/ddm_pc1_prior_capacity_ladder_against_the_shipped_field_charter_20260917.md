# ddm_pc1 — the PRIOR's capacity/rate trade: a capacity ladder trained against the shipped token field, byte-closed per rung (charter, MAIN 2026-09-17; Opus)

## Why (gs3 Addenda 69–72)
Move 55: S 0.13603403441098336 @ 179,255 B. Sub-0.12 needs −0.0160 S = −24,080 B at held distortion — 20 % of the token
field's ~119 kB. Every field-side admission and generation rule is measured at its knee (A65–A72; last three exact rows
−1.04e-4 → −4.6e-5 → −2.2e-5; joint admission +0.14 bar). The field is fixed by the scorer; what is NOT at a measured knee
is the PRIOR that codes it: the shipped prior (the context-mixing model in the archive's first section, ~11.6 kB coded,
plus the corrector/mixer families) was sized once and never traded against the tail it codes. The rate law of the
object: tail bits = the field's surprisal under the prior; a prior with more modelling power costs more coded weight bytes
and buys fewer tail bytes. Nobody has measured that curve on THIS field. ls1/ls2 measured the RECEIVER rung (context
model as coded at fixed capacity: 8,365 B short at that rung; "next = OBJECT") — this is the object rung: capacity.

## The rung (byte-closed at each rung; the field, renderer, carrier and receiver do not change)
1. Recover the shipped prior's training recipe by PROACTIVE RECALL (the lab has it: grep `.omx/research` and `experiments/`
   for the prior's producer — the section named in FORMAT.md of submissions/mrs7 as "a small network that predicts a
   five-class label", the `IntegerPrior` / `SparsePrior` loader in submissions/mrs7/inflate.py, the sealed tree's
   `runtime/rc2_hpac_semistatic_mixing.py` / `rc1_adaptive_model_sections.py` / `hpac_inference.py`, and the sj1/rlc1 pricer
   `experiments/ddm_sj1_rlc1_price.py`). Record the producer path + sha, the training data (the shipped field of move 55
   = the parse-back token planes), the loss (surprisal of the field under the prior), and the quantization/coding of the
   weights (the per-channel bit-depth codes FORMAT.md describes). If the recipe cannot be found, STOP and report what exists.
2. Baseline rung 1.0×: retrain the shipped architecture from the recipe on move 55's field; prove you reproduce the
   shipped tail bits within the measured noise (the instrument check; report the gap) — else the instrument is under test.
3. Ladder: 1.5×, 2×, 3× (and 0.7× as the control direction) of the prior's parameter count along the architecture's own
   width/depth knob; each rung: train to convergence on the field (timing smoke first; declare the budget; MLX/CPU on the
   host — no Modal); quantize + code the weights with the SHIPPED weight coder (real bytes); re-encode the shipped field
   under the new prior with the REAL coder (twins) → tail bytes; rung value = Δ(prior bytes) + Δ(tail bytes) in real bytes
   and S (25/37,545,489 per byte). Report the curve; the falsifier: no rung nets ≤ −1 bar (−25 B) after real coding — then
   the capacity axis is closed at this object (verdict_scope: formulation — the shipped architecture's width/depth knob).
4. The winning rung: the decoder must still decode the same field → raws IDENTICAL to move 55's (600/600) — this is a
   pure rate move; byte-close on move 55's tree (the prior section replaced; the receiver reads the new prior with its
   existing loader if the architecture knob is one it parses — if the receiver would need a code change, STOP and
   report: a receiver change is a different unit); twins; cold n600 parse-back; census; smokes; retention ≤ 2 GiB;
   NORMAL seal inheriting move 55's leg (`/Volumes/APDataStore/pact/ddm_pd8/SEAL_ddm_pd8_price_first_pass3_contest_cuda.json.decode_wall_clock.json`,
   1,072.3 s) — if decode time changes with prior size, say so: the leg must be re-measured by MAIN. MAIN fires.
5. Memo `.omx/research/ddm_pc1_prior_capacity_ladder_20260917.md`; serializer commits; lane `ddm_pc1_prior_capacity_ladder_20260917`;
   checkpoint `ddm_pc1`.

## Boundaries
No Modal, fire, packet, authorize_*; upstream/, PR trees, sealed trees, contract/receiver code read-only (the receiver's
loader may be READ to learn what it parses; not edited); pd*/jr*/psa*/mrs* stores read-only; work under
`/Volumes/APDataStore/pact/ddm_pc1/`; heavy steps via `tools/launch_detached_process.py --done-receipt <bare-name>`
(resumable, per-stage checkpoints, EMA if the recipe uses it); never SIGTERM running detached encodes; APDataStore ≥ 8 GiB +
retention (~22 GiB now); retain ≤ 2 GiB with sha256; never write Vertigo; the host scorer is not needed for rate rungs
(psa2's search runs at nice 10). Serializer commits, two review passes per .py, `[no-triality] [p0-ledger-ok]`; never a
co-author trailer or AI attribution. Label MEASURED / DERIVED / INFERRED / ASSUMED.

## OPTIMAL FORM
Reference form: the shipped prior's own training recipe and weight coder (recovered in step 1 — record path + sha); the
real tail coder for every byte. Declared deltas: the capacity knob only (SCOPE: 4 rungs + control); no architecture change
(a new architecture is a receiver change = another unit). Provenance pins: move 55 packet (095b81a1b); archive sha
ddadf998ddacab9b356b9b6a01a78c845ab3d6f643dd1e3cf8f37840ae550b8a; FORMAT.md of submissions/mrs7 (sha 66a005316e34b663) as
the plain-language description of the sections.

## Prior negatives accounted (operator 2026-08-15)
ls1/ls2 (receiver-rung context modelling: 8,365 B short — the OBJECT rung is this charter); cb1/ren2/rw1 (renderer refits
lose — this touches the prior, not the renderer); m143 (refit high-capacity sections — the prior IS one); constants-are-
poison (train from the recipe, never borrow a size); the container lottery (twins); pr19 (inherit the leg once; re-measure
if decode time changes).

Final message: the recipe provenance, the instrument check (rung 1.0× vs shipped tail bits), the ladder table (params,
coded prior bytes, tail bytes, Δ total, S), the winning rung's identity 600/600 and seal (or the fired falsifier with
verdict_scope), decode-time note, retained bytes, every boundary, serializer shas, ending with the current own-vehicle
frontier line.

<!-- # FORMALIZATION_PENDING: charter, not a finding; the ladder lands as EmpiricalAnchors on the rate law -->
