# ddm_so2 — the successor object, second design: every byte estimate from the LEARNED prior, every distortion estimate from the real render→scorer, one $0 rung that runs the real trainer (charter, MAIN 2026-09-13; operator full-authority GO + "creative and divergent"; codex astra xhigh)

## What so1 got wrong, measured (read `.omx/research/ddm_so1_successor_object_with_preimage_freedom_20260912.md` and gs3 Addendum 54)
so1 ranked "coarse field + regularizer/pattern table + residual" first on paper at 112–151 KB. Its own first rung refused at stage one: the
coarse plane ALONE coded to 169,100 B under a generic coder race — 1.79× the 94,292 B field gate and 1.42× the incumbent's FINE-field
stream (118,896 B). The lesson is not "coarse fields lose"; it is that a generic coder is not the incumbent: the incumbent's tail is a
LEARNED context model (the HPAC prior, 11.6 KB, trained on the field; `tools/train_ddm_cl1_hpac_capacity.py`, cl2's law) and any
representation must be priced by a prior trained on IT. Paper coders mislead by ≥ 1.8×.

## What the day settled about the live object (gs3 Addenda 47–59; read them)
Every axis is a measured conditional optimum at the exchange 6.66e-7 S/B: renderer weights (ren2) and realization both ways (rq1 — bytes buy
a position, not a spectrum); prior at its λ knee (cl2, dpi1); mixer (tmx1); coder/container (jt23); tail oracle 8,365 B short (ls1/ls2);
accuracy intercept (md1); carrier lattice/rank/basis/re-solve (pc2/pc3/br1/cb1/cr1); per-pair pose actuators (pp1: the hard pairs' pose is
a FLOOR outside every actuator — br1's free-field ceiling says only ~27 % of d_pose is reachable by ANY warp at the band-limit); the field's
single-token family (pass 8). The token field is the renderer's PRE-IMAGE (rendering the true partition scores 2.8× worse — ren1). Demand
for sub-0.12: −24,514 B at held distortion (d_seg 1.03e-4, d_pose 4.55e-6), or the equivalent along the exchange. THE CROSS (memory
`the-cross-two-objects-each-hold-one-half-of-sub012`, canonical memos) is the problem: byte feasibility (born/generator objects ~100–122 KB)
and reachable distortion (the live lineage) have a measured-empty intersection (n=4 with so1).

## Deliverable ($0 design + ONE $0 rung that MAIN fires; NO Modal; NO training launched by you)
1. A DESIGN MEMO `.omx/research/ddm_so2_successor_object_priced_with_the_learned_prior_20260913.md` proposing ≤ 3 constructions that hold
   BOTH halves, each with (a) a representation of the 600 label planes whose entropy under a prior TRAINED ON THAT REPRESENTATION is the byte
   estimate — cite what the trainer needs (a token-field cache; `--expected-cache-content-sha256`; the profile table) and how the candidate
   representation maps onto it (a coarser lattice, a different alphabet, a residual-vs-generator plane, a lane-separated plane, …); (b) a
   distortion MECHANISM that keeps pre-image freedom (the sj1 multipass machinery must be runnable on it: per-token proposals, realized
   composite verify) and a DERIVED distortion band anchored to a measured number on THIS object (e.g. rq1's cells-per-byte for grid changes;
   ren1's true-partition 2.825×; pass 6/7/8's repair fractions), never to the born object's; (c) the single falsifier that kills it.
   Weird is welcome (e.g. the field as a residual against the shipped renderer's OWN argmax of a coarse field; a two-alphabet field where
   Lane — 0.59 % area, 33.5 % bits, 40× over-represented in every damage class — is coded on its own lattice with its own prior; a
   pose-side split where the 12 hard pairs get their own object), but every number must be DERIVED from a cited measured constant.
2. **The ONE $0 rung, with an exact command MAIN can fire**: build the candidate representation of the CURRENT move-49 field losslessly (so the
   distortion leg is zero by construction and only the rate leg is tested), write the trainer cache for it, train the HPAC prior on it under
   cl2's law (60 epochs, Metal, ~1 h; `tools/train_ddm_cl1_hpac_capacity.py` — grep its argparse; the profile mechanism; per-stage checkpoints),
   and encode the field with the real coder (`experiments/ddm_sj1_rlc1_price.py`-class rail or the trainer's own packer; twins). PASS bar:
   (prior bytes + stream bytes + any side info) < the incumbent's 130,525 B (11,629 + 118,896) by ≥ 5 % — anything less does not change the
   cross. Give the command sketch against real tools (never invent flags), inputs by sha, the expected number, and the falsification threshold.
3. Serializer commit LAST, once (`REVIEW_GATE_OVERRIDE=1` for the .md; `[no-triality] [p0-ledger-ok]`; NO co-author trailer, NO AI
   attribution); a Git-object write denial (rc 17) is NOT a stop — keep the file in the working tree, leave the bundle, report; MAIN lands.
   Checkpoint `ddm_so2`. Read `docs/operating_manual_craft_handoff.md`. Never edit `upstream/`, the PR tree, sealed trees, contract code.

## OPTIMAL FORM
`# OPTIMAL_FORM_NA: design memo + one rung specification; nothing is built or trained by this arm; reference forms are the incumbent's own trainer and coder, cited by path and sha in the memo.`

## Prior negatives accounted (operator 2026-08-15)
so1 (paper coder ≥ 1.8× wrong); the cross n=4; md1/mc1/bd1/ls1-2/ren2/rq1/fb1 as listed in so1's charter; "small" does not predict distortion
(NR1); the born object's distortion 0.33 does not transfer as a number (ancestor law L18). A design that re-proposes any of these without a
new mechanism is not a candidate.

Final message: the ≤ 3 constructions with derived bytes/distortion/falsifier, the ONE rung with its exact command, memo path + sha, commit rc,
and `composition S 0.13632299781031237 @ 179,153 B [contest-CUDA T4 n600] (move 49)` unchanged.

<!-- # FORMALIZATION_PENDING: charter, not a finding -->
