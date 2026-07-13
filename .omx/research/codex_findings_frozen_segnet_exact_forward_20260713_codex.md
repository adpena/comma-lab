# Task #456 — exact frozen-SegNet forward CPU thread control

**Outcome:** **GO** for one registered local formulation: set the frozen Torch-fp32 SegNet forward
to one CPU thread only after the finite canary tournament and full paired exact-argmax gate pass.
On the first 64 receiver-realized witness pairs, the matched median forward fell from
936.312 ms at six threads to 312.677 ms at one thread, a **2.995x** speedup with zero flips in
12,582,912 argmax pixels. **NEEDS-MORE** for the requested 95% forward-cost kill, for any training
integration, and for transfer to contest CPU, CUDA, MLX, Metal, MPS, another host, another Torch
build, or unseen pairs.

`review_status: fresh-eyes-reviewed(1)-CLEAN` on the exact post-fix bytes. The recovery reviewer
re-ran the receipt validator and independently re-derived the speedup, gap, and zero-flip result.

`verdict_scope: registered formulation/substrate` — macOS arm64 CPU; Torch 2.12.1; fp32 frozen
SegNet; autograd-enabled input-dependent forward; first 64 receiver-realized pairs from the named
packet; six-thread baseline against the finite candidate set one through six. This is not a
family verdict for quantization, kernel fusion, or activation banking.

**STORES CONSULTED:** unified `tools/corpus_query.py` retrieval over research, equations, memory,
DAG, council, tasks, and docs; full `CLAUDE.md`; full `AGENTS.md`; the operating manual; the v7.5
and v8 canonical specifications; top-10 Claude memory entries; latest Codex findings and session
summary; latest T3 council and design memo; canonical frontier, lane, subagent, gradient-anchor,
dispatch, posterior, and blocker surfaces; the frozen-SegNet alternatives memo; the goldmine memo;
the measured block profile; the trainer call sites; the frozen model implementation and weights;
the receiver-realized raw bytes and their source packet; the #212 fused-R source surface; the
probe, tests, and final receipt. Deliberately did not load or actuate paid/cloud GPU state, the
protected run, the live trainer, or `upstream/evaluate.py`.

## Grounded result

- **MEASURED, fresh-eyes-reviewed(1)-CLEAN:** n=64 real pairs; 12,582,912 pixels; zero argmax
  flips; flip rate 0; identical reference/candidate argmax SHA-256
  `b7df9cbf5732c2f006e7f355e1c54062fc4461c318514ec8856756268e718f66`; zero maximum and mean
  logit delta across the admitted n64 comparison.
- **MEASURED, fresh-eyes-reviewed(1)-CLEAN:** six-thread median 936.312 ms; one-thread median
  312.677 ms; matched speedup 2.995x; forward-time reduction 66.605%; matched gap 623.635 ms.
  The composed run-local timing floor was 610.671 ms, so the measured gap exceeded it by 12.964 ms.
- **MEASURED, pre-registered-only:** the supplied profile anchor is 1,656.184 ms forward and
  456.318 ms input-gradient backward. It is a single-sample hook-instrumented measurement, not
  the matched denominator above.
- **DERIVED, fresh-eyes-reviewed(1)-CLEAN:** relative to that unmatched 1,656.184 ms anchor, the
  observed one-thread median is 5.297x lower and removes 81.121% of forward time. This does not
  satisfy the 95% kill target.
- **DERIVED, fresh-eyes-reviewed(1)-CLEAN:** if the 456.318 ms backward anchor stayed unchanged,
  the scorer slice would improve 2.747x. Exact-call validation and surrogate validation would
  receive a 5.297x anchor-relative multiplier. Idealized YOPO with backward removed would be
  6.756x faster than the old combined anchor, and this forward change multiplies its old-forward
  economics by 5.297x. Matched backward and end-to-end training speed remain **UNKNOWN**.
- **MEASURED, fresh-eyes-reviewed(1)-CLEAN, pair-0 diagnostic only:** CPU fp16 took 14,050.229 ms
  and flipped 21,852 of 196,608 pixels, a rate of 11.1145%. CPU bfloat16 took 14,160.986 ms and
  flipped 884 pixels, a rate of 0.449626%. `verdict_scope: pair-0 CPU dtype formulation only`;
  these negatives reject these two direct local casts, not quantization as a family.
- **MEASURED, fresh-eyes-reviewed(1)-CLEAN:** the positive same-arm rerun had zero flips and zero
  logit delta. The negative class-axis rotation flipped all 196,608 pair-0 pixels.
- **MEASURED, source inspection this session:** MLX failed closed because no Metal device was
  available. The located #212 code contains fused-R machinery, but no source-level fused-R plus
  SegNet-stem primitive was found. `verdict_scope: current checkout and current local substrate`;
  this is a blocker for this session, not a kernel-fusion family verdict.
- **DERIVED:** exact deep-activation reuse across changed witness inputs is not generally an
  exact forward. No banking claim was admitted without task #454's certified ball and a paired
  exact-argmax gate. `verdict_scope: uncertified direct reuse formulation`; the family remains
  open.
- **MEASURED:** `score_claim=false`; `promotion_eligible=false`; the canonical defensive bank is
  0.1880443979880752; the submittable contest-CPU pointer is 0.19108282419209976; both pointers
  are unmoved.

## Control law and falsifier

The control is an event-conditioned tested predicate with finite completion. Enumerate every
integer thread count from one through the six-thread anchor. Measure four evenly spaced canaries.
Choose the minimum canary median among zero-flip arms. Admit it only if it differs from baseline,
the full alternating-order n64 comparison has zero argmax flips, the timing gap exceeds the
composed p95-minus-p05 widths, and both canaries pass. Otherwise use the baseline. The named recess
measurement is `task456_even_quartile_canary_recess`; the full n64 gate is the authority. The
pre-registered kill criterion was any n64 argmax flip or a speed gain at or below the composed
timing floor.

## Eightfold and triality disposition

- P1: the receipt is the one empirical key; the canonical equation references it rather than
  creating a parallel result.
- P2: the comparison records the composed timing floor; across-seed and unseen-pair variance are
  **UNKNOWN**.
- P3: the argmax distortion allocation is exactly zero flips for admission.
- P4: same-arm rerun and class-axis rotation are positive and negative canaries.
- P5: baseline and candidate ran in the same process with alternating order on every real pair.
- P6: all 64 receiver-sequence pairs are retained in order; no per-frame cherry-pick admits the
  arm.
- P7: the zero-flip and above-floor falsifier existed before the final run.
- P8: the target floor is a 95% forward reduction. The measured 81.121% anchor-relative reduction
  leaves the target open; no claim closes the surface.
- Clause A: no new video-derived payload exists; the probe streams the one receiver-realized raw
  artifact and records its single geometric home and SHA-256.
- Clause B: no archive bits were allocated; marginal distortion-per-bit is not applicable.

This landing is `research_only=true`. It does not change a trainer lever, so the witness DSL is
deliberately unchanged under the no-live-trainer-edit boundary. The measured finding is encoded as
canonical equation `segnet_exact_forward_cpu_thread_control_v1`. The DAG FEED names the same
receipt. No external theorem or method was imported, so no literature citation is claimed.

## Custody and risk

The final receipt is
`experiments/results/segnet_exact_forward_20260713T020000Z/receipt.json`, SHA-256
`3b04a40c7c9e656cfc417dc60f2b73781e251a21fa02689a9e78523218ad3134`. The probe is
`tools/probe_segnet_exact_forward.py`, SHA-256
`2597500491e48059b1c4350973c23ebe4c934b13b86569aa05b124b98c97da2d`. The raw input SHA-256 is
`3819479cf6afc44b0366b01a1f1babfd25cd8fcc180825a24097e10b10d98975`. The model weights SHA-256
is `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`.

The primary risk is scope transfer: CPU thread oversubscription is host- and build-dependent. A
second risk is the single deterministic n64 spine, which gives no across-seed or unseen-pair
probability bound. A third risk is timing fragility: the speed gap cleared the conservative
run-local floor by only 12.964 ms. A fourth risk is that the anchor-relative multiplier is not a
matched full-loop measurement. None of those unknowns is promoted into score or contest-hardware
authority.
