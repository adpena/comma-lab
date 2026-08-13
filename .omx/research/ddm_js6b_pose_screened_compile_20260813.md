# JS6B pose-screened compile — full-bank non-fire closure (2026-08-13)

## Outcome

The scorer-free screen completed over the full sealed JS6 census: **200/200 proposals HELD,
0 survivors**. No proposal was compiled, no candidate archive was created, no Modal job was
fired, and no sealed fire order was emitted. This is the required fail-closed branch of the
charter: a proposal whose screened bound cannot beat the zero-edit control is held rather than
converted into a fake candidate.

The strongest favorable arithmetic still failed:

- Best row under the *lower* measured pose-damage calibration:
  `js6_0072_f790b6493122`, 33 optimistic target flips / 5 semantic cells,
  **screened net ΔS = +5.25553385416666e-7** at zero rate cost.
- Best row under the admission *upper* pose-risk bound:
  `js6_0086_80a9667afaeb`, 6 optimistic target flips / 2 semantic cells,
  **screened net ΔS = +6.291373697916667e-5** at zero rate cost.

Both calculations favor the candidate: every target-support pixel is credited as a successful
Seg flip and archive Δbytes is set to zero. Adding real rate cost or receiver/scorer failures can
only make these rows worse. The verdict scope is **FORMULATION**: the sealed 200-row unprojected
JS6 semantic-cell bank on CP135 under the measured two-instance family envelope. It does not kill
the Q3 actuator family or jointly constructed pose-aware edits.

## Build delivered

`experiments/ddm_re1t_t4_sign_gate_worker.py` now has an explicit
`--retain-pose-vectors` mode. In that mode the same hash-pinned dispatch retains:

- the exact decoded candidate payload and the existing n600 SegNet argmax field;
- official PoseNet first-six vectors for GT, candidate pass 1, and candidate pass 2;
- every PoseNet input and full 12-value output batch used to derive those vectors;
- per-pair candidate error RMS and repeat-noise RMS arrays;
- distinct stage checkpoints through the final dual-axis result.

The flag-off path preserves the legacy input census, storage JSON, checkpoint name, SegNet-only
result schema, and execution branch. The flag-on path has a separate input census containing
`POSE_SCREEN_RESULT.json`, additional storage reservation, and a separate final checkpoint.
Remote complete-S adjudication remains forbidden; the returned retained fields/vectors are for
local whole-candidate arithmetic.

`experiments/ddm_js6b_pose_screened_compile.py` performs a resumable full-census screen. It pins
the JS6 index, family law, JS6 producer, and canonical Q3 source; applies the measured
`5.7e-6…3.4e-5 S / semantic cell` pose envelope; and uses the exact #837 6×12 float-yuv6
projector only as a diagnostic. Fourteen Q3 null/visible pairs materialized before or through the
lower-calibration gate and all **66,063,872 array bytes** remain under SSD custody; subsequent rows
were already non-improving under the lower measured calibration, so their large tensors were not
read or transformed. No materialized payload was discarded.

The exact retained consumer-store total is **66,731,452 B**, and every individual materialized Q3
array has its own SHA-256 record in `POSE_SCREEN.jsonl`.

## Screen law and honesty boundary

For each proposal:

`optimistic Seg value = target_mass × 100 / (600 × 384 × 512)`

`pose-risk interval = semantic_cell_count × [5.7e-6, 3.4e-5] S`

`screened net ΔS = -optimistic Seg value + pose risk + 0 rate cost`

Admission required the upper-bound result to be strictly negative. The lower calibration was
also nonnegative for every row, so no row reached the compile gate.

This is a local prior screen, not an exact candidate measurement. Q3 is exactly null only in its
float linear yuv6 constraint; the current JS6 proposal is an integer semantic-token actuator and
does not realize the projected RGB delta. Integer quantization, receiver nonlinearities, exact
Seg flips, exact Pose vectors, archive bytes, contest-CPU, and `upstream/evaluate.py` were not
measured here. A compiled survivor would still require the extended one-dispatch T4 worker and a
locally recomputed whole-candidate S before any exact-eval handoff.

## Retained evidence

- Consumer store:
  `/Volumes/VertigoDataTier/pact/ddm_js6b_pose_screened_compile_20260813`
- `FINAL_RESULT.json`: 6,559 B, SHA-256
  `125eec7421557ab1256ac5de2d7984efe372951cafe22cb3c9b42e88c50f9ed0`
- `POSE_SCREEN.jsonl`: 316,351 B, SHA-256
  `e78d8ab1152d4cafe2734dd1d0c9ef4edbb62ac347188409ed0c87c5d07bba43`
- `NO_FIRE_ORDER.json`: 567 B, SHA-256
  `1302547f6a5713b23129aeb19063941621e42408acd63367be65614a8f9f41e5`
- Resume repeat reproduced all three hashes byte-identically.
- `SEALED_FIRE_ORDER.json`: intentionally absent; `NO_FIRE_ORDER.json` records `FOLDED`, zero
  survivors, no candidate, and the only valid reopening trigger.

## RECALL EVIDENCE

Sources and queries searched beyond the charter seeds:

- `rg -l -i 'Q3|pose-null|pose vector|semantic cell|HP3|RC64' .omx/research`
- `rg -n '8.836e-4|integer actuator|quantization|603|5.7e-6|3.4e-5'` over the matched receipts
- `.venv/bin/python tools/list_canonical_equations.py --json`, filtered for pose/null/receiver
  equations
- `rg` over `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, task ledgers, and
  `.omx/state/main_hot_state.md` for the current vehicle, consumer, and authority status
- direct code inspection of the Q3 projector, JS6 proposal producer, JO1 HP3/RC64 closure, RE1T
  dispatcher/worker lineage, and JS1B retained scorer helpers

Findings beyond the seed list changed the plan. `ddm_upstream_factor_flattening_20260808.md`
records Q3 as float-null but not exact under the integer actuator. The JS4 annex in
`ddm_js1_reseal_skeleton_20260811.md` measured nonlinear Pose leakage `8.836e-4` even after
first-order nulling and found that quantization shrink did not cure the mechanism. The ET2 metric
amendment shows that projector metric choice changes Seg preservation but not kernel membership.
Therefore Q3 energy was retained only as a diagnostic; it was not used to lower the measured
family bound or to assert an exact-pose-null JS6 candidate. The two-instance per-cell envelope,
not a guessed threshold, remained the admission authority.

## Verification

- Full scorer-free execution: 200 unique rows, 200 HELD, 0 SURVIVOR, 14 retained Q3 diagnostic
  pairs, 0 candidate archives, 0 fire orders.
- Deterministic resume: the final result, screen JSONL, and non-fire receipt hashes were unchanged.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q
  experiments/tests/test_ddm_js6b_pose_screened_compile.py
  experiments/tests/test_ddm_re1t_t4_gate_prep.py`: **21 passed**; the only warning is the repo's
  pre-existing unknown pytest `timeout` config option.
- `.venv/bin/python -m ruff check` on the two implementation files and new test: passed.
- `.venv/bin/python -m py_compile` on the two implementation files and new test: passed.
- `tac.payload_retention_gate.check_no_measure_and_discard_payload` on the new screen and extended
  worker: no findings.
- No scorer slot, Modal dispatch, `upstream/` mutation, forbidden-file edit, index mutation, or
  destructive cleanup occurred.

## Authority and frontier

**MEASURED:** full sealed proposal census; exact proposal metadata; retained Q3 decomposition for
14 rows on `[macOS-CPU scorer-free]`; family-bound arithmetic from the measured T4 n600 RE1/JO1
calibration; output bytes and SHA-256 identities.

**NOT MEASURED:** candidate d_seg, candidate d_pose, candidate archive bytes, complete candidate S,
contest-CPU, a new contest-CUDA score, or exact `upstream/evaluate.py`. The pointer did not move.

Own-vehicle frontier: **CP135 S = 0.16195513827824176 @ 186,252 B
[contest-CUDA T4, n600]**, archive SHA-256
`6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6` — unchanged.

## NEXT_IF_RESUMED

- **FOLDED** — owner: MAIN; consumer store:
  `/Volumes/VertigoDataTier/pact/ddm_js6b_pose_screened_compile_20260813/NO_FIRE_ORDER.json`;
  fire trigger: do not fire this sealed bank. Reopen only after a receiver-realizable integer Q3
  actuator or retained candidate-specific Pose-vector evidence makes the same proposal's upper
  screened net ΔS strictly negative; then perform HP3/RC64 closure and seal a fresh dual-axis order.

## LIVE-HYPOTHESES

- A receiver-realizable integer Q3/CVP actuator may escape this formulation result because #837
  established that the float pose-null subspace remains Seg-reachable. It is plausible only if
  hard-round/relinearization closes the nonlinear and integer leakage that JS4 exposed.
- Joint, relinearized whole-candidate event construction may find edits whose Pose interaction is
  below the unprojected per-cell envelope. It remains plausible because the family law covers
  fixed unprojected HP3 semantic cells, not an optimizer that consumes retained Pose vectors while
  selecting and compensating the complete event set.

## DEAD-ENDS

- The sealed 200-row **unprojected** JS6 proposal bank on CP135 is closed at FORMULATION scope:
  every row loses even under the lower measured pose calibration and zero rate cost.
- Compiling or firing the least-bad row from this bank is closed: the best upper-bound net is
  `+6.291373697916667e-5 S`, so doing so would violate the charter's hold-before-compile law.
- Treating float Q3 projection energy as exact integer/receiver Pose nullity is closed: prior JS4
  evidence measured nonlinear leakage after first-order nulling, and these proposals do not carry
  a Q3-realized actuator.
- Seg-only provisional admission is closed for this family. Future candidates must return SegNet
  fields and official PoseNet vectors from the same decoded dispatch before whole-candidate
  arithmetic.
