# ddm_wp1 next if resumed

Status at handoff: code unit built and tested; row 4 measured; rows 3, 9, and
13 queued because their required TR1 cached surfaces do not exist in the searched
scope and this charter forbids scorer/render launches.

Do next:

1. Package/commit the WP1 diff with the serializer and post-edit shas.
2. If a scorer slot is assigned later, materialize a TR1 endpoint residual/flip
   atlas on SSD, then run vh1 rows 3 and 9 from cached arrays only.
3. For solve_project row 13, add a pre-training rendered-init receipt on the
   next non-resume solve_project launch or assigned read-only initial-checkpoint
   verifier. Do not infer from the missing live-chain telemetry.

Known measured artifact:

- `.omx/research/ddm_wp1_20260805/row4_g3_transfer_tr1_endpoint_delta.json`

Key row 4 result:

- EMA top64 partially transfers: `r_delta=+0.2672903589`,
  `r_abs_delta=+0.2348742495`, top64 mean delta `+0.0006207625`.
- Live does not transfer: `r_delta=-0.1450982096`,
  `r_abs_delta=-0.0398190339`.
- Verdict: no cross-basis/default subset surrogate; EMA-only partial yes.

Own-vehicle frontier remains:

`S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`
