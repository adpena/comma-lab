# re1 Round-1 T4 sign gate — PROVISIONALLY ADMITTED (2026-08-13, MAIN)

> ⚠ ERRATA 2026-08-13: the 'new floor 0.16195344' projection is RETRACTED — a −1.7e-6
> move is below the evaluate.py 8dp canonical resolution (±3.5e-6 declared per side).
> The −2-flip sign-gate measurement itself stands (worker instrument, exact counts).


## The row [contest-CUDA T4 frozen-SegNet argmax field, n600, batch=16 — COMPONENT-ONLY]
- Candidate: re1 Round-1 HP3/RC64-closed archive, sha
  `7be3eb94b229306278a6ed204e2c716d7aafa98f6f93c82a5d2be18822467dfa` @ 186,252 B
  (byte-equal to cp135; ONE semantic-cell change through the deterministic HP3 closure).
- Dispatch: `fc-01KZY70KXAEYDWBZ9EBA3JDGNG`, run-id `ddm_re1_round1_t4_gate_20260813`,
  883.4 s Tesla T4, ~$0.16 (#381). Retention COMPLETE on volume
  `comma-ddm-js1b-argmax-retained` (raw decode + all logits + seg inputs + argmax field
  sha `3f81969c…`) — payload law honored.
- **Result: candidate field NOT identical to cp135 — 4 changed pixels, NET −2 flips vs GT
  (34,968 vs 34,970). seg ΔS = −1.6954210069444444e-06 at Δbytes = 0.**
- Verdict: `PROVISIONALLY_ADMITTED_SEG_SIGN_GATE_POSE_MEASUREMENT_REQUIRED`
  (INSTANCE scope: this archive through runtime 63b93187…). `score_claim: false`.
- Receipt: `.../round_01_singleton_best/re1t_t4_dispatch/{LOCAL_ADJUDICATION.json,
  RE1T_T4_REMOTE_RESULT.json}`.

## What this refutes / establishes
- re1x's receiver-null hypothesis is REFUTED: the HP3 semantic-cell closure SURVIVES the
  exact public F26 decode + CUDA rendering + frozen SegNet. The technique class is LIVE.
- Per hg1's cure/break lens on the 4 changed pixels: net −2 (3 cures / 1 break or 2/0 —
  per-pixel split retained in the volume field for exact decomposition).
- First measured survival of a semantic-cell edit through this lineage's full realization
  chain — the js6 proposal bank (200 ranked coupled G1+EC1 proposals) rides the SAME
  closure path and inherits this existence proof.

## Next (fired)
Pose-certifying full `evaluate.py` row DISPATCHED via the canonical paired dispatcher
(run-id `ddm_re1_round1_full_auth_20260813`, JO1-proven recipe; CPU leg expected to fail
fast at the F26 CUDA refusal — CUDA leg is the row). Composed arithmetic IF pose holds at
cp135's 6.8856e-06: S = 0.16195513827824176 − 1.6954e-06 = **0.16195344** → new floor by
−1.7e-6. If pose collapsed (pz4r law, measured never assumed): honest dead row.
