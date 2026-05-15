# PR106 Format0D Score-Table Next Build

- score_claim: `false`
- dispatch_attempted: `false`
- ready_for_exact_eval_dispatch: `false`
- evidence axis: Kaggle P100 CUDA score-table proxy plus local PacketIR materialization audit
- source format0C archive: `56cdd10bdc43708f2021458d0877b6c5e5a065a482a61280e727078462aed8e7`
- source archive bytes: `186327`
- score table: `reports/raw/kaggle_ingested/kaggle_pr106_format0c_latent_score_table_repair2_20260515T204229Z/pr106_latent_score_table/latent_run/score_table/score_table.npy`
- score table SHA-256: `cbea7c7578a275044f162ff72b2fb502cac5ef4aa0b63eff7627da553bf8e935`

## Finding

The harvested score table has real optimization signal but the current
`format0C` grammar cannot express it.

- score-table shape: `[600, 113]`
- original strict-improvement rows: `570`
- best improvement mean: `0.0014046559808775783`
- best improvement max: `0.004085123538970947`
- materialized `format0C` strict-improvement rows after grammar filter: `0`
- pairs whose original best row is incompatible with `format0C`: `570`
- compatible score-table candidates scanned: `1337`
- incompatible score-table candidates scanned: `66463`
- materialized archive changed payload: `false`
- materialized archive SHA-256: `56cdd10bdc43708f2021458d0877b6c5e5a065a482a61280e727078462aed8e7`

The local materializer now fails safe by filtering incompatible entries instead
of producing a malformed packet. That proves the current path is a no-op, not a
negative result for the score table.

## Next Build

Add explicit `format0D`: `format0C` base plus an additive extra-slot PR101
ranked/no-op stream.

Expected layout:

```text
0xFE
0x0D
pr106_len:u32
pr106_bytes
base0c_sidecar_payload: 511 bytes
extra_payload_len:u16
extra_payload: PR101 ranked/no-op stream over 600 extra slots
extra_framing_meta: <HHBB>
```

Runtime semantics: apply the decoded `format0C` correction first, then apply the
extra correction stream as a second additive latent pass.

Expected overhead is about `550` bytes, or about `0.0003662224` score on the
rate term. Exact promotion still requires paired `[contest-CUDA]` and
`[contest-CPU]` auth eval because the table is proxy evidence.

## Stop/Continue Rule

- Do not rerun `format0C` score-table materialization for this table. It is a
  grammar no-op.
- Continue with `format0D` only if runtime byte-mutation proof shows the extra
  stream changes decoded latents/output before paid exact eval.
- Promote only if exact paired eval beats the source archive on the same axis:
  `[contest-CUDA] 0.20631638661580989` and `[contest-CPU] 0.22776488386973992`.
