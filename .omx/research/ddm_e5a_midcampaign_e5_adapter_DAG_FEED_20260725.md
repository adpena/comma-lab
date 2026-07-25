---
schema: ddm_e5a_midcampaign_e5_adapter.dag_feed.v1
feed_id: FEED-603-e5a
date_utc: 2026-07-25
research_only: true
score_claim: false
main_review_required: true
---

# FEED-603-e5a — mid-campaign checkpoint to E5 packet

```text
accepted campaign checkpoint NPZ
  theta + optimizer + cursor + realized-archive identity
  19,723 B · SHA 043c2a8b...
          |
          | typed config + canonical resume-registry restore
          v
E5A checkpoint adapter (not an exporter)
          |
          v
receiver-closed WS1 W_joint live-resume state
  138,813 B · SHA 2a2c0367...
          |
          | semantic parse; generic manifests/wrappers move to inflate.py
          v
7 irreducible video-derived streams
          |
          | LA1 exact whole-stream race; every frame canonical + parse-back exact
          v
127,951 B selected frames -> 128,001 B semantic bundle
          |
          | existing E5 compiler only
          v
130,101 B archive.zip · SHA fb69964d...

The exact candidate bytes are preserved at
`/Volumes/VertigoDataTier/pact/ddm_ct1_campaign_telemetry_encode_20260725/e5a_runtime/candidate_packet/archive.zip`.
The repository intentionally commits the deterministic inflater and
`packet_cold_store_receipt.json`, not the gitignored archive bytes.
          |
          +--> repository parse-back -> SHA 2a2c0367...
          +--> embedded inflate.py parse-back -> SHA 2a2c0367...
          +--> 19 preserved render stages -> raw SHA 67666eb9...
                                           |
                                           v
frozen n600, batch32, threads4
  d_seg 0.06974277072482639
  d_pose 35.499820809591
  S_adv(packet bytes) 25.902302117302376
```

## Triality

- DSL: the hash-bound adapter config restores the canonical checkpoint resume
  registry and materializes only the checkpoint-declared
  `live_resume_state`; the E5 config accepts the resulting WS1 state through
  its existing two-stream grammar.
- DAG: checkpoint -> adapter -> receiver-closed state -> LA1 semantic bundle ->
  E5 packet -> two independent parse-backs -> preserved render stages -> exact
  advisory scorer.
- Equations: `B_packet = B_manifest_home + B_bundle_home + B_central_dir =
  1915 + 128045 + 141 = 130101`; `S_adv = 100*d_seg +
  sqrt(10*d_pose) + 25*B_packet/37545489 = 25.902302117302376`.

## Unified solver wire-in

- Sensitivity map: the exact n600 endpoint is bound to the live-resume state
  SHA and packet SHA. It must not be substituted for the campaign's EMA
  step-50 row.
- Pareto constraint: G4 is a custody/rate gate only. The packet is 688 bytes
  below the 130,789-byte coordinated reference, but its advisory objective is
  not promotion authority.
- Bit allocator: use 130,101 as the closed packet basis. The 128,254 LA1 figure
  remains a component-level prospective comparator and cannot fund 1,847
  nonexistent bytes.
- Cathedral/autopilot: R6 may consume
  `PASS_E5A_MIDCAMP_CHECKPOINT_TO_PACKET`; campaign fire remains disabled and
  MAIN review is mandatory.
- Continual-learning posterior: the earlier blocker
  `R6_BLOCKED_E5_MIDCAMP_CHECKPOINT_ADAPTER_ABSENT` is dissolved for checkpoints
  carrying canonical optimizer arrays, cursor custody, and an exact realized
  archive identity.
- Probe disambiguator: the receipt retains three distinct rate surfaces
  (selected frames, semantic bundle, complete packet). Any future coder change
  must rerun all three and exact receiver parse-back; no surface may be inferred
  from another.

## Exact custody

- G4 receipt:
  `.omx/research/ddm_e5a_midcampaign_e5_adapter_20260725/g4_receipt.json`
- Exact-packet cold-store custody:
  `.omx/research/ddm_e5a_midcampaign_e5_adapter_20260725/packet_cold_store_receipt.json`
- R6 receipt SHA:
  `f6a2b06ce120ae445d113f8fa1ff6d533574d7f12f377d40073507ff5a29cfe2`
- Export receipt SHA:
  `8da121be9bf3180e4dc0b1252cf2c2e6c0c2d57afbeae213680bd1d613d9b9ce`
- n600 receipt SHA:
  `2c4277b040157c359974a1de5d7cc07752c29d1b45aafd68ecc942e8ef12cacb`

The live campaign and its launcher/consumer surfaces were read-only; the R6
receipt proves the source tree fingerprint is byte-identical before and after.
