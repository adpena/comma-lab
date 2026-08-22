# VP1 top-2 stub — bank the split q4 model pack in a public archive

**Disposition:** QUEUED-WITH-A-FIRE-ORDER. **Axis:** scorer-free lossless byte/receiver proof.
**Score claim:** false.

## Mission

Materialize the measured unchanged-q4 model split in the public PR130 receiver and retain the final
archive rather than transferring arithmetic. The raw semantic, pose, and HPAC sections must restore
exactly. The measured model payload is 73,065 B versus 73,968 B shipped: `−903 B`, exact rate value
`−0.000601270634669321 S` before final-container interaction.

## Inputs and pins

- Base/reproduction: `.omx/research/ddm_pr130_reproduce_20260809/PR130_REPRODUCED_HERE.md` @
  `12031094d951df36bbd82459fefddc3211939ed5`.
- Exact section race: `.omx/research/ddm_pr130_reproduce_20260809/RATE_AXIS_LOSSLESS_RACE.md` @
  `0eea12ac3554ff67f0a0768881f8c5ea97b83fa3`.
- Coder closure/bar: `.omx/research/ddm_pr130_reproduce_20260809/SEMANTIC_SECTION_NO_MEMORYLESS_SLACK.md`
  @ `0df79dc0ace6420447a279537037efb334524d3e`.
- Existing public split receiver reference: `.omx/research/ddm_cx2_20260809/CX2_FINDINGS.md` @
  `442e0d593c7635da77963c4d2d50719d0838768a`.

## Acceptance

- The exact q4 semantic 40,252 B, pose 23,054 B, and HPAC 20,179 B raw sections are restored.
- Public receiver output equals the base output byte-for-byte.
- Archive/repeat archives are byte-identical and retained with hashes.
- The final ZIP is compared to 191,052 B; the 903 B section delta is not assumed to survive.
- No new coder race is admitted; this stub realizes the already measured split only.

## Ownership and trigger

Owner: MAIN PR130 lossless-pack successor. Consumer store:
`/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/`.
Fire trigger: AI1’s terminal receipt does not already bank the unchanged-q4 split archive and no
other owner holds the same pack lane. If it does, this stub is FOLDED.
