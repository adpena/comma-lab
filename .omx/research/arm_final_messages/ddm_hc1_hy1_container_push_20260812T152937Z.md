The exact frontier did not move. HY1 is byte-closed, but its `0.1475191686` proxy is refuted: the terminal archive scored **0.4044688071472634** `[contest-CUDA T4, n600]`.

| Item | Result |
|---|---:|
| Complete archive | **187,046 B**, `+794 B` vs cp135 |
| C1-on-HP3 RC64 stream | **115,237 B**, `+6 B` |
| Independent decode | **117,964,800 / 117,964,800** tokens exact |
| `d_seg` | **0.00047310** |
| `d_pose` | **0.00541086** |
| Realized survival `rho` | approximately **−1.223** |
| Recomputed S | **0.4044688071472634** |
| Delta vs cp135 | **+0.24251366886902165 S** |

F1 fired: C1 realization is harmful on both Seg and Pose. F2 did not fire: the HP3 adaptation, RC64 stream, container parser, and independent decoder are correct. The failure occurs downstream in renderer/scorer realization.

The requested n32 scorer measurement was not launched by hc1 because MAIN had already retained the stronger full-n600 authority row; hc1 launched no scorer, Modal, GPU, evaluator, or MPS job.

The terminal compose recipe was already executed through `experiments/ddm_t1r1_container_build_rehearsal.py`. T1R1’s purported pass-4 stand-in is byte-identical to the policy-terminal carrier, so no terminal substitution remained. There are no remaining `QUEUED-WITH-A-FIRE-ORDER` rows for hc1.

Artifacts:

- [Landing memo](/Users/adpena/Projects/pact/.omx/research/ddm_hc1_hy1_container_push_20260812.md)
- [Resumable state](/Volumes/APDataStore/pact/ddm_hc1_20260812/state.json)
- [Retained manifest](/Volumes/APDataStore/pact/ddm_hc1_20260812/RETAINED_MANIFEST.json)
- Commit: `de9555ae77`
- Retained archive SHA-256: `12a5b181fef4e15ad8a752161c744347beca0b5a1224c5d3d542ab148f6ece80`

Own-vehicle frontier remains **LC2 S = 0.16959899569230852 @ 187,226 B `[contest-CUDA T4, n600]`**.

## LIVE-HYPOTHESES

- C1 may remain useful as an encoder-side teacher for a representation-changing renderer, because semantic carriage is exact even though literal plane substitution fails.
- A learned Seg actuator trained and stage-gated on the shipping CUDA axis may succeed where the CPU-optimized plane and pose solve failed.
- Retaining cp135’s CUDA-solved pose carrier while changing only a shipping-axis-trained Seg mechanism is the most plausible composition.

## DEAD-ENDS

- Adding HY1’s F26 `+11 B` proxy to cp135 is closed; the complete real delta is `+794 B`.
- Re-running ANS on this HP3 state is closed; it was already 9 B worse than RC64.
- Blaming the probability object, coder, parser, or terminal carrier is closed by exact independent decode and SHA identity.
- Inferring `rho=1` from token equality is closed; measured shipping-axis `rho≈−1.223`.
- Direct C1-plane substitution on this PR135/CP135 receiver is closed on the contest-CUDA axis.
- Chasing the CPU pose solve below its CUDA disagreement floor is closed.