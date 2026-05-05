---
name: Masks dtype int8 crashes EVERYWHERE — always use .long()
description: extract_masks returns int8. Embedding, constrained_gen, optimize_poses, train_distill ALL crash on int8 masks. ALWAYS cast to .long() immediately after extraction.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
The mask extraction returns int8 tensors. Every downstream consumer needs long:
- nn.Embedding (renderer) → needs Long/Int
- constrained_gen fancy indexing → needs long
- train_distill mask loading → crashed with int8 * 63 overflow
- AV1 encoding → overflows int8 at class 4 (4*63=252 > 127)

**How to apply:** After ANY mask extraction or loading, immediately call `.long()`.
Never store masks as int8. The memory savings (4x) are not worth the crashes.
This has caused 4+ separate crashes across different components.
