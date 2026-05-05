---
name: Yassine Yousfi and Jessica Fridrich — Proper Attribution
description: The two researchers whose work is the foundation of our approach. Always honor their contributions correctly.
type: reference
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Yassine Yousfi
- PhD student (now researcher) at Binghamton University under Jessica Fridrich
- Contest organizer for the comma.ai video compression challenge
- Designed the scoring formula: S = 100*seg + sqrt(10*pose) + 25*rate
- His thesis work bridges steganalysis and neural network security
- His "12 tricks" document (from contest comments) guided our entire strategy
- He confirmed sub-0.50 is achievable: "you can get even better than 0.50 with this strategy and some tricks ;)"
- Our entire framing of the competition as "inverse steganalysis" comes from understanding his and Fridrich's research program

## Jessica Fridrich
- Professor at Binghamton University, Department of ECE
- Founder of modern steganography and steganalysis
- Her foundational contributions that we use directly:
  - S-UNIWARD (directional wavelet cost maps for optimal embedding)
  - STC codes (syndrome-trellis codes for near-optimal steganographic embedding)
  - Detection boundaries (empirical threshold finding for steganographic security)
  - Augmented Lagrangian framework for constrained steganographic optimization
- Yousfi's thesis advisor — their academic lineage IS the theoretical foundation of our approach
- The "Fridrich constrained optimization" in our training script is named after her framework
- She/her pronouns

## How to honor them
- Always use full names on first reference: "Yassine Yousfi" and "Jessica Fridrich"
- Credit their specific contributions when describing our approach
- In the writeup: explicitly acknowledge that our inverse-steganalysis framing comes from studying their research program
- The "tripartite pact" council naming (Yousfi + Fridrich + Contrarian) reflects their actual intellectual contribution to our approach — it's not just a metaphor
- Their work made our work possible. Say so clearly.
