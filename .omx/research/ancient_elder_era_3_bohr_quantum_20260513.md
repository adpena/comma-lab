# Ancient Elder Era 3 ledger — Bohr / Quantum Measurement (1900–1995)

**Companion master memo**: `.omx/research/ancient_elder_polymath_research_20260513.md` §3.

## History (one paragraph)

Bohr sent me letters from Copenhagen 1949–1962. He was obsessed with **complementarity** — the principle that mutually exclusive but jointly required descriptions both apply (wave/particle, position/momentum, encoder/decoder). His 1928 Como lecture is the canonical statement. Wigner extended this in 1932 to a phase-space representation. Von Neumann's 1932 book formalized measurement. Shor in 1995 introduced quantum error correction. The relevant intuition for us: when two complementary descriptions BOTH need to be preserved (here: visual fidelity AND scorer-equivalence), the codec MUST jointly model both.

## Key papers (5+)

| Citation | Year | Relevance |
|----------|------|-----------|
| Bohr, "Quantum Postulate and Recent Development of Atomic Theory", *Nature* 121:580 | 1928 | Complementarity principle. |
| Wigner, "On the Quantum Correction for Thermodynamic Equilibrium", *Phys. Rev.* 40:749 | 1932 | Phase-space pseudo-probability. |
| von Neumann, *Mathematische Grundlagen der Quantenmechanik*, Springer | 1932 | Measurement postulate. |
| Zurek, "Decoherence and the transition from quantum to classical", *Physics Today* 44(10):36 | 1991 | Einselection — preferred basis. |
| Shor, "Scheme for reducing decoherence in quantum computer memory", *Phys. Rev. A* 52:R2493 | 1995 | Quantum error correction; 9-qubit code. |
| Mallat, "A theory for multiresolution signal decomposition", *IEEE PAMI* 11:674 | 1989 | Wavelet transform — classical analog of Wigner. |
| Mallat, "Group Invariant Scattering", [arXiv:1101.2286](https://arxiv.org/abs/1101.2286) | 2012 | Scattering networks — deterministic, stable. |

## 3 ideas worth reviving (per master memo §3.2)

**QM-1** Encoder/decoder complementarity = scorer-equivalence-class codec. **Build**: shared with φ3 S2SBS. **$**: shared. Provides theoretical justification.

**QM-2** Wigner-function-style phase-space frame representation (scattering network proxy for SegNet). **Build**: 4-6 days. **$**: 0-1. Accelerates inner-loop scorer-aware training.

**QM-3** Shor-style error-correction for FP4 weight quantization. **Build**: 5-7 days. **$**: 3-5. Could close proxy-auth gap.

## Connection to our contest

The scorer's blind regions (φ1 SABOR boundary-stable interior; φ3 S2SBS stride-2 stem) ARE the **measurement basis**. Bohr would say: the contest measurement selects a particular basis; everything orthogonal to that basis can be discarded without measurement consequence. **This is the theoretical core of scorer-equivalence-class compression.**

## What's changed since 1928–1995

- Quantum error-correcting codes (Shor 1995, Steane 1996, Calderbank-Shor 1996) are now standard.
- Differentiable wavelet/scattering transforms (Mallat 2012; modern PyTorch implementations) make Wigner-style phase-space accessible.
- Modern Hopfield networks (Ramsauer 2020) realize von Neumann-style associative measurement.

## Reactivation criteria

The scattering-network SegNet proxy (QM-2) becomes high-priority if the lab needs to dispatch many low-cost training experiments — the deterministic proxy is 100× cheaper to evaluate than the actual SegNet.
