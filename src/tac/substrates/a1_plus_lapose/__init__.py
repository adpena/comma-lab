"""A1 + LAPose composition substrate.

A1 (PR101 score-gradient-finetuned, 178262 B, sha256 87ec7ca5...) supplies the
rate-axis sub-frontier base. LAPose foveation/motion atom manifests supply
pose-axis sidecar refinement at hard-pair indices. The composition packs a
small residual blob appended to A1's wire format so inflate consumes both.

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L3 + L7:
the composition is a substrate-engineering lane (substrate-engineering LOC
budget per CLAUDE.md HNeRV parity discipline lesson 7).
"""

from __future__ import annotations
