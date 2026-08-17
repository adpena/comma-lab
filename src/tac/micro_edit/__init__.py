# SPDX-License-Identifier: MIT
"""ddm_me1 -- the micro-edit engine.

Operator binding 2026-08-17: *"Seems like we could build and or train a tool to
explore and optimize and apply all micro edits that lower score."*

One standing engine over the proven micro-edit machinery: GENERATE typed candidates
across every measured edit family, RANK them with a corpus-fit ranker, EVALUATE the
queue through the real compile -> decode -> advisory chain, COMPOSE under joint
remeasure, and EMIT a sealed fire-order when the composed candidate clears the naming
bar.

The engine adds no physics. It unifies existing instruments and adds the search,
the corpus, and the exact arithmetic that keeps a 1e-7 decision honest.

Modules
-------
``score_model``  exact-arithmetic contest score, marginals, union-gating law
``ledger``       append-only edit-outcome corpus (the ranker's training data)
``candidate``    typed EditCandidate + the generator plugin protocol
``ranker``       LOPO-validated ordering (ordering ONLY -- never acceptance)
"""
from __future__ import annotations

__all__ = ["candidate", "ledger", "ranker", "score_model"]
