# SPDX-License-Identifier: MIT
"""tac.training - training-side helpers shared across substrates.

This package is the canonical home for training-loop helpers (hooks,
losses, schedulers) that are reused across substrate trainers. Each
module declares its own canonical public surface; consumers import
specific symbols rather than the package wildcard.

Modules:
- ``streaming_master_gradient_hook`` - SLOT MG-5 streaming sample hook
  that registers per-N-epoch master-gradient samples in the canonical
  fcntl-locked ledger at ``.omx/state/streaming_predictions.jsonl``
- ``score_weighted_reconstruction_loss`` - SLOT MG-7 sister exploit-2
  reconstruction loss helper (sister territory; see MG-7-BUNDLE)
"""
