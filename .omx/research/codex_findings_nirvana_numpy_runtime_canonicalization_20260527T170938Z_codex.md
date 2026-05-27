# Codex Findings: Nirvana Numpy Runtime Canonicalization

UTC: 2026-05-27T17:09:38Z

## Finding

The Nirvana numpy inflate path had a false-portability boundary: `inflate.py`
called a numpy parser, but the runtime archive surface still flowed through the
torch-training archive module. That could pass local torch-rich tests while
failing a torch-free contest runtime or hiding duplicate decode primitives in a
single substrate.

## Fix Landed

- Split the torch-free NRV1-v2 reader into `src/tac/substrates/nirvana/archive_numpy.py`.
- Changed Nirvana export to vendor `archive_numpy.py` plus `inflate.py` through
  `write_numpy_portable_contest_runtime`.
- Made `tac.substrates.nirvana` lazy-load training-only torch surfaces so
  importing `archive_numpy` does not import torch.
- Promoted OIHW/grouped/depthwise/pointwise numpy conv helpers into the shared
  `numpy_portable_inflate` bridge and removed the local Nirvana copies.
- This shared-bridge promotion also closes the clean-checkout dependency class
  for other numpy-portable substrate inflates that consume torch/MLX-trained
  OIHW weights without importing a training framework at decode time.
- Added a fail-closed guard rejecting decoder state dicts that duplicate the
  separate `latents` payload.
- Consolidated the duplicate Nirvana numpy inflate test surface into the
  canonical Nirvana test module and preserved its runtime parity signal there.

## Verification

- `python -m pytest src/tac/substrates/_shared/tests/test_numpy_portable_inflate.py src/tac/substrates/nirvana/tests/test_nirvana.py -q`
  - 69 passed
- `python -m ruff check ...`
  - passed
- `python -m py_compile ...`
  - passed
- `python -c "import sys; import tac.substrates.nirvana.archive_numpy; print('torch_loaded=', 'torch' in sys.modules)"`
  - `torch_loaded= False`

## Frontier Impact

This does not directly lower the current CPU frontier. It hardens the substrate
reproduction lane by making a non-NeRV/NeRV-family candidate exportable as a
true numpy-portable runtime, which is required before local MLX/Metal training
can feed byte-closed archive candidates into the final-rate attack pipeline.
