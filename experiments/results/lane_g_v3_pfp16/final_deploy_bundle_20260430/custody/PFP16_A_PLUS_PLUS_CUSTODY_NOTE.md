# PFP16 A++ Custody Note

Paper-ready claim: exact T4 CUDA auth eval recomputes score `1.043987524793892` for archive `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f` (`686635` bytes, `n=600`).

The only score authority in this bundle is `eval/contest_auth_eval.json`. It records `experiments/contest_auth_eval.py --device cuda`, Tesla T4 provenance, the upstream commit, component distances, archive bytes, and the recomputed contest formula.

Legacy remote parser/adjudication fields are quarantined in `build/provenance.json` under `legacy_parser_output_quarantined`. They must not be used for score, rank, promotion, regression, retirement, kill, or paper claims.

This custody note and manifest are metadata sidecars only; they do not alter `archive/archive.zip`.
