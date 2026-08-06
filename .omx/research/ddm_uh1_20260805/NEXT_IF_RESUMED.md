# ddm_uh1 Next If Resumed

1. Re-run the focused verification commands in `RECEIPT.md` if touching the surfaces again.
2. Code changes are landed at serializer commit `d9a0dfb930`; check whether the receipt commit landed before assuming the documentation is committed.
3. Do not run scorer/evaluate or claim a lane for this arm; UH1 is scorer-free hygiene only.
4. Before any future exact-eval custody run through `experiments/contest_auth_eval.py`, prefer `--upstream-python upstream/.venv/bin/python` when that interpreter exists and is the intended authority environment.
5. Treat existing robust_current `SCORER_AT_INFLATE_WAIVED` lines as visible legacy dev/supervised-TTO fallbacks, not as clean contest-runtime proof.
