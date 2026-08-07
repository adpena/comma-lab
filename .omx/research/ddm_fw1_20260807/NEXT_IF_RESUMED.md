# ddm_fw1 next if resumed

No scorer, exact-eval, or launch follow-on is owed by this arm.

If resuming before the commit lands:
1. Re-run `bash -n` on the three shell files named in `RECEIPT.md`.
2. Re-run `.venv/bin/python -m pytest src/tac/tests/test_preflight_hook.py -q`.
3. Re-run the helper scan over `/Volumes/VertigoDataTier/pact/*/*.sh` and `tools/*.sh`; expected warnings: 0 and 0.
4. Commit only the intended repo hunks plus this receipt package via `tools/subagent_commit_serializer.py`.

If resuming after the commit lands:
1. Before reusing ET4 or HB1 drivers, verify their SHA-256 values still match `RECEIPT.md`.
2. If either script changed, rerun the shell-driver helper scan and require either real rc propagation or a same-line `DRIVER_RC_EXIT0_OK:<reason>` waiver.

Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
