# NSCS06 Modal Harvest + Raw Contract Hardening - 2026-05-16

## Scope

Lane: `lane_nscs06_carmack_hotz_strip_everything_20260515`

Action: harvested the three live Modal T4 smoke calls from the canonical
Modal call ledger and classified the failures before any score/rank decision.

## Harvested Calls

| call_id | dispatch label | classification | score claim |
|---|---|---|---|
| `fc-01KRQDTA70GEXSZ2CEEYGWQNSR` | `substrate_nscs06_carmack_hotz_strip_everything_modal_t4_dispatch_20260516T033756Z__smoke__100ep` | compress-side SegNet full-batch OOM on T4 before archive emission | false |
| `fc-01KRQESD492907EF4X7HSSMX0W` | `substrate_nscs06_carmack_hotz_strip_everything_modal_t4_dispatch_20260516T035432Z__smoke__100ep` | archive validation failed because `inflate.sh`/`inflate.py` were packed inside `archive.zip` | false |
| `fc-01KRQMAQ7V41AFYMJH5HRK9P10` | `substrate_nscs06_carmack_hotz_strip_everything_modal_t4_dispatch_20260516T053146Z__smoke__100ep` | archive built and runtime executed, but auth eval refused partial inflate: missing `0.raw` | false |

No returned result is promotable, rank/kill eligible, or evidence of the
method family being falsified. These are measured implementation/configuration
failures only.

## Concrete Fix Landed

The latest failure showed the runtime emitted debug-style PNG frames while the
official evaluator requires `<inflated_dir>/<video>.raw` at contest raw byte
count. The runtime contract now writes a raw RGB byte stream from
`inflate_one_video(...)` and keeps PNG emission behind `NSCS06_DEBUG_PNG=1`.

The full compress path now stores contest raw geometry
`height=874,width=1164` in the CH06 archive instead of scorer geometry
`384x512`, so a full archive can emit a byte-count-valid `0.raw`.

## Verification

Commands:

```bash
.venv/bin/python -m pytest \
  src/tac/substrates/nscs06_carmack_hotz_strip_everything/tests/test_codec_roundtrip.py \
  src/tac/substrates/nscs06_carmack_hotz_strip_everything/tests/test_oom_fix.py

.venv/bin/python -m ruff check \
  experiments/train_substrate_nscs06_carmack_hotz_strip_everything.py \
  src/tac/substrates/nscs06_carmack_hotz_strip_everything

.venv/bin/python experiments/train_substrate_nscs06_carmack_hotz_strip_everything.py \
  --video-path upstream/videos/0.mkv \
  --output-dir experiments/results/nscs06_smoke_raw_contract_20260516_codex \
  --epochs 1 --device cpu --smoke
```

Results:

- `39 passed`
- `ruff`: all checks passed
- smoke wrote `18432` raw bytes, matching the synthetic expected byte count

## Next Dispatch Gate

The next NSCS06 Modal smoke is justified only after the dirty substrate package
and runtime wrapper are landed on `main`, because the previous calls mounted
dirty snapshots at three different git heads. The next smoke should be a
single claimed dispatch with:

- `score_claim=false` until exact auth eval JSON exists;
- terminal claim on all outcomes;
- CPU/CUDA axis label preserved by the canonical auth-eval helper;
- no promotion or family kill without exact result review.

## Follow-Up Hardening

The harvested-call review also found a no-signal-loss issue in already-harvested
Modal summaries: some terminal failures had `rc` and elapsed time only in
`modal_training_terminal_claim.json` / `cost_band_anchor_appended.json`, while
the aggregate call ledger row showed null fields. The Modal harvest summary
path now enriches already-harvested rows from terminal claims, cost markers, and
provider output payloads. Archive bytes/SHA are lifted from `archive.zip` when
present, with `score_claim=false` and `promotion_eligible=false` preserved.

The NSCS06 remote script also now derives the auth-eval artifact name from
`NSCS06_DEVICE`, avoiding false `contest_auth_eval_cuda.json` missing-artifact
logs when the dispatch was a CPU advisory/auth-eval run.

Additional verification:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_modal_training_harvest_summary.py \
  src/tac/tests/test_build_pr101_finetuned_archive_codec_dir.py::test_harvest_modal_calls_appends_call_id_ledger_terminal_event

.venv/bin/python -m ruff check \
  src/tac/deploy/modal/harvest_summary.py \
  tools/harvest_modal_calls.py \
  src/tac/tests/test_modal_training_harvest_summary.py

bash -n scripts/remote_lane_substrate_nscs06_carmack_hotz_strip_everything.sh

.venv/bin/python tools/harvest_modal_calls.py \
  --execute --from-ledger --repo-root . --get-timeout-seconds 20
```

The Modal call-id ledger path now also appends a supplemental terminal event
when an existing terminal row is lossy (`rc`, elapsed time, or archive facts
missing) and the already-harvested sidecars can reconstruct those fields. This
keeps the ledger append-only while making the latest event queryable.

Results: targeted harvest tests passed; ruff passed; shell syntax passed;
refreshed aggregate harvest summary now records the three NSCS06 terminal rows
with `rc=1`, `score_claim=false`, `promotion_eligible=false`, and archive facts
for the two calls that emitted `archive.zip`. The latest call-id ledger events
now carry `rc=1` / elapsed seconds for all three calls, and archive bytes/SHA
for `fc-01KRQESD492907EF4X7HSSMX0W` plus
`fc-01KRQMAQ7V41AFYMJH5HRK9P10`.
