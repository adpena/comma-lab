# CONTAMINATED — duplicate-writer collision, preserved as evidence (ddm_sm1, 2026-08-03)

Two `sb1_seg_batch.py qa03` processes wrote this JSONL concurrently. Cause: the harness
reported the first background launch as "failed exit 144" (SIGURG). The SIGURG killed the
harness's WRAPPER, not the Python child, which kept running. Relaunching on that
notification produced a SECOND writer on the same append-only store.

Damage: 28 rows / 24 unique (the file GREW from 26 to 28 after this README was first
written -- the writers were still alive while I was documenting them, which is itself
the point); duplicated instances (138,12,1) (138,12,2) (183,10,14);
one pair-state chain break at pair 138 (cell [12,2] -> [12,1]).

NOT USED for any finding. Superseded by a clean single-writer rerun under an fcntl
out-dir lock (`tools/sm1_seg_search_probe.py` / `tools/sb1_seg_batch.py` acquire it now).

LAW: a nonzero exit from the launcher is NOT evidence the job died. Check the process
table before relaunching any resumable append-only job.
