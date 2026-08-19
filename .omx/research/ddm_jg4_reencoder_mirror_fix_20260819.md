# ddm_jg4 — the re-encoder's control failed on RESUME, not on the mirror

**Date** 2026-08-19 · **Axis** `[macOS-CPU advisory / scorer-free EXACT byte measurement]` ·
**Score claim** none. This memo fixes an instrument; it moves no pointer.

## The finding, first

The `ddm_jg2` tail re-encoder was never wrong. Its **checkpoint** was.

`encode_tail` saved the arithmetic coder's probability model by calling
`corrector.state_dict()`. That method is defined **only on the flat `rr4` base class**
(`runtime/rr4_free_corrector.py:318-327`) and returns 7 arrays. The shipped
`FreeCorrector` is three subclasses deeper:

```
FreeCorrector -> Ma1WithinMissCorrector -> Fx2ModelAxisMixer
              -> FixedPointLogisticMixer -> rr4.FreeCorrector -> object
```

Neither subclass overrides `state_dict`. So the checkpoint saved 7 of the **97**
values the corrector owns and silently dropped 90: the 4000x13 logistic-mixer
`weights`, `sse_weight`, the `ma1` within-miss tables (`_miss_counts`,
`_miss_expect`, `_miss_seen`), and all 39 arrays owned by the 13 `MixerFamily`
members (`counts`/`hits`/`phat_q` each). 9.68 MB of live state, of which 3.40 MB
was written.

A resumed run therefore restarted the entire model-mixing half **cold** while
every log line and every receipt looked healthy. The coding rows from the resume
frame onward were the wrong rows, so the emitted stream was not the stream the
shipped decoder produces — and the control that exists to catch exactly that
reported the symptom without naming the cause.

## The evidence that isolates resume (measured, three 600-frame controls)

| run | resumed at | elapsed | emitted | vs shipped | prefix agrees |
|---|---|---|---|---|---|
| `ddm_jg2` | **never** (straight through) | 969.78 s | 109,696 B | **byte-identical** | 109,696 B |
| `ddm_jg3` | ~frame 275 | 494.60 s | 109,823 B | +127 B | 49,922 B |
| `ddm_jg3/harvest_r1` | frame 75 | 800.27 s | 109,736 B | +40 B | 14,553 B |

Shipped token stream: 109,696 B, sha `15054e5da33640bcb2e9d4589615c3b89b1312ce27fd9aa8e2a0ec0284b506f2`.

Two independent facts pin the mechanism:

1. **The divergence starts at the resume frame and nowhere earlier.** All three
   per-frame bit ledgers agree to the last decimal through frame 75
   (cumulative 14,552.76 B). `harvest_r1`, which resumed at 75, already differs by
   frame 100 (19,255.81 vs `jg2`'s 19,223.10). `jg3`, which resumed later, still
   matches at frame 100 and differs by frame 300.
2. **The *bit ledger* differs, not just the byte stream.** `per_frame` is computed
   from `coding_row` alone and never touches the RC64 interval. A pure coder-state
   defect could not move it. Only the corrector's probability model can — which is
   precisely the state the checkpoint dropped.

The elapsed times corroborate: 800.27 s ÷ 525 remaining frames ≈ 1.52 s/frame,
the same rate as the full 969.78 s pass over 600.

Two hypotheses the charter offered are **refuted**, and both stay refuted:
the token input (`9ba2e52b30965858`) was byte-identical to a fresh decode, and the
carrier context does not couple (the `br1` body carries the same stream and decodes
to `d_seg 0.00030309` on T4). The `+31`/`+40` B discrepancy between the two failing
runs was not nondeterminism either — it is two different resume frames losing
different amounts of accumulated model.

## The cure, and why it is structural

A longer hand-written key list would have fixed this instance and re-armed the
class: the next subclass would be forgotten the same way. So:

* **`corrector_state` walks the object, it does not enumerate it.** Every
  `__slots__` entry on every class in the MRO, plus `__dict__`, plus the same walk
  over `families`. `vars()` alone cannot see the slotted base; a `__slots__` walk
  alone cannot see the subclasses' dicts. Only the union sees the whole object.
* **`uncaptured_divergent_state` is the detector, and it does not zero on the
  cure.** It diffs the live corrector against a **cold** one — the state a resumed
  run would silently start from — and reports anything that has moved away from
  cold and is not in the capture. Run against v1's key set it names the dropped
  arrays; against v2's it returns `[]`. It runs before every checkpoint write and
  refuses rather than writing an unresumable checkpoint.
* **The schema tag fails closed.** A v1 checkpoint is refused with an explicit
  message, never resumed. It is not a slower path to the same answer; it is a
  different and wrong answer.
* **`load_corrector_state` proves the restore landed** by re-reading the state and
  comparing, so a silent partial restore cannot recur one layer down.

**Measured, `12` frames on the `br1` body:** straight-through and
checkpoint-at-6-then-resume now produce **byte-identical** streams
(sha `0f5cf77e3903e10d`, 2,316 B) and **identical** bit ledgers (max abs diff 0.0).
Before the fix this pair diverged.

## The bug ladder

* **Bug** — `encode_tail` checkpointed via `corrector.state_dict()`, dropping 90 of
  97 values (`experiments/ddm_jg2_tail_reencode.py`, resume + checkpoint blocks).
* **Class** — *a base-class serializer used on a subclassed object*: the capture is
  bounded by where `state_dict` was defined, not by what the instance owns.
  **Class population, measured** (uncovered arrays per corrector module):
  `ddm_rr2_free_corrector` **0**, `ddm_rr4_free_corrector_v2` **0**,
  `ddm_fx1_logistic_mixer_corrector` **69**, `ddm_fx2_model_axis_corrector` **81**,
  `ddm_ma1_within_miss_corrector` **90**. (The two flat modules leave only
  `boundary`, re-pinned by `begin_frame` every frame, and the constant `plane`.)
  **Live sites: 2.** `ddm_jg2_tail_reencode` (defective, fixed here) and
  `ddm_rr2_encoder_byteclose` (safe on its default flat corrector, but its
  documented `TAC_RR2_CORRECTOR_MODULE` override names exactly the defective ones —
  guarded here with a fail-closed coverage check, a pure insertion that leaves the
  sealed default path untouched).
* **Family** — **silent instrument**. The checkpoint wrote, the resume logged
  `"resumed"`, no exception fired, and a receipt was produced for a stream nobody
  could have decoded. It is the `vacuity == pass` shape: the instrument read green
  because it never reported its denominator. The cure reports both — `state_keys_saved: 97`
  on every checkpoint and `state_keys_restored: 97` on every resume — so a future
  drop is visible in the log rather than only in a failed control 800 s later.
  (Sister, not the same: the *staleness* family. The state here was not stale, it
  was absent.)
* **Meta-bug** — a **resumability claim with no falsifier**. The module docstring
  promised "`--resume` continues bit-faithfully" and nothing ever tested it. The
  claim is now a test that fails if it stops being true, and the test carries its
  own positive control so it cannot pass vacuously.

## What also changed, and why

* **Body-agnostic pointer check.** v1 pinned `POINTER_ARCHIVE_SHA = 7ce46fd7...`
  (the `up3` body). That literal blocked the live `br1` pointer entirely, even
  though both bodies carry the **same** token stream (`15054e5d...`) and the same
  109,792 B tail, differing only in the RX1 `reserved` field and 9 B of carrier.
  Hand-typing `br1`'s numbers beside `up3`'s would have moved the weld, not removed
  it. The check now asserts what it actually needs — the spliced archive's member
  must be byte-identical to `<runtime-root>/archive.zip`'s — which is strictly
  stronger than a sha literal and holds on any body. `--expect-pointer-sha256`
  remains for callers who want one body pinned.
* **The RC64 build no longer races.** The docstring said the control and the
  edited encode may run concurrently; they could not, because both compiled and
  linked to the same paths inside one `--store`. Each stage now builds in its own
  directory, which is what made this arm's two 600-frame passes concurrent.
* **`--wait-for-control-seconds`** lets the encode poll for a concurrent control
  receipt instead of reporting `UNPROVEN` purely on finishing order and forcing a
  hand reconciliation afterwards.

## Artifacts

* Fix: `experiments/ddm_jg2_tail_reencode.py`
* Sister guard: `experiments/ddm_rr2_encoder_byteclose.py`
* Test: `tests/test_ddm_jg4_reencoder_resume_fidelity.py` (5 fast + 1 `slow`)
* Custody: `/Volumes/APDataStore/pact/ddm_jg4/` — `retained/` receipts and payloads,
  `work/` streams and bit ledgers, `logs/` launch manifests and run logs.
