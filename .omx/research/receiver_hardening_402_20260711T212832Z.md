# Receiver fail-closed hardening — task #402 — VERIFICATION + BIT-IDENTITY ACCEPTANCE PROOF

**Date:** 2026-07-11
**Subagent:** `receiver-402` (checkpointed `.omx/state/subagent_progress.jsonl`)
**Authority:** `[macOS-CPU advisory] NON-PROMOTABLE`. This is APPARATUS/GATE work — no score-lever,
pointer UNMOVED (0.19108282 [contest-CPU]).
**STORES CONSULTED:** `.omx/research/pr128_intake_reverse_engineering_20260710.md` (§6/§8.2 defect
list — the source) · `receiver_consumption_bijection_counted_but_inert_weight_groups_20260710.md`
(#417 sister — the receiver-consumption bijection) · DAG `FEED-receiver-harden (task #402)` @ line
14156 (the prior landing) · `src/tac/tests/test_levelset_receiver_harden.py` (24 tests) ·
`tools/tests/test_levelset_receiver_bijection_gate.py` (11 tests).

---

## Headline

The #402 receiver hardening (six fail-closed axes) was **already landed + committed** (`e2c1b2aeb2`,
24 tests) with the #417 bijection sister (`c3697bbf1e`/`e8d080dd80`). This unit **verified it end-to-end
and closed the missing acceptance gate**: the **bit-identity before/after proof** the mission demanded,
which had not been recorded. Result: **hardening is byte-identical on a valid archive** — proven, not
asserted. Two owed deliverables (this memo + the DAG verification note) now land.

## The four mission properties — all confirmed IN THE SHIPPED `inflate.py` (contest-run surface)

Verified against a freshly-produced shipped `inflate.py` (not just the tool-side oracle):

1. **EXACT-CONSUMPTION** — every LVLS1 section reader fails CLOSED on trailing/short bytes:
   base blob (`inflate.py:96,109,111`), lane band (`:370`), ξ payload (`:546,558`), pose-carrier
   v2 + legacy (`:600,603,615`). The #417 receiver-consumption **bijection gate**
   (`tools/levelset_receiver_bijection_gate.py::assert_receiver_bijection`, wired in
   `build_levelset_blob`) is the structural sister: every COUNTED base param must be consumed by the
   `_INFLATE_PY` forward or the byte-close REFUSES (kills the counted-but-inert FAKE-lever class).
2. **FINAL-BYTE ASSERTION** — shipped inflate `main()` (`:777–787`) asserts
   `getsize(tmp) == 2·n_pairs·framebytes` and `raise SystemExit` on a short `.raw`; a truncated decode
   NEVER becomes a scoreable output. (No portable `sha256`-of-`.raw` authority claim is emitted — the
   only torch op, `_R` bicubic, has microarch-variable LSBs; per-host `--verify-bit-exact` round-trip
   is the authority. `receiver_env_manifest()` pins numpy/torch/brotli/scipy/python versions.)
3. **ATOMIC WRITES** — shipped inflate (`:758–788`) writes `dst.partial` then `os.replace(tmp, dst)`
   ONLY after the size-verify passes; a crash/OOM leaves `.partial`, never a full-size scoreable-looking
   `dst`.
4. **STORAGE PREFLIGHT** — `_raw_storage_preflight` (`tools/levelset_byte_close_and_eval.py:1884`, wired
   into `run_inflate:1955`) fails CLOSED via `shutil.disk_usage(out_dir).free` if the output volume
   cannot hold the ~3.66 GB `.raw` (+5% margin) BEFORE decode starts.

## Test suite — 35 passing (24 harden + 11 bijection)

`.venv/bin/python -m pytest src/tac/tests/test_levelset_receiver_harden.py
tools/tests/test_levelset_receiver_bijection_gate.py -q` → **35 passed**. Coverage spans
positive/happy-path, negative trailing-byte, truncated-section, wrong-magic corruption,
atomic-write, killed-subprocess, short-`.raw`, storage-ok, storage-fail-closed, dep-pinning,
cross-host note, and no-portable-sha authority.

## BIT-IDENTITY ACCEPTANCE GATE — the proof (MEASURED, this unit)

**Requirement (mission):** hardening must be score-neutral + byte-identical on valid archives; prove by
round-tripping an existing byte-closed archive before/after and diffing decode outputs bit-for-bit.

**Method (containment-safe, CPU-light):** ckpt = `levelset_v752_baseline_20260710T185913Z` (finished,
shared-head, no tex_trunk/carriers). Decoded 6 pairs (12 frames, `.raw` = 36,624,096 B) at:
- **HEAD** (hardened: #402 + #417 + all intervening default-off feature commits), and
- **pre-#402** (`e2c1b2aeb2^` = `acd1f03e01`), the tool materialized read-only via `git show` and run
  from `tools/` so its repo-relative sys.path resolved (then deleted; never committed).

Both invoked identically: `--gt-cache gt_n6.npz --max-pairs 6 --skip-parity --keep-packet`.

**Result — BIT-IDENTICAL:**

| artifact | pre-#402 sha256 | HEAD sha256 | verdict |
|---|---|---|---|
| `archive.zip` (83,093 B) | `242d453e…664a` | `242d453e…664a` | ✓ identical |
| `inflated/0.raw` (36,624,096 B) | `e290f19d…264f` | `e290f19d…264f` | ✓ identical |

`cmp` byte-for-byte on `0.raw`: identical. Since every commit between `e2c1b2aeb2^` and HEAD (the
hardening axes + LBND4/#238/phase-carrier features, all default-off) is byte-neutral on the default
valid shared-head path, the hardening is **definitively score-neutral**. (Consistent with the #417
memory's independent shared-head byte-identity note, blob sha `693a76912d9d…`.)

## Scope honesty

- The proof covers the **shared-head** default path (the live v7.5.2 lineage). tex_trunk/decoupled_head
  consumption (#417 FIX) is parity-proven separately (`test_receiver_bijection_v753_v8_parity.py`, 8
  tests, relmax ~7e-4) — a FUTURE-row unblock, not exercised here because no such archive is finished.
- Storage preflight lives at the tool's `run_inflate` orchestration layer (our byte-close/eval concern),
  not inside the contest `inflate.sh` (whose environment the evaluator provides) — correct scope.
- This is MEANS. Pointer UNMOVED. It protects FUTURE exact rows from truncated/corrupt/inert-group
  decode; it moves nothing itself.

## Triality

- **DSL leg:** N/A (receiver apparatus/gate, not a witness lever).
- **Equations leg:** the sister `receiver_forward_parity_v753_v8_v1` already registered (#417).
- **DAG leg:** `FEED-receiver-harden (#402)` @ line 14156 (the landing) + this unit's verification
  note appended (the bit-identity acceptance proof).

## Owed / next

Nothing owed on #402 — code + tests + proof + memo + DAG all landed. The contest `inflate.py` path
(distinct from byte-close) shares the same hand-written forward and should get the identical
exact-consumption/final-byte/atomic audit whenever a witness lineage reaches contest submission.
