# ddm_gb1 — groupbin8_surprise decode-scan conditioning: LOSSLESS −153 B on the dx2 body, decode identity PROVEN, C port staged

**Status:** VERDICT — **ADMITTED, THE TWENTIETH POINTER MOVE** (MAIN-authored from
receipts; the gb1 arm died to the Opus weekly limit after banking its encode receipts —
MAIN completed the decode-identity proof, the C port, the fire, and the harvest).
**verdict_scope:** INSTANCE — the dx2 pointer body (archive 976f706d…, 180,368 B), ONE new
conditioning family added to the F26 corrector.
**axis:** encode legs `[macOS-CPU advisory / scorer-free EXACT byte measurement]`
(`score_claim=false` in the receipt) · decode-identity legs `[macOS-CPU]` · authority row
`[contest-CUDA T4 n600]` PENDING (§5).
**Receipts:** `/Volumes/APDataStore/pact/ddm_gb1_groupbin8_conditioning/` —
`measurement_v1/{CANDIDATE_SEAL_gb1_groupbin8.json, VERIFY.json, PATCH_groupbin8_surprise.json}` ·
`retained/{candidate_gb1_groupbin8_surprise.zip, S1_encode_gb1_groupbin8_surprise.json}` ·
fire tree `runtime_fire_v1/` · identity harnesses
`/Volumes/APDataStore/pact/scratch_gb1_identity_20260824/` (run 5, Python corrector) +
`scratch_gb1_native_identity_20260824/` (native corrector, via inflate.sh).

## 1. The mechanism

`groupbin8_surprise` is a **decode-scan conditioning family** added to the F26 corrector:

```
groupbin8 = ((((flat % WIDTH) % 64) + 2 * ((flat // WIDTH) % 64)) * 8) // 190
family index = (cls * GROUP_BINS + groupbin8) * U_BINS + ubin        # 2,560 cells
```

It bins the SHIPPED decode-scan group plan `g(x,y) = (x mod 64) + 2·(y mod 64)` to 8 levels.
**Causal** (the position is selected before the symbol is decoded), **zero transmitted
bytes**, rule-118 free — the receiver derives it from coordinates it already holds. This is
the `mi1` law cashed in: *an input the network receives is not a feature it has consumed*
([[an-input-the-network-receives-is-not-a-feature-it-consumed]]) — the shipped model
RECEIVES these coordinates and had still not priced the decode-order structure; `mi1`'s
subtile4 negative control measured the residue, gb1 collects it.

## 2. The encode rows (arm receipts, banked before the arm died)

| candidate | bytes | Δ vs dx2 180,368 | tokens_changed | ΔS_rate |
|---|---:|---:|---:|---:|
| **groupbin8_surprise** | **180,215** | **−153** | **0** | **−1.0187641982769222e-4** |
| cls_groupbin8 (banked) | 180,268 | −100 | 0 | −6.66e-5 |

`delta_trustworthy=true`; base and candidate re-hashed by MAIN 08-24: retained zip AND the
fire-tree archive both `ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4` /
180,215 B. Against the `mi1` model-axis ceiling (2,162 B systematic at z=+11.89), −153 B
collects ~7% of the measured ceiling in one causal family at zero transmitted cost.

## 3. Decode identity — PROVEN on the macOS axis (run 5)

The Python-corrector receiver decoded the candidate end-to-end:
`out/0.raw` sha `7246a4ff8f79b03ab14b3a72f6a6e2fff18b567fcb61f12a7fe311d48f5f2de7` —
**byte-identical to the dx2 pointer's retained macOS raw** (from
`/Volumes/APDataStore/pact/ddm_dx2/r7/RESULT.json`; the T4 raw is separately `6bf8acf8…` —
decode is device-scoped in the renderer float forward only; the token/corrector layer is
device-portable). Consequence: the candidate is LOSSLESS BY CONSTRUCTION — the T4 row can
differ from dx2 only in the rate leg.

## 4. The C port (the seal's C-PORT-OWED blocker, paid)

The shipped receiver prefers the native corrector (`f26_corrector_native.so`); the repo-
canonical C at `runtime-rs/native/f26-corrector/` is the STALE 13-family rr8 original — the
SHIPPED C is the 19-family fx5 version living only in the dx2 custody tree. Port base =
shipped C. 14 exact-match patches into `runtime_fire_v1/`: `GROUP_BINS 8` · `N_FAMILIES 20` ·
enum + rule/size/limit/joint table rows · `family_rule_index` groupbin8 argument + case ·
caller `int64_t groupbin8 = (((x % 64) + 2 * (y % 64)) * 8) / 190` (all operands non-negative
→ C `/` == Python `//` exactly). `native_free_corrector.py` expected-config extended;
`assert_config_matches()` PASSES on the fire tree. Compiled clean
(`-O3 -std=c11 -ffp-contract=off -fno-fast-math -Wall -Wextra`, rc=0, no warnings).

**Native-corrector identity decode: C PORT EXACT** (2026-08-25, rc=0, 384 s wall via
`inflate.sh` — the faithful receiver path: builds both `.so`, exports
`F26_CORRECTOR_NATIVE_LIBRARY`; fallback message verified ABSENT, so the native corrector
provably ran). Output raw sha `7246a4ff8f79b03a…` — byte-for-byte identical to BOTH the
run-5 Python-corrector decode AND the dx2 pointer's retained macOS raw. The C port
reproduces the Python corrector exactly on the full real payload. Note the wall-clock:
384 s native vs run 5's ~24 min Python — the port is also the decode-budget insurance
(rr8 measured 1,419.9 s Python vs 464.6 s native on T4).

## 5. The authority row — ADMITTED (the TWENTIETH pointer move)

Canonical seal `SEAL_fire_gb1_groupbin8.json` (seal sha `9a31811d71af3dcf…`, receipt-
extracted pins, axis contest_cuda, single-axis waiver = F26 CUDA-locked lineage) validated
SEAL_VALID → ONE T4 fire via `fire_modal_auth_eval.py --seal` under single-flight, call
`fc-01M0VKA8RTDSNT6K3XVMDPRH1H`, rc=0, n_samples 600:

| quantity | measured | vs dx2 pointer |
|---|---|---|
| **S (recomputed from components, #877)** | **0.14811799921260607** | **ΔS −1.0187641982769e-4** |
| rate | 25·180,215/37,545,489 = 0.1199977 | −1.0188e-4 (the whole delta) |
| d_seg | 0.00020139 → seg 0.020139 | **byte-identical** |
| d_pose | 6.37e-6 → pose 0.0079812 | **byte-identical** |
| T4 raw sha | `6bf8acf8d4412e43…` | **== dx2's retained T4 raw** |
| wall | inflate 503.67 s (3.57× the 1,800 s CI budget) · evaluate 39.5 s · total 558 s | — |

ΔS realized EXACTLY at the deterministic rate-only projection; decode identity now proven
on BOTH axes (macOS §3 + T4 raw). **Triple identity chain: Python corrector (macOS) =
native C (macOS) = T4 authority raw.** Pointer refreshed
(`tools/refresh_canonical_frontier.py --no-update-upstream`): effective_frontier =
0.1481179992, source our_local_frontier_contest_cuda, lane `ddm_gb1_groupbin8_20260825`,
archive `ba1f3830…` / 180,215 B. Harvest receipts mirrored to
`measurement_v1/{MODAL_REMOTE_RESULT.json, FIRE_MANIFEST.json, modal_call_id.txt,
launch_manifest.json}`. Sub-0.12 arithmetic after the move: gap ≈ 0.028118; archive cap
137,986 B unchanged (distortion legs identical); demand now 42,229 B.

## 6. Five packaging gaps found while staging (the reusable part)

All five are silent-until-fire defects caught by local identity runs before any paid dispatch:

1. **Pointer archive left inside the candidate runtime tree** (×2 — scratch1, then AGAIN on
   the fire tree): `inflate.py:52` hashes `runtime/archive.zip` INSIDE the tree, not the
   data-dir copy — a candidate tree carrying the base archive is undecodable by construction.
   The `#1237` half-updated-pin genus, recurred on a tree MAIN built after curing it once.
2. **Stale inflate.py pins** — `ARCHIVE_SHA256`/`ARCHIVE_BYTES` must be programmatically set
   to the candidate (ast-verified), never inherited.
3. **Missing `data/p`** — the receiver cross-checks `data/p` == archive member `p` with
   `namelist() == ["p"]`; a data dir without it refuses.
4. **Native config drift gate** — `assert_config_matches()` refuses any tree whose C
   constants disagree with the Python SHIPPED_CONFIG (killed staging run 4, correctly).
5. **Bare `python` on a python3-only host** (#929 genus) — cured with a scratch-only
   exec-wrapper shim (never a symlink); NOT patched into inflate.sh because the T4 container
   provides `python` (minimal-diff discipline on the shipped tree).

Plus one launch-apparatus lesson re-learned: the identity waiters died to the rc=144 session
reaper twice; the canonical `tools/launch_detached_process.py` + driver-computes-verdict
pattern is the cure (the launch guard enforced it — the guard worked).

## 7. GESTALT-DELTA

The model-axis block of [[dx2-block-ceilings-are-measured-and-sum-to-5-percent]] (ceiling
2,162 B) gets its first post-mi1 collection: −153 B causal, zero-byte, realized as archive
bytes AND as an admitted authority row (the twentieth pointer move). The ceiling is not
raised — gb1 spends INSIDE it, as the block table predicts: ~2,009 B of measured
conditional structure remain. The open question it sharpens: how much of that remainder is
reachable by further causal families. Named successor: **joint 21-family re-encode of
groupbin8_surprise + cls_groupbin8** (the −100 B banked rung is NOT additive until the
model is re-encoded with both families in one config — same-axis families share the
conditional structure they price).

Sisters: [[an-input-the-network-receives-is-not-a-feature-it-consumed]] (the law that named
this lever) · [[dx2-block-ceilings-are-measured-and-sum-to-5-percent]] (the budget it spends
inside) · `#1237` (the pin genus §6 re-instances).
