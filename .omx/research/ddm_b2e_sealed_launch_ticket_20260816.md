# ddm_b2e SEALED LAUNCH TICKET — F2-ALONE short decisive row (burn-2 train-for-editability)

Status: **FIRED (MAIN, 2026-08-16) — pins resolved sha-exact; 50-step governed smoke PASS ×2 receipts (166.30 s / 50 steps end-to-end, peak RSS 2,779 MiB @ 12 GiB admission, per-stage ckpts + resume.latest + final save verified, packed export 40,252 B == sd1 EXPECTED_BASE_SEMANTIC_BYTES; first attempt refused by the admission guard for a raw launch — guard CORRECT, rerouted through tools/safe_run.py). Derived window: ≤3.33 s/step → 3,000 steps ≈ 2.8 h + 12 checkpoint saves. NOTE: was READY_TO_FIRE_PENDING_TWO_INPUT_PINS** (upgraded from BLOCKED_ON_OBJECT_REPIN)
Prepared by: ddm_b2e arm · Owner of the FIRE: **MAIN** (governed Metal slot)
Landing memo: `.omx/research/ddm_b2e_landing_and_charter_repin_20260816.md`
Commits: `ec83c44223` (levers + harness) · `f035530ef2` (memo) · `277fc58d13` (trainer wiring)

MAIN's re-pin is executed. Gates 6 and 7 are CLOSED with measured evidence. Two mechanical input
pins remain — both are lookups MAIN can close in minutes, neither is a design question.

---

## RE-PIN EXECUTED — measured results

### Gate 7 (host trainer) — CLOSED
`src/tac/pr130_lift/train_semantic_quantized_resumable.py`. F1–F4 wired as default-off flags
(commit `277fc58d13`).

### Gate 6 (warm-start object) — CLOSED, and it is the INHERITED case

**`b489c735…` is the sha256 of `/Volumes/VertigoDataTier/pact/pr135_intake_20260810/pr135/retained_fd135/pr135/canonical/semantic.wans1` (36,051 B) — EXACT MATCH.**

The provenance chain is `mz2._load_records()` → decodes `F12_BODY` → asserts equality with that
canonical WANS file → those records ARE the shipped semantic state (mp2's `parser="legacy"` returns
the template unchanged).

**Say it plainly, as MAIN asked: the shipped semantic weights are INHERITED PR135 intake weights.
We never trained them.** Consequences, all measured:

- **No semantic training checkpoint of ours exists.** MAIN's search order (a) — the hb1
  `checkpoints/{gt,tq1c}/` run — contains only HPAC **token** models: all 10 checkpoints score 2/5 on
  SemanticTokenRenderer key overlap (the coincidental `frame_embed`/`head` names), none is semantic.
- **No optimizer state exists at all** for the semantic object — not in hb1, not in the intake
  checkpoints (`has_opt=False`). The burn starts on **fresh Adam**.
- **The wd3 warm-carry law (Adam state carries ~3× pose descent) therefore does NOT apply here.**
  This is a real downgrade in expected descent rate versus the charter's assumption and MAIN should
  price the window accordingly.
- This also fully explains ns1's "never QAT'd on semantic weights": we never trained that object at
  all. The sharper statement is that it was trained *by PR135* for a **uniform q4** grid, and our
  edits move it to **mixed q3/q4** it has never seen.

### The `--init` object — located, and it is NOT bit-exact (report, do not pin over)

`--init` needs a torch checkpoint carrying architecture metadata; the `.wans1` is a raw byte stream
and cannot serve. The correct init is:

```
/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/artifacts/checkpoints/
    semantic_renderer_w96_b4_qat4_fixedtau05_tail6k_lr2e7.pt
sha256 3948ccfcd44778dc42affee18a10c3f3baa434d1a2eb2345a013146c1dbfb647
```

That sha is **exactly** sd1's pinned `EXPECTED_CHECKPOINT_SHA256`, so it is the canonical semantic
init. Its `config` is `width=96, blocks=4, frame_dim=8`, `quant_bits=4`,
`best_exact_seg=0.0002763705783420139`.

**Verification result, fail-closed:** `q4(init float weights)` reproduces the shipped decoded state
on **37 of 38 tensors exactly**. One tensor differs:

| tensor | differing elements | nature |
|---|---|---|
| `blocks.3.film.weight` (192×8) | **2 of 1,536** | rows 87 and 189, one element each |

These are **not** rounding tie-breaks — I checked: row 87 col 7 has normalized value 7.0015 (rounds to
code 7) but ships as code **5**; row 189 col 2 has normalized 0.6167 (rounds to code 1) but ships as
code **−1**, a sign flip. Both are ~2 code steps. There are no zeroed rows, so it is not a prune.

**Reading:** the shipped state carries **2 deliberately-modified elements** in the pose-critical FiLM
family relative to `q4(init)` — most likely a downstream targeted correction, or the ship came from a
marginally later checkpoint. It is 2 elements out of ~181k semantic parameters and is almost certainly
immaterial to training dynamics, **but the warm-start is NOT bit-exact to the shipped object and this
ticket does not claim that it is.** MAIN may proceed knowingly; the delta is quantified above.

---

## THE F2-ALONE SHORT ROW

**Why F2 alone is the cheapest decisive row.** It is the only lever whose mechanism is fully derived
(train the grid you deploy), it is the lever the −823 B q3/q4 pool depends on, the admission bar is
pre-registered at 50×, and it removes exactly the uniform-q4-vs-mixed-q3q4 mismatch that ns1 located.
If the collapse factor does not move materially off 1×, the regime thesis is in trouble early and
cheaply — which is the whole reason the instrument was built before the burn.

### Command shape (every flag verified against the real argparse)

```bash
.venv/bin/python -m tac.pr130_lift.train_semantic_quantized_resumable \
  --challenge-root upstream \
  --cache          /Volumes/VertigoDataTier/pact/ddm_op1r_20260809/authority_cache/gt_cache_600_official_ada.pt \
  --init  /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/artifacts/\
checkpoints/semantic_renderer_w96_b4_qat4_fixedtau05_tail6k_lr2e7.pt \
  --bits 4 \
  --weight-qat-q3q4 \
  --steps 3000 --lr 2e-7 --float-warmup-steps 0 \
  --checkpoint-every 250 --eval-every 250 \
  --device mps --seed 20260715 \
  --out  <SSD>/ddm_b2e_f2_alone/result.json \
  --save <SSD>/ddm_b2e_f2_alone/checkpoints
```

`--bits` stays **4**. F2 does not relax the int4 refusal — it requests the mixed grid through its own
flag, so the uniform path is untouched and byte-identical whenever F2 is off.

### Window derivation (stated as derivation, not measurement)

The init is itself a **6k-step tail at lr 2e-7** (`…tail6k_lr2e7`), i.e. the object is already
converged and was last moved by a low-LR tail. The F2 row is a *regrid*, not a retrain: it needs only
enough steps to relax the 4 q3-target tensors onto the coarser grid. 3,000 steps at the same tail LR
is one half of the init's own tail — the smallest window that is a genuine tail rather than a nudge.
`--steps` also sets EMA decay via `resolve_ema_policy(args.steps, …)`, so shortening the window
correctly speeds the EMA rather than leaving a stale shadow.

**Wall-clock is NOT derived here.** I have no timing measurement for this trainer on this box and I
will not invent one (first-attempt wall-clock is not a verdict, and a fabricated duration is worse
than an absent one). MAIN should take a ~50-step timing smoke to convert steps → hours before
committing the slot.

### Resumability (P0) — satisfied by construction

`--checkpoint-every 250` over 3,000 steps = 12 per-stage checkpoints. The trainer's resume payload
already carries model + EMA + optimizer + scheduler + generator + order/cursor + RNG + history +
best-EMA state, and writes the EMA shadow as the deployment state. The new lever flags are inside the
causal config, and the guard I added is **additive**: a pre-lever checkpoint resumes only while every
lever is inert; with `--weight-qat-q3q4` set it refuses, correctly, because the run genuinely differs.
17 tests pin that behaviour.

### Endpoint obligations

1. `ddm_b2e_edit_replay_admission.py replay --checkpoint <burn2 ckpt>` → weight-space deltas.
2. Measure pose/seg on the seeded stratified pair set (`… pairs` emits the exact ids).
3. `… admit --measured <rows>` → ADMITTED / REFUSED against the 50× bar.
4. Only then: byte-close and the exact-eval chain.

---

## REMAINING GATE CHECKLIST

| # | Gate | State |
|---|---|---|
| 1 | Levers built, default-off, byte-identical when off | **PASS** (83 tests) |
| 2 | F2 trains on the deployed grid | **PASS** (rtol=0/atol=0 vs `sd1.quantized_tensor`; levers-off `parameter_overrides` == PR130 `quantized_forward` exactly) |
| 3 | Admission bar pre-registered | **PASS** (50×, in code + test) |
| 4 | Edit constructions are the shipped ones | **PASS** (ns1 §A reproduced to 4 decimals) |
| 5 | Subset-bias law applied | **PASS** (stratified n32, bias-tagged) |
| 6 | Warm-start object identified + verified | **PASS with a quantified caveat** (inherited PR135; 37/38 exact; 2-element FiLM delta documented; no optimizer state) |
| 7 | Host trainer + lever wiring | **PASS** (`277fc58d13`) |
| 8 | Resume config-identity handled additively | **PASS** (17 tests; refuses active levers vs pre-lever parent) |
| 9 | Schedule + per-stage checkpoints | **PASS** (3,000 steps / 250 = 12 checkpoints; derivation above) |
| 10 | **PIN-1: `--challenge-root`** | **RESOLVED by MAIN 08-16**: `upstream/` — modules.py sha 065961ba… matches sd1 EXPECTED_UPSTREAM_MODULES_SHA256 exactly |
| 11 | **PIN-2: `--cache`** | **RESOLVED by MAIN 08-16**: `/Volumes/VertigoDataTier/pact/ddm_op1r_20260809/authority_cache/gt_cache_600_official_ada.pt` (112.5 MB) — sha 382d7dfe38b37c0cc5017e5645032faa045af6924db66e0b67549cc96c840195 matches sd1 pin exactly |
| 12 | Timing + memory preflight at the real config | **OPEN** — ~50-step smoke; MAIN fires |
| 13 | Governor admission | **NOT REQUESTED** (correctly, while 10–12 are open) |

**Nothing was fired.** No training, no Modal, no n600 scorer pass; the scorer slot was free at start
and was never used. `upstream/` untouched.
