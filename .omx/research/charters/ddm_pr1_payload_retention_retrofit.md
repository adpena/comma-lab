# ddm_pr1 — P0 DEF CON 1000: make ALWAYS-KEEP-THE-PAYLOAD structural, and count the population honestly.

**Owner:** codex arm · scorer-FREE · no Modal · no training launch · P0 (task #1001)

## OPTIMAL FORM (read first)

Reference form: a wired, positive-controlled gate whose live count is an HONEST population (one
discarded payload = one finding), plus retention actually retrofitted into the live measurement
harnesses so the class stops reproducing. Declared reductions: SCOPE only — retrofit may be ordered
by value (coder races first, they are the ANS shape) and the rest queued with a fire-order.
MECHANISM reductions are TOY-BRACKET: a count that keys on `(root, file)` and inflates one payload
into N findings; a "wired" claim without an executed rc=1 positive control; a strict flip declared
while real violations stand.

**Authority:** full online research, full OSS, full internal leverage — our own code, docs, and
unwired modules are off the shelf to use, adapt, refactor, extend, or fork. Cite path + commit.

## THE OPERATOR BINDING

2026-08-09, three messages: *"You should be constantly keeping payloads. You shouldn't be running
anything that doesn't keep the payload."* → *"That's nonnegotiable. P zero."* → *"Def con 1000."*

Landed already (commit `77160d7418`): CLAUDE.md `## ⛔ ALWAYS KEEP THE PAYLOAD` (subordinate only to
NO-FAKE and THE GOAL) · P0 ledger row `p0_always_keep_the_payload_20260809` · memory
`always_keep_the_payload_never_run_a_measure_and_discard_20260809` · `src/tac/payload_retention_gate.py`
with 24 tests and four executed controls.

**The anchor.** `ans_real_n600.py:37` `rng = len(enc.get_compressed().tobytes()); del enc` and `:41`
`an = len(ans.get_compressed().tobytes())`, then `:47` writes ONLY a scalar JSON. 681 s of n600
encode; both coder payloads discarded. Cost: TWO full re-encodes (`dt1`, `ap1`). And the recovered
ANS payload then measured **−2,120 B** against the shipped range coder — so the discard also DELAYED
a real rate win. The signature is NOT `del`; it is a scalar-only persisted artifact.

## THE THREE JOBS — in this order

**1. WIRE IT.** MAIN built the gate and did **not** add it to `preflight_all()`. State that plainly in
your landing; do not paper over it. Wire it warn-only, and execute a REAL positive control (a temp
file with the anchor shape → the gate returns rc=1 / raises). A gate nobody runs is the orphan class
this whole P0 exists to kill — see `[[orphan_sweeps_that_do_not_write_the_store_are_the_disease_20260803]]`.
Note honestly whether the commit hook actually reaches it (task #842/#852: 502 of 502 gate call sites
are skipped on a normal commit) — if it does not, say so rather than claim coverage you do not have.

**2. COUNT IT HONESTLY.** MAIN's raw scan reports **427 sites / 51,000 repo `.py`** and
**1,289 / 184,639 SSD `.py`** — both UPPER BOUNDS. The gate keys findings on `(root, file)`, so one
call like `zlib.compress(cq.tobytes() + np.float32(s).tobytes(), 9)` yields THREE roots (`zlib`,
`cq`, `np`) for ONE payload. That is the #821 "one fact counted N times" law reappearing inside our
own instrument. Fix the keying (per call-site AND per payload), re-run, and report the corrected
population **with its denominator** — never a bare count (the vacuity law).

**3. RETROFIT.** Most hits are coder races that compute `len(brotli.compress(payload, …))` and drop
the coded bytes. That is exactly the ANS shape: the race WINNER's payload is what the next step
needs. Retrofit retention (persist to the SSD tier + record sha256 and byte count in the result JSON)
into the live harnesses, ordered by value. Where a site is a genuine scalar-only probe, apply the
same-line `# MEASURE_ONLY_OK:<substantive rationale>` waiver — placeholders are rejected by design.
Drive toward a strict flip, and **do not declare one that is not earned.**

## HARD RULES

- Bulk artifacts → `/Volumes/VertigoDataTier/pact/ddm_pr1_20260809/`. No `/tmp` in evidence.
- Commits via `tools/subagent_commit_serializer.py`, POST-EDIT `--expected-content-sha256` per file,
  tags `[no-triality] [p0-ledger-ok]`, **no attribution trailer of any kind**.
- `.py` files: 2 × `tools/review_tracker.py mark-file <f> --status reviewed`; never
  `REVIEW_GATE_OVERRIDE=1` with a `.py`.
- `upstream/` IMMUTABLE. Intake clones READ-ONLY. Sister arms' landed artifacts APPEND-ONLY.
- Dogfood: run the gate on every `.py` you write in this arm. An arm that violates its own P0 while
  landing it is the worst possible outcome.

## DELIVERABLE

The wire-in with an executed positive control · the corrected population with its denominator and the
keying fix that produced it · the retrofit rows actually landed, with what you deliberately left
queued and why. If the honest population turns out far smaller than 427 — say so loudly. Correcting
my inflated number downward is a real finding and exactly the discipline this P0 encodes.
