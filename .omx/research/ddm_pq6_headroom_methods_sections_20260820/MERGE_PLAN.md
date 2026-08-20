# ddm_pq6 — MERGE PLAN: where each section goes, at what length, and what stays private

`date_utc: 2026-08-20` · `owner: ddm_pq6` · `score_claim: false` · `frontier_moved: false`

**This arm edited nothing.** MAIN applies this after `ddm_pq5`'s tone pass lands, at packet
freeze. Slot targets below were read against `PR_BODY_DRAFT.md`, `README_PUBLIC.md` and
`REPORT_PUBLIC.txt` at commit `d71363ea7d` (pq5's zero-negativity pass).

---

## 0. The two binding constraints on this merge

**Constraint 1 — the PR body stays template-terse.** The upstream contest PR template's structure
does not expand. §A–§G contribute **at most 1–2 sentences each** to the PR body, and only inside
sections the template already has. Everything else routes to the README and the report.

**Constraint 2 — the README and report carry real depth.** Operator calibration 2026-08-20: the
mechanisms and the discoveries are the signal another competitor can use, so the reasoning chain
travels with the number. Depth is not length: every sentence that lands carries a fact, a number
or a mechanism step.

**Two invariants MAIN must not break.**

- **The mirror invariant.** The ```` ```text ```` block in `PR_BODY_DRAFT.md` is byte-identical to
  `REPORT_PUBLIC.txt`. Anything landing in one lands in the other, in lockstep, and equality is
  re-checked programmatically afterwards (pq4 `MAIN_HANDOFF.md` §2).
- **Borrow-disclosure reads first.** pq4 placed "What else in this work is ours" *after* the
  borrowed-substrate section on purpose. Nothing from §C or §D may be inserted above the borrow
  disclosure.

**Hash safety.** `README.md`, `report.txt` and `BORROWED_SUBSTRATE_ACCOUNTING.md` are **outside**
the 33-row runtime manifest, so editing them cannot move `runtime_tree_sha256`
`2103073d739fc3f27d329ea0785ea3010307360c2380af0476e16d0f5b57cb9b` and cannot invalidate the
`0.14839100138338618` row. `inflate.sh` and `inflate.py` **are** in the manifest and are not
touched by this merge. Verified from `gen5_receipts/provenance.json` by pq4; MAIN should re-verify
rather than trust this line.

---

## 1. Slot table

| § | PR body — 1–2 sentences, existing section only | README — full depth | Report — mirror block |
|---|---|---|---|
| **§A** headroom | 2 sentences in **"competitive or innovative?"**: the leg split with rate at 81.05%, and the one-line statement that the token stream's residual calibration is measured as nearly spent while the 37.7% model half has no design against it. | **New section, after "What this submission is".** Full §A: exchange rate, section budget with its 176,420 B scope caveat, the three decode-identical model-axis wins, the withdrawn reservoir, the one-pixel-wide segmentation residual, the 95.9%/99.9985% split, the carrier fixed point, the floor band. | No. Report stays an evidence document. |
| **§B** directions | 1 sentence in **"additional comments"**: the two priced-and-unbuilt items (`−0.002929` third branch, `−3.243e-3` reopened rung) named with their blockers. | **New section, after §A.** Full §B items 1–9 **plus the closed-directions block**, which is the part another competitor can most directly use. | No. |
| **§C** method | 2 sentences in **"changes from upstream"**: the five move classes exist as one module set with realized-only acceptance; the mixer is a weighted geometric mean in log-odds reached by radicals so it is bit-identical across platforms. | **New section, after the borrow disclosure.** Full §C including the hull inequality, the radical construction, the sign flip, the nesting controls, the bit-share numbers, the re-encoder, the seal. | No. |
| **§D** repair | 2 sentences in **"competitive or innovative?"**, adjacent to the existing leg-split paragraph: segmentation edits cost 13.4× more pose than they buy segmentation, and the shipped candidate is admissible only because of the compensation and joint-admission machinery. | **New section, after §C. This is the highest-value section to publish in full.** The counting argument (6 equations, 12 coefficients), the shared-resize mechanism, the lattice-resampling measurement, penalty-versus-projection, all four repairs, both composition laws, the batch-shape caveat. | 1 short block: the 13.4×, the 571-of-573, and the 455-of-573 admission with its net. Mirror into the PR body's ```` ```text ```` block. |
| **§E** realization | 1 sentence in **"changes from upstream"**: the three in-loop placements (resize, uint8, colour) are enforced independently and a candidate missing any of them is refused. | **New section, after §D.** Full §E: the path, the three placements, the four measured walls, the two quantization constraints including the 38,700×-versus-2,518× amplification finding, the five in-loop guards. | No. |
| **§F** ordering | 1 sentence in **"changes from upstream"**: the pipeline order is derived from measured axis interactions, not chosen. | **New section, after §E.** Full §F: the five upstream facts, the four score-composition facts, the eight-step order with the reason each step holds its place. | No. |
| **§G** limits | **Nothing new.** pq5 already consolidated a "Known limits" section; §G's items either duplicate it or belong to the README. MAIN adds at most the two items pq5's list lacks (see §3). | **Merge into the existing limits surface**, not a new section — one limits list, not two. | Existing residual-band disagreement stays as pq5 wrote it. |

---

## 2. Length budget

| Target | Added words | Rationale |
|---|---|---|
| `PR_BODY_DRAFT.md` | **≤ 200** total across all seven sections | Template structure unchanged. No new top-level headings. |
| `README_PUBLIC.md` | **≈ 2,600–3,200** | Five new sections at real depth. This is where a reader re-derives or extends the technique. |
| `REPORT_PUBLIC.txt` | **≤ 90** | One §D block only, mirrored into the PR body. |

If the README lands materially above this band, cut whole *claims* rather than trimming
qualifiers — a shortened caveat is worse than a removed claim.

---

## 3. §G reconciliation — do not create a second limits list

pq5's consolidated "Known limits" already carries: the residual-band disagreement, the CPU-axis
absence, the single-evaluation runtime pin, the native port not being in the tree, and the
compression script's cannot-rebuild boundary.

**Items §G adds that are not in that list** — MAIN adds these two, in pq5's register, into the
existing section:

1. **24 of the 34 files in the evaluated candidate tree have no source in version control**,
   including the receiver. pq4 put a conservative version of this in the README's Reproduction
   section; the limits list does not carry it.
2. **One published mechanism is a measured null** — re-mixing the 12 stored basis dimensions
   leaves the reachable pose correction invariant to `1.9e-08`. It ships nothing.

**Items §G carries that belong in the README's method sections, not the limits list**, because
they are properties of the method rather than of this packet: the toy/naive audit grades, the
charter-premise falsifications, the re-derivation latency, the reopened-verdict tally, and the
selection-on-the-scored-clip disclosure. The last one must travel **with §C's mixer text**, in the
same subsection, so a reader meets the −560.07 B and its selection caveat together.

---

## 4. Ordering, and the two failure modes it avoids

Apply in this order:

1. Wait for `ddm_pq5` to land its tone pass. Re-read the three documents at that commit; do not
   apply against the versions this plan was written for without re-checking.
2. Apply README sections §A → §B → §C → §D → §E → §F, in that order, **all below the borrowed-
   substrate pointer**.
3. Apply the §G reconciliation into the existing limits section.
4. Apply the PR-body sentences last, into existing template sections only.
5. Apply the §D report block and re-verify the mirror equality programmatically.
6. Re-stage through `tools/stage_contest_submission_packet.py`, then re-run
   `tools/packet_census_guard.py`. Per pq3: **purge, then census, then buy the receipt, with no
   writes in between.**

The two failure modes this order avoids: applying the PR-body sentences first, which invites the
README text to be written to match a summary rather than the receipts; and editing the report
without re-checking the mirror, which is the drift pq4 flagged.

---

## 5. What does NOT go public

Checked against the Public Disclosure Hygiene rules. None of the following appears in
`SECTIONS.md`, and none may be introduced while merging.

| Class | Rule | Status in SECTIONS.md |
|---|---|---|
| Custody roots | No `/Volumes/...` path in any operator-facing or public surface | **Absent.** All receipts in `EVIDENCE.md` are repo-relative; retained payload roots are referenced by role, never by path. |
| Fleet | No Tailscale IPs, hostnames, or machine roles | **Absent.** Local measurements are labelled `[macOS-CPU advisory]` with no host identity. |
| Operator state | No `.omx/state/*` contents, session ids, hot-state, arm queue, or task-ledger internals | **Absent from SECTIONS.md.** Two ledger-sourced facts appear in `EVIDENCE.md` only, cited by content. |
| Provider transcripts / spend | No raw dispatch logs or account metadata | **Absent.** Cost appears only as the PR template's own build-cost field, which pq4/pq5 already fill. |
| Internal arm codenames | Not published without a gloss | **Absent from SECTIONS.md prose.** Codenames live in `EVIDENCE.md` receipts, which is an internal document. `jg5_joint_waterfill` is already the public submission name. |
| Unpublished operator directives | Never quoted publicly | **Absent.** The register and depth steers are recorded in SECTIONS' front matter as constraints on this draft; **MAIN removes that front matter before any text is published.** |
| Numeric task ids | Cited by content, never bare | Honoured — `EVIDENCE.md` §9 cites rows by content. |

**Two judgement calls flagged for MAIN rather than taken here.**

1. **§G item "24 of 34 files have no source in version control."** This is a genuine
   reproducibility limit and pq4 already disclosed a conservative form of it. It is also a
   statement about our own engineering hygiene. **Recommendation: publish it** — a reader
   attempting reproduction will hit it, and finding it undisclosed is worse than reading it. MAIN
   decides.
2. **The residual-band disagreement between two of our own projections.** pq5 has already written
   this into the packet. §G repeats it only so the limits list stays single-sourced. **No new
   disclosure decision is required.**

---

## 6. Handoff

- `SECTIONS.md` — the seven drafts, publication register, depth restored per the 2026-08-20
  calibration.
- `EVIDENCE.md` — receipt per number; §X records 3 claims cut, 1 number corrected against pq5,
  and 8 kept with explicit scope.
- Nothing in this directory is staged into the packet. MAIN copies text; MAIN does not copy files.
