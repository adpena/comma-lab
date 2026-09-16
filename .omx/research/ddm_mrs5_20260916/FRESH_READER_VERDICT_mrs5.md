# Fresh-reader verdict on submissions/mrs5 (Opus reader, no context, 2026-09-16 ~19:50Z) — recorded verbatim by MAIN

Verdict: **No, not as-is.** Reasons (the reader's words, condensed by MAIN only in headings):
1. Timing unknown for the thing submitted (README says the rewrite "still requires its own T4 timing").
2. Output declared device-dependent; deterministic algorithms disabled (inflate.py 2883–2884); "a score is a property of the runner".
3. Exits unless the file list is exactly ['0.mkv'] (line 2856).
4. Two implementations of the same model (corrector.c 43–84 vs the Python mixer) with no equivalence test; only 13 constants compared; a silent divergence changes bytes without an error.
5. Unreadable literals: line 1794 (1,442 chars), 1625, 1789, 1903 (`context_SHIPPED_CONFIG`) — "no reviewer reads them".
6. Undocumented switches: `outer[7] & 128` "mixer rider" (2551); `reserved & archive_PRIOR_ADAPTIVE` / `archive_RENDERER_PLANES` (2813–2817); `config[4] & 15` (2413). No format document.
7. Half-removed features: `overlay_split_selector_compensation` returns nothing for a non-empty overlay while callers unpack two values (1243–1248); `compensation_blob` threaded and never read (2529); dead guards (2496, 2543); `padding_mask` computed and discarded (1113).
8. Mechanical flattening: `render_render_video`, `miss_miss_cells`, `basis_basis_contexts`, `inference_inference_round`, ~20 module-name prefixes; four methods monkeypatched onto a live object (2164–2167); `np.testing.assert_array_equal` inside the decode loop (2490).
9. Jargon in README/code: "the original receiver's measured native set", "reviewer budget", "sealed receiver's exact supported counted parameter domain", "Free corrector", "rider", "lane".
10. Clean finding: no learned constants outside the archive; the C files carry only structural tables.
11. No build/test instructions beyond the two-command recipe; no way to check the C and Python paths agree.
Single most-improving change: a short archive-format specification naming every section, every header flag bit, and all 23 context families.
Dependencies seen: numpy, torch (CUDA by default), brotli; cc with C11/-fPIC/-ffp-contract=off/-lm; POSIX sh; correctly-rounded sqrt (asserted in both paths).
