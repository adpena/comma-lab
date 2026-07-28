# DDM EU1 — evaluator-first codec theory and eureka hunt

**Pointer honesty — MEASURED historical local-custody anchor:** `0.1910828242`
`[contest-CPU]` is **UNMOVED**. Nothing in this research-only memo creates an
archive, runs an exact evaluation, or moves any frontier pointer. The current
competitive/effective pointer is a separate pointer-file concern; `0.1910828242`
is used here only in the exact local-custody sense required by this delegation.

**DERIVED — authority boundary:** research only; no launch, no paid work, no training, no
n600 scorer job, no candidate promotion. MAIN landing review is required.

**MEASURED — phase-order receipt:** the scientific body of P1 below was written before
reading any `ddm_*` campaign artifact, TR1 specification, FD2 receipt, EE1
memo, or campaign memory. A subsequent serializer-preflight check exposed
recent commit subjects before this first commit; no P1 scientific text was
changed in response. The only scientific inputs to P1 were the pinned upstream
evaluator, the pinned scorer definitions, the pinned upstream README, and basic
non-scorer video statistics. After P1 was committed, the scientific read order
was: campaign memory index; the requested campaign memos and DAG feeds; TR1;
fd2; and EE1 last among campaign interpretation artifacts. Mechanical receipt
extraction, older-artifact recall, and external prior-art verification followed.

**MEASURED — STORES CONSULTED:** P1: `upstream/evaluate.py`,
`upstream/modules.py`, `upstream/README.md`, `upstream/videos/0.mkv` basic
container/downsample statistics only. P2–P4, read only after the P1 commit:
the complete requested `ddm_{fc1,da1,ar1,sc1,sp1,rp1,pp1,fd1,ch1,co6}`
07-28 memo/DAG set; `SPEC_tr1_trained_partition_renderer_20260728.md`; the
external fd2 receipt; EE1 **last among campaign interpretation artifacts**;
campaign `MEMORY.md`; load-bearing SSD
receipts; the older DR1 n600 realization-race receipt and exact-lattice receipt
found by recall-before-decide; and the primary papers/repos linked below.

## P1 — uncontaminated evaluator-first derivation

### P1.0 Primary-source custody

- **MEASURED:** source inspection gives `evaluate.py` SHA-256
  `7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b`;
  `modules.py` SHA-256
  `065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa`;
  README SHA-256
  `68ea239d7333696e79716e47a9c4288d2918efbcd8912f78932b0befe0af872b`.
- **MEASURED:** the sole video is `37,545,489` bytes, SHA-256
  `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`,
  HEVC Main, `1164×874`, `yuv420p`, 20 fps, 1,200 frames, 60 seconds.
- **MEASURED, BASIC-STATISTIC ONLY:** an area-downsampled `64×48` RGB read of
  all 1,200 frames has adjacent-frame pixel correlation `0.983360`,
  adjacent RGB MAD `1.7326`, two-frame RGB MAD `2.1875`, and adjacent absolute
  difference quantiles `(q50,q90,q99)=(1,4,13)`. This establishes strong
  temporal redundancy at a coarse scale; it is not scorer evidence and is not
  a distortion or score result.

### P1.1 The actual mathematical object

**DERIVED:** Let pair \(i\) contain reconstructed uint8 frames
\(y_i=(y_{i,0},y_{i,1})\). From `modules.py`, SegNet observes only
\(y_{i,1}\), resizes it bilinearly to `512×384`, and is scored only through
five-way argmax disagreement. PoseNet observes both frames through a
12-channel YUV6 representation, also at `512×384`, and only its first six
outputs enter an MSE. Therefore the exact distortion is

\[
D_{\rm seg}(y)=\frac{1}{600HW}\sum_{i,u}
  1[\arg\max G(y_{i,1})_u \ne m_{i,u}],
\qquad
D_{\rm pose}(y)=\frac{1}{600}\sum_i
  \|H(y_{i,0},y_{i,1})_{:6}-p_i\|_2^2/6,
\]

where \(m_i\) and \(p_i\) are encoder-side targets obtained from the original.
The scored program minimizes

\[
100D_{\rm seg}+\sqrt{10D_{\rm pose}}
+25\,|\mathrm{archive.zip}|/37{,}545{,}489.
\]

**DERIVED:** This is not fundamentally an RGB-reconstruction problem. It is a
minimum-description representative problem over evaluator equivalence cells:
find the shortest legal message whose decoder emits any uint8 frame pair in
the intersection

\[
\mathcal C_i(m_i,p_i)=
\{(a,b):\arg\max G(b)=m_i,\ H(a,b)_{:6}=p_i\},
\]

with controlled relaxations when exact intersection membership costs more
bytes than its score value. Pixel fidelity has no independent term.

**DERIVED:** Frame 0 is asymmetric: it has no direct SegNet obligation, while
PoseNet still couples both frames. A symmetric two-frame RGB-fidelity codec can
therefore spend rate on constraints the evaluator never imposes.

**CONJECTURE:** a promising, but not mathematically forced, optimizer
factorization is to place frame 1 in robust SegNet cells and then exploit frame
0 plus SegNet-null frame-1 directions for PoseNet. The global joint optimum need
not be lexicographic because changing either frame can alter PoseNet.

### P1.2 Byte economics

**DERIVED:** one archive byte costs
`25/37,545,489 = 6.6585895312e-7` score units. A binary kilobyte costs about
`0.0006818`; 64 KiB costs `0.04363773`.

**DERIVED:** at 64 KiB, a sub-0.15 row has only `0.10636227` total distortion
budget. If \(D_{\rm seg}=5\times10^{-4}\), PoseNet must satisfy approximately
\(D_{\rm pose}<3.1767\times10^{-4}\). If
\(D_{\rm seg}=3\times10^{-4}\), the corresponding bound relaxes to about
\(5.8312\times10^{-4}\). These are feasibility arithmetic for a hypothetical
**whole 64 KiB archive**, not the separate G1 renderer-state cap; tokens, Pose,
headers, and every other counted section must be included before applying these
thresholds.

**DERIVED:** admitting a payload increment \(\Delta B>0\) is rational only if

\[
-100\Delta D_{\rm seg}
-\left(\sqrt{10D_{\rm pose,new}}-\sqrt{10D_{\rm pose,old}}\right)
> 6.6585895312\times10^{-7}\Delta B.
\]

The square-root term makes pose byte value state-dependent; a fixed
Seg/Pose loss-weight ratio cannot be globally optimal.

### P1.3 Derived codec form

**CONJECTURE — minimum-description witness codec:** the optimal family is a
task-space witness compiler with five jointly designed parts:

1. **Encoder-side solved witnesses.** Use the frozen scorers only offline to
   solve uint8 frame pairs directly inside or near the target evaluator cells.
   The solve must include the exact resize, color conversion, clipping, and
   integer lattice; a continuous image later rounded is solving the wrong
   feasible set.
2. **A shared conditional renderer.** Amortize repeated road-scene structure
   across all 600 pairs with one compact, quantized decoder. It should emit
   piecewise-smooth regions and sharp learned boundaries rather than optimize
   generic perceptual texture.
3. **Predictive pair tokens.** Entropy-code a temporally predictive stream of
   low-dimensional pair state. The measured `0.983360` coarse adjacent
   correlation supports prediction, but the latent dimension and entropy are
   OPEN-QUESTIONs until measured on the learned tokens.
4. **Sparse evaluator certificates.** Spend exceptions only on sites/pairs
   whose evaluator-cell violation has positive score value per byte. Each
   exception needs a receiver-consumption proof; a stored target that the
   renderer does not consume has zero mechanism.
5. **Maximal generic decoder, minimal video-derived message.** The decoder
   algorithm, deterministic bases, integer rasterizer, entropy decoder, and
   generic optimization machinery belong in the free runtime. Every
   video-derived weight, token, target, exception, or learned table belongs in
   the counted archive.

**CONJECTURE — teacher-to-packet path:** first solve a large unconstrained
teacher object on the exact discrete evaluator surface; next distill its
evaluator outputs into the small renderer and predictive tokens; finally
optimize archive bytes and task distortion jointly. The teacher is useful as
an existence witness and initialization oracle, not as a shippable payload.

**CONJECTURE — cell-interior training:** because SegNet scores argmax, the
renderer should maximize the minimum winner–runner-up robustness under the
actual integer/resize neighborhood at already-correct sites, while spending
capacity on incorrect sites. Cross-entropy everywhere can waste bits increasing
already-safe margins that do not change \(D_{\rm seg}\).

**CONJECTURE — integer-native terminal optimizer:** once continuous descent
reaches a sub-quantum basin, switch to discrete coordinate/block proposals on
the uint8 lattice, ranked by exact evaluator-cell changes per coded bit.
Continuous straight-through gradients are proposal generators, not terminal
authority.

### P1.4 Falsifiers derived before campaign recall

1. **OPEN-QUESTION — G1 renderer falsifier:** on all 600 pairs and the exact
   realized uint8/resize path, can a counted renderer of at most 64 KiB reach
   native \(D_{\rm seg}\le5\times10^{-4}\), with
   \(3\times10^{-4}\) as the stretch target? A smaller sample is not a verdict.
2. **OPEN-QUESTION — amortization falsifier:** after entropy coding, do shared
   weights plus predictive pair tokens beat direct compressed witness frames
   at equal exact task distortion?
3. **OPEN-QUESTION — asymmetry falsifier:** does reserving frame 0 primarily
   for pose reduce total score relative to a symmetric renderer at identical
   archive bytes?
4. **OPEN-QUESTION — cell-interior falsifier:** at equal bytes, does
   margin/cell-aware allocation reduce realized argmax error more than dense
   logit or RGB regression?
5. **OPEN-QUESTION — discrete-finish falsifier:** after continuous convergence,
   does an integer-native local optimizer find positive net
   \(\Delta S/\mathrm{byte}\) moves that round-to-nearest/STE misses?

## P2 — verified transformed state

**MEASURED — P1 seal:** P1 is sealed at commit `e0e8232b83`. The following facts were independently
re-extracted from machine receipts rather than copied from memo prose.

### P2.0 Receipt audit

| surface | epistemic status | receipt-audited fact | receipt SHA-256 |
|---|---|---|---|
| fc1 correction object | **MEASURED support/labels; DERIVED conservative design rails** | 1,019,467 flips; support LZMA **421,366 B**; labels **41,392 B**; 0.172/0.15 archive rails **187,727/154,522 B**, reserving about **0.047 S** for total distortion rather than using pp1's exact conditional distortion | `5d0056f9…` entropy; `e10914b5…` coders |
| sc1 pose residual proxy | **MEASURED proxy, not terminal mechanism** | centered first SVD fraction **0.9986237**; AR-int5 residual proxy **2,039 B**; flat-paint uncorrected d_pose **1.9619** | `3156ee13…` |
| sp1 explicit correction support | **MEASURED on this formulation/base** | contour support **444,394 B**, worse than 421,366 B LZMA; best lossy support-only action **0.279988** | `170a53b8…` |
| pp1 direct partition | **DERIVED n600 closed-form KT length from MEASURED counts; MEASURED coder correspondence only on six frames** | best lossless context length **173,616.5 B**; 2,436 boundary px/frame; temporal disagreement **1.2456%** | `cea819c1…` |
| pp1 band law | **DERIVED interpolation over a MEASURED synthetic coder curve** | water **1.2731 B/error**; coherent crossing \(\rho_c=\)**5.0150e-4**; uniform crossing **8.5915e-4** | `63018430…` |
| rp1 GT range-carrier | **MEASURED n600 advisory, re-aggregated from five chunks** | **42,816** flips; d_seg **3.62956e-4**; d_pose **3.96473e-4**; pair-mean margins **0.033706 flipped / 5.613614 held = 166.55×** | five full hashes below |
| fd1 box-solve range-carrier | **MEASURED n600 advisory** | cell-hold flip rate **3.75739e-4**; box d_seg **1.1600e-3 → 1.2492e-3**; pair-mean margin ratio **165.23×** | `780c0382…` |
| fd1 GN window | **MEASURED formulation-scoped n600 advisory** | 2 steps × 3 multipliers = **0/6 accepted**; 5/6 candidates leave n600 d_seg bit-identical; the one mover worsens; both windows classify `BLOCK_LOCALITY_OR_REALIZATION_GAP` | `ddd1e3a9…` |
| fd2 disambiguator | **MEASURED formulation-scoped sampled advisory** | separate n600 canary passes; Q2 verdict surface is 4 block pairs + 32 sampled off-block pairs. There, ×1.0 moves 489 block sites but worsens block/off-block d_seg; ×0.5/×0.25 change description state yet produce **zero** block/off-block argmax changes and zero d_seg delta | `92deae9d…` |

**MEASURED — exact receipt custody:** full paths and SHA-256 values used above:

- `/Volumes/VertigoDataTier/pact/ddm_fc1_20260728/entropy_n600.json` —
  `5d0056f9aa1954d6cde644e631187cd1386b8503d2c4548f7b546671942c8663`;
  `/Volumes/VertigoDataTier/pact/ddm_fc1_20260728/stage2_coders_n600.json` —
  `e10914b59fd4d3b5762d39fc0e87291f2267f93514485ad4c34579f1f9a65794`.
- `/Volumes/VertigoDataTier/pact/ddm_sc1_20260728/ep_probe_receipt.json` —
  `3156ee13ac2464e9d08d2d799d5cb3f4d048c435f6115c92c28e275009124e8f`;
  `/Volumes/VertigoDataTier/pact/ddm_sp1_20260728/r1_contour_support_n600.json` —
  `170a53b8c23d4b560c9afaf57788d5f90fdec7adc6aee9d282cc2c06ad7d517a`.
- `/Volumes/VertigoDataTier/pact/ddm_pp1_20260728/r1_direct_partition_n600.json` —
  `cea819c18cdd19da9b0d10cf184a704badfdc67a7a5b8f4daaf5e51d8787cb1e`;
  `/Volumes/VertigoDataTier/pact/ddm_pp1_20260728/r2_band_lemma_curve_n600.json` —
  `63018430e7e42b25dbe21fc25635b98469319b2e977be6080d17919480a47fe9`.
- rp1 chunks `/Volumes/VertigoDataTier/pact/ddm_rp1_20260728/chunks/`:
  `chunk_gt_0000_0120.json` —
  `f5c9804368b3f03c986f78293d8e0f556fa552cb2b88a674455fbd5091efc275`;
  `chunk_gt_0120_0240.json` —
  `0334b79d8842a12c074d191dec5948832b48d7b6b050c3a3897be53210b82e72`;
  `chunk_gt_0240_0360.json` —
  `b3ef9ad42c0bce8709be008c6926598c46983083bc0575c5274f5c1f138dea81`;
  `chunk_gt_0360_0480.json` —
  `c3f0679f51704dd35df41e89c36d7a8266810178fd6570e9be5273b691fec600`;
  `chunk_gt_0480_0600.json` —
  `5393f46dd1236cccc2dabc8ddc8b2e76eafd08264dded139a4bac0086473be50`.
- `/Volumes/VertigoDataTier/pact/ddm_fd1_20260728/s0_boxsolve_band_receipt.json` —
  `780c03825212fd38a8c7e6d38275f10ac41322a219993554863a42e3ac9ae657`;
  `/Volumes/VertigoDataTier/pact/ddm_fd1_20260728/s2_gn_window/fd1_gn_window_receipt.json` —
  `ddd1e3a983cbd13db67370f6979c70c3b6eacf604ee721d2b88ac62af46f912b`.
- `/Volumes/VertigoDataTier/pact/ddm_fd2_20260728/fd2_disambiguation_receipt.json` —
  `92deae9d4cf31a54fda2ea6405d7cfdbadc5fe7b45e9edfdb8b2af309d04af4e`.

**MEASURED — recalled source custody for conditional/existing-apparatus claims:**

- `.omx/research/ddm_pp1_direct_partition_pricing_20260728.md` —
  `456d608e9ae364f797599dfdc67529d1cd85a6f0e19f49276dec42ea4a703b0d`;
  `.omx/research/SPEC_tr1_trained_partition_renderer_20260728.md` —
  `2c3760c74ca12c5b8d6698753d3f49977402091f06788ab877f6993bda277857`;
  `.omx/research/ddm_ee1_einstein_fresh_eyes_capstone_20260728.md` —
  `fc3267a254068985e4990e320e220003af8316659998a9ade820c440e52ff17c`.
- `.omx/research/neural_selfcomp_sota_20260719_codex.md` —
  `47e26f9e0f1d9a97ae9c57487f015cb32634cad1e9f5a883c58d49c34d677e74`;
  `.omx/research/autoencoder_describe_crosswalk_20260721T232351Z.md` —
  `62476f912f43a3d1a5f31aa6ea41f4b427488b925f536fef254cc70fde4afe7c`.
- `src/tac/optimization/uint8_lattice_feasibility.py` —
  `70c98632bca585c2b6d6f02e77faf45bac155494154810d9167220a223f58f54`;
  `.omx/research/g2g2_joint_multichart_solve_20260721T174416Z.json` —
  `3fb56e48acb627e3adb37c2a627c21106a098cd9b96e5ece8ad4586a82b36244`;
  `src/tac/optimization/joint_seg_pose_rate.py` —
  `9ac258b5ec0244665f34b6a0b4c709e40533eb93f956c5921e52caf80a180708`;
  `src/tac/optimization/frame1_joint_safe_cone.py` —
  `a18d62c6682c3c2f7d2451359512a4ae5a8bd4371981ed88da2c12aae9f09d09`.

**MEASURED + OPEN-QUESTION — CURRENT G1 GATE:** no OUR-line G1 renderer row exists. Native
d_seg \(\le5\times10^{-4}\), pushed toward \(3\times10^{-4}\), at
\(\le64\) KiB counted renderer weights remains unmeasured. PR130 is an
external existence proof only; none of its bytes, weights, architecture
constants, or score transfers. **DERIVED:** the renderer cap is only one
component: TR1's conditional mid table uses decimal `130,000 B` tokens +
`64,000 B` renderer + `2,000 B` Pose = `196,000 B` total and
`0.17577 S` at d_seg `3e-4` and d_pose `2.33e-5`. Separately, TR1's
conservative design rails are `187,727 B` for 0.172 and `154,522 B` for 0.15;
they reserve approximately `0.047 S` for total distortion
(`0.0470003/0.0471101` respectively), about `0.00174 S` more than pp1's
conditional `0.0452643` distortion. They are not pp1's exact byte ceilings.

**DERIVED — CONDITIONAL RATE SCENARIO:** pp1's `0.1888345` composition assumes
the unbuilt/external legs `40,000 B` renderer, `2,000 B` pose,
`d_seg=3e-4`, and `d_pose=2.33e-5`, in addition to the derived
`173,616.5 B` partition. Holding those assumptions fixed requires **58,322 B**
less to reach 0.15 and **25,282 B** less to reach 0.172; improved distortion
reduces those byte requirements. Equivalently, the scenario's exact
0.15/0.172 byte ceilings are about **157,294/190,334 B**, distinct from the
conservative TR1 rails. Separately, `173,616.5 - 116,980 =
56,636.5 B` is the explicit-token gap to the **lessons-only external** PR130
token ledger cited by pp1/TR1, not an OUR measured learned-token row. Thus
syntax-only work is unlikely to close the conditional gap, but no 57 KB
requirement is a theorem. Exact committed-source custody is the pp1 memo
`456d608e9ae364f797599dfdc67529d1cd85a6f0e19f49276dec42ea4a703b0d`
and EE1 external-ledger memo
`fc3267a254068985e4990e320e220003af8316659998a9ade820c440e52ff17c`.

**MEASURED + DERIVED — REALIZATION SHAPE:** integer range-carrier realization holds
evaluator cells at about 3.6–3.8e-4 on two solved substrates, while the current
344-active-coordinate GN direction stalls below one hard quantum on fd2's
4-pair block + 32-pair off-block sample surface; only its canary is n600. This is not
evidence that the renderer family is dead. It says the terminal variable and
acceptance surface should be tested as hard/discrete; that prescription remains
**CONJECTURE** until such a search wins.

**MEASURED — POSE BOUNDARY:** the 2,039 B sc1 value is a residual-field proxy,
not a shipped terminal-pose proof; flat paint is pose-dead.
**CONJECTURE:** retain a joint-trained/terminal six-equation solve on realized
frames until a hard row falsifies it; "cheap sidecar" describes archive shape,
not a post-hoc mechanism.

### P2.1 Recall result that changes the eureka ranking

**MEASURED — recall result:** the older DR1 full-n600 advisory race is directly
relevant and prevents a rediscovery:

- **MEASURED, INSTANCE:** post-int8 lattice control reached d_seg
  `0.026260113186` at `135,563 B`, versus `0.027468354967` for pre-uint8
  uniform/Bayer8/resize-null-sigma-delta at `137,630/137,633/137,650 B`.
  All three pre-uint8 variants had identical d_pose
  `163.060485284886`. The post-int8 row improved the realized joint action by
  `-0.12083155` for `+235 B` against its own control. This is mechanism evidence,
  **not an admissible candidate**: the same receipt records
  `composed_candidate_admissible=false` because the ordered pairwise
  conditional-byte/redundancy audit remains blocked. Receipt
  `.omx/research/ddm_dr1_realization_race_coding_gain_n600_20260723/receipt.json`,
  SHA `a84112f0fc0063e43711a0e9c0777a51abac85795b990281ca08899b122a1104`.
- **MEASURED, NARROW NEGATIVE:** the tested Bayer8 and exact-resize-derived
  pre-uint8 sigma-delta instances changed no uint8 camera value relative to
  uniform placement. This kills those exact move classes on that exact base,
  not dithering or noise-shaping as families.
- **MEASURED, SMALL-SCOPE EXISTENCE:** the exact factor-2 lattice receipt
  certifies exact numerator solutions on six frames, while a same-\(R_h\)
  support fill is scorer-inert by construction. It is a feasibility/search
  substrate, not evidence of free score. Receipt
  `.omx/research/v10_uint8_lattice_feasibility_receipt_20260718.json`, SHA
  `665ce8ecd789a863eb85fa181f11746f292626b9283f82be9d90cf10b7905779`.

### P2.2 Corrections to the uncontaminated P1 priors

1. **MEASURED + DERIVED — P1 temporal prediction was too optimistic.** Campaign evidence says
   prediction-then-residual is neutral/worse; temporal structure pays as
   decoder-available conditioning context. Predictive *tokens* remain open
   only if an actual token stream exhibits that structure.
2. **MEASURED + DERIVED — the 166× margin gap is a population mean, not a certificate.** It
   motivates a max-volume evaluator-cell quantizer, but no coordinate may be
   coarsened from that mean alone. Lower tails, cross-site coupling, Pose, hard
   parse-back, and actual bytes must be checked.
3. **DERIVED — a same-\(R\) fiber is distortion-invariant on one pinned
   evaluator/hardware/backend axis.** Both scorers consume the same resized RGB
   object. Moving only inside a proven same-\(R_h\) fiber cannot improve Seg or
   Pose on that axis; its only legitimate use is an exact archive-byte
   tie-break. CPU/CUDA numerical identity is not inferred. Distortion search
   must move between *attainable* \(R_h\)-states.
4. **MEASURED wall + CONJECTURE prescription — P1's integer-native terminal
   optimizer remains open and better motivated.** fd2 supplies a wall diagnosis
   on its 4-pair block + 32-pair sample, not an n600 verdict; DR1
   supplies advisory mechanism evidence but no admissible candidate precedent.
   Smooth GN/STE may rank moves; only a hard state can be accepted.

## P3 — eureka hunt

### P3.0 The factorization: optimize the quotient, compress the fiber

**DERIVED:** fix a pinned evaluator/hardware/backend axis \(h\) and one
video-independent universal interpreter \(U\) in free code. Let \(m=(c,z)\) be
a legal counted message: \(z\) is learned payload and \(c\) is the complete
program/architecture/config selector whenever any such choice was informed by
this video. If a single decoder/config was genuinely precommitted without
video-derived selection, \(c\) may be an implicit constant. Let
\(q=R_h(U(m))\) be the exactly attainable scorer-plane pair sequence. The
terminal problem is:

\[
\min_{q\in\mathcal Q_{U,h}}
\left[
\underbrace{100D_{{\rm seg},h}(q)+\sqrt{10D_{{\rm pose},h}(q)}}_{\text{move BETWEEN exact evaluator states}}
+
\underbrace{\lambda\min_{m:\,R_h(U(m))=q}|\mathrm{archive}(m)|}_{\text{compress WITHIN a score-invariant fiber}}
\right].
\]

If a decoder-visible architecture, width/depth, mask density, modulation basis,
entropy context, seed, or program choice is selected using this video, its
complete reproducible selector/config belongs in counted \(c\); changing free
code to embody that selection is forbidden hide-data-in-code. Encoder-only
training hyperparameters belong in the external reproducibility receipt, not
the archive, unless runtime regeneration or interpretation consumes them. For
a learned receiver, the inner minimization is over
counted weights/tokens that parse back to the chosen \(q\), not arbitrary raw
frames. Equality is axis-local; CPU and CUDA require separate verification.
This quotient/fiber separation prevents two recurring errors:
searching continuous states the public wire cannot realize, and claiming a
same-\(R_h\) null move changes the score on the pinned axis.

### P3.1 Top three

1. **CONJECTURE — G1 fixed-seed scorer-native supermask renderer — highest-upside
   reactivation/application.** LotteryCodec was already harvested in
   `neural_selfcomp_sota_20260719_codex.md` and the autoencoder crosswalk; it is
   not a newly discovered mechanism. The new post-fd2 application is to train
   its counted mask/modulations/tokens directly through hard \(R_h\)+SegNet and
   judge it at G1's whole-archive rails. The universal seed, portable integer or
   specified fp32 PRNG/weight generation, architecture, width/depth, mask
   density, modulation basis, and entropy context must be precommitted
   generically; otherwise count the reproducible decoder-visible selector/config.
   Encoder-only training rules remain provenance unless the decoder consumes them.
2. **CONJECTURE — attainable-\(R\) quotient direct search — focused extension
   of existing hard repair.** `repair_with_hard_oracle` already implements
   deterministic ±1 repair, and G2g2 already exercised coordinate polish/swap
   on six selected pairs. The new delta is fd2's 344 description-coordinate
   domain, scorer-recursive top-\(K\) proposal ranking, and full-n600 joint ZIP
   acceptance; fd2 itself diagnosed the wall only on 4 block + 32 sampled
   off-block pairs. DR1 is advisory mechanism evidence with an unresolved
   redundancy blocker, not an admissible candidate precedent.
3. **CONJECTURE — receiver-validated evaluator-cell mixed precision — turn
   margin slack into rate.** Reuse
   `joint_seg_pose_rate.derive_hyperplane_channel_band` and
   `frame1_joint_safe_cone`; add the payload-coordinate pullback and actual ZIP
   Pareto close. Empirical secants provide heuristic proposal bounds only.
   Safety is called certified only after a sound interval/Lipschitz proof or
   exhaustive finite-lattice check; otherwise the exact hard endpoint is merely
   receiver-validated.

### P3.2 Exact leverage scale

- **DERIVED:** d_seg `1e-3 → 5e-4` is `-0.05 S`; `1e-3 → 3e-4` is `-0.07 S`.
  From the 291 MB teacher/box row `1.1599986e-3`, reaching `3e-4` would be
  `-0.0860 S`, before rate/Pose changes. These are conditional differences,
  not forecasts. Teacher custody:
  `.omx/research/r6cal_asbuilt_row_receipt_20260727.json`, SHA
  `4a076f7ae53f808f1db72b7850e803e791519e60078bd24450ba09fb61e2fcb1`
  (`291,205,400 B`, exact evaluator row; never a shippable-rate precedent).
- **DERIVED:** holding distortion fixed, 10 KiB/16 KiB/64 KiB saved is
  `-0.006818/-0.010910/-0.043638 S`. With Pose held, a 10 KiB saving breaks
  even only while added d_seg stays below `6.818e-5`.
- **OPEN-QUESTION:** the exact-S leverage of a hard fd2 move is not
  transferable from DR1; require a strict negative same-object joint action.

### P3.3 Excluded axes and anti-rediscovery fences

- **MEASURED, NARROW NEGATIVE:** do **not** rerun plain pre-uint8
  uniform/Bayer8/resize-null sigma-delta on the
  DR1 move class; that instance is already neutral. Only task-conditioned,
  output-changing, entropy-aware dependent rounding is open.
- **DERIVED, AXIS-SCOPED:** do **not** sell same-\(R_h\) kernel fill as
  distortion progress. It can only reduce actual coded bytes for an unchanged
  scorer input on the pinned axis.
- **MEASURED, FORMULATION-SCOPED:** do **not** build another explicit
  flip-support/contour correction stream on the measured fc1/sp1 base:
  421–444 KB closes that formulation. The pp1 \(\rho_c\) is only the crossing
  for its tested synthetic coherent-field distribution, position coder, and
  single-error semantics; it is not a universal cutoff for parametric or
  generative support.
- **MEASURED + DERIVED:** do **not** return to copy-predict residuals, old HNeRV/full-RGB lineage,
  generic disks/patch menus, post-hoc pose, or subset verdicts. Conditioning
  context, OUR-original task renderers, terminal joint pose, and full-n600
  realized rows remain open.
- **DERIVED:** do **not** treat a soft loss, expected stochastic endpoint, unparsed tensor,
  or private receiver equality as a result. Every proposed mechanism below
  ends at deterministic hard bytes and public-wire parse-back.

## P4 — ranked eureka table

**DERIVED — honesty convention:** labels describe the proposed transfer to Pact. "External measured"
means the cited paper demonstrates its own mechanism on its own task; it is
never evidence for our score.

**CONJECTURE — concise top three, immediately before the table:** (1) reactivate
the already-harvested LotteryCodec supermask mechanism specifically against the
post-fd2 G1 hard-public-wire gate; (2) extend the existing integer hard-oracle
repair into a bounded fd2-description QDBS with full joint-byte acceptance; (3)
pull the existing Seg/Pose safe-band apparatus back to payload coordinates and
close mixed precision on actual ZIP bytes. None is a candidate, score, or newly
proven mechanism.

| rank | idea / honesty | exact mechanism | concrete primary prior art and transfer boundary | falsifier / FIRST named measurement | unlock category | conditional exact-S leverage |
|---:|---|---|---|---|---|---|
| **1** | **G1-LOTTO: fixed-seed scorer-native supermask renderer** — **CONJECTURE; reactivated external-measured mechanism, not new discovery** | Generate a large conv/coordinate renderer with a portable, precommitted PRNG/numeric recipe. Count the supermask, modulations, pair tokens, entropy parameters, and every decoder-visible video-selected selector/config. Architecture, width/depth, mask density, modulation basis, and entropy context are fixed generically or counted. Encoder-only training rules live in the reproducibility receipt unless runtime consumes them. Optimize hard masks/mods through \(R_h\)+SegNet; terminal Pose remains joint. | [LotteryCodec paper](https://proceedings.mlr.press/v267/wu25e.html) + [official repo](https://github.com/eedavidwu/LotteryCodec): overfits a binary mask and modulation in a shared random network instead of transmitting the **main synthesis-network weights**; its learned ModNet and autoregressive entropy parameters still transmit. Already harvested in-tree; generic image RD does **not** transfer to dense Seg/Pose or the G1 rail. | **`EU1-G1-LOTTO-N600`**: same n600 target/schedule and *total compressed* bits versus OUR int4. Require cross-host bit-identical public decode from specified integer/fp32 generation, n600 decode `<30 min`, peak RAM `≤16 GiB`, section ledger, and separate contest-axis closure. Component gate: d_seg `≤1e-3`, renderer state `≤64 KiB`; decisive `≤5e-4`, stretch `≤3e-4`. Whole archive must also satisfy the conservative TR1 design rail `≤187,727 B` for 0.172 or `≤154,522 B` for 0.15, reserving about `0.047 S` for distortion; these are not pp1's exact conditional ceilings. A component-only pass is not success. | leapfrog G1 / substantial paradigm change | **DERIVED conditional:** up to `0.05–0.07 S` if d_seg crosses `1e-3→5e-4/3e-4`, plus `0.0006818 S/KiB` net payload saved. |
| **2** | **FD2-QDBS: attainable-\(R_h\) quotient direct search** — **CONJECTURE; extension of existing hard repair** | Reuse `repair_with_hard_oracle` ±1 repair and G2g2 coordinate-polish/swap logic in fd2's 344-description-coordinate domain. Scorer-recursive cheap signals select a bounded proposal set; swaps/toggles/groups are accepted only on hard full-n600 joint ZIP action. | Model-based direct binary search in [video halftoning](https://engineering.purdue.edu/~bouman/publications/pdf/ei94a.pdf) and hierarchical discrete local search [Parsimonious](https://proceedings.mlr.press/v97/moon19a.html). Their objectives do not transfer. DR1 is the task-local advisory mechanism precedent found in recall, but `composed_candidate_admissible=false`; its magnitude does not transfer. | **`EU1-FD2-QDBS-N600`**: use no-new-score GN/Fisher/block signals to select exactly **16 signed singleton + 8 grouped proposals**; full-n600 verdict each (24 QDBS candidates), versus 24 precommitted integer-random candidates, maximum 48 candidate evaluations plus one shared base. Current fd2-base strict-negative action only disambiguates mechanism; repeat the same bounded terminal test on a near-rail G1 parent for frontier relevance. | unlock current fd2 wall; terminal finisher for G1 | **OPEN-QUESTION:** each `-1e-4` d_seg is `-0.01 S` before Pose/rate; the bounded row must determine scale. The `0.05–0.07 S` G1 rail is relevant only on a near-rail parent. |
| **3** | **CELLBOX: receiver-validated evaluator-cell mixed precision + exact section waterfill** — **CONJECTURE from the DERIVED score objective** | Reuse `derive_hyperplane_channel_band` and `frame1_joint_safe_cone`, pull their bounds into payload coordinates, propose a max-volume anisotropic lattice, then solve the actual ZIP-byte/nonlinear-score Pareto assignment. Empirical secants are heuristic, never certificates. | Functional/task quantization in [Misra–Goyal–Varshney](https://arxiv.org/abs/0811.3617); Pareto mixed precision in [HAWQ-V2](https://arxiv.org/abs/1911.03852). Their assumptions do not cover rare hard argmax flips or Pose. | **`EU1-CELLBOX-N600`**: same solved object; uniform precision versus heuristic anisotropic proposals, one-at-a-time section rows, all codebooks/metadata/ZIP bytes counted. Claim certification only with sound interval/Lipschitz or exhaustive finite-lattice proof; otherwise require a strict lower exact hard parsed action and label it receiver-validated. | rate unlock on current path | **DERIVED conditional:** 10–64 KiB held-distortion saving gives `0.0068–0.0436 S`; at 10 KiB, any d_seg loss above `6.818e-5` loses. |
| **4** | **G1-C3Q: soft-round/nonuniform quantization curriculum** — **CONJECTURE; reactivated external-measured latent treatment** | Apply C3-style soft-round dither/nonuniform noise to hierarchical token latents; transferring it to G1 weights or entropy-model parameters is a separate CONJECTURE. Stage-boundary hardening ends in a deterministic hard state. | [C3 project/paper](https://c3-neural-compression.github.io/) + [official repo](https://github.com/google-deepmind/c3_neural_compression): externally demonstrates the treatment on latent grids; model parameters use separate post-training quantization. Already harvested in-tree; PSNR evidence does not transfer. | **`EU1-G1-C3Q-N600`**: matched seed/config full-n600 A/B; all synthesis, entropy, token, and patch metadata counted. Only hard parse-back d_seg/Pose/bytes decide. | improve imminent G1 build | **DERIVED conditional:** same `0.05–0.07 S` rail; each net 16 KiB saving is `0.01091 S`. |
| **5** | **G1-MORIC: separatrix-native region microfields** — **CONJECTURE** | Code all-class separatrices as curved discontinuities and jointly render smooth interiors with tied tiny fields/global modulation. Generic rasterizer is free; every video-derived curve, index, parameter, and modulation is counted. This joint renderer is distinct from a post-hoc correction stream, but inherits contour-rate risk. | [MoRIC paper](https://papers.neurips.cc/paper_files/paper/2025/hash/37f6be32f832caf0f7980469fb06165b-Abstract-Conference.html) + [repo](https://github.com/eedavidwu/MoRIC) use region micro-networks, contour chain codes, and shared modulation; [DANF](https://yashbelhe.github.io/danf/index.html) models curved discontinuities. Generic reconstruction does not prove nonlocal SegNet survival. | **`EU1-G1-MORIC-N600`**: chain-coded curves + shared microfields through actual \(R_h\)/SegNet at identical total bytes. Reject if joint rendering does not amortize the counted contour stream or misses d_seg `1e-3`. | alternate G1 parametrization | **DERIVED conditional:** G1 `0.05–0.07 S`; another 16 KiB net win is `0.01091 S`. |
| **6** | **G1-KDISTILL: cell-teacher to tied local student** — **CONJECTURE with external-measured distillation mechanism** | Use the 291 MB solved object only as an encoder-side teacher; distill a student then scorer-fine-tune it. No teacher bytes ship. Parameter tying and 64 KiB compression are Pact-added conjectures. | [KiloNeRF paper](https://openaccess.thecvf.com/content/ICCV2021/html/Reiser_KiloNeRF_Speeding_Up_Neural_Radiance_Fields_With_Thousands_of_Tiny_ICCV_2021_paper.html) + [repo](https://github.com/creiser/kilonerf) distill one field into thousands of **untied** local MLPs. NeRF quality/storage/RGB objectives do not establish a 64 KiB witness. | **`EU1-G1-KDISTILL-N600`**: identical `≤64 KiB` student state and whole-archive budget; distilled versus from-scratch initialization, same scorer fine-tune/hard parse-back. Merely reproducing teacher d_seg `1.16e-3` fails G1. | G1 initialization/de-risk | **DERIVED conditional:** G1 `0.05–0.086 S`; no intrinsic rate gain. |
| **7** | **FD2-CMAM: CMA-ES-with-Margin on integer description coordinates** — **CONJECTURE fallback** | Maintain adjacent-integer sampling probability after covariance shrinks below a quantum; hard joint action is fitness. Use only if scorer-recursive QDBS stalls. | [CMA-ES with Margin](https://arxiv.org/abs/2205.13482) prevents integer-variable stagnation; [official implementation family](https://github.com/CyberAgentAILab/cmaes). Benchmark success does not address fd2 coupling/query cost. | **`EU1-FD2-CMAM-N600`**: same fd2 start/bounds/bytes; fixed total of 24 hard candidates per method versus QDBS and iid integer mutation, deterministic seeds/config. Require a strict-negative parsed endpoint. | alternate quantization-wall cure | **OPEN-QUESTION:** each `-1e-4` d_seg is `-0.01 S` before Pose/rate; only the bounded comparison can estimate attainable scale. |
| **8** | **G1-ARMROUND: hard discrete rounding policy** — **CONJECTURE** | Bernoulli policies over two adjacent choices use ARM; genuinely categorical choices use REBAR or an explicitly justified estimator. Every scored/objective endpoint is hard; REBAR's relaxed auxiliary control-variate forwards are never results. Emit one deterministic MAP/best endpoint. | [ARM](https://arxiv.org/abs/1807.11143) addresses Bernoulli gradients; [REBAR](https://proceedings.neurips.cc/paper/2017/hash/ebd6d2f5d60ff9afaeda1a81fc53e2d0-Abstract.html) supplies a broader discrete control-variate route. Endpoint gaps remain severe. | **`EU1-G1-ARMROUND-N600`**: same bits/start/query budget as nearest rounding; held-out hard finite-difference prediction plus a deterministic equal-byte lower-action endpoint required. | hard-quantized G1 finishing | **OPEN-QUESTION:** each `-1e-4` d_seg at equal bytes/Pose is `-0.01 S`; first endpoint row bounds scale, with `0.05–0.07 S` only a conditional ceiling. |
| **9** | **G1-RECOMBINER: relative-entropy-coded weight posterior** — **CONJECTURE; reactivated external-measured coding mechanism** | Use only an analytic/seed-generated video-independent prior for free; otherwise count **every** learned prior, reparameterization matrix, upsampler, posterior update, progressive block, selector, and fallback byte. Optimize hard task distortion + actual bytes and require deterministic replay. | [COMBINER](https://proceedings.neurips.cc/paper_files/paper/2023/file/060b2af0081a460f7f466f7f174d9052-Paper-Conference.pdf), [RECOMBINER](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8d58ae1b4ffcad700119f77d65611625-Abstract-Conference.html), and [repo](https://github.com/cambridge-mlg/RECOMBINER) code posterior samples progressively. Already harvested in-tree; their learned priors are not free Pact payload. | **`EU1-G1-RECOMBINER-N600`**: exact hard loss + actual archive, cross-host bit-identical decode, `<30 min`/`≤16 GiB`, versus clustered int4 at equal total bytes. | renderer-weight rate alternative | **DERIVED conditional:** 16–48 KiB held-distortion saving gives `0.0109–0.0327 S`; G1 rail remains conditional. |
| **10** | **TOK-CTW: decoder-adaptive same-token context coding** — **CONJECTURE, lowest risk** | Deterministically serialize the final multi-symbol token/precision stream bijectively to bits, then apply binary CTW over already decoded context. No target model may hide in code; decoded tensors/frames must be byte-identical. | [Context Tree Weighting](https://research.tue.nl/en/publications/the-context-tree-weighting-method-basic-properties/) mixes bounded-memory **binary** models without a learned video-specific prior. Its guarantees do not imply gain here. | **`EU1-TOK-CTW-SAME-BYTES-0`**: **$0/no scorer** on the first real G1 stream; actual compressed bytes, exact bit/token/tensor decode equality, and header amortization. Any byte loss ends the row. | cheap rate polish after G1 exists | **DERIVED conditional:** 1–10 KiB gives `0.00068–0.00682 S`, with zero distortion risk if equality holds. |

### P4.1 Recommended firing order, not authorization

**CONJECTURE — recommended order only:**

1. MAIN reviews and, if it authorizes a build, makes `EU1-G1-LOTTO-N600` one
   branch of the same G1 race as the conventional compact renderer; C3Q is a
   treatment inside both, not a separate old-vehicle campaign.
2. Use `EU1-FD2-QDBS-N600` as the hard terminal optimizer and comparator.
   Smooth GN/Fisher supplies move order only.
3. Once either renderer reaches the rail, run CELLBOX before adding capacity:
   exploit safe cell volume and exact byte waterfill first.
4. Run CTW only after a real token stream exists. It is not infrastructure
   progress and cannot substitute for G1.

**CONJECTURE — verdict:** the best post-fd2 synthesis is a three-part object: a fixed-seed
video-specific **subnetwork message** instead of counted learned weights; an
**attainable-\(R\) hard-state optimizer** instead of subquantum continuous
acceptance; and a **receiver-validated anisotropic rate allocator** instead of
uniform bit depth. The mechanisms themselves are reactivated/extended from
recalled work; their combination and post-fd2 application is the new synthesis.
All three remain proposals. The local-custody anchor
`0.1910828242 [contest-CPU]` and the competitive/effective pointer are
unchanged.
