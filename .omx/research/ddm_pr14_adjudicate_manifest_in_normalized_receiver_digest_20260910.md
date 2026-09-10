# ddm_pr14 — adjudication of `MANIFEST.sha256` in the normalized receiver digest

Date: 2026-09-10  
Axis: `[review; scorer-free; retained-byte re-derivation; prospective contract amendment]`  
`research_only=true; score_claim=false`  
Tokens: `[no-triality] [p0-ledger-ok]`

## Verdict

**AMEND, option (a), scoped to the new pre-fire timing-risk objects.** Exclude the relative path
`MANIFEST.sha256` from a new, versioned receiver-risk digest computed only by the
`candidate_prefire_timing_risk.v1` path. Keep `tac.decode_wall_clock.measure_receiver_digest`, every
completed or inherited `decode_wall_clock` leg, the candidate's full runtime digest, the candidate's
existing normalized-receiver identity, the externally verified dependency manifest, and the PR9
literal census unchanged.

This is not a relaxation of executable identity. `MANIFEST.sha256` is a derived dependency listing; it
is not read by `inflate.sh`, `inflate.py`, or any imported decode dependency. The only shipped source
that names it is encoder-side `compress.py`. Every executable/shipped source byte remains covered by
the full runtime digest and by the risk digest's individual normalized path rows. The manifest itself
remains covered by the full runtime digest and must still be regenerated and independently validated
against every raw shipped file before an intent can pass.

The controlling pr12 sentence is: “the diagnostic reference receiver [must] equal the live candidate
receiver” for the **spend-risk gate** (pr12 lines 265–275), together with the requirement that the delta
enumerate every normalized source/candidate receiver path. The controlling PR9 sentences are: “MAIN
must not seal or fire a tree whose own dependency manifest does not describe that tree” and “a
manifest-only refresh still changes the runtime-tree digest and therefore requires every downstream
receipt to be rebound” (pr9 lines 14–17). The amendment makes those compatible: risk equivalence is
computed over behavior-bearing receiver inputs, while raw-manifest correctness remains an independent,
mandatory custody gate.

`verdict_scope: FORMULATION` — the versioned risk-inheritance digest and amended risk-object joins below.  
`verdict_scope: INSTANCE` — the retained RLC1 timed reference and RLC4 180,178-byte candidate.

## Five adjudications

1. **`MANIFEST.sha256` receiver content — AMEND (a).** It is excluded only from
   `tac.candidate_seal.measure_prefire_risk_receiver_digest.v1`. It remains in
   `measure_runtime_digest`, in the candidate dependency manifest/census, and in the legacy
   `measure_receiver_digest`; no executable path is excluded.
2. **Reference re-derivation — AMEND.** Read-only recomputation over the retained RLC1 tree gives
   `9f6e71680a13d8598974ee13f78a1a72759a758681e6d86b7b288cc105442890`; the same function over RLC4's
   retained tree gives the same value. There are 49 included normalized rows and zero differences.
3. **Evidence reuse — AMEND.** Reuse the candidate/archive/encode/parse-back/raw/manifest/census/smoke/
   retention receipts byte-for-byte. Regenerate only the normalized risk-delta receipt and timing-risk
   receipt under the versioned definition, then emit a new intent. Never overwrite or re-timestamp the
   reused receipts.
4. **Move 40 legacy leg — RATIFY-AS-IS.** `tac.decode_wall_clock`, its
   `6726fd77a7c4fa80b70ce37accb30595c1004cdd91b9ef990bf9c12f21c296bf` receiver, and the completed
   `t4_direct` validator are untouched. The new risk object separately records the legacy leg digest
   and the new risk digest of the same source tree.
5. **Freeze update — AMEND.** After the implementation landing, append one immutable amendment row to
   `.omx/research/ddm_ffi1_20260910/PREFIRE_CONTRACT_FROZEN.json`; do not change or delete any existing
   key/value. The row pins this memo, the versioned definition, both recomputed endpoints, and the new
   implementation commit/manifest.

## Why this is prospective and not a tuned loosening

RLC4 is the real control that exposed a collision between two already-ratified requirements. The rule
is amended from its own stated purpose — receiver-behavior identity for timing-risk inheritance — not
from the desired candidate score. The candidate's 180,178 bytes, conditional arithmetic, and raw output
do not select the rule: the same exclusion applies to every future pre-fire risk object, and it excludes
exactly one named derived path.

The amendment lands before any resumed producer emits an intent. The original implementation commit
`a475431997d0e0c66563448524e44c2ca8ddb384` still must precede all reused production receipts. The new
amendment commit must precede the new intent timestamp and must be present in the intent's contract
block. This deliberately does **not** require old real encodes or a 3.6 GB raw comparison to acquire a
fictional new timestamp.

The amendment holds or tightens all executable checks:

- `measure_runtime_digest` still hashes raw `MANIFEST.sha256` and every shippable file.
- candidate-manifest verification still requires exact raw `{relative_path, bytes, sha256}` equality
  with the runtime tree, from a verifier outside the candidate tree.
- PR9's complete literal census still binds all 51 shipped files.
- the risk digest still hashes every current normalized receiver row except the one named derived file.
- `inflate.py` still permits only the two top-level literal archive-pin values to normalize.
- a one-byte change in `inflate.py` outside those pins changes the risk digest and refuses the risk gate.

The executed in-memory falsifier changed one non-pin byte of normalized `inflate.py`: its normalized
file SHA became `632da0dcfdd175c899f031826027fd359e0afd740c44ea67a9e3cd7e75e628ae` and the endpoint digest became
`b8a73a69ec7c35cb8489accecb618c4b331f7c55b7333ab31ffcc008dd44e66b`, not the admitted
`9f6e7168…`. A landed consumer test must exercise the typed
`PREFIRE_RISK_EVIDENCE_REFUSED` direction, not merely compare helper outputs.

## Reference re-derivation

The exact definition is:

```text
rows = []
for each sorted, regular, non-symlink file below runtime_root:
    skip archive.zip
    skip every path rejected by runtime_digest_skip_reason
    skip exactly MANIFEST.sha256
    read bytes
    if relative_path == inflate.py:
        AST-parse the file
        require exactly one literal ARCHIVE_SHA256 assignment and one literal ARCHIVE_BYTES assignment
        replace only each assignment value span with the bytes <ARCHIVE_PIN>
    append (relative_path, len(normalized_bytes), sha256(normalized_bytes))
return sha256(utf8(json.dumps(rows, separators=(",", ":"))))
```

No mtime, mode, inode, traversal root, host, environment, or generated transport path enters the value.
Both endpoints compute the same function over retained bytes; no evidence was re-timestamped.

| Endpoint | Retained root | Legacy receiver digest | Amended risk digest | Included rows | Raw/normalized bytes |
|---|---|---|---|---:|---:|
| timed RLC1 reference | `/Volumes/VertigoDataTier/pact/ddm_rlc1_rule118_cure/candidate_runtime` | `b06e59a67b60f577eda2038353a9905550967a414e546e87162a33d9d60d1e2d` | `9f6e71680a13d8598974ee13f78a1a72759a758681e6d86b7b288cc105442890` | 49 | 824,789 / 824,743 |
| RLC4 candidate | `/Volumes/VertigoDataTier/pact/ddm_rlc2_cure_on_move42/candidate_runtime` | `27948d3d5ac0c32bd6eda8c0cfd08b2b488cfaa60c44ce3e98ba2689d7edbb8a` | `9f6e71680a13d8598974ee13f78a1a72759a758681e6d86b7b288cc105442890` | 49 | 824,789 / 824,743 |

The raw input-pinset digests, computed over canonical JSON rows
`(relative_path, raw_bytes, raw_sha256)`, are
`663b36aed2656b9e56fb5c8867b8ebd083e8b241a6d8741d091c2762318f9f3d` for RLC1 and
`046621cb5c4c1d5c67f2ab2d4857ee750b97e885b95713930acfcc4c46f3a9dc` for RLC4. They differ because the
raw `inflate.py` archive pins differ. The normalized row lists are byte-identical.

Excluded derived inputs are separately pinned and remain mandatory manifest/runtime evidence:

| Endpoint | Path | Bytes | Raw SHA-256 |
|---|---|---:|---|
| RLC1 | `/Volumes/VertigoDataTier/pact/ddm_rlc1_rule118_cure/candidate_runtime/MANIFEST.sha256` | 4,570 | `98993a00b6f2eb0f0ef8454bfbeea40ccdae7c18012a5a25cdfd645dfa4fc8f4` |
| RLC4 | `/Volumes/VertigoDataTier/pact/ddm_rlc2_cure_on_move42/candidate_runtime/MANIFEST.sha256` | 4,570 | `7c2b977a968dbd0512ce817dd7951e06d817b125875aa410a4c0f0d863345147` |

### Complete included input pins

For every row except `inflate.py`, raw and normalized identities are equal at both endpoints. The table
records both raw endpoint hashes where they can differ.

| Relative path | Raw bytes | RLC1 raw SHA-256 | RLC4 raw SHA-256 | Normalized bytes | Normalized SHA-256 |
|---|---:|---|---|---:|---|
| `README.md` | 3667 | `cbad2dd78959daf3b783ed6cd0b30e2f9535ce5e6394bbebcaabde8392180609` | `cbad2dd78959daf3b783ed6cd0b30e2f9535ce5e6394bbebcaabde8392180609` | 3667 | `cbad2dd78959daf3b783ed6cd0b30e2f9535ce5e6394bbebcaabde8392180609` |
| `compress.py` | 183043 | `2056ec97a6ea803d6a44dec935015e92f8f07bc09775e9a6c951cef011888175` | `2056ec97a6ea803d6a44dec935015e92f8f07bc09775e9a6c951cef011888175` | 183043 | `2056ec97a6ea803d6a44dec935015e92f8f07bc09775e9a6c951cef011888175` |
| `cpr1/carrier_codec.py` | 7353 | `9824bd3f8913e756b1b2d76d50e7b439187e2ce16b5014b6418bda059b2eb4b3` | `9824bd3f8913e756b1b2d76d50e7b439187e2ce16b5014b6418bda059b2eb4b3` | 7353 | `9824bd3f8913e756b1b2d76d50e7b439187e2ce16b5014b6418bda059b2eb4b3` |
| `cpr1/ddm_mp2_semantic_receiver.py` | 13389 | `b337fa08249c32a7210722561ecf9a880c37456267038ded1871b21d356bd160` | `b337fa08249c32a7210722561ecf9a880c37456267038ded1871b21d356bd160` | 13389 | `b337fa08249c32a7210722561ecf9a880c37456267038ded1871b21d356bd160` |
| `cpr1/hpac_integer.py` | 17471 | `cea40a9bf2fe6db36e7269de0d25711eb77dffdc58955d3eb9228767d448db66` | `cea40a9bf2fe6db36e7269de0d25711eb77dffdc58955d3eb9228767d448db66` | 17471 | `cea40a9bf2fe6db36e7269de0d25711eb77dffdc58955d3eb9228767d448db66` |
| `cpr1/hpac_integer_sparse.py` | 8282 | `191962f37f5710bd242728366e702d25187f47c1adfaa19ab4bc200ad2f0b864` | `191962f37f5710bd242728366e702d25187f47c1adfaa19ab4bc200ad2f0b864` | 8282 | `191962f37f5710bd242728366e702d25187f47c1adfaa19ab4bc200ad2f0b864` |
| `cpr1/inflate.py` | 13810 | `a3d25a10188684e2617d71b0f50bb883d28938d4ab9f75c385e0ee78953c7d4e` | `a3d25a10188684e2617d71b0f50bb883d28938d4ab9f75c385e0ee78953c7d4e` | 13810 | `a3d25a10188684e2617d71b0f50bb883d28938d4ab9f75c385e0ee78953c7d4e` |
| `cpr1/integer_model_io.py` | 5881 | `dee0bfa4c8e46a47d10216a94f5f43f9b27eb41e781ab99f740099e089c63be1` | `dee0bfa4c8e46a47d10216a94f5f43f9b27eb41e781ab99f740099e089c63be1` | 5881 | `dee0bfa4c8e46a47d10216a94f5f43f9b27eb41e781ab99f740099e089c63be1` |
| `cpr1/rc1_adaptive_model_sections.py` | 20916 | `7e4170dfbf94863e87c4459405a35ddb65fbcdc1670e7226dffe4f2d3c3f52f2` | `7e4170dfbf94863e87c4459405a35ddb65fbcdc1670e7226dffe4f2d3c3f52f2` | 20916 | `7e4170dfbf94863e87c4459405a35ddb65fbcdc1670e7226dffe4f2d3c3f52f2` |
| `inflate.py` | 2735 | `93303ee37fb9161de3c8adfd346d8738b6fe07168dbc4a89820c3ea2e97f38c9` | `c186a94a704b32fad506cb0963665ab62e3bc855195837180c5c1cef7512f76e` | 2689 | `a7dc43ff132053ec134a961fc659ec0f2d8bdbed29d46783ff4b6ed9fbd7051c` |
| `inflate.sh` | 3380 | `0a1820f29e454e16951068924d082cae8cbc4699f218b1beeb8663cdfdb9e618` | `0a1820f29e454e16951068924d082cae8cbc4699f218b1beeb8663cdfdb9e618` | 3380 | `0a1820f29e454e16951068924d082cae8cbc4699f218b1beeb8663cdfdb9e618` |
| `runtime/__init__.py` | 57 | `3899ca964c34159ba49e75d02e9552bc9842e93f2e767da749b594b2089581a8` | `3899ca964c34159ba49e75d02e9552bc9842e93f2e767da749b594b2089581a8` | 57 | `3899ca964c34159ba49e75d02e9552bc9842e93f2e767da749b594b2089581a8` |
| `runtime/baseline.py` | 2058 | `495769155304e694e364790c7455f5cbe15815dd457daee129e0b08928f91c38` | `495769155304e694e364790c7455f5cbe15815dd457daee129e0b08928f91c38` | 2058 | `495769155304e694e364790c7455f5cbe15815dd457daee129e0b08928f91c38` |
| `runtime/bits.py` | 1578 | `834ac42b28608d7a91965799d41f3168a545b07054579a3eed8aebe58b92c55a` | `834ac42b28608d7a91965799d41f3168a545b07054579a3eed8aebe58b92c55a` | 1578 | `834ac42b28608d7a91965799d41f3168a545b07054579a3eed8aebe58b92c55a` |
| `runtime/carrier_repack.py` | 8206 | `48303a4859b26644f03960af0026f26163e17d22de290d70a5d5736d71b78337` | `48303a4859b26644f03960af0026f26163e17d22de290d70a5d5736d71b78337` | 8206 | `48303a4859b26644f03960af0026f26163e17d22de290d70a5d5736d71b78337` |
| `runtime/compensation_overlay.py` | 7074 | `7e5d905d42cd0ec65851d5df5f762ce8adac65783781015ad48585a9fc91231f` | `7e5d905d42cd0ec65851d5df5f762ce8adac65783781015ad48585a9fc91231f` | 7074 | `7e5d905d42cd0ec65851d5df5f762ce8adac65783781015ad48585a9fc91231f` |
| `runtime/ddm_wc1_advisory_runtime.py` | 25232 | `df939fd0d47d9d06b880c1dade269f706437ad184497755d3c2c67e675876bc2` | `df939fd0d47d9d06b880c1dade269f706437ad184497755d3c2c67e675876bc2` | 25232 | `df939fd0d47d9d06b880c1dade269f706437ad184497755d3c2c67e675876bc2` |
| `runtime/dx2_cabac_coefficients.py` | 12929 | `fae972756ca19f8d2a364969ce53e6c79cb20741cf8bac6eec30d9c72390233e` | `fae972756ca19f8d2a364969ce53e6c79cb20741cf8bac6eec30d9c72390233e` | 12929 | `fae972756ca19f8d2a364969ce53e6c79cb20741cf8bac6eec30d9c72390233e` |
| `runtime/entropy/__init__.py` | 54 | `1d18072f7e08c2c8be5763f50d7bda5370272ba3b4c789a51fe1a2fa6a8111de` | `1d18072f7e08c2c8be5763f50d7bda5370272ba3b4c789a51fe1a2fa6a8111de` | 54 | `1d18072f7e08c2c8be5763f50d7bda5370272ba3b4c789a51fe1a2fa6a8111de` |
| `runtime/entropy/adaptive_ans.py` | 4230 | `95b7a45e77e5ece9512111579d0bd1544defbfc11d5110be7612b2e4a2d46797` | `95b7a45e77e5ece9512111579d0bd1544defbfc11d5110be7612b2e4a2d46797` | 4230 | `95b7a45e77e5ece9512111579d0bd1544defbfc11d5110be7612b2e4a2d46797` |
| `runtime/entropy/coefficient_ar1_codec.py` | 4875 | `5e0c03b6b8f03cdf5e0789a0bba02ce3eca0d332f23f8edb742781369fdd38c3` | `5e0c03b6b8f03cdf5e0789a0bba02ce3eca0d332f23f8edb742781369fdd38c3` | 4875 | `5e0c03b6b8f03cdf5e0789a0bba02ce3eca0d332f23f8edb742781369fdd38c3` |
| `runtime/entropy/coefficient_predictor.py` | 4027 | `7a5717db1464277e10baa8244c702d93573955e71d0065ccfbfe104b361b8957` | `7a5717db1464277e10baa8244c702d93573955e71d0065ccfbfe104b361b8957` | 4027 | `7a5717db1464277e10baa8244c702d93573955e71d0065ccfbfe104b361b8957` |
| `runtime/entropy/rc64.py` | 2904 | `dad52aa013aafcc64666630724523431597f4da5997aebcaec91c63b515ade36` | `dad52aa013aafcc64666630724523431597f4da5997aebcaec91c63b515ade36` | 2904 | `dad52aa013aafcc64666630724523431597f4da5997aebcaec91c63b515ade36` |
| `runtime/entropy/rc64_backend.c` | 5638 | `05839d1416e68a49c8022d0cccb1581c3e4338fb14c867fc6c116e203c412996` | `05839d1416e68a49c8022d0cccb1581c3e4338fb14c867fc6c116e203c412996` | 5638 | `05839d1416e68a49c8022d0cccb1581c3e4338fb14c867fc6c116e203c412996` |
| `runtime/entropy/renderer_weight_codec.py` | 9656 | `8fbc79c470e40b2b00c6c4a321ad46f972ceb55dbae6cf43d9a5ea156a4cd8a1` | `8fbc79c470e40b2b00c6c4a321ad46f972ceb55dbae6cf43d9a5ea156a4cd8a1` | 9656 | `8fbc79c470e40b2b00c6c4a321ad46f972ceb55dbae6cf43d9a5ea156a4cd8a1` |
| `runtime/f26_corrector_native.c` | 50299 | `3e2705f5505036121d85329958a4f23b5ea95e6d20d45ecf92901f2b65cca92a` | `3e2705f5505036121d85329958a4f23b5ea95e6d20d45ecf92901f2b65cca92a` | 50299 | `3e2705f5505036121d85329958a4f23b5ea95e6d20d45ecf92901f2b65cca92a` |
| `runtime/f26_hpac_native.c` | 40911 | `1326fd9dd2c85c9e78da64b1a4986536f21eb9a416ad6fbfaf1bb88698d70c00` | `1326fd9dd2c85c9e78da64b1a4986536f21eb9a416ad6fbfaf1bb88698d70c00` | 40911 | `1326fd9dd2c85c9e78da64b1a4986536f21eb9a416ad6fbfaf1bb88698d70c00` |
| `runtime/f26_hpac_native.py` | 26781 | `f13fe8beee4268cbc1df4f20016a6db6635bebebb53afc85e5025ada3748ecba` | `f13fe8beee4268cbc1df4f20016a6db6635bebebb53afc85e5025ada3748ecba` | 26781 | `f13fe8beee4268cbc1df4f20016a6db6635bebebb53afc85e5025ada3748ecba` |
| `runtime/f26_inflate.py` | 29176 | `a3616bbc968cd890a466b301e78b8f3370846cdd6c36c57948976a103d8d5b99` | `a3616bbc968cd890a466b301e78b8f3370846cdd6c36c57948976a103d8d5b99` | 29176 | `a3616bbc968cd890a466b301e78b8f3370846cdd6c36c57948976a103d8d5b99` |
| `runtime/frame0_selector.py` | 4964 | `5b0f7e0f30415f7716f1913f4994f761a10c6e0c337218cc58a337f264a943c9` | `5b0f7e0f30415f7716f1913f4994f761a10c6e0c337218cc58a337f264a943c9` | 4964 | `5b0f7e0f30415f7716f1913f4994f761a10c6e0c337218cc58a337f264a943c9` |
| `runtime/free_corrector.py` | 14709 | `dd337159bd84e96e767cbde9a6dffecc909e824c2f092399e09095bebaf094a5` | `dd337159bd84e96e767cbde9a6dffecc909e824c2f092399e09095bebaf094a5` | 14709 | `dd337159bd84e96e767cbde9a6dffecc909e824c2f092399e09095bebaf094a5` |
| `runtime/fx1_logistic_mixer_corrector.py` | 34979 | `8038119d065d578b6c163d2ee515e437cab273737ecf82f8c30619844c0f7452` | `8038119d065d578b6c163d2ee515e437cab273737ecf82f8c30619844c0f7452` | 34979 | `8038119d065d578b6c163d2ee515e437cab273737ecf82f8c30619844c0f7452` |
| `runtime/fx2_model_axis_corrector.py` | 31470 | `6462ba51ddf29dbb60b091e22043d591a1d081d9583a4864348f2cb1525aa064` | `6462ba51ddf29dbb60b091e22043d591a1d081d9583a4864348f2cb1525aa064` | 31470 | `6462ba51ddf29dbb60b091e22043d591a1d081d9583a4864348f2cb1525aa064` |
| `runtime/hpac_inference.py` | 10952 | `bbd8cdb0a9f94283b1eff9168726ee7e89e58a68bc7aec170f8340abea1b2ed6` | `bbd8cdb0a9f94283b1eff9168726ee7e89e58a68bc7aec170f8340abea1b2ed6` | 10952 | `bbd8cdb0a9f94283b1eff9168726ee7e89e58a68bc7aec170f8340abea1b2ed6` |
| `runtime/ihs2.py` | 13327 | `6d3f45b1a75d1baad7a1b0510c44e6718e62c9a1f870aafdef02cdcff65aa903` | `6d3f45b1a75d1baad7a1b0510c44e6718e62c9a1f870aafdef02cdcff65aa903` | 13327 | `6d3f45b1a75d1baad7a1b0510c44e6718e62c9a1f870aafdef02cdcff65aa903` |
| `runtime/ihs2_gate_a.py` | 10527 | `4dc291aa8c9dc543be85285f42cdf1e9365e9d1c67f4ada3b8714e130779d49b` | `4dc291aa8c9dc543be85285f42cdf1e9365e9d1c67f4ada3b8714e130779d49b` | 10527 | `4dc291aa8c9dc543be85285f42cdf1e9365e9d1c67f4ada3b8714e130779d49b` |
| `runtime/native_free_corrector.py` | 15952 | `e10036adf617998cca20484a000062099ef6468c16042f26063dea480ab0df13` | `e10036adf617998cca20484a000062099ef6468c16042f26063dea480ab0df13` | 15952 | `e10036adf617998cca20484a000062099ef6468c16042f26063dea480ab0df13` |
| `runtime/rc1_adaptive_model_sections.py` | 20916 | `7e4170dfbf94863e87c4459405a35ddb65fbcdc1670e7226dffe4f2d3c3f52f2` | `7e4170dfbf94863e87c4459405a35ddb65fbcdc1670e7226dffe4f2d3c3f52f2` | 20916 | `7e4170dfbf94863e87c4459405a35ddb65fbcdc1670e7226dffe4f2d3c3f52f2` |
| `runtime/rc2_hpac_semistatic_mixing.py` | 31341 | `375c0325496ef9ce87afaa657b35dbf871609c388b0c96bd82d67285148f9589` | `375c0325496ef9ce87afaa657b35dbf871609c388b0c96bd82d67285148f9589` | 31341 | `375c0325496ef9ce87afaa657b35dbf871609c388b0c96bd82d67285148f9589` |
| `runtime/rc3_shared_mixer.py` | 8010 | `c0ac51f359f5bca5ef2e8ab803f8a1961be79dd27aa223e5496a0cfa6564a9fa` | `c0ac51f359f5bca5ef2e8ab803f8a1961be79dd27aa223e5496a0cfa6564a9fa` | 8010 | `c0ac51f359f5bca5ef2e8ab803f8a1961be79dd27aa223e5496a0cfa6564a9fa` |
| `runtime/residual_archive.py` | 34460 | `15333b33264093641d096af25efe38ab6e62549bc6eb57e515e5d2a78b3fbae4` | `15333b33264093641d096af25efe38ab6e62549bc6eb57e515e5d2a78b3fbae4` | 34460 | `15333b33264093641d096af25efe38ab6e62549bc6eb57e515e5d2a78b3fbae4` |
| `runtime/rlc1_geometry.c` | 4238 | `414cc13dfdf597cadbe1a906a790958649446ce406461883978f0604327b46e5` | `414cc13dfdf597cadbe1a906a790958649446ce406461883978f0604327b46e5` | 4238 | `414cc13dfdf597cadbe1a906a790958649446ce406461883978f0604327b46e5` |
| `runtime/rlc1_geometry.py` | 4552 | `c6fc46dd647725bd3fa256c7e712e255e135bd1928e93f70bad53c0ef56c451f` | `c6fc46dd647725bd3fa256c7e712e255e135bd1928e93f70bad53c0ef56c451f` | 4552 | `c6fc46dd647725bd3fa256c7e712e255e135bd1928e93f70bad53c0ef56c451f` |
| `runtime/rlc1_mixer.py` | 5530 | `80dd1a6be9541f833924675e9596633486e82db0ec2e09c4463161d4f100bb4d` | `80dd1a6be9541f833924675e9596633486e82db0ec2e09c4463161d4f100bb4d` | 5530 | `80dd1a6be9541f833924675e9596633486e82db0ec2e09c4463161d4f100bb4d` |
| `runtime/rr4_free_corrector.py` | 15421 | `96fd35aaf82c737a997ea41d28c2b6e83ee8b0237afcf52808ee6cdf55a874c0` | `96fd35aaf82c737a997ea41d28c2b6e83ee8b0237afcf52808ee6cdf55a874c0` | 15421 | `96fd35aaf82c737a997ea41d28c2b6e83ee8b0237afcf52808ee6cdf55a874c0` |
| `runtime/rr5_arith_basis.py` | 20793 | `c44758dfa6b530b0e3185241c05971f75b14db7f4d2a329763f1a7bc0332c0bb` | `c44758dfa6b530b0e3185241c05971f75b14db7f4d2a329763f1a7bc0332c0bb` | 20793 | `c44758dfa6b530b0e3185241c05971f75b14db7f4d2a329763f1a7bc0332c0bb` |
| `runtime/sm1_semantic_mixer.py` | 9693 | `505a0749124f94ece64f33c91f540b4bd1b2bb382a977479e293336dd80ceb26` | `505a0749124f94ece64f33c91f540b4bd1b2bb382a977479e293336dd80ceb26` | 9693 | `505a0749124f94ece64f33c91f540b4bd1b2bb382a977479e293336dd80ceb26` |
| `runtime/tc1_receiver_checkpoint.py` | 8753 | `5fc1222c83f00d779ff2a58fd7f13875a6aaa83ed8feb00614a8f2b362ef2826` | `5fc1222c83f00d779ff2a58fd7f13875a6aaa83ed8feb00614a8f2b362ef2826` | 8753 | `5fc1222c83f00d779ff2a58fd7f13875a6aaa83ed8feb00614a8f2b362ef2826` |
| `runtime/tc1_shared_mixer.py` | 8590 | `35e44da161603cf86defc174731c41d2e50ac30c8218db2a1afb153927beb8eb` | `35e44da161603cf86defc174731c41d2e50ac30c8218db2a1afb153927beb8eb` | 8590 | `35e44da161603cf86defc174731c41d2e50ac30c8218db2a1afb153927beb8eb` |

## Evidence reuse without re-timestamping

Every `REUSE` decision is conditional on a fresh read producing the exact byte count and SHA below.
Reusing a receipt means referencing those same immutable bytes; it never means copying them into a new
receipt with a new time.

| Receipt/object | Disposition | Existing `{path, bytes, sha256}` | Reason |
|---|---|---|---|
| candidate dependency manifest | **REUSE** | `.omx/research/ddm_rlc4_20260910/CANDIDATE_MANIFEST.json`, 9,444 B, `02491b221c0833e056f16d47c507097ffd641b672e937a6dadf65fb495244eee` | It binds the unchanged full runtime identity `816f72e3…`, including the valid regenerated raw manifest. |
| external manifest validation | **REUSE** | `.omx/research/ddm_rlc4_20260910/MANIFEST_VALIDATION.json`, 559 B, `15a5b13d9ff24b769649ac8c89f126064c6e2f85fb49e1eb42d12ee1e701bfd9` | PR9's independent raw-file validation remains exactly the required gate. |
| full-n600 twins and transitive execution/payload refs | **REUSE** | `.omx/research/ddm_rlc4_20260910/TWIN_ENCODE.json`, 1,189 B, `abe1246898ae180752900d06b9dd7a506fb47fe1a3feb78586bbaad30b36742e` | Encoder arithmetic and payload bytes are unaffected by a digest-definition amendment. |
| archive parse-back | **REUSE** | `.omx/research/ddm_rlc4_20260910/ARCHIVE_PARSEBACK.json`, 658 B, `63b1f50f9f8ed334fb23aa5fc098b22efaa3d8114427188dbbd103956e462323` | It binds the selected encoded bytes to the unchanged archive member. |
| cold full-n600 raw identity | **REUSE** | `.omx/research/ddm_rlc4_20260910/RAW_IDENTITY_N600.json`, 1,771 B, `959cdfbc296b2aafaf3350a5922ca91839e2c2cca7b5c8cf60da6e9cc018b02a` | It is an executed all-byte comparison on the exact unchanged runtime/archive, not a digest proxy. |
| complete literal census | **REUSE** | `.omx/research/ddm_rlc4_20260910/LITERAL_CENSUS.json`, 13,410 B, `455721e517786f402ac2d0e71e79723915bdab1baa7dd423c71e1e9c69e775c0` | It already covers all 51 raw shipped files and stays stricter than the risk digest. |
| retention manifest and retained payloads | **REUSE** | `.omx/research/ddm_rlc4_20260910/RETENTION_MANIFEST.json`, 3,538 B, `c6e66980fa0f6c6f8d11ab7e225a8d9147c56242a23820bff84704932db8611b` | Custody paths/bytes have not changed; the resumed producer must re-read every transitive ref. |
| public candidate/frontier smoke | **REUSE** | `.omx/research/ddm_rlc4_20260910/PUBLIC_SMOKE.json`, 3,957 B, `503a35331431049448acfa9a3b9a2e1ca51cbe47620d5ff1218b9709e83f2a37` | The four public-path outcomes bind the unchanged runtime/archive tree. |
| source completed T4 leg | **REUSE** | `/Volumes/VertigoDataTier/pact/ddm_sj1_compose39_price/SEAL_ddm_sj1_compose39_rp1_union_contest_cuda.json.decode_wall_clock.json`, 1,553 B, `ed929b24cc876bf8ffabc3004b856decbb5e73d0fee13c1d3c87d91d659f9521` | It is validated by unchanged legacy code and remains the sole completed source timing authority. |
| base local diagnostic | **REUSE** | `/Volumes/VertigoDataTier/pact/ddm_dwc1_decode_wall_clock/receipts/move40_quiesced_local.json`, 230,286 B, `9cefbac9f1a003a380d817308f070b6122b99e94303bb78eeac7304fc3bc43fd` | Its actual refused verdict and legacy receiver identity remain historical inputs to the same arithmetic. |
| RLC1 g3 diagnostic | **REUSE** | `/Volumes/VertigoDataTier/pact/ddm_rlc1_rule118_cure/receipts/rlc1_g3_threads4_local.json`, 228,309 B, `b8efd88ceb21f48f78b57a3add3b90a665ea7258c3fed44449d306fed37986b9` | It stays a refused `[macOS-CPU advisory]` receipt; the new risk digest joins its retained receiver tree to the live candidate. |
| RLC1 g4 diagnostic | **REUSE** | `/Volumes/VertigoDataTier/pact/ddm_rlc1_rule118_cure/receipts/rlc1_g4_threads4_local.json`, 235,786 B, `a8dc34ebd3913031eb6da7c2008c0355a8732311ad610f490041f3f582f98db7` | Same boundary as g3; no timing authority is promoted. |
| old normalized receiver delta | **HISTORICAL-ONLY; REGENERATE NEW FILE** | `.omx/research/ddm_rlc4_20260910/NORMALIZED_RECEIVER_DELTA.json`, 14,974 B, `b6f69f287d23df7453931d191d8feb404bea893d78f0a2bc286cb761bda4d1ce` | It uses the legacy manifest-inclusive endpoint definition. Keep it as refusal evidence; create a new v2 delta with the versioned definition and no `MANIFEST.sha256` row. |
| old timing-risk receipt | **HISTORICAL-ONLY; REGENERATE NEW FILE** | `.omx/research/ddm_rlc4_20260910/TIMING_RISK.json`, 2,630 B, `33110f1d95cb6f548b1c1d281f4b78209dd2af68ca09597708e12be4d22014e3` | Its candidate/reference fields intentionally preserve the refused legacy mismatch. A new immutable receipt must name the v2 definition, the new delta, and both legacy joins. |
| conflict and producer refusal | **REUSE AS HISTORICAL NEGATIVE ONLY** | `.omx/research/ddm_rlc4_20260910/RECEIVER_MANIFEST_CONFLICT.json`, 1,278 B, `573dfb6a5b6199074124614d49500f3864357528f437a5a571397dfe352be6b1` | It proves why the amendment exists. It is not a passing risk receipt or intent. |
| pre-fire intent | **EMIT NEW** | absent | The failed producer deleted it; there are no prior intent bytes to reuse. It must be emitted only after the amendment and freeze append land. |

The candidate runtime/archive remain exactly
`/Volumes/VertigoDataTier/pact/ddm_rlc2_cure_on_move42/candidate_runtime` and archive
`eaf17a7038a4671bf39c3071192b15eec7d7513d471d520b6d13bc9acf94a60b` / 180,178 B. Any runtime,
archive, manifest, or referenced-receipt drift cancels every reuse decision.

## Clause → code → test

| Clause | Current/new code location | Required executed control |
|---|---|---|
| PR9 manifest regeneration remains mandatory | current `candidate_seal.py:1919-1935`; no semantic change | existing `test_evidence_byte_drift`; new `test_prefire_risk_manifest_exclusion_does_not_bypass_dependency_manifest_validation` |
| risk comparison uses behavior-bearing normalized content | replace the direct equality at `candidate_seal.py:2053-2065` with the versioned helper and explicit legacy joins | new `test_prefire_risk_accepts_only_regenerated_manifest_indirection` |
| every executable byte remains load-bearing | new helper beside `candidate_seal.py:2007-2037`; only `MANIFEST.sha256` is excluded | new `test_prefire_risk_refuses_nonpin_inflate_byte_change` |
| legacy `t4_direct` remains byte-for-byte unchanged | `decode_wall_clock.py:80-119,326-343,389-408`; **no edit** | existing `test_completed_fixture_seal_uses_unchanged_direct_builder`; new `test_prefire_amendment_keeps_legacy_t4_direct_manifest_inclusive` |
| reused evidence may predate the amendment, but the intent may not | amendment-aware `_pf_contract` at `candidate_seal.py:1761-1798` | new `test_prefire_amendment_allows_pinned_old_evidence_but_refuses_pre_amendment_intent` |
| freeze is append-only and contract-custodied | amendment-aware freeze read in `_pf_contract` and `build_prefire_intent` | new `test_prefire_amendment_requires_latest_frozen_row_and_live_committed_sources` |

Focused baseline before amendment: `115 passed` from
`.venv/bin/python -m pytest src/tac/tests/test_candidate_prefire_intent.py src/tac/tests/test_candidate_seal.py -q`.
This review does not claim the proposed patch has passed until MAIN lands it and runs the named controls.

## Literal implementation patch contract

MAIN's implementation landing must make the following exact semantic edits. The function and field
names below are normative; substituting an implicit flag or changing the legacy function is a refusal.

### `src/tac/candidate_seal.py`

Add these constants with the other pre-fire constants:

```python
PREFIRE_RISK_RECEIVER_DIGEST_DEFINITION = (
    "tac.candidate_seal.measure_prefire_risk_receiver_digest.v1"
)
PREFIRE_RISK_RECEIVER_EXCLUDED_PATHS = frozenset({"MANIFEST.sha256"})
PREFIRE_CONTRACT_AMENDMENT_SCHEMA = "prefire_contract_amendment.v1"
PREFIRE_CONTRACT_AMENDMENT_ID = "ddm_pr14_manifest_in_receiver_risk_digest"
```

Replace the duplicated row materializer at current `candidate_seal.py:2007-2037` with a single private
materializer and two public definitions:

```python
def _materialize_prefire_receiver_rows(
    root: Path, *, excluded_paths: frozenset[str] = frozenset()
) -> list[tuple[str, int, str]]:
    import ast

    rows = []
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root).as_posix()
        if (
            rel == "archive.zip"
            or runtime_digest_skip_reason(rel)
            or rel in excluded_paths
        ):
            continue
        data = path.read_bytes()
        if rel == "inflate.py":
            tree = ast.parse(data)
            lines = data.splitlines(keepends=True)
            offsets = [sum(map(len, lines[:i])) for i in range(len(lines))]
            edits = []
            seen = set()
            for node in tree.body:
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in {"ARCHIVE_SHA256", "ARCHIVE_BYTES"}
                ):
                    name = node.targets[0].id
                    _pf_require(
                        name not in seen,
                        "PREFIRE_RISK_EVIDENCE_REFUSED",
                        f"duplicate archive pin {name}",
                    )
                    seen.add(name)
                    value = node.value
                    expected = str if name == "ARCHIVE_SHA256" else int
                    _pf_require(
                        isinstance(value, ast.Constant) and type(value.value) is expected,
                        "PREFIRE_RISK_EVIDENCE_REFUSED",
                        f"nonliteral archive pin {name}",
                    )
                    edits.append(
                        (
                            offsets[value.lineno - 1] + value.col_offset,
                            offsets[value.end_lineno - 1] + value.end_col_offset,
                        )
                    )
            _pf_require(
                seen == {"ARCHIVE_SHA256", "ARCHIVE_BYTES"},
                "PREFIRE_RISK_EVIDENCE_REFUSED",
                "both archive pins required",
            )
            for start, end in sorted(edits, reverse=True):
                data = data[:start] + b"<ARCHIVE_PIN>" + data[end:]
        rows.append((rel, len(data), hashlib.sha256(data).hexdigest()))
    _pf_require(bool(rows), "PREFIRE_RISK_EVIDENCE_REFUSED", "receiver tree empty")
    return rows


def _prefire_receiver_rows_digest(rows: list[tuple[str, int, str]]) -> str:
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def prefire_receiver_rows(root: Path) -> list[tuple[str, int, str]]:
    """Legacy row view used to prove parity with decode_wall_clock."""
    from tac.decode_wall_clock import measure_receiver_digest

    rows = _materialize_prefire_receiver_rows(root)
    _pf_require(
        _prefire_receiver_rows_digest(rows) == measure_receiver_digest(root),
        "PREFIRE_RISK_EVIDENCE_REFUSED",
        "legacy normalized row parity failed",
    )
    return rows


def prefire_risk_receiver_rows(root: Path) -> list[tuple[str, int, str]]:
    """Behavior-bearing rows for pre-fire spend-risk inheritance only."""
    return _materialize_prefire_receiver_rows(
        root, excluded_paths=PREFIRE_RISK_RECEIVER_EXCLUDED_PATHS
    )


def measure_prefire_risk_receiver_digest(root: Path) -> str:
    """Versioned risk digest; excludes only the derived dependency manifest."""
    return _prefire_receiver_rows_digest(prefire_risk_receiver_rows(root))
```

Amend `validate_prefire_risk` so the legacy source leg and old diagnostic receipts remain explicit,
while every new risk endpoint uses the versioned helper. The normative nested shapes are:

```python
source_receiver = {
    "digest_definition": PREFIRE_RISK_RECEIVER_DIGEST_DEFINITION,
    "sha256": measure_prefire_risk_receiver_digest(Path(leg["runtime_dir"])),
    "t4_direct_digest_definition": "tac.decode_wall_clock.measure_receiver_digest",
    "t4_direct_sha256": leg["receiver_sha256"],
}
candidate_receiver = {
    "digest_definition": PREFIRE_RISK_RECEIVER_DIGEST_DEFINITION,
    "sha256": measure_prefire_risk_receiver_digest(
        Path(intent["candidate"]["runtime"]["path"])
    ),
}
diagnostic_reference_receiver = {
    "path": str(Path(risk["diagnostic_reference_receiver"]["path"])),
    "digest_definition": PREFIRE_RISK_RECEIVER_DIGEST_DEFINITION,
    "sha256": candidate_receiver["sha256"],
    "receipt_digest_definition": "tac.decode_wall_clock.measure_receiver_digest",
    "receipt_sha256": measure_receiver_digest(
        Path(risk["diagnostic_reference_receiver"]["path"])
    ),
}
```

Require exact equality between each retained object and the corresponding shape above. Build the delta
from `prefire_risk_receiver_rows` at both endpoints; require its endpoint hashes to equal
`source_receiver["sha256"]` and `candidate_receiver["sha256"]`. Validate the base diagnostic's retained
`receiver_sha256` against `source_receiver["t4_direct_sha256"]`; validate every candidate diagnostic's
retained `receiver_sha256` against `diagnostic_reference_receiver["receipt_sha256"]`. The risk
calculation and RLC2 exact-number checks remain unchanged.

Do not change `_pf_identity`, `build_prefire_intent`'s existing candidate normalized-receiver field, or
any non-risk evidence endpoint. That is what permits the seven real receipts above to remain exact-byte
references instead of rewritten history.

Amend `_pf_contract` and `build_prefire_intent` to carry an exact `contract.amendment` object copied from
the last row of the frozen receipt. The base implementation commit/manifest remain the materialization
contract and are verified from their committed blobs. The amendment implementation commit/manifest
must match live source and committed blobs. Enforce:

```text
base implementation commit <= reused candidate production timestamp <= new intent timestamp
base implementation commit is ancestor of amendment implementation commit
amendment implementation commit <= new intent timestamp
amendment implementation commit is ancestor of current HEAD
the exact amendment row is committed in current HEAD
the candidate production source commit descends from the base implementation commit
the new intent blob, when required, is byte-identical to current HEAD
```

The old live-file equality check moves from the base manifest to the amendment manifest; the base
manifest is still re-read and checked against `git show <base-commit>:<path>`. This is the only ordering
that both proves the old evidence was produced under a frozen contract and avoids inventing new evidence
timestamps after the amendment.

### Tests

Add exactly these tests to `src/tac/tests/test_candidate_prefire_intent.py`:

```text
test_prefire_risk_accepts_only_regenerated_manifest_indirection
test_prefire_risk_refuses_nonpin_inflate_byte_change
test_prefire_risk_manifest_exclusion_does_not_bypass_dependency_manifest_validation
test_prefire_amendment_keeps_legacy_t4_direct_manifest_inclusive
test_prefire_amendment_allows_pinned_old_evidence_but_refuses_pre_amendment_intent
test_prefire_amendment_requires_latest_frozen_row_and_live_committed_sources
```

The first test stages two otherwise-identical trees whose valid raw manifests differ only on the raw
`inflate.py` archive-pin hash and requires equal risk digests plus unequal legacy digests. The second
mutates one byte of `inflate.py` outside the two pin AST spans and must receive
`PREFIRE_RISK_EVIDENCE_REFUSED` from `validate_prefire_risk`. The third corrupts/stales the raw manifest
and must receive `PREFIRE_NON_TIMING_GATE_REFUSED` from full intent validation even though the risk
helper excludes that path. The fourth builds and validates the existing legacy `t4_direct` fixture,
then changes its manifest and proves the unchanged legacy validator refuses. The last two exercise both
chronology directions and exact freeze/manifest Git custody.

Run:

```text
.venv/bin/python -m pytest src/tac/tests/test_candidate_prefire_intent.py src/tac/tests/test_candidate_seal.py src/tac/tests/test_decode_wall_clock_t4_direct.py -q
.venv/bin/python -m ruff check src/tac/candidate_seal.py src/tac/tests/test_candidate_prefire_intent.py
```

`src/tac/decode_wall_clock.py`, `tools/make_candidate_seal.py`, and
`src/tac/tests/test_candidate_seal.py` receive no source edit. Their unchanged code and existing tests
are part of the regression gate.

## Freeze append

The append is necessarily a second, mechanical landing: its `implementation_commit` and implementation-
manifest hash do not exist until the source/test landing exists, and this memo cannot contain its own
post-edit SHA without a self-hash cycle. MAIN must first land the implementation, generate a sorted
replacement implementation manifest from that commit, and then append the following row as the sole
element of a new top-level `amendments` array. Metavariables below are patch instructions and are
**forbidden literal values** in the JSON; replace them from measured files/Git or refuse.

```json
{
  "schema": "prefire_contract_amendment.v1",
  "amendment_id": "ddm_pr14_manifest_in_receiver_risk_digest",
  "adjudication_memo": {
    "path": "/Users/adpena/Projects/pact/.omx/research/ddm_pr14_adjudicate_manifest_in_normalized_receiver_digest_20260910.md",
    "bytes": "${POST_EDIT_PR14_MEMO_BYTES}",
    "sha256": "${POST_EDIT_PR14_MEMO_SHA256}"
  },
  "definition_change": {
    "scope": "candidate_prefire_timing_risk.v1 only; legacy decode_wall_clock unchanged",
    "digest_definition": "tac.candidate_seal.measure_prefire_risk_receiver_digest.v1",
    "excluded_relative_paths": ["MANIFEST.sha256"],
    "raw_manifest_still_required": true,
    "executable_difference_policy": "REFUSE"
  },
  "reference_receiver": {
    "path": "/Volumes/VertigoDataTier/pact/ddm_rlc1_rule118_cure/candidate_runtime",
    "legacy_sha256": "b06e59a67b60f577eda2038353a9905550967a414e546e87162a33d9d60d1e2d",
    "amended_sha256": "9f6e71680a13d8598974ee13f78a1a72759a758681e6d86b7b288cc105442890",
    "included_file_count": 49,
    "raw_input_pinset_sha256": "663b36aed2656b9e56fb5c8867b8ebd083e8b241a6d8741d091c2762318f9f3d",
    "excluded_manifest": {"bytes": 4570, "sha256": "98993a00b6f2eb0f0ef8454bfbeea40ccdae7c18012a5a25cdfd645dfa4fc8f4"}
  },
  "candidate_receiver": {
    "path": "/Volumes/VertigoDataTier/pact/ddm_rlc2_cure_on_move42/candidate_runtime",
    "legacy_sha256": "27948d3d5ac0c32bd6eda8c0cfd08b2b488cfaa60c44ce3e98ba2689d7edbb8a",
    "amended_sha256": "9f6e71680a13d8598974ee13f78a1a72759a758681e6d86b7b288cc105442890",
    "included_file_count": 49,
    "raw_input_pinset_sha256": "046621cb5c4c1d5c67f2ab2d4857ee750b97e885b95713930acfcc4c46f3a9dc",
    "excluded_manifest": {"bytes": 4570, "sha256": "7c2b977a968dbd0512ce817dd7951e06d817b125875aa410a4c0f0d863345147"}
  },
  "implementation_commit": "${FULL_IMPLEMENTATION_LANDING_COMMIT}",
  "implementation_manifest": {
    "path": "${ABSOLUTE_AMENDED_IMPLEMENTATION_MANIFEST_PATH}",
    "bytes": "${AMENDED_IMPLEMENTATION_MANIFEST_BYTES}",
    "sha256": "${AMENDED_IMPLEMENTATION_MANIFEST_SHA256}"
  },
  "score_claim": false
}
```

Before the freeze commit, mechanically require: memo path/bytes/SHA match; both retained endpoints
recompute to the pinned amended digest; the implementation manifest contains every
`PREFIRE_IMPLEMENTATION_PATHS` source; every row equals both the live file and
`git show ${FULL_IMPLEMENTATION_LANDING_COMMIT}:<path>`; and the implementation commit contains this
memo at the pinned SHA. Then commit the freeze append. A single landing that guesses its own commit hash
is forbidden.

## Provenance pins

| Surface | SHA-256 / commit |
|---|---|
| pr12 memo | `50d00e3956dc7ae5d3b15379d2ae6f8704119817b0b58f50aa30413b97eacadc` |
| frozen receipt | `59158b8fce89e12c06eb4ae3e1cb71f347daa78a2cc5e8a061e1899be1150c2a` |
| pr13 memo | `3156992449ea4f71966dfe96adea1471aa3b735dae3bac4a9f49c4a16ce4eba3` |
| RLC4 conflict | `573dfb6a5b6199074124614d49500f3864357528f437a5a571397dfe352be6b1` |
| original contract implementation | `a475431997d0e0c66563448524e44c2ca8ddb384` |
| current reviewed `candidate_seal.py` | `4c0baadd883717a693c77825adb207941d9c63796a53d90e930229b5c6144efb` |
| current reviewed `decode_wall_clock.py` | `f1ce9122a0114e13481eb1f3fe8ba3951b5c0f1549ec49538994393acceb89ff` |
| current reviewed `make_candidate_seal.py` | `e4b7f48ffa7df4a35586d803b50fb94bc25d36a667b9dfb1a6b8cec08d343791` |
| pointer move 42 landing | `d2803c2148b6153e6fc26d18428cd2d3cda3ea3e` |
| pointer archive | `f111ab4259c757409e791247d33978a714ceb1cd66e50c149e2e65fbf208756f` |

## RECALL EVIDENCE

Searched the full `.omx/research/` Markdown/JSON corpus by content for
`manifest.{0,80}(receiver|digest)|receiver.{0,80}manifest|normalized receiver|content-only digest|environment-free digest|prefire|first measurement|t4_direct`;
read the pr8–pr13/RLC/FFI lineage and the exact RLC4 receipts; searched
`CANONICAL_RESEARCH_INDEX*` and every `sub015_DAG_*` file for the same terms; searched docs, design
documents, and `SPEC*` surfaces; searched `canonical_task_status.jsonl`, `lane_registry.json`, the
operator P0 ledger, active dispatch claims, and live hot state; and generated the complete canonical
equation registry with `.venv/bin/python tools/list_canonical_equations.py --json`, filtering it for
`receiver|digest|manifest|runtime|content-only|t4_direct|prefire|first measurement`.

Beyond the charter seeds, R9M's first exact row records that two validators must compute the same
environment-free content identity; the candidate-seal contract says a content digest must be invariant
under its own consumer and keep per-file pins for actionable drift. Those precedents changed the plan
from “ignore the mismatch in `validate_prefire_risk`” to a named, shared helper used on both risk
endpoints. Direct source search also found that only encoder-side `compress.py` reads
`MANIFEST.sha256`; the public decoder does not. The live task ledger confirms RLC4's refusal and routes
this exact decision before any resumed intent or authorization. No canonical equation, DAG FEED,
design/SPEC document, or older task row supplied a competing pre-fire risk-digest definition in the
searched scope.

## Boundaries

- This was scorer-free read-only adjudication of code and retained/live trees. Only this memo,
  checkpoint state, and serializer custody were produced.
- No `src/`, `tools/`, `upstream/`, PR tree, candidate runtime, sealed tree, volume path, pointer, lane,
  spend ledger, or shared staged index was edited.
- No Modal/provider call, fire, authorization, encoder, decoder, scorer, evaluator, timing window,
  payload materialization, movement, deletion, or cleanup ran.
- The recomputed digests are `[macOS-CPU scorer-free; retained-byte derivation]`, not score or timing
  authority. The 1,032.725-second value remains diagnostic spend-risk arithmetic only.
- The RLC4 archive has no contest score. Its `0.13743655372199698` value remains conditional arithmetic.
- The exact pointer did not move. This amendment is apparatus work, not goal progress.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: `MAIN / FFI implementation arm`; consumer store: this memo, the amended source/tests, a new committed implementation manifest, and `.omx/research/ddm_ffi1_20260910/PREFIRE_CONTRACT_FROZEN.json`; fire trigger: MAIN lands this memo, then the literal scoped implementation and all named positive/red tests pass. Land the implementation first, generate its exact manifest, then append and commit the freeze row; never edit the legacy `decode_wall_clock` implementation.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: `rlc5 resumed producer`; consumer store: `/Volumes/VertigoDataTier/pact/ddm_rlc2_cure_on_move42`, the reused RLC4 evidence refs, a new versioned normalized-risk delta, a new timing-risk receipt, and a new intent; fire trigger: the amendment implementation and freeze append are committed, the pointer still names move 42, and every reused `{path,bytes,sha256}` revalidates. Regenerate only the two risk receipts, emit/self-validate the real intent, and execute both normal-seal refusal controls; do not authorize or dispatch.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: `MAIN`; consumer store: the exact committed intent, a new `candidate_first_measurement_authorization.v1`, the authorized SSD output directory, nonce-consumption record, call ledger, and retained result; fire trigger: the real intent is committed byte-identically, both normal refusal controls pass, provider cost is strictly below USD 5, the T4 lane is actively claimed, single-flight/cloud state is clear, and the pointer is unchanged. Authorize, commit authorization, dry-run, fire once, and harvest; never retry an ambiguous reservation.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: `MAIN`; consumer store: the completed `candidate_seal.v3`, exact-result adjudication, evaluation ledger, and pointer packet; fire trigger: the harvested result passes unchanged cold n600 `t4_direct` at no more than 1,260 seconds and every identity/non-timing/pointer/score-bar recheck. Promote only if the exact contest-CUDA score qualifies.

Frontier unchanged: `composition S 0.1374765052591843 @ 180,238 B [contest-CUDA T4 n600] (move 42)`.

## LIVE-HYPOTHESES

- The retained 180,178-byte archive may beat move 42 after the custody amendment because two real
  encodes agree and the full cold macOS public output is byte-identical; exact T4 scoring remains
  untested.
- The candidate may complete below 1,260 seconds on T4 because the unchanged completed source leg and
  retained RLC1 diagnostics give a 1,032.725-second spend-risk estimate; that is not timing authority or
  a confidence bound.
- The scoped versioned digest should remain stable across producer and MAIN because it is a pure function
  of retained bytes and both endpoints already produce the same 49 normalized rows; the real cross-actor
  intent/authorization path remains untested.

## DEAD-ENDS

- Keeping `MANIFEST.sha256` verbatim in the risk digest is closed for this formulation: it reintroduces
  normalized archive pins through a derived hash row and makes PR9 regeneration incompatible with
  behavior identity.
- Restoring the old RLC1 manifest in RLC4 is closed: it is stale for the real `inflate.py` pins and would
  repeat PR9's exact failed condition.
- Recomputing or rewriting the completed move-40 `t4_direct` leg is closed: pr12 preserves it unchanged,
  and a versioned new-object risk digest avoids touching it.
- Excluding `MANIFEST.sha256` from the full runtime digest, dependency manifest, or literal census is
  closed: those are the independent raw-custody gates that make the scoped risk exclusion safe.
- Re-running the twins, cold 3.6 GB raw comparison, parse-back, smokes, or census merely to acquire a new
  timestamp is closed: exact byte/hash revalidation is the honest reuse rule.
- Treating the conditional score, diagnostic timing projection, fixture tests, or an emitted intent as
  promotion evidence is closed: only the harvested exact row plus unchanged `t4_direct` and a valid v3
  seal can move the pointer.
