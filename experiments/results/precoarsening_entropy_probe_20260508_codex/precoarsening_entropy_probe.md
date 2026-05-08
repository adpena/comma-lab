# Pre-Coarsening Entropy-Coder Probe: PR101/PR106

- schema: `precoarsening_entropy_coder_probe_v1`
- evidence: `CPU/proxy_precoarsening_entropy_accounting`
- score_claim: `false`
- ready_for_exact_eval_dispatch: `false`

## Target Summary

| target | archive bytes | decoder bytes | source q11 | canonical q11 | proxy total | constriction total | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| PR101 hnerv_ft_microcodec | 178258 | 162164 | 162164 | 162395 | 174751 | 174800 | `measured_precoarsening_static_config_retired` |
| PR106 belt_and_suspenders | 186239 | 170278 | 170278 | 170226 | 182488 | 182532 | `measured_precoarsening_static_config_retired` |

## HStack/VStack Review

- HStack status: `decoder_stream_only_horizontal_candidate`
- VStack status: `terminal_entropy_stage_after_representation_quantization`
- verdict: `do_not_build_archive_from_precoarsening_static_entropy_probe`

### HStack Synergy

- Touches parser-proven renderer decoder bytes only, so it can be evaluated independently from latent/sidecar HStack lanes once a runtime adapter exists.
- Could HStack with low-level latent/sidecar repacks because those operate on disjoint logical sections.

### HStack Antagonism

- Competes with PR101 split-Brotli, HDM-style decoder recodes, and any decoder replacement that owns the same renderer weight stream.
- Any custom entropy runtime adds code bytes and dependency risk that are not charged in this probe.

### VStack Synergy

- Best positioned after byte-map/order derivation and after any lossy coarsening, because those transforms change the symbol distribution being coded.
- A post-coarsening rerun could shrink frequency tables and improve practical static-model economics.

### VStack Antagonism

- Pre-coarsening static coding is intentionally a stricter screen; it may understate a later coarsened-stack coder but cannot justify dispatch by itself.
- Zero-order per-tensor models ignore sequential context that Brotli already exploits, so source-layout Brotli remains the reference to beat.

## Blockers

- `no_score_claim`
- `no_candidate_archive_emitted`
- `no_runtime_decoder_adapter`
- `decoder_code_bytes_not_charged`
- `exact_cuda_auth_eval_missing`
