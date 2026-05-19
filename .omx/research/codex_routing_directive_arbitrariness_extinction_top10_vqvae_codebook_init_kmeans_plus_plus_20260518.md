# Codex Routing Directive — TOP-10 Arbitrariness Extinction: VQ-VAE Codebook Init `torch.randn * 0.1` → K-means++

**Subagent**: `lane_arbitrariness_extinction_meta_lens_systematic_audit_20260518`
**Value ID**: `vqvae_codebook_init_random_uniform_vs_kmeans`
**Resolution path**: `self_alien_tech`
**Predicted ΔS**: [-0.003, -0.0005]
**Cost envelope**: $0
**Rank score per dollar**: 3.0

## Bug class

`src/tac/neural_weight_codec.py:143` initializes codebook with `torch.randn(K, latent_dim) * 0.1` — generic random normal. van den Oord 2017 VQ-VAE paper §3 explicitly recommends DATA-AWARE init (K-means++ on a sample of real latents) to avoid the codebook-collapse problem (most codewords never used → effective K << nominal K).

## 5-path analysis

1. **experimental** — sweep init schemes. Slow.
2. **analytical_solve** — N/A.
3. **formula** — N/A.
4. **learned** — codebook itself is learned; init is what we're addressing.
5. **self_alien_tech** [RECOMMENDED] — K-means++ on real data sample. OR Product Quantization (Jegou et al 2011). OR Online K-means via Sinkhorn-Knopp (Asano et al 2020 SeLa).

## Concrete next step ($0)

Add data-aware init:

```python
class NeuralWeightCodec(nn.Module):
    def init_codebook_from_data(self, sample_latents: Tensor, method: str = "kmeans++"):
        """Initialize codebook via K-means++ on a sample of real latents.

        Per van den Oord 2017 §3: random init causes codebook collapse;
        data-aware init eliminates the collapse.
        """
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=self.cfg.codebook_size, init="k-means++", n_init=10)
        km.fit(sample_latents.cpu().numpy())
        self.codebook.data = torch.from_numpy(km.cluster_centers_).to(self.codebook.device)
```

Wire into trainer: run init pass with first 1000 latent samples before SGD starts.

## Coupling

- Composes with row `vq_codebook_K_64_hardcoded_neural_weight_codec` (TOP candidates list): once K is optimized, init dominates the codebook utility curve

## Exit criteria

1. `init_codebook_from_data` canonical method added
2. Trainer pre-flight init pass wired
3. Smoke confirms codebook utility (% of codewords used) ↑
4. Empirical anchor confirms predicted ΔS
