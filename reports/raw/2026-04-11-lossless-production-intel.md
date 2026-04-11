# Lossless Production Intel

## Highest-confidence deployed or near-production references

- **Meta / Facebook: Zstandard plus continuous dictionary retraining**
  - clear production precedent for fixed-corpus, offline-trained exact compression
  - strongest transferable lesson: collect representative samples, retrain dictionaries/plans offline, keep runtime decode simple
  - sources:
    - https://engineering.fb.com/2016/08/31/core-data/smaller-and-faster-data-compression-with-zstandard/
    - https://engineering.fb.com/2018/12/19/core-data/zstandard/

- **Meta: OpenZL / Managed Compression**
  - high-confidence near-production signal for format-aware transforms, periodic retraining, and bounded runtime control points
  - strongest transferable lesson: structure-first compression is preferable to one flat autoregressive model
  - source:
    - https://engineering.fb.com/2025/10/06/developer-tools/openzl-open-source-format-aware-compression-framework/

- **Google: Lyra in Duo**
  - real deployed learned compression, though for speech rather than video
  - strongest transferable lesson: lightweight causal predictors can ship; giant monolithic predictors are not the only route
  - source:
    - https://research.google/blog/lyra-a-new-very-low-bitrate-codec-for-speech-compression/

- **NVIDIA: nvCOMP**
  - production GPU compression infrastructure, exact not learned
  - strongest transferable lesson: accelerate search and evaluation loops, but do not mistake faster compression for a better code
  - sources:
    - https://developer.nvidia.com/nvcomp
    - https://developer.nvidia.com/blog/accelerating-lossless-gpu-compression-with-new-flexible-interfaces-in-nvidia-nvcomp/

- **Orange / INRIA: Cool-Chic**
  - strongest public OSS example of a codec engineered like a real system, with explicit entropy coding and practical decoder engineering
  - strongest transferable lesson: low-complexity learned pieces can sit inside a codec, but entropy coding remains central
  - sources:
    - https://github.com/Orange-OpenSource/Cool-Chic
    - https://orange-opensource.github.io/Cool-Chic/code_documentation/overview.html
    - https://orange-opensource.github.io/Cool-Chic/encoding/architecture.html

## Published-only but especially relevant

- **Google LMCodec**
  - hierarchical coarse-to-fine RVQ plus conditional entropy coding
  - strongest transferable lesson: encode a cheap backbone stream first, then predict finer residual streams conditionally
  - source:
    - https://research.google/pubs/lmcodec-a-low-bitrate-speech-codec-with-causal-transformer-models/

- **Samsung Compress & Cache**
  - offline indexing / summary-token production for later generation and retrieval
  - strongest transferable lesson: fixed corpora reward offline amortization and reusable summaries, not just online next-token modeling
  - source:
    - https://research.samsung.com/research-papers/Compress-Cache-Vision-token-compression-for-efficient-generation-and-retrieval

- **Adobe GPU-accelerated lossless image compression**
  - no strong evidence of a deployed learned codec, but relevant for exact-compression throughput
  - strongest transferable lesson: exact compressors can justify serious systems engineering to make search cheaper
  - source:
    - https://research.adobe.com/publication/gpu-accelerated-lossless-image-compression-with-massive-parallelization-2/

## Actionable takeaways for commaVQ

- Do **not** overcommit to one flat GPT over one flattened stream.
- Prefer:
  - structure-aware transforms
  - hierarchical or residual streams
  - cheap exact coders on easier streams
  - stronger models only on hard residual streams
- Fixed-corpus amortization is underused and likely high-EV:
  - dictionaries
  - offline clustering
  - reusable prototypes
  - summary / regime tokens
- Random-access and multi-stream decomposition are recurring practical themes.
- The most plausible 2026 upgrade over 2024 commavq GPT-style entries is:
  - `format-aware transform -> hierarchical streams -> cheap/contextual exact coder -> stronger model only on hard residuals`

## Practical implication for this branch

- Current best local evidence already points the same way:
  - `position_major` beats `frame_major`
  - simple conditional coding beats static coding
- So the next frontier should stay on:
  - source factorization
  - stronger exact conditional coders
  - corpus-amortized transforms or summaries
- Not:
  - a bigger flat GPT as the first move
