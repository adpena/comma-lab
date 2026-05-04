// Citations + influences + repo lineage. Grouped by what each source actually
// helped us with (codec, model, sidecar, methodology). Keep tight: every entry
// has to either cite a primitive we used or a paper that shaped how we thought.

interface Ref {
  title: string;
  authors: string;
  venue?: string;
  url?: string;
  note?: string; // What we took from it
}

interface Group {
  label: string;
  blurb: string;
  refs: Ref[];
}

const GROUPS: Group[] = [
  {
    label: "Direct repo lineage",
    blurb: "Submissions whose code we built on directly.",
    refs: [
      { title: "qzs3 range mask (PR #81)", authors: "erichasinternet",
        url: "https://github.com/commaai/comma_video_compression_challenge/pull/81",
        note: "We forked the C++ adaptive 9-context binary arithmetic coder from this PR and added five composable optimizations on top." },
      { title: "emir_flatpack (PR #73)", authors: "emir_flatpack",
        url: "https://github.com/commaai/comma_video_compression_challenge/pull/73",
        note: "Inspired our flat-FP4 model packer that strips the pickle envelope. Different architecture, same idea." },
      { title: "Rang Mask Optimizations (PR #84)", authors: "ottokunkel",
        url: "https://github.com/commaai/comma_video_compression_challenge/pull/84",
        note: "Our closest competitor at submission time (0.275). Same codec base but different tile layout." },
      { title: "Quantizr (PR #53)", authors: "comma.ai compression challenge",
        url: "https://github.com/commaai/comma_video_compression_challenge/pull/53",
        note: "The architecture our autoresearch loop used as a starting point: C1=56, C2=64, FiLM in Head 1, mask emb + coords stem. The PR that flipped the whole submission from a codec problem into a generator problem for us." },
    ],
  },
  {
    label: "Methodology",
    blurb: "How we ran the search, not what we searched for.",
    refs: [
      { title: "nanoGPT-speedrun & autoresearch protocol", authors: "Andrej Karpathy",
        url: "https://github.com/KellerJordan/modded-nanogpt",
        note: "The 'search algorithms, not hyperparameters' rule. We ran 195 5-minute proxy experiments under exactly this discipline." },
      { title: "Karpathy autoresearch reference implementation", authors: "Karpathy 2024",
        url: "https://github.com/karpathy/nanoGPT",
        note: "Express schedules as fractions of total steps, never absolute epochs. Lock hyperparameters, vary algorithms." },
    ],
  },
  {
    label: "Lossless coding & entropy primitives",
    blurb: "What the mask codec is built on, mathematically.",
    refs: [
      { title: "Asymmetric Numeral Systems", authors: "Jarosław Duda", venue: "arXiv:1311.2540, 2013",
        url: "https://arxiv.org/abs/1311.2540",
        note: "Foundational ANS paper. Background reading for why arithmetic coding spends fractional bits." },
      { title: "Understanding Entropy Coding With ANS", authors: "Robert Bamler", venue: "arXiv:2201.01741, 2022",
        url: "https://arxiv.org/abs/2201.01741",
        note: "Modern accessible companion. rANS within 0.1% of Shannon; informs why brotli + bit-packing on the sidecar leaves only ~1KB on the table." },
      { title: "Modified Rice-Golomb Code for Predictive Coding of Integers", authors: "Tewari et al.", venue: "arXiv:1210.6705",
        url: "https://arxiv.org/abs/1210.6705",
        note: "The Laplacian-distribution residual coder family our delta encoding approximates with simpler bit-packing." },
      { title: "Brotli", authors: "Alakuijala et al. (Google)", venue: "RFC 7932",
        url: "https://datatracker.ietf.org/doc/html/rfc7932",
        note: "Outer wrap on our model + pose streams. Beats raw deflate by ~10% on entropy-coded payloads." },
    ],
  },
  {
    label: "Neural compression",
    blurb: "The literature lens on 'small frozen decoder + tiny per-instance side info'.",
    refs: [
      { title: "Variational Image Compression with a Scale Hyperprior", authors: "Ballé, Minnen, Singh, Hwang, Johnston", venue: "ICLR 2018, arXiv:1802.01436",
        url: "https://arxiv.org/abs/1802.01436",
        note: "Canonical 'side information' framing for neural compression. The right way to think about per-pair sidecars as a learned hyperprior." },
      { title: "Instance-Adaptive Video Compression", authors: "van Rozendaal, Hill, Cohen, et al. (Qualcomm AI)", venue: "arXiv:2111.10302",
        url: "https://arxiv.org/abs/2111.10302",
        note: "Direct analog to our sidecar. Encode a per-instance parameter delta on top of a frozen decoder. Validated the architectural pattern." },
      { title: "Improving Inference for Neural Image Compression", authors: "Yang, Bamler, Mandt", venue: "arXiv:2006.04240",
        url: "https://arxiv.org/abs/2006.04240",
        note: "Latent refinement at test time (Stochastic Gumbel Annealing). Future direction for the sidecar's int8 grid search." },
      { title: "COIN: Compression with Implicit Neural Representations", authors: "Dupont, Goliński, Alizadeh, Teh, Doucet", venue: "arXiv:2103.03123",
        url: "https://arxiv.org/abs/2103.03123",
        note: "Small SIREN networks as a parameterization for delta maps. The literature anchor for 'tiny network is a sidecar'." },
      { title: "Neural Distributed Source Coding", authors: "Mital, Özyılkan, Garjani, Gündüz", venue: "arXiv:2106.02797",
        url: "https://arxiv.org/abs/2106.02797",
        note: "Wyner-Ziv setup: encode given side info available at the decoder. Frames the upper bound on what our sidecar can save." },
    ],
  },
  {
    label: "Architecture & quantization",
    blurb: "Specific blocks and tricks we used inside the H3 generator.",
    refs: [
      { title: "FiLM: Visual Reasoning with a General Conditioning Layer", authors: "Perez, Strub, de Vries, Dumoulin, Courville", venue: "AAAI 2018",
        url: "https://arxiv.org/abs/1709.07871",
        note: "Pose conditioning into Head 1. Two FiLMRes blocks, FiLM Linear wrapped in 1×1 QConv to get FP4 byte treatment." },
      { title: "EfficientNet", authors: "Tan & Le", venue: "ICML 2019",
        url: "https://arxiv.org/abs/1905.11946",
        note: "SegNet's encoder backbone. Knowing this shaped how we designed frame 2 to satisfy its argmax." },
      { title: "FastViT", authors: "Vasu, Gabriel, Zhu, Tuzel, Ranjan", venue: "ICCV 2023",
        url: "https://arxiv.org/abs/2303.14189",
        note: "PoseNet's backbone (fastvit_t12). ViT-style first conv stride that the F1 frame is implicitly designed to please." },
      { title: "LSQ+: Improving Low-Bit Quantization", authors: "Bhalgat, Lee, Nagel, Blankevoort, Kwak", venue: "CVPRW 2020",
        url: "https://arxiv.org/abs/2004.09576",
        note: "Learnable quantization offsets. Background for FP4 QAT and why our zero-init constraint matters." },
      { title: "Self-Compressing Neural Networks", authors: "Csefalvay & Imber", venue: "arXiv:2301.13142",
        url: "https://arxiv.org/abs/2301.13142",
        note: "Trains under a continuous bit-budget. Adjacent paradigm to our static FP4 codebook approach." },
    ],
  },
  {
    label: "Adversarial structure of the sidecar",
    blurb: "Why poking single pixels into a frozen ViT moves its predictions.",
    refs: [
      { title: "Patch-Fool: Are Vision Transformers Always Robust Against Adversarial Perturbations?", authors: "Fu, Zhang, Hu, Li, Lin", venue: "ICLR 2022, arXiv:2203.08392",
        url: "https://arxiv.org/abs/2203.08392",
        note: "ViTs are less robust to localized sparse perturbations than to L∞ noise. Direct theoretical justification for our 2-byte F1 warps and pixel patches." },
      { title: "Sparse vs Contiguous Adversarial Pixel Perturbations", authors: "Botocan, Patrăscu, Buduleanu", venue: "arXiv:2407.18251",
        url: "https://arxiv.org/abs/2407.18251",
        note: "Empirical confirmation that contiguous patches > sparse pixels for ViT attacks. Why 2×2 X2 mask blocks work." },
      { title: "Differentiable Patch Selection for Image Recognition", authors: "Cordonnier, Mahendran, Dosovitskiy, et al.", venue: "CVPR 2021",
        url: "https://arxiv.org/abs/2104.03059",
        note: "Perturbed-optimizer differentiable top-K. Future direction for replacing our discrete greedy patch selection." },
      { title: "Towards Evaluating the Robustness of Neural Networks", authors: "Carlini & Wagner", venue: "S&P 2017",
        url: "https://arxiv.org/abs/1608.04644",
        note: "Foundational adversarial-attack methodology. The 'invert the discriminator' framing inherits directly from this lineage." },
    ],
  },
  {
    label: "U-Net & receptive fields",
    blurb: "Why a tiny U-Net trunk is enough.",
    refs: [
      { title: "U-Net: Convolutional Networks for Biomedical Image Segmentation", authors: "Ronneberger, Fischer, Brox", venue: "MICCAI 2015",
        url: "https://arxiv.org/abs/1505.04597",
        note: "Original U-Net. Our trunk is a 92K-param tiny U-Net with one downsample." },
      { title: "Computing Receptive Fields of Convolutional Neural Networks", authors: "Araujo, Norris, Sim", venue: "Distill, 2019",
        url: "https://distill.pub/2019/computing-receptive-fields/",
        note: "Reference for why a single mask flip propagates to a sizable feature region through the cascade." },
      { title: "Real-Time Single Image Super-Resolution (PixelShuffle)", authors: "Shi et al.", venue: "CVPR 2016",
        url: "https://arxiv.org/abs/1609.05158",
        note: "Sub-pixel upsampling. Considered for the upsample path; we kept bilinear because the eval pipeline upscales anyway." },
    ],
  },
];

export default function References() {
  return (
    <section id="references" className="border-t border-white/10 bg-comma-surface">
      <div className="max-w-[920px] mx-auto px-6 lg:px-10 py-20 lg:py-28">
        <header className="mb-12">
          <div className="mono text-comma-green text-[12px] tracking-[0.25em] mb-3">
            §A  ·  REFERENCES &amp; INFLUENCES
          </div>
          <h2 className="h-display text-[40px] md:text-[56px] lg:text-[68px] text-white leading-[0.95]">
            What this stands on
          </h2>
          <p className="mt-6 text-white/65 text-[18px] md:text-[20px] leading-[1.55]" style={{ letterSpacing: "-0.012em" }}>
            Every citation below is here because it shaped a specific decision in the submission, not because it sounds impressive. Grouped by where in the pipeline it landed.
          </p>
        </header>

        <div className="space-y-12">
          {GROUPS.map((g) => (
            <div key={g.label}>
              <div className="mono text-[11px] uppercase tracking-widest text-comma-green mb-1">{g.label}</div>
              <p className="mono text-[12px] text-white/45 mb-4">{g.blurb}</p>
              <div className="space-y-3">
                {g.refs.map((r, i) => (
                  <div key={i} className="border-l-2 border-white/10 hover:border-comma-green pl-5 py-2 transition-colors">
                    <div className="flex flex-wrap items-baseline gap-x-3">
                      {r.url ? (
                        <a href={r.url} target="_blank" rel="noreferrer" className="link-green text-[15px] font-semibold">
                          {r.title}
                        </a>
                      ) : (
                        <span className="text-white text-[15px] font-semibold">{r.title}</span>
                      )}
                      <span className="text-white/55 text-[13px]">{r.authors}</span>
                      {r.venue && <span className="mono text-[11px] text-white/40">{r.venue}</span>}
                    </div>
                    {r.note && <p className="mt-1 text-[13px] text-white/65 leading-snug">{r.note}</p>}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

      </div>
    </section>
  );
}
