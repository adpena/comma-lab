// Sits between Hero and §00. Tight, formal abstract + a big central-insight
// callout. The callout is the thesis statement: small, contrarian, hard to
// forget. Inspired by the way other writeups frame their main claim up top.

import { Code } from "./Prose";

export default function Abstract() {
  return (
    <section className="border-t border-white/10 bg-black">
      <div className="max-w-[920px] mx-auto px-6 lg:px-10 py-16 lg:py-20">
        {/* The abstract */}
        <header className="mb-8">
          <div className="mono text-comma-green text-[12px] tracking-[0.25em] mb-3">ABSTRACT</div>
        </header>

        <div
          className="text-white/85 text-[18px] md:text-[19px] leading-[1.65]"
          style={{ letterSpacing: "-0.012em" }}
        >
          <p>
            We submit <Code>vibe_coder_final_boss</Code>, a 197,160-byte archive that scores <strong className="text-comma-green">0.22878</strong> on the comma 2026 video compression challenge: 0.046 below the leader at the time of submission and a roughly 19× reduction from the repo baseline. The pipeline is built around four components. (1) A lossless 9-context binary arithmetic mask coder forked from <a className="link-green" href="https://github.com/commaai/comma_video_compression_challenge/pull/81">PR #81</a> with five composable encoder-side wins layered on top, packing 600 frames of segmentation mask into 135 KB. (2) A 92K-parameter FP4-quantized U-Net generator with two output heads, found by 195 short-budget proxy experiments under a Karpathy-style autoresearch loop, then continued through targeted FP4-friendly fine-tunes and a low-rank SVD-warm-started pose MLP, packed into 57 KB via a custom flat-FP4 packer that bypasses the pickle envelope entirely. (3) A 2.4 KB per-pair sidecar of learned mask flips, pose deltas, and frame-1 warps that invert the SegNet/PoseNet eval discriminators directly. (4) Per-dim N-bit quantization of the pose stream and a one-letter <Code>ZIP_STORED</Code> envelope for byte-level zip overhead. We document the architecture, training curriculum, sidecar pipeline, and every dead end we hit on the way.
          </p>
        </div>

        {/* The central-insight callout */}
        <div className="mt-12 p-8 lg:p-12 border-l-4 border-comma-green bg-gradient-to-br from-comma-green/8 to-transparent">
          <div className="mono text-[11px] uppercase tracking-[0.3em] text-comma-green mb-4">THE THESIS</div>
          <h2
            className="h-display text-[32px] md:text-[44px] lg:text-[52px] text-white leading-[1.05]"
            style={{ letterSpacing: "-0.02em" }}
          >
            When the metric is a frozen neural network,<br />
            the whole game changes.
          </h2>
          <p className="mt-6 text-white/75 text-[16px] md:text-[18px] max-w-[720px] leading-snug">
            You are not compressing a video so a human can watch it. You are
            compressing a video so two specific networks (SegNet for segmentation,
            PoseNet for pose) produce the same outputs they would on the original.
            That changes <em>every</em> downstream decision: pixel realism is unrewarded,
            saturated edge-friendly hallucinations score better than honest reconstructions,
            and the eval networks become oracles you can search against. Knowing the
            discriminator changes the optimization. The whole submission is a study in
            taking that idea seriously.
          </p>
        </div>
      </div>
    </section>
  );
}
