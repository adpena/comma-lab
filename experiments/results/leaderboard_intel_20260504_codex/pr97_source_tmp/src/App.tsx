import Hero from "./components/Hero";
import Section from "./components/Section";
import { Prose, SubHeading, Code, Pull, Table, Figure } from "./components/Prose";
import ScoreCalculator from "./interactive/ScoreCalculator";
import CascadeAnimator from "./interactive/CascadeAnimator";
import TileLayoutInteractive from "./interactive/TileLayoutInteractive";
import CodecTimeline from "./interactive/CodecTimeline";
import SVDRankSlider from "./interactive/SVDRankSlider";
import SidecarPipeline from "./interactive/SidecarPipeline";
import DropPruning from "./interactive/DropPruning";
import ScoreJourney from "./interactive/ScoreJourney";
import ArchiveComposition from "./interactive/ArchiveComposition";
import Architecture from "./interactive/Architecture";
import References from "./components/References";
import Abstract from "./components/Abstract";
import Postscript from "./components/Postscript";

export default function App() {
  return (
    <main className="min-h-screen bg-black text-white antialiased">
      <Hero />

      <Abstract />

      {/* orientation + score formula */}
      <Section
        id="story"
        number="00"
        kicker="THE SCORE WE SHIPPED"
        title="0.22878"
        lede={<>The challenge measures <Code>100 · seg + √(10 · pose) + 25 · rate</Code>. We shipped 0.22878, which is 0.046 below the leader's 0.275 at the time of submission. This page is a writeup of how the score got there: a lossless mask codec, a 92K-parameter FP4 generator that learns to please the eval discriminators, autoresearch-driven architecture search, low-rank fine-tune, and a 2.4 KB sidecar layer of per-pair corrections.</>}
      >
        <Prose>
          <p>
            Compression problems usually trade off bytes against a fidelity metric you can compute from the decoded image: PSNR, SSIM, MS-SSIM. This challenge is different. The fidelity metric is two <em>frozen neural networks</em>: SegNet, which classifies every pixel of frame 2 into 5 semantic classes, and PoseNet, which extracts a 6-dimensional ego-motion vector from the (frame 1, frame 2) pair. The submission's job is to reconstruct frames such that those two networks produce nearly the same outputs they would on the originals.
          </p>
          <p>
            That single fact reshapes everything downstream. <strong>You don't have to make pretty pictures.</strong> You have to make <em>SegNet-shaped pictures and PoseNet-shaped flows</em>. We will see in §2 that the trained generator outputs frames that look unrecognizable to a human eye but score 0.027 on SegNet, because saturated colors on the right side of class boundaries is exactly what SegNet's first conv layer responds to.
          </p>
          <p>
            Below is the live formula. Drag any of the three sliders and watch the score recompute. The presets jump to anchor values: the repo baseline, the current PR #84 leader, and our submission with and without the sidecar.
          </p>
        </Prose>

        <div className="mt-8">
          <ScoreCalculator />
        </div>

        <Prose className="mt-12">
          <p>
            Three observations from playing with that calculator that shaped the rest of the design:
          </p>
          <ol className="list-decimal pl-6 space-y-2">
            <li>The <Code>√(10 · pose)</Code> term is concave, so the marginal score-improvement-per-pose-distortion-reduction <em>grows</em> as pose distortion shrinks. This is why the sidecar's last 2.4 KB are worth more per byte than the first 100 KB.</li>
            <li>The rate term is linear. Saving 1 KB is always worth the same amount: <Code>25 / 37,545,489 ≈ 6.7 · 10⁻⁷</Code> score units. So engineering a smaller archive is a freely composable optimization where every byte saved is a fixed gain.</li>
            <li>SegNet distortion enters with a 100× multiplier and is roughly twice as easy to push down as pose distortion. The former responds to the model's mask conditioning, the latter requires geometric consistency between two synthesized frames. Most teams optimize seg first; we did too.</li>
          </ol>
        </Prose>
      </Section>

      {/* archive */}
      <Section
        id="archive"
        number="01"
        kicker="WHERE THE BYTES GO"
        title="The 197 KB archive"
        lede="Every component negotiates space against every other. Mask, model, sidecar, pose, and a tiny 116-byte zip envelope. Hover the donut for the byte cost and what each slice actually stores."
        background="surface"
      >
        <ArchiveComposition />

        <Prose className="mt-12">
          <p>
            The <strong>mask</strong> at 135 KB is by far the biggest component. It has to be, because the model can only reconstruct what it's conditioned on, and the conditioning has to encode every distinguishing feature of the scene that SegNet might check (lane edges, sky/foliage boundary, vehicle silhouettes). The codec design (§2) is everything in this slice.
          </p>
          <p>
            The <strong>model</strong> at 57 KB is what it costs to ship a 92 K-parameter generator with FP4-quantized convolutional weights and FP16 biases. The number you'd <em>think</em> you're shipping (47 KB, what the score formula's <Code>estimate_model_bytes()</Code> charges) is what the formula <em>charges</em>; the number you actually ship if you naively <Code>torch.save()</Code> a state dict is 327 KB. We close that 7× pickle gap with a flat-FP4 packer (§5.1).
          </p>
          <p>
            The <strong>sidecar</strong> at 2.4 KB is the surprise. 1.2% of the archive carries 2.4% of the score improvement, because the rate-distortion tradeoff is steeper at this end of the curve.
          </p>
        </Prose>
      </Section>

      {/* codec */}
      <Section
        id="codec"
        number="02"
        kicker="LOSSLESS MASK CODEC"
        title="219 KB lossy AV1 → 135 KB lossless"
        lede="Five composable optimizations on top of erichasinternet's range coder (PR #81). All lossless, all byte-exact roundtrip. Three encoder-side wins that don't touch the codec's math, plus two C++ tweaks (a deeper cascade and a lazier adapt rate)."
      >
        <Prose>
          <p>
            Every neural-reconstruction submission stores a per-frame <em>something</em> that the model decodes back into pixels. The naive choice is a grayscale video where class IDs <Code>{`{0,1,2,3,4}`}</Code> are mapped to gray levels and run through AV1: the resulting <Code>mask.obu.br</Code> is 219 KB but only 99.96% pixel-accurate (44,330 out of 117M pixels get the wrong class on decode). Those 44 K wrong pixels then propagate into the generator and cost ~0.03 of seg term in our pipeline when we briefly trained against the lossy decoded mask distribution.
          </p>
          <p>
            Replacing the lossy AV1 mask with a <strong>lossless arithmetic coder</strong> wins twice: the model receives <em>exact</em> masks at inflate time (eliminating the 0.03 seg drift), and the encoded stream is <em>smaller</em> than AV1 because the codec exploits the discrete 5-class structure that a video codec cannot.
          </p>
        </Prose>

        <SubHeading kicker="2.1" id="cascade">The base coder</SubHeading>
        <Prose>
          <p>
            We adapted the adaptive 9-context binary arithmetic coder from <a className="link-green" href="https://github.com/commaai/comma_video_compression_challenge/pull/81" target="_blank" rel="noreferrer">erichasinternet's PR #81</a>. Credit to that submission for the original C++ source <Code>range_mask_codec.cpp</Code>. The choice of <em>arithmetic</em> coding (rather than huffman, brotli, lzma, etc) is the key. Arithmetic coding can spend less than one bit per symbol when probabilities are skewed, and dashcam mask pixels are <em>very</em> skewed (a road pixel almost always neighbors another road pixel). The codec keeps a per-context probability table that adapts to actual frequencies as it encodes, converging to near-optimal bits-per-pixel after a short warm-up.
          </p>
          <p>
            The encoder walks each pixel through a fail-fast cascade: three binary "is this class equal to a neighbor?" predictions, then a fallback over the 5 classes if none match. The animation below is one synthetic mask region cycling through scenarios that exercise each cascade exit. ~70% of pixels exit on UP (vertical structure carries the strongest signal in driving footage); ~5% fall through to the fallback.
          </p>
        </Prose>

        <div className="mt-8"><CascadeAnimator /></div>

        <Prose className="mt-12">
          <p>
            The 9 context dimensions are <Code>(prev, left, up, ul, ur, pr, pd, up2, left2)</Code>: same-pixel-previous-frame, both 4-connected and 8-connected spatial neighbors, the previous-frame right and down neighbors, and the 2-step up/left neighbors. That gives 6⁹ ≈ 10M context buckets, each with its own adaptive frequency table. For a typical mask where 50% of pixels are class 2 (sky/road) and ~75% match their UP neighbor, the cascade fails fast and most pixels emit just one bit.
          </p>
        </Prose>

        <SubHeading kicker="2.2" id="optimizations">The five optimizations</SubHeading>
        <Prose>
          <p>The PR #81 baseline was 159 KB on this dataset. Five composable changes pushed it to 135 KB.</p>
        </Prose>

        <div className="mt-8"><CodecTimeline /></div>

        <SubHeading>A. Transposed scan orientation (−17 KB)</SubHeading>
        <Prose>
          <p>
            The codec's context model assumes UP and LEFT carry strong structural information. For dashcam masks the dominant structure is <em>vertical</em> (horizon line, lane markings, sky/foliage band), so encoding columns first instead of rows lets the UP predictor fire more often. Empirically: transposing each per-frame mask from (384, 512) to (512, 384) before encoding drops 174 KB → 157 KB. The decoder transposes back after decoding.
          </p>
        </Prose>

        <SubHeading>B. Tile splitting (−18 KB)</SubHeading>
        <Prose>
          <p>
            A single arithmetic stream uses one shared adaptive model. When the data switches from "uniform sky" to "busy horizon" mid-stream, the model takes thousands of pixels to re-converge to the new local statistics, and pays bits for every guess it gets wrong while it adapts. The fix is to <em>give different regions different models</em>: split the mask into spatial tiles aligned with the natural sky/horizon/road layout, encode each tile as its own bitstream with its own adaptive model, pack the resulting bytes back-to-back.
          </p>
          <p>
            22 tiles arranged as 4 horizontal bands × per-band W splits = <Code>[3, 8, 8, 3]</Code>. The asymmetric W allocation (more splits for the busy middle bands, fewer for the sky/road bands) was found by sweep; uniform 8×8 was 0.6 KB worse and uniform 4×2 was 2.4 KB worse. Hover any tile below to see its byte cost and the lossless flip its data gets before being range-coded (that's optimization D).
          </p>
        </Prose>

        <div className="mt-8"><TileLayoutInteractive /></div>

        <Prose className="mt-12">
          <Table
            headers={["H band", "rows", "W splits", "tiles", "role"]}
            rows={[
              ["0", "0..127",   "3", "3", "mostly sky"],
              ["1", "128..255", "8", "8", "upper horizon"],
              ["2", "256..383", "8", "8", "lower horizon"],
              ["3", "384..511", "3", "3", "mostly road"],
            ]}
          />
        </Prose>

        <SubHeading>C. Tuned initial priors (−0.1 KB)</SubHeading>
        <Prose>
          <p>
            The cascade has five <Code>#define</Code>'d initial-belief values. Default <Code>(3, 4, 3, 60000, 3)</Code> is well-tuned for general-purpose semantic masks; our optimum on dashcam data is <Code>(5, 10, 6, 60000, 2)</Code>. The intuition: UP and LEFT are even stronger predictors than the defaults assume, and when the cascade falls all the way through, the most-likely fallback class is something the cascade already considered (so giving lower prior to "any class except up/left/prev" helps).
          </p>
        </Prose>

        <SubHeading>D. Per-tile static scan transforms (−2.5 KB)</SubHeading>
        <Prose>
          <p>
            Once tiles existed as independent streams, the next observation was that the cascade has a directional bias. Pixels at the top of a tile have no UP neighbor; pixels at the left have no LEFT neighbor. So a tile that has its busiest region near the bottom-right wastes bits encoding a low-information top-left region first.
          </p>
          <p>
            We searched per-tile lossless transforms: <Code>revT</Code> (reverse the 600-frame time axis), <Code>revH</Code> (vertical flip), <Code>revW</Code> (horizontal flip), and combinations. Each of the 22 tiles got the transform that minimised its individual bytes, and because the transform schedule is hard-coded in both encoder and decoder, it costs zero archive bytes. 16 of 22 tiles got a non-identity transform; the sky/road bands stayed at <Code>id</Code> because they had no asymmetry to exploit. Net: <strong>−2,462 bytes</strong>.
          </p>
        </Prose>

        <SubHeading>E. Deeper cascade + lazier adapt (−0.6 KB)</SubHeading>
        <Prose>
          <p>The original cascade asked three "is class equal to neighbor X?" binary questions before falling back. The deeper cascade adds six more before falling through:</p>
          <pre className="mono text-[14px] bg-black border border-white/10 p-4 text-comma-green overflow-x-auto">{`up → left → prev → ul → ur → pd → pr → up2 → left2 → fallback`}</pre>
          <p>
            Diagonal (<Code>ul, ur</Code>) and shifted-temporal (<Code>pd, pr</Code>, the prev-frame down/right neighbors) matches were previously eaten by the expensive fallback path; with these added, more pixels exit the cascade with one extra bit instead of ~2.3 bits. Pair this with reducing the adaptive update increment from <Code>+20</Code> to <Code>+8</Code> (which makes the per-context probability tables converge slower, and therefore stay closer to true frequency longer once they've reached it) and we shave another 642 bytes.
          </p>
          <p>
            We also explored deeper interventions: reordering the cascade itself, modifying the context model, per-tile mode selection across the codec's 11 alternative adaptive modes. Putting <Code>prev</Code> first (intuitive for video) costs +83 KB; the spatial predictors are far stronger than temporal because the codec runs frame-by-frame and PREV is the same-position pixel from the previous frame, which on a moving dashcam is rarely the same pixel content. ~20 codec source variants explored; the gain ceiling for anything else is ≈300 bytes.
          </p>
        </Prose>
      </Section>

      {/* model */}
      <Section
        id="model"
        number="03"
        kicker="THE GENERATOR"
        title="92K parameters, two heads"
        lede="Two facts shape the architecture: SegNet only sees frame 2, and Conv2d weights compress to FP4 while Linear weights stay FP16. Both push the design toward a tiny conv U-Net with the pose-conditioning capacity wrapped in 1×1 QConv to get FP4 byte treatment."
        background="surface"
      >
        <Prose>
          <p>
            The challenge scores submissions by <Code>100 · seg + √(10 · pose) + 25 · rate</Code>, where seg/pose distortions are measured by frozen SegNet and PoseNet "discriminators". Two facts shaped the entire architecture:
          </p>
          <ol className="list-decimal pl-6 space-y-2">
            <li><strong>SegNet only sees frame 2</strong> of each pair. So frame 2 needs to look "right" to a U-Net+EfficientNet, but frame 1 doesn't have to look like a frame at all. It just needs to make <Code>posenet(yuv6(f1) ⊕ yuv6(f2))</Code> output the correct pose vector.</li>
            <li><strong>Conv2d weights are stored as FP4</strong> when sizing the model (4-bit codebook + 16-bit per-block scale, then brotli), but <Code>nn.Linear</Code> is stored at FP16. So Linear capacity costs ~4× more per parameter than Conv2d capacity. But capacity in pose-conditioning Linears (FiLM, MLP) is much cheaper to compress than capacity in the trunk.</li>
          </ol>
          <p>
            We compress each video as a per-pair tuple <Code>(mask, pose)</Code> plus a single shared 57 KB generator. The generator is a 92 K-parameter U-Net with two heads.
          </p>
        </Prose>

        <div className="not-prose mt-8"><Architecture /></div>

        <Pull label="THE CENTRAL TRICK">
          When your evaluator is a neural network, you don't optimize for pixel fidelity. You optimize for whatever activation the evaluator happens to read out.
        </Pull>

        <Prose>
          <p>
            What the trained generator actually produces is visually unrecognizable: saturated and hallucinated, but it hits <Code>seg ≈ 0.027</Code> and <Code>pose ≈ 0.08</Code> on held-out pairs. SegNet's first conv mostly responds to high-contrast color edges aligned with class boundaries, so the easiest way to make SegNet predict "road" for a region is to fill it with a single saturated color whose YUV gradient matches a road-edge pattern. The model converges to a palette of those edge-friendly colors and ignores everything else. Same story for PoseNet: it cares about geometric flow between f1 and f2, so the generator learns to put high-contrast features in places that produce the right optical flow at PoseNet's quarter-resolution input.
          </p>
        </Prose>

        <Figure src="/writeup_assets/hero_pair_60.png" alt="hero pair 60" caption="Left: ground truth. Right: our reconstruction. Different palette; same SegNet output." />
        <Figure src="/writeup_assets/gen_reconstruction.gif" alt="generator reconstruction" caption="Road geometry, sky/foliage band, and lane markings emerge purely as the easiest way for the generator to satisfy SegNet's argmax. Never supervised by a pixel-level loss." />
      </Section>

      {/* autoresearch */}
      <Section
        id="autoresearch"
        number="04"
        kicker="ARCHITECTURE SEARCH"
        title="Karpathy-style autoresearch"
        lede="The architecture wasn't designed by hand. It was found by ~195 short-budget proxy experiments, each a self-contained 5-minute training run. An LLM agent reads the previous result, proposes a single change, runs the proxy, decides keep/revert, repeats."
      >
        <Prose>
          <p>
            The loop follows Karpathy's nanoGPT-speedrun rule: <strong>search algorithms, not hyperparameters</strong>. This distinction is the whole game. Hyperparameter values (learning rate, EMA decay, schedule timings) over-fit to the specific proxy budget. A learning rate optimal at 5 minutes is usually wrong at 12 hours, and the proxy-best LR sweep produces a misleading map. Algorithmic changes (architecture, loss formulation, optimizer choice) tend to transfer: if focal loss beats cross-entropy at 5 minutes, it almost always beats cross-entropy at 12 hours too, because the underlying gradient dynamics it exploits don't depend on schedule.
          </p>
          <p>
            An LLM agent reads the previous result, proposes a single change to <Code>train.py</Code>, runs the proxy, decides keep/revert based on whether the score improved by more than the run-to-run noise floor (≈ ±0.15), commits or resets, and repeats. The vast majority of attempts fail; a handful of architectural decisions did most of the work.
          </p>
        </Prose>

        <Figure src="/writeup_assets/autoresearch_progression.png" alt="autoresearch" caption="195 experiments, baseline 2.14 → 1.36. Green = kept, gray = reverted, red = running best." />

        <Prose>
          <Table
            headers={["exp", "change", "proxy score", "category"]}
            rows={[
              ["11",  "pose-finetune loss MSE → SmoothL1 (β=0.1)", "1.57", "algorithmic"],
              ["17",  "pose_mlp 2-layer → 3-layer", "1.40", "architectural"],
              ["37",  "Head1 concat-pose merge", "1.46", "architectural"],
              ["81",  "3-step ERR_BOOST schedule", "1.39", "schedule"],
              ["112", "COND_DIM 48 → 64", "1.52", "architectural"],
              ["118", "trunk-output FiLM modulating h1 only", "1.42", "architectural"],
              ["168", "trunk_film FP4 via QLinear", "1.61", "rate (algorithmic)"],
              ["169", "FiLMRes.film FP4 zero-init", "1.51", "rate (algorithmic)"],
              ["171", "focal loss (γ=2) replaces ERR_BOOST", "1.45", "algorithmic"],
              ["182", <span className="text-comma-green">"dual FiLMRes Head1 (final shape)"</span>, <span className="text-comma-green">"1.36"</span>, "architectural"],
            ]}
          />
        </Prose>

        <Pull label="HIGHEST-LEVERAGE CATEGORIES">
          FP4-wrapping zero-initialized Linears (saves ~9 KB per Linear) and focal loss + boundary weighting (concentrates gradient on the pixels where one logit perturbation flips an argmax). Together responsible for ~25 KB in rate and a sustained ~0.013 score improvement.
        </Pull>
      </Section>

      {/* training */}
      <Section
        id="training"
        number="05"
        kicker="TRAINING"
        title="A100 12h + 3090 fine-tunes"
        lede="Once the architecture froze at exp 182, we ran a three-stage curriculum (anchor / fine-tune / joint) on Colab A100 for 12 hours at batch size 16. Then a 3090 continuation cut another 0.08 from the score."
        background="surface"
      >
        <Figure src="/writeup_assets/training_curves.png" alt="training curves" caption="Loss spike at minute ~193 = QAT enabling the FP4 codebook. Re-converges within an epoch." />

        <Prose>
          <ol className="list-decimal pl-6 space-y-2">
            <li><strong>Anchor stage (55%).</strong> Train Head 2 + trunk against <Code>frame2 → segnet</Code> with focal loss. QAT enables at 70% of this stage.</li>
            <li><strong>Finetune stage (27%).</strong> Freeze Head 2, train Head 1 + pose_mlp against the pose target only.</li>
            <li><strong>Joint stage (13%).</strong> Unfreeze everything, train against the full <Code>100 · seg + √(10 · pose)</Code> loss.</li>
          </ol>
          <p>
            The A100 12h run produced a model at score ≈ 0.41. We noticed pose loss was still trending down, so we ran a 4-hour 3090 "joint+" continuation with lower LR. That alone took 0.41 → 0.33. The clean lesson: with 0.7 EMA decay used in the final phase, the model was still in a noisy regime; a longer, lower-LR finish on more data converges to a meaningfully better point. A second 3090 continuation at <Code>lr=2e-6</Code> (16× smaller) brought it to <strong>0.2988</strong>. The targeted fine-tune doesn't change architecture; it repaints the weights into a flatter local minimum that FP4 quantization rounds onto more cleanly.
          </p>
        </Prose>

        <SubHeading kicker="5.1" id="h3">H3: LowRank pose_mlp + SVD warm-start</SubHeading>
        <Prose>
          <p>
            The last training improvement compressed the pose conditioning MLP. After autoresearch converged on a 3-layer <Code>Linear(6,64) → Linear(64,64) → Linear(64,64)</Code> for <Code>pose_mlp</Code>, we replaced the two square 64×64 layers with a low-rank factorization at rank 16. To preserve the trained representation, we warm-started from a truncated SVD: take <Code>W (64×64)</Code>, compute <Code>U, S, V = svd(W)</Code>, set <Code>A = √S[:16] · V[:16]</Code> and <Code>B = U[:, :16] · √S[:16]</Code> so <Code>B@A</Code> exactly reconstructs the rank-16 truncation of W.
          </p>
        </Prose>

        <pre className="mono text-[13px] bg-black border border-white/10 p-4 text-white/85 overflow-x-auto not-prose"><code>{`class LowRankLinear(nn.Module):
    def __init__(self, in_f, out_f, rank, bias=True):
        super().__init__()
        self.a = nn.Linear(in_f, rank, bias=False)   # 64×16 = 1024 params
        self.b = nn.Linear(rank, out_f, bias=bias)   # 16×64 + 64 = 1088 params
    def forward(self, x):
        return self.b(self.a(x))`}</code></pre>

        <Prose className="mt-8">
          <p>
            <strong>Why rank-16 specifically?</strong> The singular value spectrum of the trained matrix decays steeply. The first 16 singular values together account for ~99.5% of the matrix's L2 energy. Drag the slider below to see the bar chart re-color (kept σᵢ in green, dropped in gray) and watch the energy-retained percentage update.
          </p>
        </Prose>

        <div className="mt-8"><SVDRankSlider /></div>

        <Pull label="WHY THIS WORKS">
          If a layer can be cleanly low-rank-approximated, that's a sign the layer was over-parameterized to begin with. Shrinking it is free.
        </Pull>

        <Figure src="/writeup_assets/h3_progression.png" alt="h3 progression" caption="Pose distortion dropped from 0.0833 → 0.0778 despite removing 4096 parameters from the conditioning path. Rate-saving and pose-improving in the same change." />
      </Section>

      {/* sidecar */}
      <Section
        id="sidecar"
        number="06"
        kicker="LEARNED CORRECTIONS"
        title="Sidecar: invert the oracles"
        lede="The eval networks are frozen and fully differentiable. We treat them as oracles to invert: for each video pair, find a tiny byte-level edit that lowers seg+pose more than its bytes cost in rate. 2.4 KB. −0.005 score."
      >
        <Prose>
          <p>
            This is unusual for a "compression" problem. We are not making a better model or a better codec. We are exploiting the fact that the <em>evaluators</em> are known and bounded, so for each pair we can search a small discrete space of decoder-side edits that move the reconstruction toward what the eval networks score well on. <strong>Knowing the discriminator changes the optimization.</strong>
          </p>
          <p>
            The seg and pose discriminators each have a fixed input pipeline. That pipeline has <em>blind spots</em> (input changes that don't affect the network's prediction) and <em>leverage points</em> (tiny perturbations the gradient says are big jumps in the output). A 2×2 mask-block flip doesn't visually change the SegNet target much but changes the generator's downstream feature map across the entire receptive field of that block. A correctly chosen 2-byte int8 (qx, qy) translation of frame 1 doesn't change SegNet's output at all (SegNet only reads frame 2) but shows up in PoseNet as a fractional-pixel pose delta, useful for cancelling residual quantization noise.
          </p>
        </Prose>

        <SubHeading kicker="6.1" id="five">The five edit families</SubHeading>
        <Prose>
          <p>Below, each method auto-cycles to show what it does, how many pairs got it, and how many bytes it contributed to the final 2.4 KB.</p>
        </Prose>

        <div className="mt-8"><SidecarPipeline /></div>

        <SubHeading kicker="6.2">Per-pair method selection</SubHeading>
        <Prose>
          <p>
            The naive composition (apply every method to every pair) <em>hurts</em> score. About 1/3 of pairs end up worse than baseline once two or more correction layers stack: each method optimizes against a slightly different objective, and they compound destructively.
          </p>
          <p>
            The fix is per-pair subset selection. For each of the top 600 pairs (ranked by pose error), enumerate the 16 subsets of <Code>{`{X2, CMA-ES, S2, C3}`}</Code>, batch all viable subsets into a single generator forward, pick the subset that minimizes pose loss. F1 warps are searched separately <em>after</em> the subset is chosen, because they apply to the upsampled output rather than the generator input.
          </p>
          <p>
            This took our 600-pair selection from "everything everywhere" to a sparse plan: <Code>x2=316</Code>, <Code>cmaes=39</Code>, <Code>pattern=79</Code>, <Code>pose=132</Code>. About 1/3 of pairs ended up with no mask/pose patches at all because every subset (including the empty one) was net-positive on the rate-distortion tradeoff.
          </p>
        </Prose>

        <SubHeading kicker="6.3" id="drop">Drop-pruning</SubHeading>
        <Prose>
          <p>
            After per-pair selection there are still pairs whose patches individually improve pose by less than the bytes they cost. The unified pipeline picks methods <em>per pair</em> but does not consider the global byte budget, and the bitpack format has per-pair fixed overhead (3 bytes for pair_id + flags), so a pair contributing only one mask flip might cost 6 bytes for ~0.0001 pose improvement = net negative.
          </p>
          <p>
            The drop-pruning pass evaluates the marginal contribution of each pair. Greedy phase: drop the worst pair, recompute, repeat. Local-swap phase: replace one kept pair with one dropped pair if it lowers the bound further. Watch the optimization converge over 15 steps below:
          </p>
        </Prose>

        <div className="mt-8"><DropPruning /></div>

        <Prose className="mt-8">
          <p>
            Result on H3: 63 pairs dropped. Sidecar shrinks from 2,652 → 2,376 bytes. Crucially, the <em>measured</em> score after applying drops on real frames matched the autoresearch prediction within 1e-4. The linear estimate was a good enough surrogate to drive the optimization without re-running the full 30-min eval each step.
          </p>
        </Prose>

        <SubHeading kicker="6.4">The bitpack format</SubHeading>
        <Prose>
          <p>
            Pack everything into a single blob: per-pair record <Code>[delta-encoded pair_id][1B flags][optional X2 patches][optional CMA patches][optional pattern patches][optional 3 int8 pose deltas][optional int8 qx, qy]</Code>. Flags say which sub-fields are present, so a pair with only a warp costs 4 bytes total (1B delta + 1B flags + 2B qx,qy).
          </p>
          <p>
            Mask patches pack <Code>(x:9, y:9, class:3)</Code> into 3 bytes. Pattern patches add a 3-bit shape id, still 3 bytes. Pose deltas are three int8s. Pair IDs are delta-encoded against the previous (sorted) pair_id with a 1-byte delta or <Code>0xFF + 2-byte pair_id</Code> escape: saves 1 byte per pair vs. raw u16 IDs because most adjacent kept pairs are within 256.
          </p>
          <Table
            headers={["component", "raw bytes", "per-pair cost"]}
            rows={[
              ["pair IDs (delta-encoded)", "481", "1 B/pair avg"],
              ["flags", "481", "1 B/pair"],
              ["X2 patches (272 pairs × 2.6/pair)", "2,295", "8.4 B/pair"],
              ["CMA patches (33 × 2/pair)", "198", "6 B/pair"],
              ["pattern patches (69 × 2.4/pair)", "510", "7.4 B/pair"],
              ["pose deltas (109 × 3 int8)", "327", "3 B/pair"],
              ["F1 warps (227 × 2 int8)", "454", "2 B/pair"],
              [<strong>"all packed (raw)"</strong>, <strong>"3,482"</strong>, ""],
              [<span className="text-comma-green font-bold">lzma xz preset=6</span>, <span className="text-comma-green font-bold">2,376</span>, ""],
            ]}
          />
          <p>
            The xz outer wrap squeezes out the remaining redundancy from the packed integers (mostly zero high bits of pair-id deltas and class fields). Saved ~1.1 KB on top of the bitpack.
          </p>
        </Prose>

        <SubHeading kicker="6.5">Final numbers</SubHeading>
        <Prose>
          <Table
            headers={["stage", "sidecar bytes", "seg dist", "pose dist", "score"]}
            rows={[
              ["baseline (no sidecar)",       "0",     "0.000271", "0.000604", "0.23447"],
              ["unified pipeline (no opt)",   "2,652", "0.000273", "0.000497", "0.22921"],
              ["+ drop pruning",              "2,388", "0.000272", "0.000497", "0.22895"],
              [<strong>"+ warp refine (radius=2)"</strong>, <strong>"2,376"</strong>, <strong>"0.000272"</strong>, <strong>"0.000495"</strong>, <strong className="text-comma-green">"0.22878"</strong>],
            ]}
          />
        </Prose>

        <SubHeading kicker="6.6">What we tried but didn't ship</SubHeading>
        <Prose>
          <ul className="list-disc pl-6 space-y-2">
            <li><strong>Adversarial decode at eval time.</strong> Run gradient descent through PoseNet at inflate time, refining each frame in-place rather than storing patches. Killed because doing this means shipping the actual SegNet+PoseNet weights inside <Code>archive.zip</Code> for the decoder to backprop through. Those discriminator weights would dwarf our entire current archive several times over. Not feasible in the rate budget.</li>
            <li><strong>Channel-only RGB patches.</strong> Modify one color channel at one pixel of the output frame. Net-negative on bytes for our H3 model after per-pair selection. Most "improvements" were within FP4 quantization noise of the model itself.</li>
            <li><strong>F2 frame warps.</strong> Same as F1 warps but for frame 2. SegNet does read frame 2, so a translation that helps PoseNet hurts SegNet; cost outweighed gain.</li>
            <li><strong>Higher-qscale warp refinement</strong> (qscale=20, qscale=40): finer displacement quantization. Marginal improvements (&lt;0.0001 score), not worth the integration complexity.</li>
            <li><strong>Restack ordering.</strong> Different orderings of the bitpack flags. Symmetry made all orderings equivalent in bytes.</li>
          </ul>
        </Prose>
      </Section>

      {/* byte squeezing */}
      <Section
        id="bytes"
        number="07"
        kicker="EVERY-BYTE-COUNTS"
        title="Squeezing the non-mask bytes"
        lede="Mask is 135 KB. The remaining 60 KB is model weights, pose data, and zip overhead, and three tiny interventions there pulled the archive from a naive 500 KB down to 197 KB."
        background="surface"
      >
        <SubHeading kicker="7.1" id="flat-fp4">Flat-FP4 model packing (−270 KB raw)</SubHeading>
        <Prose>
          <p>
            This is the trap nobody catches until they actually look at their archive size. The scoring formula's <Code>estimate_model_bytes()</Code> charges a Conv2d weight at 4 bits + a 16-bit per-block scale. That's the number we <em>think</em> we're shipping. But <Code>torch.save(state_dict)</Code> writes a pickle: full Python class hierarchy, dtype name strings, dict key strings, tensor metadata, and FP32 weight bytes. None of that compresses well, and brotli can't unpack the FP32 → FP4 transform on its own.
          </p>
          <p>
            For our 92K-parameter model: theoretical 47 KB vs. actual 327 KB after brotli. <strong>A 7× discrepancy you only see when you measure the actual <Code>archive.zip</Code></strong>, and it completely changes the rate term. Most submissions in the leaderboard are paying this tax.
          </p>
          <p>
            Our flat-FP4 packer removes the pickle. A small Python helper builds an ordered schema <Code>(name, kind, shape)</Code> once, both encoder and decoder agree on it, and the on-disk format is just raw bytes:
          </p>
          <pre className="mono text-[13px] bg-black border border-white/10 p-4 text-white/85 overflow-x-auto not-prose"><code>{`for each entry in SCHEMA (order baked into inflate.py):
    if kind == 'fp4_w':
        emit packed 4-bit nibbles (2 weights/byte)
        emit per-block fp16 scales
    elif kind in ('fp16_w', 'fp16_b'):
        emit raw fp16 bytes`}</code></pre>
          <p>
            That's 61,096 raw bytes → <strong>57,238 after brotli</strong>: a 5.7× reduction relative to the pickle baseline, and within ~10 KB of the score formula's theoretical model bound. This single change moved the actual-archive-size score from <strong>0.44 to 0.26</strong> in one swap.
          </p>
        </Prose>

        <SubHeading kicker="7.2" id="pose-bits">Per-dim pose quantization (−10.9 KB)</SubHeading>
        <Prose>
          <p>
            The naive baseline <Code>brotli(np.save(poses))</Code> gives 13 KB. The insight is that uniform-precision floating point is wrong for this data: dimension 0 (speed) has magnitude ~30; dimensions 1–5 (rotation rates) are around 0.05. Storing both at the same precision wastes bits on the rotations and underdelivers on speed. Per-dim quantization with separate <Code>(lo, scale, bits)</Code> headers fixes this: give speed 14 bits and rotation 4 bits each.
          </p>
          <Table
            headers={["dim", "role", "range", "std", "bits chosen", "step size"]}
            rows={[
              ["0", "speed (m/s)", "~12", "1.25",  "14", "0.0007"],
              ["1", "rot ω₁",     "0.205", "0.036", "4", "0.014"],
              ["2", "rot ω₂",     "0.162", "0.030", "4", "0.011"],
              ["3", "rot ω₃",     "0.069", "0.010", "4", "0.005"],
              ["4", "rot ω₄",     "0.063", "0.007", "4", "0.004"],
              ["5", "rot ω₅",     "0.123", "0.029", "4", "0.008"],
            ]}
          />
          <p>
            We measured the model's tolerance for quantization noise by sweeping bits-per-dim against full re-eval. The model simply doesn't care about the 5th decimal place of the pose input. The FP4 weight quantization <em>inside</em> the model adds noise an order of magnitude larger than what 4-bit pose quantization introduces.
          </p>
        </Prose>

        <SubHeading kicker="7.3" id="zip">Single-file zip with one-letter filename (−~150 B)</SubHeading>
        <Prose>
          <p>
            Inside <Code>archive.zip</Code> we ship a single member named <Code>p</Code> using <Code>ZIP_STORED</Code>. Filename saves ~6 bytes per character vs. a descriptive name; one stored member instead of three saves ~50 bytes of duplicated zip metadata; <Code>ZIP_STORED</Code> saves ~30 bytes of zip-level "compression" overhead that would otherwise re-encode our already-entropy-coded data. Inside <Code>p</Code>, our own length-prefix layout is plain concatenation, no per-component zip overhead at all.
          </p>
        </Prose>
      </Section>

      {/* score journey */}
      <Section
        id="journey"
        number="08"
        kicker="HOW WE GOT HERE"
        title="From 4.39 to 0.229"
        lede="An ~19× reduction across 8 stages. The animation below cycles through every milestone, or click any bar to jump there."
      >
        <ScoreJourney />

        <Prose className="mt-12">
          <p>
            The biggest single jumps came from autoresearch finding the right architecture (4.39 → 1.36, ~3.0 score) and the A100 12h training (1.36 → 0.41, ~0.95 score). After that the gains narrow to fractions: 3090 fine-tunes pull another 0.08, the targeted FP4-friendly fine-tune gets 0.03, the H3 LowRank pose_mlp gets 0.01, codec optimizations bring another 0.06, and the sidecar layer adds the final 0.006. Each later improvement is harder per byte than the one before. The marginal ROI tilts steeply against you as the score approaches the rate floor.
          </p>
        </Prose>
      </Section>

      <Postscript />

      <References />

      <footer className="border-t border-white/10">
        <div className="max-w-[920px] mx-auto px-6 lg:px-10 py-12 grid md:grid-cols-2 gap-8 mono text-[12px] text-white/55">
          <div>
            <div className="mono text-[10px] uppercase tracking-widest text-comma-green mb-3">submission</div>
            <p className="text-white"><Code>submissions/vibe_coder_final_boss/</Code></p>
            <p className="mt-2">archive.zip · 197,160 B · score 0.22878</p>
          </div>
          <div>
            <div className="mono text-[10px] uppercase tracking-widest text-comma-green mb-3">credits</div>
            <p>codec source: <a className="link-green" href="https://github.com/commaai/comma_video_compression_challenge/pull/81">PR #81 erichasinternet</a> (range_mask_codec.cpp)</p>
            <p className="mt-2">flat-pack inspiration: <a className="link-green" href="https://github.com/commaai/comma_video_compression_challenge/pull/73">PR #73 emir_flatpack</a></p>
            <p className="mt-2"><a className="link-green" href="https://github.com/commaai/comma_video_compression_challenge">comma.ai · video compression challenge</a></p>
          </div>
        </div>
      </footer>
    </main>
  );
}
