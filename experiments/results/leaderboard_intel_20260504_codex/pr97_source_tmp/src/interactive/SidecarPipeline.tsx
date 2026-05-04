// Auto-cycling 5-method showcase. No buttons. Each method holds for ~3.2s.

import { useEffect, useState } from "react";

interface Method {
  key: string;
  short: string;
  full: string;
  pairs: number;
  bytes: number;
  perPair: string;
  blurb: string;
  color: string;
  visualize: () => React.ReactNode;
}

const METHODS: Method[] = [
  {
    key: "x2", short: "X2", full: "2×2 mask block flips",
    pairs: 272, bytes: 2295, perPair: "8.4 B/pair",
    color: "#fde68a",
    blurb: "Flip a 2×2 patch of the input mask to a different class. One block changes the entire feature map under the U-Net's receptive field, so leverage per byte is high. Found by verified greedy: take gradient of pose loss, score top-5 candidates, re-run the generator, accept whichever lowers loss.",
    visualize: () => (
      <svg viewBox="0 0 80 80" className="w-32 h-32">
        {Array.from({ length: 16 }).map((_, i) => {
          const r = Math.floor(i / 4), c = i % 4;
          const flipped = (r === 1 && c === 1) || (r === 1 && c === 2) || (r === 2 && c === 1) || (r === 2 && c === 2);
          return <rect key={i} x={c * 20 + 1} y={r * 20 + 1} width={18} height={18} fill={flipped ? "#51FF00" : "#444"} />;
        })}
        <rect x={20} y={20} width={40} height={40} fill="none" stroke="#FF4133" strokeWidth={2} strokeDasharray="3 2" />
      </svg>
    ),
  },
  {
    key: "cmaes", short: "CMA-ES", full: "single-pixel mask flips",
    pairs: 33, bytes: 198, perPair: "6 B/pair",
    color: "#fbcfe8",
    blurb: "For pairs X2 didn't fully fix. CMA-ES over K=2 single-pixel flip locations from the top-30 gradient positions, batched per generation. CMA-ES coordinates the two flips when they need to compose.",
    visualize: () => (
      <svg viewBox="0 0 80 80" className="w-32 h-32">
        {Array.from({ length: 64 }).map((_, i) => {
          const r = Math.floor(i / 8), c = i % 8;
          const isFlip = (r === 2 && c === 5) || (r === 5 && c === 2);
          return <rect key={i} x={c * 10 + 1} y={r * 10 + 1} width={8} height={8} fill={isFlip ? "#fbcfe8" : "#333"} />;
        })}
      </svg>
    ),
  },
  {
    key: "s2", short: "S2", full: "variable-shape patterns",
    pairs: 69, bytes: 510, perPair: "7.4 B/pair",
    color: "#bfdbfe",
    blurb: "Mask flips with shape vocabulary {1×1, 3×3, 1×4, 4×1, 2×2}. Strips capture road edges. 3×3 catches isolated misclassifications in homogeneous regions. CMA-ES over (shape, position, class).",
    visualize: () => (
      <svg viewBox="0 0 80 80" className="w-32 h-32">
        <rect x={6} y={10} width={5} height={5} fill="#bfdbfe" />
        {[0,1,2,3].map(i => <rect key={`h${i}`} x={20 + i * 7} y={32} width={5} height={5} fill="#bfdbfe" />)}
        {[0,1,2,3].map(i => <rect key={`v${i}`} x={56} y={10 + i * 7} width={5} height={5} fill="#bfdbfe" />)}
        {Array.from({ length: 9 }).map((_, i) => {
          const r = Math.floor(i / 3), c = i % 3;
          return <rect key={`s${i}`} x={20 + c * 7} y={56 + r * 7} width={5} height={5} fill="#bfdbfe" />;
        })}
      </svg>
    ),
  },
  {
    key: "c3", short: "C3", full: "pose-vector deltas",
    pairs: 109, bytes: 327, perPair: "3 B/pair",
    color: "#bbf7d0",
    blurb: "int8 grid search over 3 dominant pose dims (1, 2, 5). Per pair, evaluate 7³ = 343 candidate (Δd₁, Δd₂, Δd₅) tuples in a single batched generator forward, keep the lowest-loss. Adam-via-autograd through the FP4 generator was unstable; gradient-free grid worked.",
    visualize: () => (
      <div className="text-center mono">
        <div className="text-comma-green text-[24px] font-bold">7×7×7</div>
        <div className="text-white/55 mt-1 text-[11px]">candidate cube</div>
        <div className="mt-3 text-white/85 text-[12px]">Δd<sub>1</sub>, Δd<sub>2</sub>, Δd<sub>5</sub></div>
        <div className="text-white/40 mt-1 text-[10px]">int8 each, 3 B/pair</div>
      </div>
    ),
  },
  {
    key: "warp", short: "F1 warps", full: "translate output frame 1",
    pairs: 227, bytes: 454, perPair: "2 B/pair",
    color: "#fed7aa",
    blurb: "Per-pair int8 (qx, qy) translation applied to output frame 1. PoseNet sees a translation as a smooth pose delta; SegNet ignores frame 1. So 2 bytes can cancel a fractional-pixel pose residual at zero cost to seg.",
    visualize: () => (
      <svg viewBox="0 0 80 80" className="w-32 h-32">
        <rect x={5} y={5} width={50} height={50} fill="none" stroke="rgba(255,255,255,0.3)" strokeWidth={1} strokeDasharray="2 2" />
        <rect x={20} y={20} width={50} height={50} fill="rgba(254, 215, 170, 0.3)" stroke="#fed7aa" strokeWidth={1.5} />
        <line x1={30} y1={30} x2={45} y2={45} stroke="#fed7aa" strokeWidth={2} markerEnd="url(#arrowhead)" />
        <defs>
          <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="6" refY="3" orient="auto">
            <polygon points="0 0, 6 3, 0 6" fill="#fed7aa" />
          </marker>
        </defs>
        <text x={50} y={75} fontSize="6" fill="#fed7aa" fontFamily="monospace">qx, qy</text>
      </svg>
    ),
  },
];

const HOLD_MS = 3300;

export default function SidecarPipeline() {
  const [active, setActive] = useState(0);

  useEffect(() => {
    const t = window.setTimeout(() => {
      setActive((i) => (i + 1) % METHODS.length);
    }, HOLD_MS);
    return () => clearTimeout(t);
  }, [active]);

  const m = METHODS[active];
  const totalPairs = METHODS.reduce((a, m) => a + m.pairs, 0);
  const totalBytes = METHODS.reduce((a, m) => a + m.bytes, 0);

  return (
    <div className="card !p-6 lg:!p-8 not-prose">
      <div className="flex items-center justify-between mb-6">
        <div className="mono text-[11px] uppercase tracking-widest text-comma-green">interactive · auto</div>
        <div className="mono text-[11px] uppercase text-white/40">5 sidecar edit families</div>
      </div>

      {/* Method tabs */}
      <div className="grid grid-cols-5 gap-2 mb-6">
        {METHODS.map((mm, i) => (
          <div
            key={mm.key}
            className={`p-3 border-l-2 transition-all ${i === active ? "bg-white/5" : ""}`}
            style={{ borderLeftColor: i === active ? mm.color : "rgba(255,255,255,0.15)", minHeight: 80 }}
          >
            <div className="mono text-[10px] uppercase tracking-widest" style={{ color: i === active ? mm.color : "rgba(255,255,255,0.55)" }}>{mm.short}</div>
            <div className={`text-[13px] mt-1 leading-tight ${i === active ? "text-white" : "text-white/60"}`}>{mm.full}</div>
            <div className="mono text-[10px] mt-1 text-white/40">{mm.pairs} pairs · {mm.bytes} B</div>
          </div>
        ))}
      </div>

      {/* Active detail: fixed-height container */}
      <div className="grid lg:grid-cols-[200px_1fr_220px] gap-6 items-start" style={{ minHeight: 200 }}>
        <div className="border border-white/10 bg-black h-48 flex items-center justify-center">
          {m.visualize()}
        </div>
        <div>
          <div className="mono text-[11px] uppercase tracking-widest mb-2" style={{ color: m.color }}>{m.short} · {m.full}</div>
          <p className="text-[14px] text-white/80 leading-relaxed">{m.blurb}</p>
        </div>
        <div className="border border-white/10 bg-black p-4 mono text-[12px] space-y-3">
          <div>
            <div className="text-white/40 text-[10px] uppercase tracking-widest">pairs in final blob</div>
            <div className="text-[24px] tabular-nums" style={{ color: m.color }}>{m.pairs}</div>
            <div className="text-white/40 text-[10px]">{((m.pairs / totalPairs) * 100).toFixed(1)}% of {totalPairs} total</div>
          </div>
          <div className="border-t border-white/10 pt-3">
            <div className="text-white/40 text-[10px] uppercase tracking-widest">bytes</div>
            <div className="text-[24px] tabular-nums" style={{ color: m.color }}>{m.bytes}</div>
            <div className="text-white/40 text-[10px]">{m.perPair} · {((m.bytes / totalBytes) * 100).toFixed(1)}% of sidecar</div>
          </div>
        </div>
      </div>
    </div>
  );
}
