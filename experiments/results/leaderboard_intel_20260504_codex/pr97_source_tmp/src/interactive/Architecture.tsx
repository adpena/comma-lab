// SVG flowchart of the H3 generator with bigger labels, an animated data
// pulse on the active edge, and a fullscreen toggle (because the diagram is
// genuinely dense and reads better when it has room to breathe).

import { useEffect, useRef, useState } from "react";

interface Block {
  id: string;
  name: string;
  layer: string;
  shape: string;
  params: number;
  kind: "fp4" | "fp16" | "linear" | "io";
  desc: string;
  x: number; y: number; w: number; h: number;
}

const BLOCKS: Block[] = [
  // Inputs
  { id: "mask",     name: "mask",          layer: "input",            shape: "(B, 384, 512)",  params: 0,    kind: "io",
    desc: "5-class SegNet semantic mask, decoded losslessly from the range-coded archive.",
    x: 40, y: 50, w: 180, h: 80 },
  { id: "pose",     name: "pose",          layer: "input",            shape: "(B, 6)",          params: 0,    kind: "io",
    desc: "6-D ego-motion vector. Per-dim N-bit quant in archive: speed gets 14 bits, rotations 4 each.",
    x: 40, y: 170, w: 180, h: 80 },

  // Pose MLP
  { id: "pose_mlp", name: "pose_mlp",      layer: "Linear → SiLU → LowRank → SiLU → LowRank", shape: "(B, 64)", params: 4544, kind: "linear",
    desc: "Three-layer MLP producing the cond vector. Last two layers are low-rank (64×16 + 16×64 each) SVD-warm-started from the trained dense weights. Drops 4096 params, improves pose distortion.",
    x: 280, y: 170, w: 320, h: 80 },

  // Trunk row 1
  { id: "embed",    name: "QEmb",          layer: "Embedding(5, 6)",  shape: "(B, 6, 384, 512)", params: 30, kind: "fp16",
    desc: "Per-class learned embedding into 6-D feature space. Tiny enough that we skip quantization.",
    x: 280, y: 50, w: 180, h: 80 },
  { id: "coords",   name: "+ coords",      layer: "(xx, yy) channels", shape: "(B, 8, 384, 512)", params: 0, kind: "io",
    desc: "Concat 2 normalized coord channels onto the embedding so the trunk knows pixel position.",
    x: 500, y: 50, w: 180, h: 80 },
  { id: "stem",     name: "stem",          layer: "DSConv(8, 56)",     shape: "(B, 56, 384, 512)", params: 800, kind: "fp4",
    desc: "Depthwise-separable conv. 3×3 depthwise + 1×1 pointwise expand from 8 to 56 channels.",
    x: 720, y: 50, w: 180, h: 80 },
  { id: "s1",       name: "s1",            layer: "Res(56)",           shape: "(B, 56, 384, 512)", params: 5800, kind: "fp4",
    desc: "First residual block. Output 's' is saved as the U-Net skip connection used after the bottleneck.",
    x: 940, y: 50, w: 180, h: 80 },

  // U-Net down/bottleneck/up column
  { id: "down",     name: "down",          layer: "DSConv(56, 64, s=2)", shape: "(B, 64, 192, 256)", params: 4200, kind: "fp4",
    desc: "Strided depthwise-separable conv. Halves spatial dims, expands to 64 channels (the U-Net bottleneck).",
    x: 940, y: 170, w: 180, h: 80 },
  { id: "d1",       name: "d1",            layer: "Res(64)",           shape: "(B, 64, 192, 256)", params: 7600, kind: "fp4",
    desc: "Bottleneck residual block. The deepest part of the U-Net.",
    x: 940, y: 290, w: 180, h: 80 },
  { id: "up",       name: "up",            layer: "Upsample×2 + DSConv(64, 56)", shape: "(B, 56, 384, 512)", params: 4400, kind: "fp4",
    desc: "Bilinear upsample back to model resolution + DSConv to project 64 to 56 channels.",
    x: 940, y: 410, w: 180, h: 80 },

  // Fuse + f1 row
  { id: "fuse",     name: "fuse",          layer: "DSConv(112, 56)",   shape: "(B, 56, 384, 512)", params: 6800, kind: "fp4",
    desc: "U-Net fusion: concat the up-path (56) with the saved skip 's' (56) → 112 channels → DSConv back to 56.",
    x: 720, y: 410, w: 180, h: 80 },
  { id: "f1",       name: "f1 (feat)",     layer: "Res(56)",           shape: "(B, 56, 384, 512)", params: 5800, kind: "fp4",
    desc: "Final trunk residual. The output 'feat' is the shared trunk feature both heads consume.",
    x: 500, y: 410, w: 180, h: 80 },

  // Head 1
  { id: "h1_r1",    name: "FiLMRes #1",    layer: "FiLMRes(56, cond=64)", shape: "(B, 56, 384, 512)", params: 9000, kind: "fp4",
    desc: "Pose-conditioned residual block. The FiLM Linear (zero-init) projects cond → 2×56 (gain, bias) channel-wise modulation. Wrapped in 1×1 QConv to get FP4 byte treatment.",
    x: 280, y: 530, w: 180, h: 80 },
  { id: "h1_r2",    name: "FiLMRes #2",    layer: "FiLMRes(56, cond=64)", shape: "(B, 56, 384, 512)", params: 9000, kind: "fp4",
    desc: "Second FiLM block. Added in autoresearch exp 182 (the dual-FiLM Head1 was the final architectural change).",
    x: 500, y: 530, w: 180, h: 80 },
  { id: "h1_out",   name: "out",           layer: "QConv2d(56, 3, 1)",  shape: "(B, 3, 384, 512)", params: 168, kind: "fp16",
    desc: "Final 1×1 conv to RGB. Output layer not quantized.",
    x: 720, y: 530, w: 180, h: 80 },
  { id: "h1_up",    name: "↑ upsample",    layer: "F.interpolate(bilinear)", shape: "(B, 3, 874, 1164)", params: 0, kind: "io",
    desc: "Bilinear upsample to camera resolution. Sidecar's optional F1 warp may translate this output by (qx, qy)/qscale pixels.",
    x: 940, y: 530, w: 180, h: 80 },
  { id: "frame1",   name: "FRAME 1 →",     layer: "PoseNet input",      shape: "uint8 (874, 1164, 3)", params: 0, kind: "io",
    desc: "Frame 1 doesn't have to look real. It just has to make PoseNet output the correct 6-D ego-motion vector.",
    x: 1160, y: 530, w: 240, h: 80 },

  // Head 2
  { id: "h2_r1",    name: "Res",            layer: "Res(56)",          shape: "(B, 56, 384, 512)", params: 5800, kind: "fp4",
    desc: "Plain residual block. Head 2 skips pose conditioning (frame 2 is the SegNet target, which doesn't read pose).",
    x: 280, y: 650, w: 180, h: 80 },
  { id: "h2_pre",   name: "pre",            layer: "DSConv(56, 52)",   shape: "(B, 52, 384, 512)", params: 3700, kind: "fp4",
    desc: "Project to head_hidden=52 channels.",
    x: 500, y: 650, w: 180, h: 80 },
  { id: "h2_out",   name: "out",            layer: "QConv2d(52, 3, 1)", shape: "(B, 3, 384, 512)", params: 156, kind: "fp16",
    desc: "Final 1×1 conv to RGB. Output layer not quantized.",
    x: 720, y: 650, w: 180, h: 80 },
  { id: "h2_up",    name: "↑ upsample",     layer: "F.interpolate(bilinear)", shape: "(B, 3, 874, 1164)", params: 0, kind: "io",
    desc: "Bilinear upsample to camera resolution.",
    x: 940, y: 650, w: 180, h: 80 },
  { id: "frame2",   name: "FRAME 2 →",      layer: "SegNet input",      shape: "uint8 (874, 1164, 3)", params: 0, kind: "io",
    desc: "Frame 2 has to look 'right' to SegNet's argmax. Saturated colors aligned with class boundaries.",
    x: 1160, y: 650, w: 240, h: 80 },
];

const EDGES: [string, string, "data" | "skip" | "cond"][] = [
  ["mask", "embed", "data"],
  ["embed", "coords", "data"],
  ["coords", "stem", "data"],
  ["stem", "s1", "data"],
  ["s1", "down", "data"],
  ["down", "d1", "data"],
  ["d1", "up", "data"],
  ["up", "fuse", "data"],
  ["s1", "fuse", "skip"],
  ["fuse", "f1", "data"],
  ["pose", "pose_mlp", "data"],
  ["pose_mlp", "h1_r1", "cond"],
  ["pose_mlp", "h1_r2", "cond"],
  ["f1", "h1_r1", "data"],
  ["h1_r1", "h1_r2", "data"],
  ["h1_r2", "h1_out", "data"],
  ["h1_out", "h1_up", "data"],
  ["h1_up", "frame1", "data"],
  ["f1", "h2_r1", "data"],
  ["h2_r1", "h2_pre", "data"],
  ["h2_pre", "h2_out", "data"],
  ["h2_out", "h2_up", "data"],
  ["h2_up", "frame2", "data"],
];

const KIND_COLOR: Record<Block["kind"], string> = {
  fp4: "#10b981",
  fp16: "#3b82f6",
  linear: "#a855f7",
  io: "#9ca3af",
};

const KIND_LABEL: Record<Block["kind"], string> = {
  fp4: "FP4 conv",
  fp16: "FP16",
  linear: "FP16 LowRank",
  io: "I/O",
};

const TOTAL_PARAMS = BLOCKS.reduce((a, b) => a + b.params, 0);

// SVG canvas. Picked deliberately wide so labels have breathing room.
const SVG_W = 1440;
const SVG_H = 770;

const HOLD_MS = 1500;

export default function Architecture() {
  const [activeIdx, setActiveIdx] = useState(0);
  const [fullscreen, setFullscreen] = useState(false);
  const userTouchedRef = useRef(false);

  useEffect(() => {
    if (userTouchedRef.current) return;
    const t = window.setTimeout(() => {
      setActiveIdx((i) => (i + 1) % BLOCKS.length);
    }, HOLD_MS);
    return () => clearTimeout(t);
  }, [activeIdx]);

  // ESC closes fullscreen
  useEffect(() => {
    if (!fullscreen) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setFullscreen(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [fullscreen]);

  const cur = BLOCKS[activeIdx];
  const onSelect = (id: string) => {
    userTouchedRef.current = true;
    const i = BLOCKS.findIndex(b => b.id === id);
    if (i >= 0) setActiveIdx(i);
  };

  const blockById = (id: string) => BLOCKS.find(b => b.id === id)!;

  // Routing for the SVG edge paths (simple orthogonal step routing).
  const edgePath = (from: string, to: string, kind: string) => {
    const a = blockById(from);
    const b = blockById(to);
    let ax = a.x + a.w, ay = a.y + a.h / 2;
    let bx = b.x, by = b.y + b.h / 2;
    // Vertical-down segments (trunk down/d1/up, fan-out into heads)
    const verticalChain = new Set([
      "s1->down", "down->d1", "d1->up",
      "f1->h1_r1", "f1->h2_r1",
    ]);
    if (verticalChain.has(`${from}->${to}`)) {
      ax = a.x + a.w / 2; ay = a.y + a.h;
      bx = b.x + b.w / 2; by = b.y;
    }
    // Skip from s1 (top) to fuse (bottom)
    if (kind === "skip") {
      ax = a.x + a.w / 2; ay = a.y + a.h;
      bx = b.x + b.w / 2; by = b.y;
    }
    // Cond fan-out from pose_mlp (down)
    if (kind === "cond") {
      ax = a.x + a.w / 2; ay = a.y + a.h;
      bx = b.x + b.w / 2; by = b.y;
    }
    // up → fuse: go up-then-left
    if (from === "up" && to === "fuse") {
      ax = a.x; ay = a.y + a.h / 2;
      bx = b.x + b.w; by = b.y + b.h / 2;
    }

    const midX = (ax + bx) / 2;
    return { d: `M ${ax} ${ay} L ${midX} ${ay} L ${midX} ${by} L ${bx} ${by}`, ax, ay, bx, by };
  };

  const Diagram = (
    <svg
      viewBox={`0 0 ${SVG_W} ${SVG_H}`}
      className="w-full h-auto"
      preserveAspectRatio="xMidYMid meet"
    >
      {/* Edges */}
      {EDGES.map(([from, to, kind], i) => {
        const { d } = edgePath(from, to, kind);
        const isActive = (cur.id === from || cur.id === to);
        const stroke =
          kind === "skip" ? "#facc15" :
          kind === "cond" ? "#a855f7" :
          isActive ? "#51FF00" : "rgba(255,255,255,0.28)";
        return (
          <g key={i}>
            <path d={d} stroke={stroke} strokeWidth={isActive ? 2.5 : 1.5} fill="none"
              strokeDasharray={kind === "skip" ? "6 4" : kind === "cond" ? "3 4" : undefined} />
            {isActive && kind === "data" && (
              <circle r={5} fill="#51FF00">
                <animateMotion dur="0.8s" repeatCount="1" path={d} />
              </circle>
            )}
          </g>
        );
      })}

      {/* Skip + cond legend lines */}
      <text x={1180} y={120} fontSize="12" fill="#facc15" fontFamily="monospace">- - U-Net skip</text>
      <text x={1180} y={140} fontSize="12" fill="#a855f7" fontFamily="monospace">· · pose conditioning</text>

      {/* Block rectangles */}
      {BLOCKS.map((b) => {
        const isActive = b.id === cur.id;
        const color = KIND_COLOR[b.kind];
        return (
          <g key={b.id} onClick={() => onSelect(b.id)} style={{ cursor: "pointer" }}>
            <rect
              x={b.x} y={b.y} width={b.w} height={b.h}
              fill={isActive ? color : "rgba(0,0,0,0.55)"}
              fillOpacity={isActive ? 0.18 : 1}
              stroke={isActive ? "#51FF00" : color}
              strokeWidth={isActive ? 2.5 : 1.5}
            />
            <text x={b.x + b.w / 2} y={b.y + 30} textAnchor="middle"
              fontSize="18" fontWeight={700}
              fill={isActive ? "#51FF00" : "#fff"}
              fontFamily="Inter, sans-serif">
              {b.name}
            </text>
            <text x={b.x + b.w / 2} y={b.y + 52} textAnchor="middle"
              fontSize="12" fill={isActive ? "#51FF00" : "rgba(255,255,255,0.55)"}
              fontFamily="monospace">
              {b.shape}
            </text>
            {b.params > 0 && (
              <text x={b.x + b.w / 2} y={b.y + 70} textAnchor="middle"
                fontSize="12" fill={color} fontFamily="monospace">
                {b.params.toLocaleString()} params
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );

  return (
    <>
      <div className="card !p-6 lg:!p-8 not-prose">
        <div className="flex items-center justify-between mb-6">
          <div className="mono text-[11px] uppercase tracking-widest text-comma-green">interactive · auto</div>
          <div className="mono text-[11px] uppercase text-white/40">H3 generator architecture · 92K params</div>
        </div>

        {/* Explainer + legend */}
        <div className="mb-5 grid lg:grid-cols-[1fr_auto] gap-6 items-start">
          <p className="text-[14px] text-white/75 leading-relaxed max-w-[760px]">
            The actual H3 generator from <span className="mono text-comma-green">autoresearch/train.py</span>: a 92K-parameter
            U-Net trunk with two output heads. <span className="text-comma-green">Head 1</span> takes pose conditioning
            and writes frame 1 (the PoseNet target); <span className="text-comma-green">Head 2</span> writes frame 2 (the
            SegNet target). The trunk is shared. Click any block to pin it, or hit the expand button for fullscreen.
          </p>
          <div className="flex flex-col items-end gap-3">
            <button
              onClick={() => setFullscreen(true)}
              className="mono text-[11px] uppercase tracking-wider px-3 py-2 border border-comma-green text-comma-green hover:bg-comma-green hover:text-black transition-colors"
            >
              ⛶ expand fullscreen
            </button>
            <div className="border border-white/10 bg-black p-3 mono text-[10px] space-y-1 min-w-[170px]">
              <div className="text-white/40 uppercase tracking-widest mb-2">legend</div>
              {(["fp4", "linear", "fp16", "io"] as const).map((k) => (
                <div key={k} className="flex items-center gap-2">
                  <div className="w-3 h-3" style={{ background: KIND_COLOR[k] }} />
                  <span className="text-white/70">{KIND_LABEL[k]}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Inline SVG */}
        <div className="bg-black border border-white/10 p-2">
          {Diagram}
        </div>

        {/* Detail panel */}
        <div className="mt-5 grid lg:grid-cols-[1fr_280px] gap-5 items-start" style={{ minHeight: 130 }}>
          <div className="border border-white/10 bg-black p-4">
            <div className="mono text-[10px] uppercase tracking-widest text-comma-green mb-1">{cur.layer}</div>
            <div className="text-xl font-bold text-white">{cur.name}</div>
            <p className="text-[14px] text-white/80 mt-2 leading-snug">{cur.desc}</p>
          </div>
          <div className="border border-white/10 bg-black p-4 mono text-[12px] grid grid-cols-2 gap-y-2">
            <div className="text-white/40">output shape</div>
            <div className="text-white text-right tabular-nums">{cur.shape}</div>
            <div className="text-white/40">params</div>
            <div className="text-white text-right tabular-nums">{cur.params.toLocaleString()}</div>
            <div className="text-white/40">storage</div>
            <div className="text-right" style={{ color: KIND_COLOR[cur.kind] }}>{KIND_LABEL[cur.kind]}</div>
            <div className="text-white/40 col-span-2 border-t border-white/10 pt-2 mt-1">
              {((cur.params / TOTAL_PARAMS) * 100).toFixed(1)}% of {(TOTAL_PARAMS / 1000).toFixed(0)}K total
            </div>
          </div>
        </div>
      </div>

      {/* Fullscreen modal */}
      {fullscreen && (
        <div
          className="fixed inset-0 z-50 bg-black/95 backdrop-blur-sm flex flex-col"
          onClick={() => setFullscreen(false)}
        >
          <div className="px-6 py-4 flex items-center justify-between border-b border-white/15">
            <div className="mono text-[11px] uppercase tracking-widest text-comma-green">H3 generator architecture · 92K params</div>
            <button
              className="mono text-[11px] uppercase tracking-wider px-3 py-2 border border-white/30 hover:border-comma-green hover:text-comma-green text-white"
              onClick={(e) => { e.stopPropagation(); setFullscreen(false); }}
            >
              ✕ close (esc)
            </button>
          </div>
          <div className="flex-1 overflow-auto p-6" onClick={(e) => e.stopPropagation()}>
            <div className="bg-black">
              {Diagram}
            </div>
            <div className="mt-5 grid lg:grid-cols-[1fr_320px] gap-5 items-start max-w-[1600px] mx-auto">
              <div className="border border-white/10 bg-black p-4">
                <div className="mono text-[10px] uppercase tracking-widest text-comma-green mb-1">{cur.layer}</div>
                <div className="text-xl font-bold text-white">{cur.name}</div>
                <p className="text-[14px] text-white/80 mt-2 leading-snug">{cur.desc}</p>
              </div>
              <div className="border border-white/10 bg-black p-4 mono text-[12px] grid grid-cols-2 gap-y-2">
                <div className="text-white/40">output shape</div>
                <div className="text-white text-right tabular-nums">{cur.shape}</div>
                <div className="text-white/40">params</div>
                <div className="text-white text-right tabular-nums">{cur.params.toLocaleString()}</div>
                <div className="text-white/40">storage</div>
                <div className="text-right" style={{ color: KIND_COLOR[cur.kind] }}>{KIND_LABEL[cur.kind]}</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
