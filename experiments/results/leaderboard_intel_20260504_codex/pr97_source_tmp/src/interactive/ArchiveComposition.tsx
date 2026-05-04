// Auto-cycling donut + always-visible detail panel (fixed height, no
// reflow on hover). Hover overrides the auto-cycle.

import { useEffect, useRef, useState } from "react";

interface Slice {
  key: string;
  name: string;
  bytes: number;
  color: string;
  desc: string;
}

const SLICES: Slice[] = [
  { key: "mask", name: "mask",  bytes: 135_120, color: "#3b82f6",
    desc: "22 range-coded tiles. 600 frames × 384×512 px lossless 5-class semantic mask compressed by an adaptive 9-context binary arithmetic coder with per-tile static scan transforms." },
  { key: "model", name: "model", bytes: 57_238, color: "#10b981",
    desc: "92K-param FP4 H3 generator. Flat per-tensor packing, no pickle, no Python class hierarchy. Decoder agrees on a hard-coded SCHEMA so weights are just raw nibbles + per-block FP16 scales." },
  { key: "sidecar", name: "sidecar", bytes: 2_376, color: "#ef4444",
    desc: "Per-pair learned corrections. Up to 5 edit types per pair (X2 mask flips, CMA-ES flips, pattern flips, pose deltas, F1 warps), bit-packed and lzma'd." },
  { key: "pose", name: "pose", bytes: 2_310, color: "#f59e0b",
    desc: "600 × 6 float32 pose vectors quantized per-dim (14, 4, 4, 4, 4, 4) bits, bit-packed, brotli'd. Speed gets 14 bits because its magnitude is ~30; rotations get 4 because their range is ~0.05." },
  { key: "envelope", name: "headers + zip", bytes: 116, color: "#94a3b8",
    desc: "Single ZIP_STORED member named 'p', plus our own 4-byte length prefixes for each component." },
];

const TOTAL = SLICES.reduce((a, s) => a + s.bytes, 0);

const COMPETITORS = [
  { name: "ours (vibe_coder_final_boss)", bytes: 197_160, score: 0.229, color: "#51FF00", us: true },
  { name: "PR #84 ottokunkel",            bytes: 215_735, score: 0.275, color: "#3b82f6" },
  { name: "PR #82 henosis_frontier",      bytes: 296_789, score: 0.30,  color: "#8b5cf6" },
  { name: "PR #79 qpose14_segactions",    bytes: 277_388, score: 0.31,  color: "#f59e0b" },
  { name: "PR #74 ph4ntom_drv",           bytes: 321_311, score: 0.35,  color: "#ef4444" },
];

const HOLD_MS = 2400;

export default function ArchiveComposition() {
  const [activeIdx, setActiveIdx] = useState(0);
  const userTouchedRef = useRef(false);

  useEffect(() => {
    if (userTouchedRef.current) return;
    const t = window.setTimeout(() => {
      setActiveIdx((i) => (i + 1) % SLICES.length);
    }, HOLD_MS);
    return () => clearTimeout(t);
  }, [activeIdx]);

  const cx = 150, cy = 150, r = 110, rInner = 70;
  let cumulative = 0;
  const arcs = SLICES.map((s, i) => {
    const startA = (cumulative / TOTAL) * Math.PI * 2 - Math.PI / 2;
    cumulative += s.bytes;
    const endA = (cumulative / TOTAL) * Math.PI * 2 - Math.PI / 2;
    const large = endA - startA > Math.PI ? 1 : 0;
    const x1 = cx + r * Math.cos(startA);
    const y1 = cy + r * Math.sin(startA);
    const x2 = cx + r * Math.cos(endA);
    const y2 = cy + r * Math.sin(endA);
    const xi1 = cx + rInner * Math.cos(endA);
    const yi1 = cy + rInner * Math.sin(endA);
    const xi2 = cx + rInner * Math.cos(startA);
    const yi2 = cy + rInner * Math.sin(startA);
    const path = `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2} L ${xi1} ${yi1} A ${rInner} ${rInner} 0 ${large} 0 ${xi2} ${yi2} Z`;
    return { ...s, path, i };
  });

  const cur = SLICES[activeIdx];
  const onSelect = (i: number) => { userTouchedRef.current = true; setActiveIdx(i); };

  return (
    <div className="card !p-6 lg:!p-8 not-prose">
      <div className="flex items-center justify-between mb-6">
        <div className="mono text-[11px] uppercase tracking-widest text-comma-green">interactive · auto</div>
        <div className="mono text-[11px] uppercase text-white/40">197 KB archive composition</div>
      </div>

      <div className="grid lg:grid-cols-[300px_1fr] gap-8 items-start">
        <div className="relative">
          <svg viewBox="0 0 300 300" className="w-full max-w-[300px] mx-auto">
            {arcs.map((a) => (
              <path
                key={a.key} d={a.path}
                fill={a.color}
                opacity={activeIdx === a.i ? 1 : 0.35}
                onClick={() => onSelect(a.i)}
                style={{ cursor: "pointer", transition: "opacity 250ms" }}
              />
            ))}
            <text x={cx} y={cy - 5} textAnchor="middle" fill="#fff" fontSize="11" fontFamily="monospace">{cur.name.toUpperCase()}</text>
            <text x={cx} y={cy + 18} textAnchor="middle" fill={cur.color} fontSize="22" fontWeight="bold" fontFamily="monospace">{cur.bytes.toLocaleString()}</text>
            <text x={cx} y={cy + 35} textAnchor="middle" fill="#fff" opacity={0.5} fontSize="10" fontFamily="monospace">{((cur.bytes / TOTAL) * 100).toFixed(1)}%</text>
          </svg>
        </div>

        <div className="space-y-1">
          {/* List rows: no hover background, fixed row size */}
          {SLICES.map((s, i) => (
            <button
              key={s.key}
              onClick={() => onSelect(i)}
              className={`w-full grid grid-cols-[16px_1fr_120px_60px] gap-3 items-center px-3 py-2 transition-opacity ${i === activeIdx ? "opacity-100" : "opacity-50"}`}
              style={{ minHeight: 40 }}
            >
              <div className="w-3 h-3" style={{ background: s.color }} />
              <div className="text-left text-white text-[14px]">{s.name}</div>
              <div className="mono text-[12px] text-white/55 text-right tabular-nums">{s.bytes.toLocaleString()} B</div>
              <div className="mono text-[12px] text-right tabular-nums" style={{ color: s.color }}>{((s.bytes / TOTAL) * 100).toFixed(1)}%</div>
            </button>
          ))}

          {/* Always-visible detail panel: fixed height */}
          <div className="border border-white/10 bg-black p-4 mt-3" style={{ minHeight: 110 }}>
            <div className="mono text-[10px] uppercase tracking-widest mb-1" style={{ color: cur.color }}>{cur.name}</div>
            <p className="text-[13px] text-white/85 leading-snug">{cur.desc}</p>
          </div>
        </div>
      </div>

      {/* Competitor comparison */}
      <div className="mt-10 border-t border-white/10 pt-8">
        <div className="mono text-[10px] uppercase tracking-widest text-white/40 mb-4">vs. competitor submissions (lower-left = better)</div>
        <div className="space-y-3">
          {COMPETITORS.sort((a, b) => a.score - b.score).map((c) => {
            const pct = (c.bytes / 350_000) * 100;
            return (
              <div key={c.name} className={`grid grid-cols-[260px_1fr_100px] gap-3 items-center ${c.us ? "" : "opacity-80"}`}>
                <div className={`mono text-[12px] ${c.us ? "text-comma-green" : "text-white/70"}`}>{c.name}</div>
                <div className="relative h-7 bg-black border border-white/10">
                  <div
                    className="absolute inset-y-0 left-0"
                    style={{ width: `${pct}%`, background: c.color, opacity: c.us ? 1 : 0.55 }}
                  />
                  <div className="absolute inset-0 flex items-center px-3 mono text-[11px] mix-blend-difference text-white">
                    {(c.bytes / 1024).toFixed(1)} KB
                  </div>
                </div>
                <div className={`mono text-[12px] tabular-nums text-right ${c.us ? "text-comma-green" : "text-white/55"}`}>
                  score {c.score.toFixed(3)}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
