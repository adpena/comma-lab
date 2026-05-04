// Auto-cycling codec evolution timeline. No buttons. Behaves like a video.

import { useEffect, useState } from "react";

interface Stage {
  key: string;
  name: string;
  bytes: number;
  delta?: string;
  blurb: string;
  lossy?: boolean;
}

const STAGES: Stage[] = [
  { key: "av1",        name: "AV1 grayscale (lossy)",        bytes: 219_588, blurb: "Encode the 5-class mask as grayscale and pass it through libaom-av1. ~44k pixels per video decode to the wrong class.", lossy: true },
  { key: "range",      name: "Range-coded adaptive9bin",     bytes: 174_281, delta: "−45 KB", blurb: "PR #81's binary arithmetic coder. 9-context fail-fast cascade, lossless. Already smaller than AV1." },
  { key: "transpose",  name: "+ transposed orientation",     bytes: 156_656, delta: "−17 KB", blurb: "Encode columns first instead of rows. Dashcam scenes have vertical structure, so the UP predictor fires more often." },
  { key: "tiles",      name: "+ 22 tiles, splits [3,8,8,3]", bytes: 138_355, delta: "−18 KB", blurb: "Split mask into 22 spatial tiles, each with its own adaptive model. Asymmetric W splits found by sweep." },
  { key: "priors",     name: "+ tuned init priors",          bytes: 138_231, delta: "−0.1 KB", blurb: "UP/LEFT/PREV true-init counts swept on this dataset. Trust UP and LEFT more than the codec defaults assume." },
  { key: "transforms", name: "+ per-tile scan transforms",   bytes: 135_769, delta: "−2.5 KB", blurb: "Each tile gets a hard-coded lossless flip (revT/revH/revW). Costs zero archive bytes because the schedule is baked into both encoder and decoder." },
  { key: "cascade",    name: "+ deeper cascade + adapt=8",   bytes: 135_120, delta: "−0.6 KB", blurb: "Extend cascade from 3 to 9 questions, drop adaptive update increment from +20 to +8." },
];

const AV1_BASELINE = STAGES[0].bytes;
const FLOOR = 100_000;
const STEP_MS = 2200;

export default function CodecTimeline() {
  const [idx, setIdx] = useState(0);

  useEffect(() => {
    const t = window.setTimeout(() => {
      setIdx((i) => (i + 1) % STAGES.length);
    }, STEP_MS);
    return () => clearTimeout(t);
  }, [idx]);

  const cur = STAGES[idx];
  const pctOfBaseline = ((cur.bytes - FLOOR) / (AV1_BASELINE - FLOOR)) * 100;
  const totalSaved = AV1_BASELINE - cur.bytes;

  return (
    <div className="card !p-6 lg:!p-8 not-prose">
      <div className="flex items-center justify-between mb-6">
        <div className="mono text-[11px] uppercase tracking-widest text-comma-green">interactive · auto</div>
        <div className="mono text-[11px] uppercase text-white/40">codec evolution</div>
      </div>

      {/* Big bytes display: fixed-height cluster */}
      <div className="grid lg:grid-cols-[1fr_auto] gap-8 items-end mb-6" style={{ minHeight: 110 }}>
        <div>
          <div className="mono text-[11px] uppercase tracking-widest text-white/40 mb-1 min-h-[16px]">{cur.name}</div>
          <div className="text-[58px] leading-none font-bold text-white tabular-nums">
            {cur.bytes.toLocaleString()}
            <span className="text-[24px] text-white/40 ml-3 font-normal">bytes</span>
          </div>
          <div className="mono text-[13px] text-comma-green mt-2 min-h-[18px]">
            {cur.delta && <>{cur.delta} from previous · {((1 - cur.bytes / AV1_BASELINE) * 100).toFixed(1)}% smaller than AV1 baseline</>}
          </div>
        </div>
        <div className="text-right">
          <div className="mono text-[10px] uppercase tracking-widest text-white/40">total saved vs AV1</div>
          <div className="mono text-[32px] text-comma-green font-bold tabular-nums">{totalSaved.toLocaleString()} B</div>
        </div>
      </div>

      {/* Animated bar */}
      <div className="relative h-12 bg-black border border-white/10 mb-3">
        <div
          className="absolute inset-y-0 left-0 transition-[width] duration-700 ease-out"
          style={{
            width: `${Math.max(pctOfBaseline, 4)}%`,
            background: "linear-gradient(90deg, #51FF00, #51FF00 70%, transparent)",
            opacity: cur.lossy ? 0.35 : 1,
          }}
        />
        <div className="absolute inset-0 flex items-center justify-between px-3 mono text-[11px]">
          <span className="text-black mix-blend-difference">{cur.bytes.toLocaleString()} B</span>
          <span className="text-white/55">→ AV1 baseline {AV1_BASELINE.toLocaleString()} B</span>
        </div>
      </div>

      {/* Blurb area: fixed height to prevent layout shift */}
      <p className="text-[14px] text-white/80 leading-snug" style={{ minHeight: 60 }}>{cur.blurb}</p>

      {/* Stage indicators */}
      <div className="grid grid-cols-7 gap-1 mt-4">
        {STAGES.map((s, i) => (
          <div
            key={s.key}
            className={`mono text-[10px] uppercase tracking-wider px-2 py-2 border-l-2 transition-all ${i === idx ? "border-comma-green bg-comma-green/10 text-comma-green" : "border-white/15 text-white/40"}`}
            style={{ minHeight: 56 }}
          >
            <div className="text-[9px] mb-1 opacity-60">stage {i}</div>
            <div className="text-[10px] leading-tight">{(s.bytes / 1024).toFixed(1)}K</div>
          </div>
        ))}
      </div>
    </div>
  );
}
