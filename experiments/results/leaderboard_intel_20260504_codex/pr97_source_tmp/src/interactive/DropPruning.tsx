// Auto-playing 15-step drop-pruning trajectory. Loops. No buttons.

import { useEffect, useState } from "react";

interface Step {
  step: number;
  drops: number;
  score: number;
  bytes: number;
  action: string;
}

const TRACE: Step[] = [
  { step: 0, drops: 0, score: 0.285424, bytes: 2434, action: "starting score, all 400 sidecar pairs" },
  { step: 1, drops: 50, score: 0.285201, bytes: 2277, action: "greedy phase: 50 drops in first wave" },
  { step: 2, drops: 51, score: 0.285196, bytes: 2265, action: "local: drop pair #441" },
  { step: 3, drops: 52, score: 0.285191, bytes: 2249, action: "local: drop pair #159" },
  { step: 4, drops: 53, score: 0.285187, bytes: 2238, action: "local: drop pair #119" },
  { step: 5, drops: 54, score: 0.285183, bytes: 2218, action: "local: drop pair #128" },
  { step: 6, drops: 55, score: 0.285180, bytes: 2212, action: "local: drop pair #186" },
  { step: 7, drops: 56, score: 0.285177, bytes: 2205, action: "local: drop pair #180" },
  { step: 8, drops: 57, score: 0.285174, bytes: 2199, action: "local: drop pair #389" },
  { step: 9, drops: 58, score: 0.285172, bytes: 2196, action: "local: drop pair #468" },
  { step: 10, drops: 59, score: 0.285170, bytes: 2189, action: "local: drop pair #171" },
  { step: 11, drops: 60, score: 0.285169, bytes: 2185, action: "local: drop pair #87" },
  { step: 12, drops: 60, score: 0.285167, bytes: 2175, action: "swap: replace #541 with #126 (−10 B)" },
  { step: 13, drops: 61, score: 0.285166, bytes: 2158, action: "local: drop pair #125" },
  { step: 14, drops: 62, score: 0.285165, bytes: 2153, action: "local: drop pair #415" },
  { step: 15, drops: 63, score: 0.285164, bytes: 2152, action: "converged: 63 drops total, 282 B saved" },
];

const MAX_BYTES = 2434;
const MIN_BYTES = 2100;
const MAX_SCORE = 0.285424;
const MIN_SCORE = 0.285150;

const STEP_MS = 600;
const HOLD_END_MS = 2000;

export default function DropPruning() {
  const [idx, setIdx] = useState(0);

  useEffect(() => {
    const isEnd = idx >= TRACE.length - 1;
    const t = window.setTimeout(() => {
      setIdx(isEnd ? 0 : idx + 1);
    }, isEnd ? HOLD_END_MS : STEP_MS);
    return () => clearTimeout(t);
  }, [idx]);

  const cur = TRACE[idx];
  const start = TRACE[0];

  return (
    <div className="card !p-6 lg:!p-8 not-prose">
      <div className="flex items-center justify-between mb-6">
        <div className="mono text-[11px] uppercase tracking-widest text-comma-green">interactive · auto</div>
        <div className="mono text-[11px] uppercase text-white/40">drop-pruning trajectory</div>
      </div>

      <div className="grid grid-cols-3 gap-4 mb-6" style={{ minHeight: 120 }}>
        <Metric label="step" value={cur.step.toString()} sub="of 15" />
        <Metric label="sidecar bytes" value={cur.bytes.toLocaleString()} sub={`${start.bytes - cur.bytes} saved`} highlight />
        <Metric label="predicted score" value={cur.score.toFixed(6)} sub={`${(start.score - cur.score).toFixed(6)} better`} highlight />
      </div>

      <div className="bg-black border border-white/10 p-5 mb-6" style={{ minHeight: 240 }}>
        <div className="flex items-baseline justify-between mb-3">
          <div className="mono text-[10px] uppercase tracking-widest text-white/40">trajectory · 15 steps</div>
          <div className="mono text-[10px] text-white/40">x = step  ·  red = bytes  ·  green = score</div>
        </div>
        <svg viewBox="0 0 600 180" className="w-full h-44">
          <BytesPath />
          <ScorePath />
          <line
            x1={(cur.step / 15) * 580 + 10}
            x2={(cur.step / 15) * 580 + 10}
            y1={5} y2={175}
            stroke="rgba(255,255,255,0.4)" strokeWidth={1} strokeDasharray="2 2"
          />
          <circle
            cx={(cur.step / 15) * 580 + 10}
            cy={170 - ((cur.bytes - MIN_BYTES) / (MAX_BYTES - MIN_BYTES)) * 75}
            r={4} fill="#ef4444" stroke="#000" strokeWidth={1}
          />
          <circle
            cx={(cur.step / 15) * 580 + 10}
            cy={85 - ((cur.score - MIN_SCORE) / (MAX_SCORE - MIN_SCORE)) * 75}
            r={4} fill="#51FF00" stroke="#000" strokeWidth={1}
          />
        </svg>
      </div>

      <div className="bg-black border border-white/10 p-4" style={{ minHeight: 64 }}>
        <div className="mono text-[10px] uppercase tracking-widest text-white/40 mb-2">step {cur.step}</div>
        <div className="mono text-[14px] text-white">{cur.action}</div>
      </div>
    </div>
  );
}

function BytesPath() {
  const path = TRACE.map((s, i) => {
    const x = (s.step / 15) * 580 + 10;
    const y = 170 - ((s.bytes - MIN_BYTES) / (MAX_BYTES - MIN_BYTES)) * 75;
    return `${i === 0 ? "M" : "L"} ${x} ${y}`;
  }).join(" ");
  return <path d={path} stroke="#ef4444" strokeWidth={2} fill="none" />;
}

function ScorePath() {
  const path = TRACE.map((s, i) => {
    const x = (s.step / 15) * 580 + 10;
    const y = 85 - ((s.score - MIN_SCORE) / (MAX_SCORE - MIN_SCORE)) * 75;
    return `${i === 0 ? "M" : "L"} ${x} ${y}`;
  }).join(" ");
  return <path d={path} stroke="#51FF00" strokeWidth={2} fill="none" />;
}

function Metric({ label, value, sub, highlight = false }: { label: string; value: string; sub: string; highlight?: boolean }) {
  return (
    <div className="border border-white/10 bg-black p-4">
      <div className="mono text-[10px] uppercase tracking-widest text-white/40">{label}</div>
      <div className={`text-[28px] font-bold tabular-nums leading-none mt-1 ${highlight ? "text-comma-green" : "text-white"}`}>{value}</div>
      <div className="mono text-[11px] text-white/55 mt-1">{sub}</div>
    </div>
  );
}
