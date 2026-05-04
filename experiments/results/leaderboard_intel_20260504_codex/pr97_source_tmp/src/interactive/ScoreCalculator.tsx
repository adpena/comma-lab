// Live calculator for the challenge score formula:
//   score = 100 * seg_dist + sqrt(10 * pose_dist) + 25 * (archive_bytes / 37,545,489)
// Drag the sliders, watch each term and the total update.
// Pre-loaded with our final submission's values so reader can perturb from there.

import { useState } from "react";

const UNCOMPRESSED = 37_545_489;
const LEADER = 0.275;

// Anchor presets the reader can jump to
const PRESETS = [
  { name: "baseline_fast", seg: 0.0023, pose: 0.0042, bytes: 7_350_000 },
  { name: "PR #84 leader", seg: 0.00086, pose: 0.00097, bytes: 215_735 },
  { name: "ours (no sidecar)", seg: 0.000271, pose: 0.000604, bytes: 194_780 },
  { name: "ours (final)", seg: 0.000272, pose: 0.000495, bytes: 197_160 },
];

export default function ScoreCalculator() {
  const [seg, setSeg] = useState(0.000272);
  const [pose, setPose] = useState(0.000495);
  const [bytes, setBytes] = useState(197_160);

  const segTerm = 100 * seg;
  const poseTerm = Math.sqrt(10 * pose);
  const rateTerm = 25 * (bytes / UNCOMPRESSED);
  const total = segTerm + poseTerm + rateTerm;
  const beatsLeader = total < LEADER;

  return (
    <div className="card !p-6 lg:!p-8 not-prose">
      <div className="flex items-center justify-between mb-6">
        <div className="mono text-[11px] uppercase tracking-widest text-comma-green">interactive</div>
        <div className="mono text-[11px] uppercase text-white/40">score formula calculator</div>
      </div>

      <div className="grid lg:grid-cols-[1fr_320px] gap-8 items-start" style={{ minHeight: 460 }}>
        {/* Sliders */}
        <div className="space-y-5">
          <Slider
            label="SegNet distortion"
            value={seg}
            min={0} max={0.003} step={0.000001}
            format={(v) => v.toFixed(6)}
            color="#51FF00"
            onChange={setSeg}
            term={`100 × ${seg.toFixed(6)} = ${segTerm.toFixed(4)}`}
          />
          <Slider
            label="PoseNet distortion"
            value={pose}
            min={0} max={0.005} step={0.000001}
            format={(v) => v.toFixed(6)}
            color="#51FF00"
            onChange={setPose}
            term={`√(10 × ${pose.toFixed(6)}) = ${poseTerm.toFixed(4)}`}
          />
          <Slider
            label="archive.zip bytes"
            value={bytes}
            min={100_000} max={1_000_000} step={100}
            format={(v) => v.toLocaleString()}
            color="#51FF00"
            onChange={setBytes}
            term={`25 × ${bytes.toLocaleString()} / ${UNCOMPRESSED.toLocaleString()} = ${rateTerm.toFixed(4)}`}
          />

          <div className="pt-3">
            <div className="mono text-[10px] uppercase tracking-widest text-white/40 mb-2">jump to preset</div>
            <div className="flex flex-wrap gap-2">
              {PRESETS.map((p) => (
                <button
                  key={p.name}
                  className="mono text-[11px] uppercase tracking-wider px-3 py-2 border border-white/15 hover:border-comma-green hover:text-comma-green transition-colors"
                  onClick={() => { setSeg(p.seg); setPose(p.pose); setBytes(p.bytes); }}
                >
                  {p.name}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Score readout */}
        <div className="border border-white/10 p-6 bg-black">
          <div className="mono text-[10px] uppercase tracking-widest text-white/40 mb-3">final score</div>
          <div className={`text-[68px] leading-none font-bold tracking-tight ${beatsLeader ? "text-comma-green" : "text-white"}`}>
            {total.toFixed(5)}
          </div>
          <div className="mt-3 mono text-[12px] text-white/55">
            <div className="flex justify-between gap-2"><span>seg term</span><span>{segTerm.toFixed(4)}</span></div>
            <div className="flex justify-between gap-2"><span>+ pose term</span><span>{poseTerm.toFixed(4)}</span></div>
            <div className="flex justify-between gap-2 border-t border-white/10 pt-1 mt-1"><span>+ rate term</span><span>{rateTerm.toFixed(4)}</span></div>
          </div>
          <div className="mt-5 mono text-[11px]">
            {beatsLeader ? (
              <span className="text-comma-green">▼ {(LEADER - total).toFixed(4)} below the {LEADER.toFixed(3)} leader</span>
            ) : (
              <span className="text-comma-error">▲ {(total - LEADER).toFixed(4)} above the {LEADER.toFixed(3)} leader</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format: (v: number) => string;
  color: string;
  onChange: (v: number) => void;
  term: string;
}
function Slider({ label, value, min, max, step, format, color, onChange, term }: SliderProps) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div>
      <div className="flex items-baseline justify-between mb-2">
        <span className="mono text-[12px] uppercase tracking-widest text-white/55">{label}</span>
        <span className="mono text-[14px] text-white">{format(value)}</span>
      </div>
      <input
        type="range"
        min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full appearance-none h-[3px] bg-white/15 outline-none cursor-pointer"
        style={{
          background: `linear-gradient(to right, ${color} 0%, ${color} ${pct}%, rgba(255,255,255,0.15) ${pct}%, rgba(255,255,255,0.15) 100%)`,
        }}
      />
      <div className="mono text-[11px] text-white/40 mt-1">{term}</div>
    </div>
  );
}
