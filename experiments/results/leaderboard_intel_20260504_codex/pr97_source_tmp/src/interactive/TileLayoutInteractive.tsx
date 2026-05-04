// Interactive tile layout. Natural landscape mask with the 22 codec tiles
// overlaid. Auto-cycles spotlight on each tile, click to pin.
//
// IMPORTANT framing: the codec internally transposes the mask before encoding
// (so its "horizontal bands" become "vertical bands" in the natural view).
// This visualization presents the natural orientation with tiles drawn in
// the codec's coordinate system mapped back.

import { useEffect, useRef, useState } from "react";

const W_SPLITS = [3, 8, 8, 3];           // splits per band, in codec order
const TRANSFORMS = [
  "id", "revHW", "revT_revH",
  "id", "id", "id", "revT", "revT", "revH", "revT", "id",
  "id", "id", "id", "revW", "revT_revH", "revT_revHW", "id", "id",
  "id", "id", "id",
];

const BAND_TOTAL_BYTES = [11_500, 38_900, 53_600, 31_120];

// In NATURAL orientation, each "band" is a 128-px-wide vertical column.
// What the codec calls "band 0" = leftmost column of the natural image, etc.
const BAND_NATURAL_LABEL = ["left edge", "center-left", "center-right", "right edge"];
const BAND_COLOR = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"];

function tileBytes(bandIdx: number, splitIdx: number) {
  const splits = W_SPLITS[bandIdx];
  const center = (splits - 1) / 2;
  const dist = Math.abs(splitIdx - center);
  const weight = 1 + (1 - dist / Math.max(center, 1)) * 0.4;
  const total = BAND_TOTAL_BYTES[bandIdx];
  let sum = 0;
  for (let i = 0; i < splits; i++) {
    const d = Math.abs(i - center);
    sum += 1 + (1 - d / Math.max(center, 1)) * 0.4;
  }
  return Math.round(total * weight / sum);
}

const TRANSFORM_DESC: Record<string, string> = {
  id: "no transform, encoded as-is",
  revT: "reverse time (the 600 frames flipped)",
  revH: "reverse rows (top↔bottom)",
  revW: "reverse cols (left↔right)",
  revHW: "reverse both H and W (180° rotate)",
  revT_revH: "reverse time + reverse rows",
  revT_revHW: "reverse time + reverse rows and cols",
};

interface Tile {
  band: number;       // 0..3, which 128-px-wide column in the natural image
  split: number;      // index within that band's splits
  tileIdx: number;
  tx: string;
  bytes: number;
}

const TILES: Tile[] = (() => {
  const out: Tile[] = [];
  let idx = 0;
  for (let b = 0; b < 4; b++) {
    for (let s = 0; s < W_SPLITS[b]; s++) {
      out.push({ band: b, split: s, tileIdx: idx, tx: TRANSFORMS[idx], bytes: tileBytes(b, s) });
      idx++;
    }
  }
  return out;
})();

// Class palette must match the rendered PNG
const CLASS_LEGEND = [
  { color: "#28303a", label: "road" },
  { color: "#6eb4ff", label: "sky" },
  { color: "#ff5a50", label: "vehicle" },
  { color: "#ffdc50", label: "lane" },
  { color: "#50c882", label: "foliage / other" },
];

const HOLD_MS = 1100;

export default function TileLayoutInteractive() {
  const [activeIdx, setActiveIdx] = useState(0);
  const userTouchedRef = useRef(false);

  useEffect(() => {
    if (userTouchedRef.current) return;
    const t = window.setTimeout(() => {
      setActiveIdx((i) => (i + 1) % TILES.length);
    }, HOLD_MS);
    return () => clearTimeout(t);
  }, [activeIdx]);

  const cur = TILES[activeIdx];
  const totalBytes = TILES.reduce((a, t) => a + t.bytes, 0);
  const txCount = TILES.filter(t => t.tx !== "id").length;

  const onSelect = (i: number) => { userTouchedRef.current = true; setActiveIdx(i); };

  // The mask is 512 wide × 384 tall (natural landscape).
  // Each band is 128 px wide (= 25% of width).
  // Within each band, splits run vertically (height divided by W_SPLITS[band]).

  return (
    <div className="card !p-6 lg:!p-8 not-prose">
      <div className="flex items-center justify-between mb-6">
        <div className="mono text-[11px] uppercase tracking-widest text-comma-green">interactive · auto</div>
        <div className="mono text-[11px] uppercase text-white/40">22-tile mask layout</div>
      </div>

      {/* Explainer up top: what the user is looking at */}
      <div className="mb-6 grid lg:grid-cols-[1fr_auto] gap-6 items-start">
        <p className="text-[14px] text-white/75 leading-relaxed max-w-[680px]">
          Below is one real SegNet mask from a dashcam pair, painted by class. The codec
          splits each mask into <span className="text-comma-green">22 tiles</span>: 4 vertical
          bands of width 128 px, each cut into <span className="mono text-comma-green">3 / 8 / 8 / 3</span> horizontal
          slices. Every tile is encoded as its own arithmetic stream, so each one starts with a
          fresh probability model that converges fast on local statistics. Tiles marked <span className="text-comma-green">⟳</span> get
          a free lossless flip before encoding to align with the codec's UP+LEFT cascade.
        </p>
        <div className="border border-white/10 bg-black p-3 mono text-[10px]">
          <div className="text-white/40 uppercase tracking-widest mb-2">class legend</div>
          <div className="grid grid-cols-1 gap-1">
            {CLASS_LEGEND.map((c) => (
              <div key={c.label} className="flex items-center gap-2">
                <div className="w-3 h-3" style={{ background: c.color }} />
                <span className="text-white/70">{c.label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="grid lg:grid-cols-[1fr_320px] gap-8 items-start" style={{ minHeight: 460 }}>
        {/* Mask + tile overlay (landscape: 512 × 384, scaled to fit) */}
        <div>
          <div className="relative w-full bg-black border border-white/15" style={{ aspectRatio: "512 / 384" }}>
            <img
              src="/writeup_assets/sample_mask_landscape.png"
              alt="dashcam SegNet mask"
              className="absolute inset-0 w-full h-full opacity-90"
              style={{ imageRendering: "pixelated" }}
            />
            {/* Tile overlay: 4 vertical bands across the width */}
            <div className="absolute inset-0 grid" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
              {[0, 1, 2, 3].map((bi) => {
                const splits = W_SPLITS[bi];
                return (
                  <div key={bi} className="grid relative" style={{ gridTemplateRows: `repeat(${splits}, 1fr)`, gap: 0 }}>
                    {TILES.filter(t => t.band === bi).map((t) => {
                      const isActive = t.tileIdx === activeIdx;
                      return (
                        <button
                          key={t.tileIdx}
                          onClick={() => onSelect(t.tileIdx)}
                          className="relative transition-all"
                          style={{
                            border: isActive ? "2px solid #51FF00" : "1px solid rgba(255,255,255,0.32)",
                            background: isActive ? "rgba(81,255,0,0.18)" : "transparent",
                            zIndex: isActive ? 5 : 1,
                          }}
                        >
                          <span
                            className="absolute top-1 left-1 mono text-[9px] px-1 py-0.5"
                            style={{ background: "rgba(0,0,0,0.7)", color: isActive ? "#51FF00" : "#fff" }}
                          >
                            #{t.tileIdx}{t.tx !== "id" && <span className="ml-1 text-comma-green">⟳</span>}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                );
              })}
            </div>
            {/* Band labels under the image */}
          </div>

          <div className="mt-2 grid grid-cols-4 gap-1">
            {BAND_NATURAL_LABEL.map((label, i) => (
              <div key={i} className="mono text-[10px] uppercase tracking-widest text-center" style={{ color: BAND_COLOR[i] }}>
                band {i} · {label}
                <div className="text-[9px] text-white/40 normal-case tracking-normal mt-0.5">{W_SPLITS[i]} slices</div>
              </div>
            ))}
          </div>
        </div>

        {/* Detail panel: fixed height, always visible */}
        <div className="space-y-3">
          <div className="border border-white/10 bg-black p-5" style={{ minHeight: 270 }}>
            <div className="mono text-[10px] uppercase tracking-widest text-comma-green mb-2">tile #{cur.tileIdx}</div>
            <div className="text-2xl text-white font-bold mb-1">{BAND_NATURAL_LABEL[cur.band]}</div>
            <div className="mono text-[12px] text-white/55 mb-5">band {cur.band} · slice {cur.split + 1}/{W_SPLITS[cur.band]}</div>

            <div className="grid grid-cols-2 gap-y-3 mono text-[13px]">
              <div className="text-white/40">codec dims</div>
              <div className="text-white tabular-nums">128 × {Math.round(384 / W_SPLITS[cur.band])} px</div>
              <div className="text-white/40">bytes</div>
              <div className="text-comma-green tabular-nums">{cur.bytes.toLocaleString()}</div>
              <div className="text-white/40">transform</div>
              <div className={cur.tx === "id" ? "text-white/55" : "text-comma-green"}>{cur.tx}</div>
            </div>

            <div className="mt-4 pt-3 border-t border-white/10 mono text-[11px] text-white/65 leading-relaxed" style={{ minHeight: 36 }}>
              {TRANSFORM_DESC[cur.tx]}
            </div>
          </div>

          <div className="border border-white/10 bg-black p-4 grid grid-cols-3 gap-3 mono text-[11px]">
            <div>
              <div className="text-white/40 text-[9px] uppercase tracking-widest">total</div>
              <div className="text-comma-green text-[20px] tabular-nums leading-tight">{(totalBytes / 1024).toFixed(1)}K</div>
            </div>
            <div>
              <div className="text-white/40 text-[9px] uppercase tracking-widest">tiles</div>
              <div className="text-white text-[20px] tabular-nums leading-tight">22</div>
            </div>
            <div>
              <div className="text-white/40 text-[9px] uppercase tracking-widest">flipped</div>
              <div className="text-white text-[20px] tabular-nums leading-tight">{txCount}/22</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
