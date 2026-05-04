// Pixel-block typography. Letters rendered as a 5xN grid of cells per glyph.
// Inspired by the PROMPTING IS ALL YOU NEED hero, but rendered as CSS grids
// (no canvas, no animation loop) so it composes cleanly into a static page.

const PIXEL_MAP: Record<string, number[][]> = {
  // 5 rows tall, variable width
  P: [[1,1,1,1],[1,0,0,1],[1,1,1,1],[1,0,0,0],[1,0,0,0]],
  R: [[1,1,1,1],[1,0,0,1],[1,1,1,1],[1,0,1,0],[1,0,0,1]],
  O: [[1,1,1,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,1,1,1]],
  M: [[1,0,0,0,1],[1,1,0,1,1],[1,0,1,0,1],[1,0,0,0,1],[1,0,0,0,1]],
  T: [[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]],
  I: [[1,1,1],[0,1,0],[0,1,0],[0,1,0],[1,1,1]],
  N: [[1,0,0,0,1],[1,1,0,0,1],[1,0,1,0,1],[1,0,0,1,1],[1,0,0,0,1]],
  G: [[1,1,1,1,1],[1,0,0,0,0],[1,0,1,1,1],[1,0,0,0,1],[1,1,1,1,1]],
  S: [[1,1,1,1],[1,0,0,0],[1,1,1,1],[0,0,0,1],[1,1,1,1]],
  A: [[0,1,1,0],[1,0,0,1],[1,1,1,1],[1,0,0,1],[1,0,0,1]],
  L: [[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,1,1,1]],
  Y: [[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]],
  U: [[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,1,1,1]],
  D: [[1,1,1,0],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,1,1,0]],
  E: [[1,1,1,1],[1,0,0,0],[1,1,1,1],[1,0,0,0],[1,1,1,1]],
  C: [[1,1,1,1],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,1,1,1]],
  V: [[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0]],
  B: [[1,1,1,0],[1,0,0,1],[1,1,1,0],[1,0,0,1],[1,1,1,0]],
  H: [[1,0,0,1],[1,0,0,1],[1,1,1,1],[1,0,0,1],[1,0,0,1]],
  F: [[1,1,1,1],[1,0,0,0],[1,1,1,1],[1,0,0,0],[1,0,0,0]],
  K: [[1,0,0,1],[1,0,1,0],[1,1,0,0],[1,0,1,0],[1,0,0,1]],
  W: [[1,0,0,0,1],[1,0,0,0,1],[1,0,1,0,1],[1,1,0,1,1],[1,0,0,0,1]],
  X: [[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0],[0,1,0,1,0],[1,0,0,0,1]],
  Z: [[1,1,1,1,1],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],[1,1,1,1,1]],
  Q: [[1,1,1,1],[1,0,0,1],[1,0,0,1],[1,0,1,1],[1,1,1,1]],
  J: [[0,0,0,1],[0,0,0,1],[0,0,0,1],[1,0,0,1],[0,1,1,0]],
  // Digits
  "0": [[1,1,1,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,1,1,1]],
  "1": [[0,0,1,0],[0,1,1,0],[0,0,1,0],[0,0,1,0],[0,1,1,1]],
  "2": [[1,1,1,1],[0,0,0,1],[0,1,1,0],[1,0,0,0],[1,1,1,1]],
  "3": [[1,1,1,1],[0,0,0,1],[0,1,1,1],[0,0,0,1],[1,1,1,1]],
  "4": [[1,0,0,1],[1,0,0,1],[1,1,1,1],[0,0,0,1],[0,0,0,1]],
  "5": [[1,1,1,1],[1,0,0,0],[1,1,1,1],[0,0,0,1],[1,1,1,1]],
  "6": [[1,1,1,1],[1,0,0,0],[1,1,1,1],[1,0,0,1],[1,1,1,1]],
  "7": [[1,1,1,1],[0,0,0,1],[0,0,1,0],[0,1,0,0],[0,1,0,0]],
  "8": [[1,1,1,1],[1,0,0,1],[1,1,1,1],[1,0,0,1],[1,1,1,1]],
  "9": [[1,1,1,1],[1,0,0,1],[1,1,1,1],[0,0,0,1],[1,1,1,1]],
  ".": [[0],[0],[0],[0],[1]],
  "-": [[0,0,0],[0,0,0],[1,1,1],[0,0,0],[0,0,0]],
};

interface Props {
  text: string;
  pixelSize?: number;        // px per cell
  letterSpacing?: number;    // pixel cols between letters
  wordSpacing?: number;      // pixel cols between words
  color?: string;
  className?: string;
  onColor?: string;          // separate color for "on" cells (defaults to color)
  showOff?: boolean;         // render the "off" cells faintly (grid feel)
  offOpacity?: number;
  reveal?: boolean;          // animate pixels popping in on mount
  revealDuration?: number;   // total animation duration in ms (default 1100)
  revealStartDelay?: number; // delay before reveal begins (ms)
}

export default function PixelText({
  text,
  pixelSize = 8,
  letterSpacing = 1,
  wordSpacing = 3,
  color = "#fff",
  onColor,
  className = "",
  showOff = false,
  offOpacity = 0.06,
  reveal = false,
  revealDuration = 1100,
  revealStartDelay = 0,
}: Props) {
  const upper = text.toUpperCase();
  const words = upper.split(" ");
  const fg = onColor ?? color;

  // Compute total width to center
  let totalCols = 0;
  words.forEach((word, wi) => {
    word.split("").forEach((ch, i) => {
      const map = PIXEL_MAP[ch];
      if (!map) return;
      totalCols += map[0].length + (i < word.length - 1 ? letterSpacing : 0);
    });
    if (wi < words.length - 1) totalCols += wordSpacing;
  });

  const ROWS = 5;
  const grid: ({ on: boolean; word: number; idxInWord: number } | null)[][] =
    Array.from({ length: ROWS }, () => Array(totalCols).fill(null));

  let col = 0;
  words.forEach((word, wi) => {
    word.split("").forEach((ch, ci) => {
      const map = PIXEL_MAP[ch];
      if (!map) return;
      for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < map[r].length; c++) {
          grid[r][col + c] = { on: !!map[r][c], word: wi, idxInWord: ci };
        }
      }
      col += map[0].length;
      if (ci < word.length - 1) col += letterSpacing;
    });
    if (wi < words.length - 1) col += wordSpacing;
  });

  return (
    <div
      className={className}
      style={{
        display: "grid",
        gridTemplateColumns: `repeat(${totalCols}, ${pixelSize}px)`,
        gridTemplateRows: `repeat(${ROWS}, ${pixelSize}px)`,
        gap: 0,
        color: fg,
        lineHeight: 0,
      }}
      aria-label={text}
    >
      {grid.flat().map((cell, i) => {
        if (!cell) return <div key={i} />;
        const baseOpacity = cell.on ? 1 : (showOff ? offOpacity : 0);
        const bg = cell.on ? fg : (showOff ? color : "transparent");
        if (!reveal) {
          return (
            <div
              key={i}
              style={{
                width: pixelSize,
                height: pixelSize,
                backgroundColor: bg,
                opacity: baseOpacity,
              }}
            />
          );
        }
        // Random per-cell delay so pixels pop in shuffled order
        const cellDelay = Math.random() * revealDuration * 0.85;
        return (
          <div
            key={i}
            style={{
              width: pixelSize,
              height: pixelSize,
              backgroundColor: bg,
              opacity: 0,
              animation: `pixelReveal ${revealDuration * 0.18}ms ease-out forwards`,
              animationDelay: `${revealStartDelay + cellDelay}ms`,
              ["--reveal-final-opacity" as string]: String(baseOpacity),
            }}
          />
        );
      })}
    </div>
  );
}
