/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        comma: {
          green: "#51FF00",
          greenDark: "#51B124",
          black: "#000000",
          surface: "#121212",
          card: "#1A1A1A",
          cardLight: "#F8F9FA",
          textMuted: "#9E9E9E",
          textSubtle: "#525252",
          textDark: "#333333",
          error: "#FF4133",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["'JetBrains Mono'", "ui-monospace", "monospace"],
        display: ["'Monument Extended'", "Inter", "sans-serif"],
      },
      letterSpacing: {
        comma: "-0.04em",
        commaTight: "-0.02em",
      },
      borderRadius: {
        none: "0px",
      },
      keyframes: {
        cellPulse: {
          from: { opacity: "var(--opacity-min, 0.05)" },
          to: { opacity: "var(--opacity-max, 0.5)" },
        },
        scan: {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(100%)" },
        },
        flicker: {
          "0%,100%": { opacity: "1" },
          "5%": { opacity: "0.85" },
          "10%": { opacity: "1" },
          "15%": { opacity: "0.95" },
        },
        blink: {
          "0%,49%": { opacity: "1" },
          "50%,100%": { opacity: "0" },
        },
      },
      animation: {
        cellPulse: "cellPulse 5s infinite alternate",
        scan: "scan 8s linear infinite",
        flicker: "flicker 6s infinite",
        blink: "blink 1s steps(1) infinite",
      },
    },
  },
  plugins: [],
};
