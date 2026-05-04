---
name: Git History Visualization Tools
description: Gource for video, D3.js custom timeline for scores, git-of-theseus for code evolution
type: reference
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Highest impact (implement first)

### 1. Custom D3.js score timeline (Cloudflare site)
- Map commits to score evolution with breakthrough/failure annotations
- Extend existing timeline.html
- Color: green=improvement, red=regression, yellow=infrastructure

### 2. Gource animated video
```bash
brew install gource ffmpeg
gource -1920x1080 --seconds-per-day 3 --auto-skip-seconds 1 \
  --title "PACT: Video Compression Challenge" \
  --key --highlight-all-users --hide mouse,progress \
  --file-idle-time 0 --max-files 0 \
  --background-colour 111111 --font-size 18 \
  -o - | ffmpeg -y -r 60 -f image2pipe -vcodec ppm -i - \
  -vcodec libx264 -preset medium -pix_fmt yuv420p -crf 18 \
  gource_pact.mp4
```

## Supporting artifacts

### 3. git-of-theseus — code survival analysis
```bash
uv pip install git-of-theseus
git-of-theseus-analyze .
git-of-theseus-stack-plot cohorts.json
```
Shows which code survived vs was thrown away. "Ship of Theseus" for the paper.

### 4. git-story — Manim animated commit DAG
```bash
uv pip install git-story
git-story --commits 50 --title "PACT Research Timeline"
```

### 5. Hercules — file coupling analysis
Reveals which files always change together (the research frontier).

## Academic precedent
- "Git as open electronic laboratory notebook" (Digital Discovery, 2023)
- "Git can facilitate reproducibility in science" (Nature, 500+ citations)
- Our CLAUDE.md: "git history IS our research timeline" — publishable methodology
