/* FLOW (Tab 3) — the full n600 drive as a scrub/play VIDEO (WebGPU, canvas2d fallback).
 *
 * Inlined into the dashboard page by tools/dashboard_server.py::_flow_client_js (CSP-strict:
 * self-contained, no external <script src>, no CDN). The server renders the WHOLE 600-pair field
 * sequence from the BEST checkpoint in a detached governed subprocess and serves it ONCE via
 * GET /api/flow_sequence; a lightweight {type:"flow_ready"} WS ping (window.__flow.ready) tells us
 * a new sequence exists. We fetch it once, decode every frame, then scrub/play LOCALLY — zero
 * network + zero re-render per scrub.
 *
 * sequence = { ok, epoch, n, w, h, fps, classes:[{i,label,hex}], mean_dseg, hardest:[...],
 *              per_pair_dseg:[...], frames:[{ i, dseg, mode, render_b64(jpeg),
 *              fields_b64(RGB png: R=witness partition, G=SegNet argmax, B=GT argmax),
 *              frag_b64(L png: margin fragility 0..255) }] }
 *
 * Controls: FRAME SLIDER = the segment timeline (0 -> n-1); PLAY animates it at ~fps (loops);
 * LAYER = witness render / witness partition / SegNet argmax / disagreement-vs-GT / margin heat;
 * class isolation; margin-fragility threshold. The PARTITION + SegNet + disagreement layers use the
 * CANONICAL comma10k / openpilot segmentation palette (from sequence.classes) — the colors Yousfi +
 * comma read instantly; only the margin-heat layer uses an inferno colormap.
 *
 * AUTHORITY: [macOS-CPU advisory · NON-PROMOTABLE] imagery. A viz moves no pointer (0.19110).
 */
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };

  var DEFAULT_PAL = [[64, 32, 32], [255, 0, 0], [128, 128, 96], [0, 255, 102], [204, 0, 255]];
  var CLASS_LABELS = ["0 Road", "1 Lane", "2 Undrivable", "3 Movable", "4 MyCar"];

  // ---- sequence + decode state ----
  var SEQ = null, PAL = DEFAULT_PAL.slice(), loadedEpoch = null, fetching = false;
  var cache = [];            // cache[i] = decoded obj {bmp,wit,seg,gt,frag,w,h} | in-flight Promise
  var _pf = 0;              // background prefetch cursor

  // ---- control state ----
  var curFrame = 0, mode = 2, iso = -1, thr = 0.55;
  var LAYER_LABELS = ["witness render", "witness partition", "SegNet argmax", "disagreement vs GT", "margin heat"];
  var playing = false, rafId = null, lastT = 0, playAccum = 0;

  var inited = false, wired = false, gpuOk = false, gpuLost = false, CANVAS = null;
  var fcanvas = null, fctx = null;    // offscreen for field getImageData

  // ---------- small helpers ----------
  function b64ToBlob(b64, type) {
    var s = atob(b64), n = s.length, u = new Uint8Array(n);
    for (var i = 0; i < n; i++) u[i] = s.charCodeAt(i);
    return new Blob([u], { type: type });
  }
  function hexToRgb(h) {
    h = String(h || "").replace("#", "");
    if (h.length !== 6) return [128, 128, 128];
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  }
  function inferno(x) {
    var t = Math.min(1, Math.max(0, x));
    var c0 = [0.00021894, 0.00165100, -0.01948090], c1 = [0.10651342, 0.56395644, 3.93271239],
      c2 = [11.60249308, -3.97285397, -15.94239411], c3 = [-41.70399613, 17.43639888, 44.35414520],
      c4 = [77.16293570, -33.40235894, -81.80730926], c5 = [-71.31942824, 32.62606426, 73.20951986],
      c6 = [25.13112622, -12.24266895, -23.07032500], o = [0, 0, 0];
    for (var k = 0; k < 3; k++) {
      var v = c0[k] + t * (c1[k] + t * (c2[k] + t * (c3[k] + t * (c4[k] + t * (c5[k] + t * c6[k])))));
      o[k] = Math.min(1, Math.max(0, v));
    }
    return o;
  }
  function fmtSig(v, n) {
    if (v == null || !isFinite(v)) return "—";
    if (v === 0) return "0";
    var neg = v < 0, a = Math.abs(v), d = n - 1 - Math.floor(Math.log10(a));
    if (d < 0) d = 0; if (d > 20) d = 20;
    var s = a.toFixed(d);
    if (s.indexOf(".") >= 0) s = s.replace(/0+$/, "").replace(/\.$/, "");
    return (neg ? "-" : "") + s;
  }
  function flowVisible() { var s = $("tab-flow"); return !!(s && !s.classList.contains("hide")); }
  function classLabel(c) {
    if (SEQ && SEQ.classes && SEQ.classes[c]) return SEQ.classes[c].label;
    return CLASS_LABELS[c] || ("class " + c);
  }
  function palRGB(c) { var p = PAL[c] || [128, 128, 128]; return [p[0], p[1], p[2]]; }

  // ---------- fetch + ingest the n600 sequence ----------
  function statusMsg(meta) {
    if (!meta) return "waiting for the first n600 sequence…";
    if (meta.status === "rendering") {
      var prog = (typeof meta.pair === "number" && typeof meta.n === "number" && meta.n > 0)
        ? " — pair " + meta.pair + "/" + meta.n +
          (typeof meta.secs === "number" && meta.pair > 0
            ? " (~" + Math.max(1, Math.round((meta.n - meta.pair) * (meta.secs / meta.pair) / 60)) + " min left)"
            : "")
        : " (~14 min governed pass)";
      return "rendering the n600 video from the best checkpoint…" + prog + "; auto-appears when done";
    }
    if (meta.status === "error") return "sequence render error — " + (meta.err || "see server log");
    return "the n600 video renders on the next best checkpoint…";
  }
  function onFlowReady(meta) {
    window.__flow = window.__flow || {}; window.__flow.ready = meta || null;
    if (meta && meta.ok) {
      if (SEQ && loadedEpoch === meta.epoch) { updateStatus(); return; }  // already have this one
      fetchSequence(meta);
    } else if (!SEQ) {
      if (inited) showMsg(statusMsg(meta));
      updateStatus();
    }
  }
  function fetchSequence(meta) {
    if (fetching) return;
    fetching = true;
    if (inited) showMsg("fetching the n600 video sequence… (one-time)");
    fetch("/api/flow_sequence" + (location.search || ""), { credentials: "same-origin" })
      .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
      .then(function (seq) {
        if (!seq || !seq.ok || !Array.isArray(seq.frames) || !seq.frames.length) throw new Error("empty sequence");
        onSequence(seq);
      })
      .catch(function (e) { if (!SEQ && inited) showMsg("sequence fetch failed: " + (e && e.message || e)); })
      .then(function () { fetching = false; });
  }
  function onSequence(seq) {
    SEQ = seq; loadedEpoch = seq.epoch;
    cache = new Array(seq.n); _pf = 0;
    PAL = (seq.classes && seq.classes.length) ? seq.classes.map(function (c) { return hexToRgb(c.hex); }) : DEFAULT_PAL.slice();
    buildLegend(seq.classes || []);
    var fr = $("flowframe");
    if (fr) { fr.max = String(Math.max(0, seq.n - 1)); if (curFrame >= seq.n) curFrame = 0; fr.value = String(curFrame); }
    setFrameUI(curFrame);
    prefetchDecode();
    renderNow();
  }

  // ---------- decode ----------
  function ensureFieldCanvas(w, h) {
    if (!fcanvas) { fcanvas = document.createElement("canvas"); fctx = fcanvas.getContext("2d", { willReadFrequently: true }); }
    if (fcanvas.width !== w || fcanvas.height !== h) { fcanvas.width = w; fcanvas.height = h; }
  }
  function decodeFrame(i) {
    if (!SEQ) return Promise.reject(new Error("no sequence"));
    var c = cache[i];
    if (c) { return c.then ? c : Promise.resolve(c); }
    var f = SEQ.frames[i], w = SEQ.w, h = SEQ.h;
    // OPAQUE images (RGB class fields + L fragility) -> no alpha premultiply corrupting class indices.
    var p = Promise.all([
      createImageBitmap(b64ToBlob(f.render_b64, "image/jpeg")),
      createImageBitmap(b64ToBlob(f.fields_b64, "image/png")),
      createImageBitmap(b64ToBlob(f.frag_b64, "image/png"))
    ]).then(function (r) {
      var bmp = r[0], fb = r[1], gb = r[2];
      ensureFieldCanvas(w, h);
      fctx.clearRect(0, 0, w, h); fctx.drawImage(fb, 0, 0);
      var fd = fctx.getImageData(0, 0, w, h).data;
      var wit = new Uint8Array(w * h), seg = new Uint8Array(w * h), gt = new Uint8Array(w * h);
      for (var k = 0, o = 0; k < w * h; k++, o += 4) { wit[k] = fd[o]; seg[k] = fd[o + 1]; gt[k] = fd[o + 2]; }
      fctx.clearRect(0, 0, w, h); fctx.drawImage(gb, 0, 0);
      var gd = fctx.getImageData(0, 0, w, h).data;
      var frag = new Uint8Array(w * h);
      for (var k2 = 0, o2 = 0; k2 < w * h; k2++, o2 += 4) frag[k2] = gd[o2];
      if (fb.close) fb.close(); if (gb.close) gb.close();
      var obj = { bmp: bmp, wit: wit, seg: seg, gt: gt, frag: frag, w: w, h: h };
      cache[i] = obj; return obj;
    });
    cache[i] = p; return p;
  }
  function prefetchDecode() {
    _pf = 0;
    (function pump() {
      if (!SEQ) return;
      var budget = 6;
      while (budget-- > 0 && _pf < SEQ.n) { var i = _pf++; if (!cache[i]) decodeFrame(i).catch(function () {}); }
      if (_pf < SEQ.n) setTimeout(pump, 24);
    })();
  }

  // ---------- canvas sizing (field aspect, dpr-crisp) ----------
  function sizeCanvas(W, H) {
    var c = CANVAS, cssW = c.clientWidth || 512, dpr = Math.min(window.devicePixelRatio || 1, 2);
    var pxW = Math.max(1, Math.round(cssW * dpr)), pxH = Math.max(1, Math.round(pxW * H / W));
    if (c.width !== pxW || c.height !== pxH) { c.width = pxW; c.height = pxH; }
  }

  // ---------- WebGPU path ----------
  var dev = null, gctx = null, fmt = null, pipe = null, uni = null, bgl = null, samp = null;
  var texR = null, texW = null, texS = null, texG = null, texF = null, tW = 0, tH = 0;

  var WGSL = [
    "struct U { head:vec4f, ctl:vec4f, pal:array<vec4f,5>, };",
    "@group(0) @binding(0) var<uniform> u : U;",
    "@group(0) @binding(1) var texR : texture_2d<f32>;",   // witness render (rgba8unorm)
    "@group(0) @binding(2) var texW : texture_2d<u32>;",   // witness partition argmax
    "@group(0) @binding(3) var texS : texture_2d<u32>;",   // SegNet argmax
    "@group(0) @binding(4) var texG : texture_2d<u32>;",   // GT argmax
    "@group(0) @binding(5) var texF : texture_2d<f32>;",   // margin fragility (r8unorm)
    "@group(0) @binding(6) var samp : sampler;",
    "struct VSOut { @builtin(position) pos:vec4f, @location(0) uv:vec2f, };",
    "@vertex fn vs(@builtin(vertex_index) i:u32) -> VSOut {",
    "  var p = array<vec2f,3>(vec2f(-1.0,-1.0), vec2f(3.0,-1.0), vec2f(-1.0,3.0));",
    "  var o:VSOut; o.pos = vec4f(p[i],0.0,1.0);",
    "  var uv = 0.5*(p[i]+vec2f(1.0,1.0)); uv.y = 1.0-uv.y; o.uv = uv; return o;",
    "}",
    "fn pal(c:i32) -> vec3f {",
    "  if (c==0){return u.pal[0].rgb;} if (c==1){return u.pal[1].rgb;} if (c==2){return u.pal[2].rgb;}",
    "  if (c==3){return u.pal[3].rgb;} if (c==4){return u.pal[4].rgb;} return vec3f(0.5,0.5,0.5);",
    "}",
    "fn inferno(x:f32) -> vec3f {",
    "  let t = clamp(x,0.0,1.0);",
    "  let c0 = vec3f(0.00021894,0.00165100,-0.01948090);",
    "  let c1 = vec3f(0.10651342,0.56395644,3.93271239);",
    "  let c2 = vec3f(11.60249308,-3.97285397,-15.94239411);",
    "  let c3 = vec3f(-41.70399613,17.43639888,44.35414520);",
    "  let c4 = vec3f(77.16293570,-33.40235894,-81.80730926);",
    "  let c5 = vec3f(-71.31942824,32.62606426,73.20951986);",
    "  let c6 = vec3f(25.13112622,-12.24266895,-23.07032500);",
    "  return clamp(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))), vec3f(0.0), vec3f(1.0));",
    "}",
    "@fragment fn fs(inp:VSOut) -> @location(0) vec4f {",
    "  let dims = u.head.xy; let thr = u.head.z; let md = i32(round(u.head.w));",
    "  let isoc = i32(round(u.ctl.x));",
    "  let ic = clamp(vec2i(floor(inp.uv*dims)), vec2i(0,0), vec2i(dims)-vec2i(1,1));",
    "  let wit = i32(textureLoad(texW, ic, 0).r);",
    "  let seg = i32(textureLoad(texS, ic, 0).r);",
    "  let gt  = i32(textureLoad(texG, ic, 0).r);",
    "  let frag = textureLoad(texF, ic, 0).r;",
    "  var col = vec3f(0.0,0.0,0.0); var cls = seg;",
    "  if (md == 0) { col = textureSampleLevel(texR, samp, inp.uv, 0.0).rgb; return vec4f(col,1.0); }",
    "  else if (md == 1) { col = pal(wit); cls = wit; }",
    "  else if (md == 2) { col = pal(seg); cls = seg; }",
    "  else if (md == 3) { let dis = seg != gt; col = select(pal(seg)*0.32, vec3f(1.0,0.16,0.16), dis); cls = seg; }",
    "  else { let g = select(frag*0.22, frag, frag >= thr); return vec4f(inferno(g), 1.0); }",
    "  if ((md == 1 || md == 2) && frag >= thr) {",
    "    let k = 0.55*(frag-thr)/max(1e-3, 1.0-thr);",
    "    col = mix(col, vec3f(0.98,0.95,0.6), clamp(k,0.0,0.9));",
    "  }",
    "  if (isoc >= 0 && cls != isoc) { col = col * 0.12; }",
    "  return vec4f(col, 1.0);",
    "}"
  ].join("\n");

  function initGPU() {
    return Promise.resolve().then(function () {
      CANVAS = $("flowcanvas");
      if (!navigator.gpu) throw new Error("navigator.gpu absent");
      return navigator.gpu.requestAdapter();
    }).then(function (adapter) {
      if (!adapter) throw new Error("no WebGPU adapter");
      return adapter.requestDevice();
    }).then(function (device) {
      dev = device;
      if (dev.lost && dev.lost.then) dev.lost.then(function () { gpuLost = true; });
      gctx = CANVAS.getContext("webgpu");
      if (!gctx) throw new Error("no webgpu canvas context");
      fmt = navigator.gpu.getPreferredCanvasFormat();
      gctx.configure({ device: dev, format: fmt, alphaMode: "opaque" });
      var mod = dev.createShaderModule({ code: WGSL });
      bgl = dev.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "uint" } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "uint" } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "uint" } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } }
      ] });
      var pl = dev.createPipelineLayout({ bindGroupLayouts: [bgl] });
      pipe = dev.createRenderPipeline({
        layout: pl,
        vertex: { module: mod, entryPoint: "vs" },
        fragment: { module: mod, entryPoint: "fs", targets: [{ format: fmt }] },
        primitive: { topology: "triangle-list" }
      });
      uni = dev.createBuffer({ size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      samp = dev.createSampler({ magFilter: "linear", minFilter: "linear" });
    });
  }
  function ensureTex(W, H) {
    if (tW === W && tH === H && texR) return;
    [texR, texW, texS, texG, texF].forEach(function (t) { if (t && t.destroy) t.destroy(); });
    var UB = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST;
    texR = dev.createTexture({ size: [W, H], format: "rgba8unorm", usage: UB | GPUTextureUsage.RENDER_ATTACHMENT });
    texW = dev.createTexture({ size: [W, H], format: "r8uint", usage: UB });
    texS = dev.createTexture({ size: [W, H], format: "r8uint", usage: UB });
    texG = dev.createTexture({ size: [W, H], format: "r8uint", usage: UB });
    texF = dev.createTexture({ size: [W, H], format: "r8unorm", usage: UB });
    tW = W; tH = H;
  }
  function drawGPU(obj) {
    var W = obj.w, H = obj.h;
    ensureTex(W, H);
    dev.queue.copyExternalImageToTexture({ source: obj.bmp }, { texture: texR }, { width: W, height: H });
    dev.queue.writeTexture({ texture: texW }, obj.wit, { bytesPerRow: W, rowsPerImage: H }, { width: W, height: H });
    dev.queue.writeTexture({ texture: texS }, obj.seg, { bytesPerRow: W, rowsPerImage: H }, { width: W, height: H });
    dev.queue.writeTexture({ texture: texG }, obj.gt, { bytesPerRow: W, rowsPerImage: H }, { width: W, height: H });
    dev.queue.writeTexture({ texture: texF }, obj.frag, { bytesPerRow: W, rowsPerImage: H }, { width: W, height: H });
    var buf = new Float32Array(28);
    buf[0] = W; buf[1] = H; buf[2] = thr; buf[3] = mode;
    buf[4] = iso; buf[5] = 0; buf[6] = 0; buf[7] = 0;
    for (var c = 0; c < 5; c++) { var p = PAL[c] || [128, 128, 128]; var b = 8 + c * 4; buf[b] = p[0] / 255; buf[b + 1] = p[1] / 255; buf[b + 2] = p[2] / 255; buf[b + 3] = 1; }
    dev.queue.writeBuffer(uni, 0, buf);
    sizeCanvas(W, H);
    var bg = dev.createBindGroup({ layout: bgl, entries: [
      { binding: 0, resource: { buffer: uni } },
      { binding: 1, resource: texR.createView() },
      { binding: 2, resource: texW.createView() },
      { binding: 3, resource: texS.createView() },
      { binding: 4, resource: texG.createView() },
      { binding: 5, resource: texF.createView() },
      { binding: 6, resource: samp }
    ] });
    var enc = dev.createCommandEncoder();
    var view = gctx.getCurrentTexture().createView();
    var pass = enc.beginRenderPass({ colorAttachments: [{ view: view,
      clearValue: { r: 0.05, g: 0.06, b: 0.08, a: 1 }, loadOp: "clear", storeOp: "store" }] });
    pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.draw(3); pass.end();
    dev.queue.submit([enc.finish()]);
  }

  // ---------- canvas2d fallback ----------
  var ctx2d = null, off2d = null, off2dCtx = null;
  function ensure2dOff(W, H) {
    if (!off2d) { off2d = document.createElement("canvas"); off2dCtx = off2d.getContext("2d"); }
    if (off2d.width !== W || off2d.height !== H) { off2d.width = W; off2d.height = H; }
  }
  function colorPixel(obj, k) {
    var wit = obj.wit[k], seg = obj.seg[k], gt = obj.gt[k], frag = obj.frag[k] / 255, rgb, cls;
    if (mode === 1) { rgb = palRGB(wit); cls = wit; }
    else if (mode === 2) { rgb = palRGB(seg); cls = seg; }
    else if (mode === 3) { rgb = (seg !== gt) ? [255, 41, 41] : palRGB(seg).map(function (v) { return v * 0.32; }); cls = seg; }
    else { var g = (frag >= thr) ? frag : frag * 0.22; var c = inferno(g); return [c[0] * 255, c[1] * 255, c[2] * 255]; }
    if ((mode === 1 || mode === 2) && frag >= thr) {
      var kk = Math.min(0.9, 0.55 * (frag - thr) / Math.max(1e-3, 1 - thr));
      rgb = [rgb[0] * (1 - kk) + 250 * kk, rgb[1] * (1 - kk) + 242 * kk, rgb[2] * (1 - kk) + 153 * kk];
    }
    if (iso >= 0 && cls !== iso) rgb = [rgb[0] * 0.12, rgb[1] * 0.12, rgb[2] * 0.12];
    return rgb;
  }
  function draw2d(obj) {
    var W = obj.w, H = obj.h;
    sizeCanvas(W, H);
    if (mode === 0) {   // witness render: draw the photo bitmap (smoothed)
      ctx2d.imageSmoothingEnabled = true;
      ctx2d.clearRect(0, 0, CANVAS.width, CANVAS.height);
      ctx2d.drawImage(obj.bmp, 0, 0, W, H, 0, 0, CANVAS.width, CANVAS.height);
      return;
    }
    ensure2dOff(W, H);
    var img = off2dCtx.createImageData(W, H), d = img.data;
    for (var k = 0, o = 0; k < W * H; k++, o += 4) {
      var rgb = colorPixel(obj, k);
      d[o] = rgb[0] | 0; d[o + 1] = rgb[1] | 0; d[o + 2] = rgb[2] | 0; d[o + 3] = 255;
    }
    off2dCtx.putImageData(img, 0, 0);
    ctx2d.imageSmoothingEnabled = (mode === 4);   // heat smooth; class fields crisp
    ctx2d.clearRect(0, 0, CANVAS.width, CANVAS.height);
    ctx2d.drawImage(off2d, 0, 0, W, H, 0, 0, CANVAS.width, CANVAS.height);
  }
  function initFallback() {
    var old = $("flowcanvas");
    var n = old.cloneNode(false);       // a canvas holds one context type; clone for a clean 2d ctx
    old.parentNode.replaceChild(n, old);
    CANVAS = n; ctx2d = CANVAS.getContext("2d");
  }

  // ---------- UI ----------
  function setBadge() {
    var b = $("flowbadge"); if (!b) return;
    if (gpuOk && !gpuLost) { b.className = "flowbadge gpu"; b.textContent = "WebGPU"; }
    else { b.className = "flowbadge cpu"; b.textContent = "canvas2d"; }
  }
  function showMsg(t) { var m = $("flowmsg"); if (m) { m.classList.remove("hide"); m.textContent = t; } }
  function hideMsg() { var m = $("flowmsg"); if (m) m.classList.add("hide"); }
  function buildLegend(classes) {
    var lg = $("flowlegend"); if (!lg) return;
    if (lg.dataset.n === String(classes.length) && classes.length) { syncLegendActive(); return; }
    lg.innerHTML = "";
    var all = document.createElement("span"); all.className = "lc" + (iso < 0 ? " on" : "");
    all.dataset.cls = "-1"; all.innerHTML = "<span class='dot' style='background:transparent;border-style:dashed'></span>all";
    lg.appendChild(all);
    classes.forEach(function (c) {
      var el = document.createElement("span"); el.className = "lc" + (iso === c.i ? " on" : "");
      el.dataset.cls = String(c.i);
      el.innerHTML = "<span class='dot' style='background:" + c.hex + "'></span>" + c.label;
      lg.appendChild(el);
    });
    lg.dataset.n = String(classes.length);
    Array.prototype.forEach.call(lg.querySelectorAll(".lc"), function (el) {
      el.addEventListener("click", function () { setIso(parseInt(el.dataset.cls, 10)); });
    });
  }
  function syncLegendActive() {
    var lg = $("flowlegend"); if (!lg) return;
    Array.prototype.forEach.call(lg.querySelectorAll(".lc"), function (el) {
      el.classList.toggle("on", parseInt(el.dataset.cls, 10) === iso);
    });
  }
  function setIso(v) {
    iso = v; var s = $("flowiso"); if (s) s.value = String(v);
    var lab = $("flowiso_v"); if (lab) lab.textContent = (v < 0 ? "all classes" : classLabel(v));
    syncLegendActive(); renderNow();
  }
  function setFrameUI(i) {
    var fr = $("flowframe"); if (fr && fr.value !== String(i)) fr.value = String(i);
    var lab = $("flowframe_v");
    if (lab) { var n = SEQ ? SEQ.n : 600; lab.textContent = i + " / " + (n - 1); }
  }
  function frameDseg(i) {
    if (SEQ && SEQ.per_pair_dseg && SEQ.per_pair_dseg[i] != null) return SEQ.per_pair_dseg[i];
    if (cache[i] && !cache[i].then && SEQ && SEQ.frames[i]) return SEQ.frames[i].dseg;
    return (SEQ && SEQ.frames[i]) ? SEQ.frames[i].dseg : null;
  }
  function updateStatus(obj) {
    var el = $("flowstatus"); if (!el) return;
    if (!SEQ) {
      var meta = (window.__flow || {}).ready;
      el.textContent = meta && meta.status ? meta.status : "no sequence yet";
      return;
    }
    var i = curFrame | 0, ds = frameDseg(i), path = (gpuOk && !gpuLost) ? "WebGPU" : "canvas2d";
    var md = SEQ.frames[i] ? SEQ.frames[i].mode : "";
    el.textContent = "frame " + i + "/" + (SEQ.n - 1) + " · d_seg " + fmtSig(ds, 5) +
      (md ? " · " + md : "") + " · ep " + (SEQ.epoch != null ? SEQ.epoch : "?") +
      " · " + SEQ.h + "×" + SEQ.w + " · " + path;
  }

  function renderNow() {
    if (!SEQ) { updateStatus(); return; }
    setFrameUI(curFrame);
    if (!flowVisible() || !CANVAS || CANVAS.clientWidth === 0) { updateStatus(); return; }
    var i = Math.max(0, Math.min(SEQ.n - 1, curFrame | 0));
    decodeFrame(i).then(function (obj) {
      if ((curFrame | 0) !== i && !playing) { /* moved during decode (scrub) */ }
      try {
        if (gpuOk && !gpuLost) drawGPU(obj); else draw2d(obj);
        hideMsg();
      } catch (e) {
        if (gpuOk) { gpuOk = false; try { initFallback(); } catch (e2) {} setBadge(); }
        try { draw2d(obj); hideMsg(); } catch (e3) { showMsg("frame render error: " + (e3 && e3.message || e3)); }
      }
      updateStatus(obj);
    }).catch(function (e) { showMsg("decode error: " + (e && e.message || e)); });
  }

  // ---------- play loop (segment timeline, ~fps, loops) ----------
  function playStep(t) {
    if (!playing) { rafId = null; return; }
    if (!lastT) lastT = t;
    playAccum += (t - lastT); lastT = t;
    var fps = (SEQ && SEQ.fps) || 12, interval = 1000 / fps, n = SEQ ? SEQ.n : 1;
    if (playAccum >= interval && n > 1) {
      var steps = Math.floor(playAccum / interval); playAccum -= steps * interval;
      curFrame = (curFrame + steps) % n;
      renderNow();
    }
    rafId = requestAnimationFrame(playStep);
  }
  function togglePlay() {
    if (!SEQ || SEQ.n < 2) return;
    playing = !playing;
    var b = $("flowplay");
    if (b) { b.classList.toggle("on", playing); b.setAttribute("aria-pressed", String(playing)); b.innerHTML = playing ? "▍▍ pause" : "▶ play"; }
    if (playing) { lastT = 0; playAccum = 0; if (!rafId) rafId = requestAnimationFrame(playStep); }
  }
  function stopPlay() {
    if (!playing) return;
    playing = false;
    var b = $("flowplay");
    if (b) { b.classList.remove("on"); b.setAttribute("aria-pressed", "false"); b.innerHTML = "▶ play"; }
  }

  // ---------- wiring (once) ----------
  function wireControls() {
    if (wired) return; wired = true;
    var md = $("flowmode");
    if (md) md.addEventListener("change", function () {
      mode = parseInt(md.value, 10) || 0;
      var v = $("flowmode_v"); if (v) v.textContent = LAYER_LABELS[mode] || "";
      renderNow();
    });
    var is = $("flowiso");
    if (is) is.addEventListener("change", function () { setIso(parseInt(is.value, 10)); });
    var tr = $("flowthr");
    if (tr) tr.addEventListener("input", function () {
      thr = parseFloat(tr.value); var v = $("flowthr_v"); if (v) v.textContent = thr.toFixed(2); renderNow();
    });
    var fr = $("flowframe");
    if (fr) fr.addEventListener("input", function () {
      curFrame = parseInt(fr.value, 10) || 0; stopPlay(); renderNow();
    });
    var pl = $("flowplay");
    if (pl) pl.addEventListener("click", togglePlay);
    window.addEventListener("resize", function () { if (flowVisible()) renderNow(); });
  }

  // ---------- public hooks ----------
  function activate() {
    if (inited) { renderNow(); return; }
    inited = true;
    wireControls();
    var force2d = /[?&]flow2d=1/.test(location.search || "");
    var start = (!force2d && navigator.gpu) ? initGPU().then(function () { gpuOk = true; }).catch(function () { gpuOk = false; }) : Promise.resolve();
    start.then(function () {
      if (!gpuOk) initFallback();
      setBadge();
      var v;
      v = $("flowmode_v"); if (v) v.textContent = LAYER_LABELS[mode];
      v = $("flowthr_v"); if (v) v.textContent = thr.toFixed(2);
      var meta = (window.__flow || {}).ready;
      if (meta && meta.ok && !SEQ) fetchSequence(meta);
      else if (!SEQ) showMsg(statusMsg(meta));
      renderNow();
    });
  }

  window.__flowActivate = activate;
  window.__flowReady = onFlowReady;
})();
