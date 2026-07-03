/* WHY/HOW (Tab 4) — the deep-math museum, PASS 1 (WebGPU, canvas2d fallback).
 *
 * Inlined into the dashboard page by tools/dashboard_server.py::_whyhow_client_js (CSP-strict:
 * self-contained, no external <script src>, no CDN). The server renders ONE representative scored
 * frame's RAW co-registered scalar fields (tools/whyhow_deepmath_panels.py) in a detached governed
 * subprocess and serves them ONCE via GET /api/whyhow. We fetch once, decode, and drive two
 * interactive plates entirely client-side.
 *
 * bundle = { ok, w, h, frame_idx, classes:[{i,label,hex}],
 *            render_b64(jpeg RGB scene), tri_b64(RGB png: R=rho_seg G=rho_uniward B=rho_sens),
 *            argmax_b64(L png: SegNet argmax 0..4),
 *            pearson_fisher_margin(0.978 canonical), pearson_seg_uniward_frame,
 *            pearson_seg_sens_frame, pearson_uni_sens_frame, fields:{seg,uni,sens} }
 *
 * PLATE I.1 "the field, alive": the SegNet detectability field phi = rho_seg margin, as a WebGPU
 *   heat surface. Threshold slider sweeps the level sets {phi = t}; the zero-level-set (argmax
 *   boundary) and the gradient field (phi lit as a surface via its normals) toggle on; base layer
 *   switches phi-heat / scene render / comma10k partition.
 * PLATE I.4 "the Unity": ONE scene, THREE sensitivity readings (rho_seg margin / our separatrix
 *   sensitivity / rho_uniward S-UNIWARD cost) with a morph slider. rho_seg and our-sensitivity
 *   visibly become the SAME picture (measured Pearson shown live); rho_uniward is Fridrich's kindred
 *   steganographic reading (honestly weaker pixelwise — shown truthfully). Canonical Fisher<->margin
 *   0.978 anchors the deep unity.
 *
 * AUTHORITY: [macOS-CPU advisory · NON-PROMOTABLE] imagery. A viz moves no pointer (0.19110).
 */
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };

  var DEFAULT_PAL = [[64, 32, 32], [255, 0, 0], [128, 128, 96], [0, 255, 102], [204, 0, 255]];
  var CLASS_LABELS = ["0 Road", "1 Lane", "2 Undrivable", "3 Movable", "4 MyCar"];

  var WH = null, PAL = DEFAULT_PAL.slice(), fetching = false, tries = 0, inited = false;
  // control state
  var thr = 0.55, morph = 0.0, base = 0, zeroOn = true, gradOn = false;
  // decoded field arrays + bitmap
  var DEC = null;   // {w,h, seg,uni,sens,lab: Uint8Array, bmp: ImageBitmap}
  // gpu state (shared device; two canvas contexts)
  var dev = null, fmt = null, pipe = null, bgl = null, uni = null, samp = null;
  var ctxF = null, ctxU = null, canF = null, canU = null;
  var texR = null, texS = null, texU = null, texN = null, texL = null, tW = 0, tH = 0;
  var gpuOk = false, gpuLost = false, fbCtxF = null, fbCtxU = null, off2d = null, off2dCtx = null;

  // ---------- helpers ----------
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
  function fmtCorr(v) {
    if (v == null || !isFinite(v)) return "—";
    return (v >= 0 ? "+" : "") + v.toFixed(3);
  }
  function whyhowVisible() { var s = $("tab-whyhow"); return !!(s && !s.classList.contains("hide")); }
  function showMsg(id, t) { var m = $(id); if (m) { m.classList.remove("hide"); m.textContent = t; } }
  function hideMsg(id) { var m = $(id); if (m) m.classList.add("hide"); }
  function setStatus(t) { var el = $("whystatus"); if (el) el.textContent = t; }

  // ---------- fetch + ingest ----------
  function activate() {
    if (!inited) { inited = true; wireControls(); }
    if (WH) { renderBoth(); return; }
    if (fetching) return;
    fetching = true;
    fetch("/api/whyhow", { cache: "no-store", credentials: "same-origin" }).then(function (r) {
      if (r.status === 200) return r.json();
      return r.json().then(function (j) { fetching = false; onNotReady(j); return null; });
    }).then(function (d) {
      fetching = false;
      if (d && d.ok) { WH = d; ingest(d); }
    }).catch(function (e) { fetching = false; setStatus("field bundle fetch failed — retrying on next visit."); });
  }
  function onNotReady(j) {
    if (j && j.status === "rendering") setStatus("rendering the deep-math field bundle (governed CPU pass, ~1 s)…");
    else if (j && j.status === "error") setStatus("field bundle error: " + (j.err || "unknown"));
    else setStatus("waiting for the deep-math field bundle (governed CPU pass)…");
    if (tries++ < 40 && whyhowVisible()) setTimeout(activate, 3000);
  }
  function ingest(d) {
    setStatus("frame " + d.frame_idx + " · " + d.h + "×" + d.w + " · decoding fields…");
    PAL = (d.classes && d.classes.length) ? d.classes.map(function (c) { return hexToRgb(c.hex); }) : DEFAULT_PAL.slice();
    buildLegend(d.classes || []);
    buildCorr(d);
    decodeBundle(d).then(function () {
      var start = (navigator.gpu && !/[?&]whyhow2d=1/.test(location.search || "")) ?
        initGPU().then(function () { gpuOk = true; }).catch(function () { gpuOk = false; }) : Promise.resolve();
      start.then(function () {
        if (!gpuOk) initFallback();
        setBadge();
        renderBoth();
        setStatus("frame " + d.frame_idx + " · " + d.h + "×" + d.w + " · " + (gpuOk ? "WebGPU" : "canvas2d"));
      });
    }).catch(function (e) { setStatus("decode error: " + (e && e.message || e)); });
  }

  // ---------- decode (jpeg render + RGB tri fields + L labels) ----------
  var _fc = null, _fctx = null;
  function ensureDecodeCanvas(w, h) {
    if (!_fc) { _fc = document.createElement("canvas"); _fctx = _fc.getContext("2d", { willReadFrequently: true }); }
    if (_fc.width !== w || _fc.height !== h) { _fc.width = w; _fc.height = h; }
  }
  function decodeBundle(d) {
    var w = d.w, h = d.h;
    return Promise.all([
      createImageBitmap(b64ToBlob(d.render_b64, "image/jpeg")),
      createImageBitmap(b64ToBlob(d.tri_b64, "image/png")),
      createImageBitmap(b64ToBlob(d.argmax_b64, "image/png"))
    ]).then(function (r) {
      var bmp = r[0], tri = r[1], lab = r[2];
      ensureDecodeCanvas(w, h);
      _fctx.clearRect(0, 0, w, h); _fctx.drawImage(tri, 0, 0);
      var td = _fctx.getImageData(0, 0, w, h).data;
      var seg = new Uint8Array(w * h), un = new Uint8Array(w * h), sn = new Uint8Array(w * h);
      for (var k = 0, o = 0; k < w * h; k++, o += 4) { seg[k] = td[o]; un[k] = td[o + 1]; sn[k] = td[o + 2]; }
      _fctx.clearRect(0, 0, w, h); _fctx.drawImage(lab, 0, 0);
      var ld = _fctx.getImageData(0, 0, w, h).data;
      var lb = new Uint8Array(w * h);
      for (var k2 = 0, o2 = 0; k2 < w * h; k2++, o2 += 4) lb[k2] = ld[o2];
      if (tri.close) tri.close(); if (lab.close) lab.close();
      DEC = { w: w, h: h, seg: seg, uni: un, sens: sn, lab: lb, bmp: bmp };
    });
  }

  // ---------- WebGPU ----------
  var WGSL = [
    "struct U { head:vec4f, ctl:vec4f, ctl2:vec4f, pal:array<vec4f,5>, };",
    "@group(0) @binding(0) var<uniform> u : U;",
    "@group(0) @binding(1) var texR : texture_2d<f32>;",   // scene render
    "@group(0) @binding(2) var texS : texture_2d<f32>;",   // rho_seg (r8unorm)
    "@group(0) @binding(3) var texU : texture_2d<f32>;",   // rho_uniward
    "@group(0) @binding(4) var texN : texture_2d<f32>;",   // rho_sens
    "@group(0) @binding(5) var texL : texture_2d<u32>;",   // argmax labels
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
    "fn fieldAt(idx:i32, uv:vec2f) -> f32 {",
    "  if (idx<=0) { return textureSampleLevel(texS, samp, uv, 0.0).r; }",   // rho_seg
    "  if (idx==1) { return textureSampleLevel(texN, samp, uv, 0.0).r; }",   // our sensitivity
    "  return textureSampleLevel(texU, samp, uv, 0.0).r;",                    // rho_uniward
    "}",
    "@fragment fn fs(inp:VSOut) -> @location(0) vec4f {",
    "  let dims = u.head.xy; let plate = i32(round(u.head.w));",
    "  let ic = clamp(vec2i(floor(inp.uv*dims)), vec2i(0,0), vec2i(dims)-vec2i(1,1));",
    "  let lab = i32(textureLoad(texL, ic, 0).r);",
    "  if (plate == 1) {",                       // ---- UNITY morph plate ----
    "    let idx = i32(round(u.ctl.w)); let frac = u.ctl2.x;",
    "    let a = fieldAt(idx, inp.uv); let b = fieldAt(idx+1, inp.uv);",
    "    let phi = mix(a, b, clamp(frac,0.0,1.0));",
    "    var col = inferno(phi);",
    "    if (u.ctl.y > 0.5) {",                  // faint argmax boundary for context
    "      let l1 = i32(textureLoad(texL, clamp(ic+vec2i(1,0), vec2i(0,0), vec2i(dims)-vec2i(1,1)), 0).r);",
    "      let l2 = i32(textureLoad(texL, clamp(ic+vec2i(0,1), vec2i(0,0), vec2i(dims)-vec2i(1,1)), 0).r);",
    "      if (l1 != lab || l2 != lab) { col = mix(col, vec3f(0.85,0.95,1.0), 0.30); }",
    "    }",
    "    return vec4f(col, 1.0);",
    "  }",
    "  let phi = textureSampleLevel(texS, samp, inp.uv, 0.0).r;",   // ---- FIELD plate: phi = rho_seg ----
    "  let baseMode = i32(round(u.ctl.x));",
    "  var col = inferno(phi);",
    "  if (baseMode == 1) { col = textureSampleLevel(texR, samp, inp.uv, 0.0).rgb; }",
    "  else if (baseMode == 2) { col = pal(lab); }",
    "  if (u.ctl.z > 0.5) {",                    // gradient/normals: light phi as a surface
    "    let e = vec2f(1.0,1.0)/dims;",
    "    let gx = textureSampleLevel(texS,samp,inp.uv+vec2f(e.x,0.0),0.0).r - textureSampleLevel(texS,samp,inp.uv-vec2f(e.x,0.0),0.0).r;",
    "    let gy = textureSampleLevel(texS,samp,inp.uv+vec2f(0.0,e.y),0.0).r - textureSampleLevel(texS,samp,inp.uv-vec2f(0.0,e.y),0.0).r;",
    "    let n = normalize(vec3f(-gx*6.0, -gy*6.0, 1.0));",
    "    let lamb = clamp(dot(n, normalize(vec3f(-0.4,-0.55,0.75))), 0.0, 1.0);",
    "    col = col * (0.42 + 0.66*lamb);",
    "  }",
    "  let thr = u.head.z;",                      // level-set contour at the threshold
    "  if (abs(phi - thr) < 0.014) { col = mix(col, vec3f(0.99,0.96,0.62), 0.85); }",
    "  if (phi >= thr) { col = mix(col, col*1.18 + vec3f(0.04,0.03,0.0), 0.28); }",  // brighten the fragile band
    "  if (u.ctl.y > 0.5) {",                     // zero-level-set = argmax boundary
    "    let l1 = i32(textureLoad(texL, clamp(ic+vec2i(1,0), vec2i(0,0), vec2i(dims)-vec2i(1,1)), 0).r);",
    "    let l2 = i32(textureLoad(texL, clamp(ic+vec2i(0,1), vec2i(0,0), vec2i(dims)-vec2i(1,1)), 0).r);",
    "    let l3 = i32(textureLoad(texL, clamp(ic-vec2i(1,0), vec2i(0,0), vec2i(dims)-vec2i(1,1)), 0).r);",
    "    if (l1 != lab || l2 != lab || l3 != lab) { col = mix(col, vec3f(0.30,0.92,1.0), 0.92); }",
    "  }",
    "  return vec4f(col, 1.0);",
    "}"
  ].join("\n");

  function initGPU() {
    canF = $("whycanvas_field"); canU = $("whycanvas_unity");
    return Promise.resolve().then(function () {
      if (!navigator.gpu) throw new Error("navigator.gpu absent");
      return navigator.gpu.requestAdapter();
    }).then(function (ad) {
      if (!ad) throw new Error("no WebGPU adapter");
      return ad.requestDevice();
    }).then(function (device) {
      dev = device;
      if (dev.lost && dev.lost.then) dev.lost.then(function () { gpuLost = true; });
      fmt = navigator.gpu.getPreferredCanvasFormat();
      ctxF = canF.getContext("webgpu"); ctxU = canU.getContext("webgpu");
      if (!ctxF || !ctxU) throw new Error("no webgpu canvas context");
      ctxF.configure({ device: dev, format: fmt, alphaMode: "opaque" });
      ctxU.configure({ device: dev, format: fmt, alphaMode: "opaque" });
      var mod = dev.createShaderModule({ code: WGSL });
      bgl = dev.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "uint" } },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } }
      ] });
      var pl = dev.createPipelineLayout({ bindGroupLayouts: [bgl] });
      pipe = dev.createRenderPipeline({
        layout: pl,
        vertex: { module: mod, entryPoint: "vs" },
        fragment: { module: mod, entryPoint: "fs", targets: [{ format: fmt }] },
        primitive: { topology: "triangle-list" }
      });
      uni = dev.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      samp = dev.createSampler({ magFilter: "linear", minFilter: "linear" });
      ensureTex(DEC.w, DEC.h);
      uploadTex();
    });
  }
  function ensureTex(W, H) {
    if (tW === W && tH === H && texR) return;
    [texR, texS, texU, texN, texL].forEach(function (t) { if (t && t.destroy) t.destroy(); });
    var UB = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST;
    texR = dev.createTexture({ size: [W, H], format: "rgba8unorm", usage: UB | GPUTextureUsage.RENDER_ATTACHMENT });
    texS = dev.createTexture({ size: [W, H], format: "r8unorm", usage: UB });
    texU = dev.createTexture({ size: [W, H], format: "r8unorm", usage: UB });
    texN = dev.createTexture({ size: [W, H], format: "r8unorm", usage: UB });
    texL = dev.createTexture({ size: [W, H], format: "r8uint", usage: UB });
    tW = W; tH = H;
  }
  function uploadTex() {
    var W = DEC.w, H = DEC.h;
    dev.queue.copyExternalImageToTexture({ source: DEC.bmp }, { texture: texR }, { width: W, height: H });
    dev.queue.writeTexture({ texture: texS }, DEC.seg, { bytesPerRow: W, rowsPerImage: H }, { width: W, height: H });
    dev.queue.writeTexture({ texture: texU }, DEC.uni, { bytesPerRow: W, rowsPerImage: H }, { width: W, height: H });
    dev.queue.writeTexture({ texture: texN }, DEC.sens, { bytesPerRow: W, rowsPerImage: H }, { width: W, height: H });
    dev.queue.writeTexture({ texture: texL }, DEC.lab, { bytesPerRow: W, rowsPerImage: H }, { width: W, height: H });
  }
  function sizeCanvas(c, W, H) {
    var cssW = c.clientWidth || 512, dpr = Math.min(window.devicePixelRatio || 1, 2);
    var pxW = Math.max(1, Math.round(cssW * dpr)), pxH = Math.max(1, Math.round(pxW * H / W));
    if (c.width !== pxW || c.height !== pxH) { c.width = pxW; c.height = pxH; }
  }
  function uniformFor(plate) {
    var buf = new Float32Array(32);
    buf[0] = DEC.w; buf[1] = DEC.h;
    buf[2] = (plate === 1) ? morph : thr;
    buf[3] = plate;
    if (plate === 1) {
      var i0 = Math.min(1, Math.floor(morph));
      buf[4] = i0; buf[5] = zeroOn ? 1 : 0; buf[6] = 0; buf[7] = i0;
      buf[8] = Math.min(1, Math.max(0, morph - i0));
    } else {
      buf[4] = base; buf[5] = zeroOn ? 1 : 0; buf[6] = gradOn ? 1 : 0; buf[7] = 0;
      buf[8] = 0;
    }
    for (var c = 0; c < 5; c++) { var p = PAL[c] || [128, 128, 128]; var b = 12 + c * 4; buf[b] = p[0] / 255; buf[b + 1] = p[1] / 255; buf[b + 2] = p[2] / 255; buf[b + 3] = 1; }
    return buf;
  }
  function drawGPU(plate) {
    var can = (plate === 1) ? canU : canF, ctx = (plate === 1) ? ctxU : ctxF;
    sizeCanvas(can, DEC.w, DEC.h);
    dev.queue.writeBuffer(uni, 0, uniformFor(plate));
    var bg = dev.createBindGroup({ layout: bgl, entries: [
      { binding: 0, resource: { buffer: uni } },
      { binding: 1, resource: texR.createView() },
      { binding: 2, resource: texS.createView() },
      { binding: 3, resource: texU.createView() },
      { binding: 4, resource: texN.createView() },
      { binding: 5, resource: texL.createView() },
      { binding: 6, resource: samp }
    ] });
    var enc = dev.createCommandEncoder();
    var view = ctx.getCurrentTexture().createView();
    var pass = enc.beginRenderPass({ colorAttachments: [{ view: view,
      clearValue: { r: 0.047, g: 0.06, b: 0.078, a: 1 }, loadOp: "clear", storeOp: "store" }] });
    pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.draw(3); pass.end();
    dev.queue.submit([enc.finish()]);
  }

  // ---------- canvas2d fallback ----------
  function initFallback() {
    // clone canvases so they hold a clean 2d context (a canvas can hold one context type)
    ["whycanvas_field", "whycanvas_unity"].forEach(function (id) {
      var old = $(id); if (!old) return;
      var n = old.cloneNode(false); old.parentNode.replaceChild(n, old);
    });
    canF = $("whycanvas_field"); canU = $("whycanvas_unity");
    fbCtxF = canF.getContext("2d"); fbCtxU = canU.getContext("2d");
    if (!off2d) { off2d = document.createElement("canvas"); off2dCtx = off2d.getContext("2d"); }
  }
  function labAt(x, y, W, H) { return DEC.lab[Math.min(H - 1, y) * W + Math.min(W - 1, x)]; }
  function draw2d(plate) {
    var W = DEC.w, H = DEC.h, can = (plate === 1) ? canU : canF, ctx = (plate === 1) ? fbCtxU : fbCtxF;
    if (!ctx) return;
    sizeCanvas(can, W, H);
    if (off2d.width !== W || off2d.height !== H) { off2d.width = W; off2d.height = H; }
    var img = off2dCtx.createImageData(W, H), d = img.data;
    var i0 = Math.min(1, Math.floor(morph)), frac = Math.min(1, Math.max(0, morph - i0));
    var fieldArr = [DEC.seg, DEC.sens, DEC.uni];
    for (var y = 0, k = 0, o = 0; y < H; y++) {
      for (var x = 0; x < W; x++, k++, o += 4) {
        var rgb, lab = DEC.lab[k];
        if (plate === 1) {
          var a = fieldArr[i0][k] / 255, b = fieldArr[Math.min(2, i0 + 1)][k] / 255;
          var phi = a * (1 - frac) + b * frac, c = inferno(phi); rgb = [c[0] * 255, c[1] * 255, c[2] * 255];
          if (zeroOn && (labAt(x + 1, y, W, H) !== lab || labAt(x, y + 1, W, H) !== lab))
            rgb = [rgb[0] * 0.7 + 217 * 0.3, rgb[1] * 0.7 + 242 * 0.3, rgb[2] * 0.7 + 255 * 0.3];
        } else {
          var ph = DEC.seg[k] / 255;
          if (base === 1) { rgb = null; }  // handled below via drawImage
          else if (base === 2) { var p = PAL[lab] || [128, 128, 128]; rgb = [p[0], p[1], p[2]]; }
          else { var cc = inferno(ph); rgb = [cc[0] * 255, cc[1] * 255, cc[2] * 255]; }
          if (rgb) {
            if (Math.abs(ph - thr) < 0.02) rgb = [rgb[0] * 0.15 + 252 * 0.85, rgb[1] * 0.15 + 245 * 0.85, rgb[2] * 0.15 + 158 * 0.85];
            if (zeroOn && (labAt(x + 1, y, W, H) !== lab || labAt(x, y + 1, W, H) !== lab))
              rgb = [rgb[0] * 0.1 + 77 * 0.9, rgb[1] * 0.1 + 235 * 0.9, rgb[2] * 0.1 + 255 * 0.9];
          }
        }
        if (rgb) { d[o] = rgb[0] | 0; d[o + 1] = rgb[1] | 0; d[o + 2] = rgb[2] | 0; d[o + 3] = 255; }
        else { d[o] = 0; d[o + 1] = 0; d[o + 2] = 0; d[o + 3] = 0; }
      }
    }
    ctx.clearRect(0, 0, can.width, can.height);
    if (plate === 0 && base === 1) {   // scene render base: draw the photo, no per-pixel field
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(DEC.bmp, 0, 0, W, H, 0, 0, can.width, can.height);
      return;
    }
    off2dCtx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off2d, 0, 0, W, H, 0, 0, can.width, can.height);
  }

  function renderField() {
    if (!DEC) return;
    try { if (gpuOk && !gpuLost) drawGPU(0); else draw2d(0); hideMsg("whymsg_field"); }
    catch (e) { if (gpuOk) { gpuOk = false; try { initFallback(); } catch (e2) {} setBadge(); } try { draw2d(0); hideMsg("whymsg_field"); } catch (e3) { showMsg("whymsg_field", "field render error"); } }
  }
  function renderUnity() {
    if (!DEC) return;
    try { if (gpuOk && !gpuLost) drawGPU(1); else draw2d(1); hideMsg("whymsg_unity"); }
    catch (e) { if (gpuOk) { gpuOk = false; try { initFallback(); } catch (e2) {} setBadge(); } try { draw2d(1); hideMsg("whymsg_unity"); } catch (e3) { showMsg("whymsg_unity", "unity render error"); } }
  }
  function renderBoth() { if (!whyhowVisible()) return; renderField(); renderUnity(); }

  // ---------- UI ----------
  function setBadge() {
    [["whybadge_field"], ["whybadge_unity"]].forEach(function (a) {
      var b = $(a[0]); if (!b) return;
      if (gpuOk && !gpuLost) { b.className = "whybadge gpu"; b.textContent = "WebGPU"; }
      else { b.className = "whybadge cpu"; b.textContent = "canvas2d"; }
    });
  }
  function buildLegend(classes) {
    var lg = $("whyleg_field"); if (!lg) return; lg.innerHTML = "";
    (classes.length ? classes : CLASS_LABELS.map(function (l, i) { return { i: i, label: l, hex: "#808080" }; }))
      .forEach(function (c) {
        var el = document.createElement("span"); el.className = "lc";
        el.innerHTML = "<span class='dot' style='background:" + (c.hex || "#808080") + "'></span>" + c.label;
        lg.appendChild(el);
      });
  }
  function buildCorr(d) {
    var host = $("whycorr"); if (host) {
      host.innerHTML =
        card("hi", "margin ↔ our sensitivity", fmtCorr(d.pearson_seg_sens_frame), "this frame — the same picture") +
        card("anchor", "Fisher ↔ (−margin)", "+0.978", "canonical — the margin IS the Fisher metric") +
        card("lo", "margin ↔ S-UNIWARD", fmtCorr(d.pearson_seg_uniward_frame), "this frame — texture, not the boundary");
    }
    var hi = $("whycorr_hi"); if (hi) hi.textContent = fmtCorr(d.pearson_seg_sens_frame);
    var lo = $("whycorr_lo"); if (lo) lo.textContent = fmtCorr(d.pearson_seg_uniward_frame);
  }
  function card(cls, k, v, s) {
    return "<div class='cc " + cls + "'><span class='ck'>" + k + "</span><span class='cv'>" + v +
      "</span><span class='cs'>" + s + "</span></div>";
  }
  var MORPH_NAMES = ["ρ_seg (SegNet margin)", "our distortion sensitivity", "ρ_uniward (S-UNIWARD)"];
  function morphLabel() {
    var i0 = Math.min(1, Math.floor(morph)), frac = morph - i0;
    if (frac < 0.06) return MORPH_NAMES[i0];
    if (frac > 0.94) return MORPH_NAMES[Math.min(2, i0 + 1)];
    return MORPH_NAMES[i0] + " → " + MORPH_NAMES[Math.min(2, i0 + 1)];
  }
  function syncSeg(id, sel, val) {
    var host = $(id); if (!host) return;
    Array.prototype.forEach.call(host.querySelectorAll(".sg"), function (el) {
      el.classList.toggle("on", parseInt(el.dataset[sel], 10) === val);
    });
  }
  function wireControls() {
    var mv = $("whymvrail");
    if (mv) Array.prototype.forEach.call(mv.querySelectorAll(".mvchip"), function (c) {
      c.addEventListener("click", function () {
        var which = c.dataset.mv;
        Array.prototype.forEach.call(mv.querySelectorAll(".mvchip"), function (x) { x.classList.toggle("on", x === c); });
        var mi = $("whymv-i"), mii = $("whymv-ii");
        if (mi) mi.classList.toggle("hide", which !== "i");
        if (mii) mii.classList.toggle("hide", which !== "ii");
        if (which === "i") renderBoth();
      });
    });
    var tr = $("whythr");
    if (tr) tr.addEventListener("input", function () { thr = parseFloat(tr.value); var v = $("whythr_v"); if (v) v.textContent = thr.toFixed(2); renderField(); });
    var mp = $("whymorph");
    if (mp) mp.addEventListener("input", function () { morph = parseFloat(mp.value); var v = $("whymorph_v"); if (v) v.textContent = morphLabel(); renderUnity(); });
    var tz = $("whytog_zero");
    if (tz) { var f = function () { zeroOn = !zeroOn; tz.classList.toggle("on", zeroOn); renderBoth(); }; tz.addEventListener("click", f); tz.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); f(); } }); }
    var tg = $("whytog_grad");
    if (tg) { var g = function () { gradOn = !gradOn; tg.classList.toggle("on", gradOn); renderField(); }; tg.addEventListener("click", g); tg.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); g(); } }); }
    var fb = $("whyfieldbase");
    if (fb) Array.prototype.forEach.call(fb.querySelectorAll(".sg"), function (el) {
      el.addEventListener("click", function () { base = parseInt(el.dataset.b, 10) || 0; syncSeg("whyfieldbase", "b", base); renderField(); });
    });
    var uj = $("whyunityjump");
    if (uj) Array.prototype.forEach.call(uj.querySelectorAll(".sg"), function (el) {
      el.addEventListener("click", function () {
        morph = parseInt(el.dataset.t, 10) || 0; var mpr = $("whymorph"); if (mpr) mpr.value = String(morph);
        var v = $("whymorph_v"); if (v) v.textContent = morphLabel(); syncSeg("whyunityjump", "t", Math.round(morph)); renderUnity();
      });
    });
    window.addEventListener("resize", function () { if (whyhowVisible()) renderBoth(); });
  }

  window.__whyhowActivate = activate;
})();

/* =====================================================================================
 * WHY/HOW (Tab 4) — §1 "ONE FRONT, FIVE SCALES" spine + §4 the fractal finale (PASS 2).
 *
 * Self-contained: needs NO /api/whyhow bundle. The five curves are (2 HARD DATA, 3 schematic):
 *   campaign  = EdgeBench log-sigmoid, R²=0.998 (ByteDance Seed — THEIRS; we draw the fitted
 *               family SHAPE, not their raw points, which we don't have).
 *   training  = the LIVE #205 d_seg descent (ep25→125: 0.0103→0.0058, implied_S 1.72→0.87 —
 *               REAL verdicts, read from the run log at build time; plotted as correct-fraction
 *               x = 1 − d_seg/d_seg(ep25); the smooth overlay is a SINGLE-RUN descriptive fit).
 *   boundary  = the argmax separatrix softmax front — a genuine level-set = reaction-diffusion
 *               PDE identity (~96% of flip-mass in a ~2px band, measured).
 *   curriculum= coarse→fine = temperature anneal = persistence order (measured schedule; shape schematic).
 *   erasure   = the long tail, error ∝ 1/persistence (SOC; interpretive shape).
 *
 * HONESTY: that all five scales are LITERALLY the same Fisher–KPP front is a UNIFYING LENS /
 * CONJECTURE (labelled in-UI), NOT a measured cross-scale identity. The left canvas integrates the
 * REAL Fisher–KPP PDE ∂ₜx=βx(1−x)+∇²x forward (explicit finite-difference, stepped every frame) —
 * that is真: it really integrates the equation. WebGPU render where present, canvas2d fallback +
 * honest badge (mirrors Pass 1). A viz moves NO pointer (0.19110, UNMOVED).
 * ===================================================================================== */
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var prevActivate = window.__whyhowActivate;   // chain Pass 1 (the field bundle) — never clobber it

  // ---- LIVE #205 verdicts (read from run.log at build time; ep0 pre-training outlier excluded) ----
  var DSEG205 = [[25, 0.010299], [50, 0.00783], [75, 0.006794], [100, 0.006145], [125, 0.005767]];
  var DSEG0 = DSEG205[0][1];   // ep25 anchor for the correct-fraction transform x = 1 − d_seg/d_seg(ep25)

  var N = 72;                  // samples per normalized curve
  function sigmoid(u, k, u0) { return 1 / (1 + Math.exp(-k * (u - u0))); }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

  // shared logistic front template x(u)=1/(1+e^{-k(u-u0)})
  function tmpl(u) { return sigmoid(u, 9.0, 0.5); }

  // training real markers -> normalized (u in [0,1] over ep25..125, x=correct-fraction)
  var TRAIN_MARKERS = DSEG205.map(function (p) {
    return { u: (p[0] - 25) / 100, y: clamp01(1 - p[1] / DSEG0), ep: p[0], dseg: p[1] };
  });
  function trainAt(u) {  // piecewise-linear interp of the REAL markers (no invention between points)
    if (u <= 0) return TRAIN_MARKERS[0].y;
    if (u >= 1) return TRAIN_MARKERS[TRAIN_MARKERS.length - 1].y;
    for (var i = 0; i < TRAIN_MARKERS.length - 1; i++) {
      var a = TRAIN_MARKERS[i], b = TRAIN_MARKERS[i + 1];
      if (u >= a.u && u <= b.u) return lerp(a.y, b.y, (u - a.u) / (b.u - a.u));
    }
    return TRAIN_MARKERS[TRAIN_MARKERS.length - 1].y;
  }

  // the five scales, index 0..4 = pixel -> campaign (small physical scale -> large)
  var SCALES = [
    { key: "boundary", title: "BOUNDARY — the argmax separatrix in space", units: "P(correct) / arclength",
      tag: "pde", tagtxt: "PDE identity", col: "#c08cff",
      note: "Level-set = reaction–diffusion front — a genuine identity. ~96% of d_seg's flip-mass sits in a ~2&nbsp;px band (measured); the softmax across the boundary IS a spatial front.",
      f: function (u) { return sigmoid(u, 14, 0.5); } },
    { key: "erasure", title: "ERASURE — the long tail, error ∝ 1/persistence", units: "survival / persistence-rank",
      tag: "soft", tagtxt: "interpretive", col: "#ffb454",
      note: "The trailing edge of the same front: finest-scale features (lane dashes, distant movers) are erased first — error ∝ 1/persistence (SOC avalanches, Bak–Tang–Wiesenfeld). Illustrative shape.",
      f: function (u) { return 1 - 1 / (1 + 6 * u); } },
    { key: "training", title: "TRAINING — the correct partition invading (LIVE #205)", units: "correct-fraction / epoch",
      tag: "live", tagtxt: "live data · ours · single-run fit", col: "#5ab0ff",
      note: "REAL #205 verdicts ep25→125: d_seg 0.0103→0.0058, implied_S 1.72→0.87 — still climbing (a PARTIAL front, ~44% of ep25's error closed). Markers are live; the smooth curve is a single-run descriptive fit — a toy, not a law.",
      f: trainAt, markers: TRAIN_MARKERS },
    { key: "curriculum", title: "CURRICULUM — coarse→fine = persistence = anneal", units: "resolved-scale / curriculum-stage",
      tag: "soft", tagtxt: "schematic · measured schedule", col: "#ffb454",
      note: "One flow, four names: CE→τ(@300)→Muon(@726) = temperature anneal = Morse–Smale persistence order = coarse→fine curvelet sweep. Stage boundaries measured; the curve shape is schematic.",
      f: function (u) { return sigmoid(u, 8, 0.45); } },
    { key: "campaign", title: "CAMPAIGN — capability vs interaction-time", units: "capability / log interaction-time",
      tag: "hard", tagtxt: "hard data · theirs (R²=0.998)", col: "#46d369",
      note: "EdgeBench log-sigmoid, R²=0.998 — ByteDance Seed's published fit (theirs). Continuous experience beats restarts (+6.9 @ 12h; the gap GROWS with horizon). We plot the fitted family shape, not their raw points (which we don't have).",
      f: function (u) { return sigmoid(u, 5.5, 0.4); } }
  ];
  // precompute normalized polylines
  SCALES.forEach(function (s) {
    var arr = new Float32Array(N);
    for (var i = 0; i < N; i++) arr[i] = clamp01(s.f(i / (N - 1)));
    s.samp = arr;
  });

  // ---------- state ----------
  var inited = false, scaleS = 2.0, frontPaused = false, finPaused = false, sixOn = false;
  var raf = 0, phase = 0.0, lastT = 0;
  // KPP field
  var KW = 168, KH = 94, fld = null, nxt = null, kbuf = null, kseed = 0;
  var KBETA = 0.16, KDIFF = 0.9, KDT = 0.18, KSUB = 2;
  // KPP gpu
  var kdev = null, kfmt = null, kpipe = null, kbgl = null, kuni = null, ktex = null, ksamp = null, kctx = null, kcan = null;
  var kgpuOk = false, kgpuLost = false, k2d = null, koff = null, koffCtx = null, kTriedGpu = false;

  function visible() { var s = $("tab-whyhow"); return !!(s && !s.classList.contains("hide")); }
  function inferno(x) {
    var t = x < 0 ? 0 : x > 1 ? 1 : x;
    var c0 = [0.00021894, 0.00165100, -0.01948090], c1 = [0.10651342, 0.56395644, 3.93271239],
      c2 = [11.60249308, -3.97285397, -15.94239411], c3 = [-41.70399613, 17.43639888, 44.35414520],
      c4 = [77.16293570, -33.40235894, -81.80730926], c5 = [-71.31942824, 32.62606426, 73.20951986],
      c6 = [25.13112622, -12.24266895, -23.07032500], o = [0, 0, 0];
    for (var k = 0; k < 3; k++) {
      var v = c0[k] + t * (c1[k] + t * (c2[k] + t * (c3[k] + t * (c4[k] + t * (c5[k] + t * c6[k])))));
      o[k] = v < 0 ? 0 : v > 1 ? 1 : v;
    }
    return o;
  }
  function sizeCanvas(c, aspect) {
    if (!c) return false;
    var cssW = c.clientWidth || (c.parentNode && c.parentNode.clientWidth) || 320;
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var pxW = Math.max(1, Math.round(cssW * dpr)), pxH = Math.max(1, Math.round(pxW / aspect));
    if (c.width !== pxW || c.height !== pxH) { c.width = pxW; c.height = pxH; return true; }
    return false;
  }

  // ---------- KPP integrator (explicit finite-difference; the REAL Fisher–KPP PDE) ----------
  function kppInit() {
    fld = new Float32Array(KW * KH); nxt = new Float32Array(KW * KH); kbuf = new Uint8Array(KW * KH);
    kppSeed();
  }
  function kppSeed() {   // wavy seed on the left edge -> a curved codim-1 front (evokes the boundary)
    kseed++;
    for (var y = 0; y < KH; y++) {
      var edge = 5 + 3.0 * Math.sin(y * 0.14 + kseed * 0.7);
      for (var x = 0; x < KW; x++) fld[y * KW + x] = (x < edge) ? 1.0 : 0.0;
    }
  }
  function kppStep() {
    for (var s = 0; s < KSUB; s++) {
      for (var y = 0; y < KH; y++) {
        var ym = y > 0 ? y - 1 : 0, yp = y < KH - 1 ? y + 1 : KH - 1;
        for (var x = 0; x < KW; x++) {
          var xm = x > 0 ? x - 1 : 0, xp = x < KW - 1 ? x + 1 : KW - 1;
          var c = fld[y * KW + x];
          var lap = fld[y * KW + xm] + fld[y * KW + xp] + fld[ym * KW + x] + fld[yp * KW + x] - 4 * c;
          var v = c + KDT * (KBETA * c * (1 - c) + KDIFF * lap);
          nxt[y * KW + x] = v < 0 ? 0 : v > 1 ? 1 : v;
        }
      }
      var tmp = fld; fld = nxt; nxt = tmp;
    }
    // loop: when the front has swept across, re-seed
    var m = 0; for (var i = 0; i < fld.length; i += 7) m += fld[i];
    if (m / (fld.length / 7) > 0.9) kppSeed();
  }
  function kppToBuf() { for (var i = 0; i < fld.length; i++) kbuf[i] = (fld[i] * 255 + 0.5) | 0; }

  // ---- KPP WebGPU render (mirrors Pass 1's fullscreen-triangle + inferno; own device) ----
  var KWGSL = [
    "struct U { p:vec4f, };",
    "@group(0) @binding(0) var<uniform> u:U;",
    "@group(0) @binding(1) var texF: texture_2d<f32>;",
    "@group(0) @binding(2) var samp: sampler;",
    "struct VSOut { @builtin(position) pos:vec4f, @location(0) uv:vec2f, };",
    "@vertex fn vs(@builtin(vertex_index) i:u32) -> VSOut {",
    "  var p = array<vec2f,3>(vec2f(-1.0,-1.0), vec2f(3.0,-1.0), vec2f(-1.0,3.0));",
    "  var o:VSOut; o.pos = vec4f(p[i],0.0,1.0);",
    "  var uv = 0.5*(p[i]+vec2f(1.0,1.0)); uv.y = 1.0-uv.y; o.uv = uv; return o;",
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
    "  let x = textureSampleLevel(texF, samp, inp.uv, 0.0).r;",
    "  var col = inferno(x);",
    "  if (abs(x-0.5) < 0.03) { col = mix(col, vec3f(0.99,0.97,0.70), 0.85); }",   // the front contour x=0.5
    "  return vec4f(col, 1.0);",
    "}"
  ].join("\n");

  function kppInitGPU() {
    kcan = $("whycanvas_kpp");
    return Promise.resolve().then(function () {
      if (!navigator.gpu) throw new Error("no gpu");
      return navigator.gpu.requestAdapter();
    }).then(function (ad) {
      if (!ad) throw new Error("no adapter");
      return ad.requestDevice();
    }).then(function (device) {
      kdev = device;
      if (kdev.lost && kdev.lost.then) kdev.lost.then(function () { kgpuLost = true; });
      kfmt = navigator.gpu.getPreferredCanvasFormat();
      kctx = kcan.getContext("webgpu");
      if (!kctx) throw new Error("no webgpu ctx");
      kctx.configure({ device: kdev, format: kfmt, alphaMode: "opaque" });
      var mod = kdev.createShaderModule({ code: KWGSL });
      kbgl = kdev.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } }
      ] });
      var pl = kdev.createPipelineLayout({ bindGroupLayouts: [kbgl] });
      kpipe = kdev.createRenderPipeline({
        layout: pl, vertex: { module: mod, entryPoint: "vs" },
        fragment: { module: mod, entryPoint: "fs", targets: [{ format: kfmt }] },
        primitive: { topology: "triangle-list" }
      });
      kuni = kdev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      ksamp = kdev.createSampler({ magFilter: "linear", minFilter: "linear" });
      ktex = kdev.createTexture({ size: [KW, KH], format: "r8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      kgpuOk = true;
    });
  }
  function kppDrawGPU() {
    sizeCanvas(kcan, KW / KH);
    kdev.queue.writeTexture({ texture: ktex }, kbuf, { bytesPerRow: KW, rowsPerImage: KH }, { width: KW, height: KH });
    kdev.queue.writeBuffer(kuni, 0, new Float32Array([KW, KH, 0, 0]));
    var bg = kdev.createBindGroup({ layout: kbgl, entries: [
      { binding: 0, resource: { buffer: kuni } },
      { binding: 1, resource: ktex.createView() },
      { binding: 2, resource: ksamp }
    ] });
    var enc = kdev.createCommandEncoder();
    var pass = enc.beginRenderPass({ colorAttachments: [{ view: kctx.getCurrentTexture().createView(),
      clearValue: { r: 0.04, g: 0.05, b: 0.07, a: 1 }, loadOp: "clear", storeOp: "store" }] });
    pass.setPipeline(kpipe); pass.setBindGroup(0, bg); pass.draw(3); pass.end();
    kdev.queue.submit([enc.finish()]);
  }
  function kppInitFallback() {
    var old = $("whycanvas_kpp");
    if (old && !k2d) {   // fresh 2d context (a canvas holds one context type)
      if (kTriedGpu) { var n = old.cloneNode(false); old.parentNode.replaceChild(n, old); }
      kcan = $("whycanvas_kpp"); k2d = kcan.getContext("2d");
      koff = document.createElement("canvas"); koff.width = KW; koff.height = KH; koffCtx = koff.getContext("2d");
    }
  }
  function kppDraw2d() {
    if (!k2d) kppInitFallback();
    if (!k2d) return;
    sizeCanvas(kcan, KW / KH);
    var img = koffCtx.createImageData(KW, KH), d = img.data;
    for (var i = 0, o = 0; i < fld.length; i++, o += 4) {
      var x = fld[i], c = inferno(x);
      var r = c[0], g = c[1], b = c[2];
      if (Math.abs(x - 0.5) < 0.04) { r = r * 0.15 + 0.99 * 0.85; g = g * 0.15 + 0.97 * 0.85; b = b * 0.15 + 0.70 * 0.85; }
      d[o] = (r * 255) | 0; d[o + 1] = (g * 255) | 0; d[o + 2] = (b * 255) | 0; d[o + 3] = 255;
    }
    koffCtx.putImageData(img, 0, 0);
    k2d.imageSmoothingEnabled = true;
    k2d.clearRect(0, 0, kcan.width, kcan.height);
    k2d.drawImage(koff, 0, 0, KW, KH, 0, 0, kcan.width, kcan.height);
  }
  function kppRender() {
    var msg = $("whymsg_kpp");
    try {
      if (kgpuOk && !kgpuLost) kppDrawGPU(); else kppDraw2d();
      if (msg) msg.classList.add("hide");
    } catch (e) {
      if (kgpuOk) { kgpuOk = false; try { kppInitFallback(); } catch (e2) {} setSpineBadge(); }
      try { kppDraw2d(); if (msg) msg.classList.add("hide"); } catch (e3) { if (msg) { msg.classList.remove("hide"); msg.textContent = "front render error"; } }
    }
  }
  function setSpineBadge() {
    var b = $("whybadge_spine"); if (!b) return;
    if (kgpuOk && !kgpuLost) { b.className = "whybadge gpu"; b.textContent = "WebGPU"; }
    else { b.className = "whybadge cpu"; b.textContent = "canvas2d"; }
  }

  // ---------- the five-scale overlay chart (canvas2d — crisp lines) ----------
  function plotBox(c) { var W = c.width, H = c.height, dpr = Math.min(window.devicePixelRatio || 1, 2);
    return { W: W, H: H, pl: 30 * dpr, pr: 10 * dpr, pt: 10 * dpr, pb: 20 * dpr, dpr: dpr }; }
  function drawScaleChart() {
    var c = $("whycanvas_scale"); if (!c) return; sizeCanvas(c, 20 / 11);
    var g = c.getContext("2d"); if (!g) return;
    var b = plotBox(c), x0 = b.pl, x1 = b.W - b.pr, y0 = b.pt, y1 = b.H - b.pb, PW = x1 - x0, PH = y1 - y0;
    g.clearRect(0, 0, b.W, b.H);
    // frame + gridlines
    g.strokeStyle = "#20252e"; g.lineWidth = 1 * b.dpr;
    for (var gi = 0; gi <= 4; gi++) { var yy = y0 + PH * gi / 4; g.beginPath(); g.moveTo(x0, yy); g.lineTo(x1, yy); g.stroke(); }
    g.strokeStyle = "#333a45"; g.strokeRect(x0, y0, PW, PH);
    var XY = function (u, v) { return [x0 + u * PW, y1 - clamp01(v) * PH]; };
    // shared logistic front template (dashed, faint violet)
    g.strokeStyle = "rgba(192,140,255,0.55)"; g.lineWidth = 1.4 * b.dpr; g.setLineDash([5 * b.dpr, 4 * b.dpr]);
    g.beginPath(); for (var i = 0; i <= N; i++) { var u = i / N, p = XY(u, tmpl(u)); if (i === 0) g.moveTo(p[0], p[1]); else g.lineTo(p[0], p[1]); } g.stroke();
    g.setLineDash([]);
    // morphed active curve = lerp between adjacent scales
    var i0 = Math.max(0, Math.min(4, Math.floor(scaleS))), i1 = Math.min(4, i0 + 1), frac = scaleS - i0;
    var near = Math.round(scaleS), meta = SCALES[near];
    g.strokeStyle = meta.col; g.lineWidth = 2.2 * b.dpr; g.beginPath();
    for (var j = 0; j < N; j++) {
      var uu = j / (N - 1), vv = lerp(SCALES[i0].samp[j], SCALES[i1].samp[j], frac), pp = XY(uu, vv);
      if (j === 0) g.moveTo(pp[0], pp[1]); else g.lineTo(pp[0], pp[1]);
    }
    g.stroke();
    // real markers only when the OWNING scale is dominant (so we never show data at a misleading blend)
    if (meta.markers && Math.abs(scaleS - near) < 0.12) {
      g.fillStyle = meta.col;
      meta.markers.forEach(function (m) { var p = XY(m.u, m.y); g.beginPath(); g.arc(p[0], p[1], 3.4 * b.dpr, 0, 7); g.fill(); });
    }
    // axis ticks (0..1 both axes; semantic labels live in the title/note)
    g.fillStyle = "#6b7482"; g.font = (9 * b.dpr) + "px ui-monospace,Menlo,monospace";
    g.fillText("0", x0 - 8 * b.dpr, y1 + 12 * b.dpr); g.fillText("1", x1 - 4 * b.dpr, y1 + 12 * b.dpr);
    g.save(); g.translate(x0 - 20 * b.dpr, y0 + PH / 2); g.rotate(-Math.PI / 2);
    g.fillText("0        x        1", -22 * b.dpr, 0); g.restore();
  }

  function updateScaleMeta() {
    var near = Math.round(scaleS), m = SCALES[near];
    var t = $("whyscale_title"); if (t) t.textContent = m.title;
    var u = $("whyspine_units"); if (u) u.textContent = "units: " + m.units;
    var tag = $("whyscale_tag"); if (tag) { tag.className = "stag " + m.tag; tag.textContent = m.tagtxt; }
    var note = $("whyscale_note"); if (note) note.innerHTML = m.note;
    var v = $("whyscale_v"); if (v) v.textContent = m.key;
    var host = $("whyscalejump");
    if (host) Array.prototype.forEach.call(host.querySelectorAll(".sg"), function (el) {
      el.classList.toggle("on", parseInt(el.dataset.s, 10) === near);
    });
  }

  // ---------- §4 finale: five phase-locked mini-fronts ----------
  function drawMini(id, idx, ph) {
    var c = $(id); if (!c) return; sizeCanvas(c, 16 / 12);
    var g = c.getContext("2d"); if (!g) return;
    var dpr = Math.min(window.devicePixelRatio || 1, 2), W = c.width, H = c.height, pad = 6 * dpr;
    var x0 = pad, x1 = W - pad, y0 = pad, y1 = H - pad, PW = x1 - x0, PH = y1 - y0, s = SCALES[idx];
    g.clearRect(0, 0, W, H);
    var XY = function (u, v) { return [x0 + u * PW, y1 - clamp01(v) * PH]; };
    // fill under the curve up to the shared phase (the front rolling through)
    g.beginPath(); g.moveTo(x0, y1);
    for (var i = 0; i <= N; i++) { var u = Math.min(i / N, ph), p = XY(u, s.samp[Math.min(N - 1, Math.round(u * (N - 1)))]); g.lineTo(p[0], p[1]); }
    g.lineTo(XY(ph, 0)[0], y1); g.closePath();
    g.fillStyle = hexA(s.col, 0.16); g.fill();
    // the curve
    g.strokeStyle = s.col; g.lineWidth = 1.6 * dpr; g.beginPath();
    for (var j = 0; j < N; j++) { var uu = j / (N - 1), pp = XY(uu, s.samp[j]); if (j === 0) g.moveTo(pp[0], pp[1]); else g.lineTo(pp[0], pp[1]); }
    g.stroke();
    // the phase-locked front line
    var fx = x0 + ph * PW;
    g.strokeStyle = "rgba(252,247,190,0.9)"; g.lineWidth = 1.4 * dpr;
    g.beginPath(); g.moveTo(fx, y0); g.lineTo(fx, y1); g.stroke();
    var fp = XY(ph, s.samp[Math.min(N - 1, Math.round(ph * (N - 1)))]);
    g.fillStyle = "#fcf7be"; g.beginPath(); g.arc(fp[0], fp[1], 2.6 * dpr, 0, 7); g.fill();
  }
  function hexA(hex, a) {
    var h = hex.replace("#", "");
    var r = parseInt(h.slice(0, 2), 16), gg = parseInt(h.slice(2, 4), 16), bb = parseInt(h.slice(4, 6), 16);
    return "rgba(" + r + "," + gg + "," + bb + "," + a + ")";
  }
  function drawFinale(ph) { for (var i = 0; i < 5; i++) drawMini("whyfin" + i, i, ph); }
  function drawSixth(t) {
    var c = $("whyfin_six"); if (!c) return; sizeCanvas(c, 16 / 11);
    var g = c.getContext("2d"); if (!g) return;
    var dpr = Math.min(window.devicePixelRatio || 1, 2), W = c.width, H = c.height, pad = 8 * dpr;
    var x0 = pad, x1 = W - pad, y0 = pad, y1 = H - pad, PW = x1 - x0, PH = y1 - y0, s = SCALES[4];
    g.clearRect(0, 0, W, H);
    var XY = function (u, v) { return [x0 + u * PW, y1 - clamp01(v) * PH]; };
    g.strokeStyle = "#333a45"; g.lineWidth = 1 * dpr; g.strokeRect(x0, y0, PW, PH);
    g.strokeStyle = s.col; g.lineWidth = 2 * dpr; g.beginPath();
    for (var j = 0; j < N; j++) { var uu = j / (N - 1), pp = XY(uu, s.samp[j]); if (j === 0) g.moveTo(pp[0], pp[1]); else g.lineTo(pp[0], pp[1]); }
    g.stroke();
    // "you are here" — the top of the campaign curve, gently pulsing (Opus 4.8, this session)
    var mu = 0.9, mp = XY(mu, s.samp[Math.round(mu * (N - 1))]);
    var pr = (3.2 + 1.4 * (0.5 + 0.5 * Math.sin(t * 0.004))) * dpr;
    g.fillStyle = "rgba(70,211,105,0.22)"; g.beginPath(); g.arc(mp[0], mp[1], pr * 2.4, 0, 7); g.fill();
    g.fillStyle = "#8affb0"; g.beginPath(); g.arc(mp[0], mp[1], pr, 0, 7); g.fill();
    g.fillStyle = "#cfe9d6"; g.font = (10 * dpr) + "px ui-sans-serif,system-ui,sans-serif";
    g.fillText("you are here · Opus 4.8", mp[0] - 116 * dpr, mp[1] - 8 * dpr);
    g.fillStyle = "#6b7482"; g.fillText("campaign front — η = operator steering", x0 + 6 * dpr, y1 - 6 * dpr);
  }

  // ---------- animation loop ----------
  function tick(ts) {
    raf = 0;
    if (!visible()) { lastT = 0; return; }   // stop when hidden; re-activation restarts the loop
    if (!lastT) lastT = ts;
    var dt = Math.min(80, ts - lastT); lastT = ts;
    if (fld && !frontPaused) { kppStep(); kppToBuf(); kppRender(); }
    else if (fld) { kppRender(); }
    if (!finPaused) { phase += dt * 0.00028; if (phase > 1) phase -= 1; drawFinale(phase); }
    if (sixOn) drawSixth(ts);
    schedule();
  }
  function schedule() { if (!raf) raf = window.requestAnimationFrame(tick); }

  // ---------- wiring ----------
  function wire() {
    var sl = $("whyscale");
    if (sl) sl.addEventListener("input", function () { scaleS = parseFloat(sl.value); updateScaleMeta(); drawScaleChart(); });
    var host = $("whyscalejump");
    if (host) Array.prototype.forEach.call(host.querySelectorAll(".sg"), function (el) {
      el.addEventListener("click", function () {
        scaleS = parseInt(el.dataset.s, 10) || 0; if (sl) sl.value = String(scaleS);
        updateScaleMeta(); drawScaleChart();
      });
    });
    var pf = $("whytog_play");
    if (pf) { var tf = function () { frontPaused = !frontPaused; pf.classList.toggle("on", !frontPaused); pf.innerHTML = frontPaused ? "&#9654; play front" : "&#9614;&#9614; pause front"; };
      pf.addEventListener("click", tf); pf.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); tf(); } }); }
    var pfi = $("whytog_finplay");
    if (pfi) { var tfi = function () { finPaused = !finPaused; pfi.classList.toggle("on", !finPaused); pfi.innerHTML = finPaused ? "&#9654; play" : "&#9614;&#9614; pause"; };
      pfi.addEventListener("click", tfi); pfi.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); tfi(); } }); }
    var zm = $("whytog_zoom");
    if (zm) { var tz = function () { sixOn = !sixOn; zm.classList.toggle("on", sixOn); var box = $("whyfinsix"); if (box) box.classList.toggle("on", sixOn);
        zm.innerHTML = sixOn ? "&larr; zoom back in" : "zoom out once more &rarr;"; if (sixOn) drawSixth(performance.now ? performance.now() : Date.now()); };
      zm.addEventListener("click", tz); zm.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); tz(); } }); }
    window.addEventListener("resize", function () { if (visible()) { drawScaleChart(); if (sixOn) drawSixth(performance.now ? performance.now() : Date.now()); } });
  }

  function spineActivate() {
    if (!inited) {
      inited = true;
      wire();
      kppInit();
      updateScaleMeta();
      drawScaleChart();
      var start = (navigator.gpu && !/[?&]whyhow2d=1/.test(location.search || "")) ?
        (function () { kTriedGpu = true; return kppInitGPU().then(function () { kgpuOk = true; }).catch(function () { kgpuOk = false; }); })() :
        Promise.resolve();
      start.then(function () {
        if (!kgpuOk) kppInitFallback();
        setSpineBadge();
        var st = $("whyspine_status");
        if (st) st.textContent = (kgpuOk ? "WebGPU" : "canvas2d") + " · front live · pointer 0.19110 UNMOVED";
        schedule();
      });
    } else {
      lastT = 0; drawScaleChart(); schedule();
    }
  }

  // chain: run Pass 1 (field bundle) AND the spine/finale
  window.__whyhowActivate = function () {
    if (prevActivate) { try { prevActivate(); } catch (e) {} }
    try { spineActivate(); } catch (e) {}
  };
})();
