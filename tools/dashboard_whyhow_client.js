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
