/* FLOW (Tab 3) — client-side WebGPU interactive renderer of the witness level-set field.
 *
 * Inlined into the dashboard page by tools/dashboard_server.py::_flow_client_js (CSP-strict:
 * self-contained, no external <script src>, no CDN). Reads the field data the server ships over
 * the SAME WebSocket as base64 (window.__flow, populated by _pushFlow in the main script):
 *   latest = { w,h,downsample,classes,pairs:[{pair_idx,our_dseg,margin_vmax,
 *              our_margin_b64(f16), our_argmax_b64(u8), gt_argmax_b64(u8)}] }
 *   history = rolling last-N pair-0 frames { epoch, our_dseg, our_margin_b64, our_argmax_b64 }
 *
 * The SegNet margin (top1-top2 logit gap) is the scorer's own sensitivity field; small margin =
 * the codim-1 separatrix where the argmax can flip = where d_seg lives. FLOW uploads margin +
 * argmax as GPU textures and shades them with an inline WGSL fragment shader, driven by sliders;
 * the epoch scrubber animates the rolling history = the level-set FLOWING toward the argmax
 * partition as the witness trains. A canvas2d path renders the identical colormap where WebGPU is
 * absent (progressive enhancement). Force the fallback for testing with ?flow2d=1.
 *
 * AUTHORITY: [macOS-CPU advisory · NON-PROMOTABLE] imagery. A viz moves no pointer (0.19110).
 */
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };

  // ---- control state (bound to shader uniforms / 2d params) ----
  var mode = 0;        // 0 heat · 1 partition · 2 blend · 3 disagreement
  var iso = -1;        // class isolation: -1 = all, else class index
  var thr = 0.55;      // margin-fragility threshold (isolates the fragile band)
  var blend = 0.5;     // partition <-> heat blend
  var curPair = 0;     // displayed pair
  var scrubIdx = -1;   // -1 = live (follow newest); >=0 = pinned history index
  var playing = false, rafId = null, playAccum = 0, lastT = 0;

  var inited = false, wired = false, gpuOk = false, gpuLost = false;
  var CANVAS = null;

  // ---- decode cache (avoid re-decoding the same field on every slider tweak) ----
  var _dkey = null, _dmBytes = null, _dmF32 = null, _dA = null, _dG = null;
  var PAL = [[64, 32, 32], [255, 0, 0], [128, 128, 96], [0, 255, 102], [204, 0, 255]];

  // ---------- small helpers ----------
  function b64ToU8(b64) {
    var s = atob(b64), n = s.length, u = new Uint8Array(n);
    for (var i = 0; i < n; i++) u[i] = s.charCodeAt(i);
    return u;
  }
  function half2float(h) {
    var s = (h & 0x8000) >> 15, e = (h & 0x7C00) >> 10, f = h & 0x03FF;
    if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 0x1F) return f ? NaN : ((s ? -1 : 1) * Infinity);
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  function hexToRgb(h) {
    h = String(h || "").replace("#", "");
    if (h.length !== 6) return [128, 128, 128];
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  }
  // inferno colormap polynomial fit (Matt Zucker) -> [r,g,b] in 0..1
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

  // ---------- field selection from window.__flow ----------
  function currentField() {
    var F = window.__flow || {}, L = F.latest;
    if (!L || !Array.isArray(L.pairs) || !L.pairs.length) return null;
    var pairs = L.pairs;
    var p0 = pairs.reduce(function (a, b) { return a.pair_idx < b.pair_idx ? a : b; });
    var base = null;
    for (var i = 0; i < pairs.length; i++) if (pairs[i].pair_idx === curPair) base = pairs[i];
    if (!base) base = p0;
    var W = L.w, H = L.h, classes = L.classes || [], vmax = base.margin_vmax || 1;
    var hist = F.history || [];
    var scrubOn = (curPair === p0.pair_idx && hist.length > 0);
    if (scrubOn) {
      var idx = (scrubIdx < 0 || scrubIdx >= hist.length) ? hist.length - 1 : scrubIdx;
      var h = hist[idx];
      return { W: W, H: H, classes: classes, vmax: vmax, marginB64: h.our_margin_b64,
        argB64: h.our_argmax_b64, gtB64: base.gt_argmax_b64, epoch: h.epoch, dseg: h.our_dseg,
        idx: idx, len: hist.length, scrubOn: true, live: (idx === hist.length - 1) };
    }
    return { W: W, H: H, classes: classes, vmax: vmax, marginB64: base.our_margin_b64,
      argB64: base.our_argmax_b64, gtB64: base.gt_argmax_b64, epoch: L.epoch, dseg: base.our_dseg,
      idx: 0, len: 1, scrubOn: false, live: true };
  }
  function decodeField(field) {
    var key = field.marginB64 + "|" + field.argB64 + "|" + field.gtB64;
    if (key === _dkey) return;
    _dmBytes = b64ToU8(field.marginB64);
    var u16 = new Uint16Array(_dmBytes.buffer, _dmBytes.byteOffset, _dmBytes.byteLength >> 1);
    _dmF32 = new Float32Array(u16.length);
    for (var i = 0; i < u16.length; i++) _dmF32[i] = half2float(u16[i]);
    _dA = b64ToU8(field.argB64);
    _dG = b64ToU8(field.gtB64);
    _dkey = key;
  }

  // ---------- canvas sizing (field aspect, dpr-crisp) ----------
  function sizeCanvas(W, H) {
    var c = CANVAS, cssW = c.clientWidth || 512, dpr = Math.min(window.devicePixelRatio || 1, 2);
    var pxW = Math.max(1, Math.round(cssW * dpr)), pxH = Math.max(1, Math.round(pxW * H / W));
    if (c.width !== pxW || c.height !== pxH) { c.width = pxW; c.height = pxH; }
  }

  // ---------- WebGPU path ----------
  var dev = null, gctx = null, fmt = null, pipe = null, uni = null, bgl = null;
  var texM = null, texA = null, texG = null, texW = 0, texH = 0;

  var WGSL = [
    "struct U { dims:vec2f, threshold:f32, mode:f32, classIso:f32, marginVmax:f32, blend:f32, pad:f32, };",
    "@group(0) @binding(0) var<uniform> u : U;",
    "@group(0) @binding(1) var texM : texture_2d<f32>;",
    "@group(0) @binding(2) var texA : texture_2d<u32>;",
    "@group(0) @binding(3) var texG : texture_2d<u32>;",
    "struct VSOut { @builtin(position) pos:vec4f, @location(0) uv:vec2f, };",
    "@vertex fn vs(@builtin(vertex_index) i:u32) -> VSOut {",
    "  var p = array<vec2f,3>(vec2f(-1.0,-1.0), vec2f(3.0,-1.0), vec2f(-1.0,3.0));",
    "  var o:VSOut; o.pos = vec4f(p[i],0.0,1.0);",
    "  var uv = 0.5*(p[i]+vec2f(1.0,1.0)); uv.y = 1.0-uv.y; o.uv = uv; return o;",
    "}",
    "fn palette(c:i32) -> vec3f {",
    "  switch(c){",
    "    case 0: { return vec3f(0.251,0.125,0.125); }",
    "    case 1: { return vec3f(1.0,0.0,0.0); }",
    "    case 2: { return vec3f(0.502,0.502,0.376); }",
    "    case 3: { return vec3f(0.0,1.0,0.4); }",
    "    case 4: { return vec3f(0.8,0.0,1.0); }",
    "    default: { return vec3f(0.5,0.5,0.5); }",
    "  }",
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
    "fn loadM(ip:vec2i) -> f32 {",
    "  let d = vec2i(textureDimensions(texM));",
    "  let q = clamp(ip, vec2i(0,0), d - vec2i(1,1));",
    "  return textureLoad(texM, q, 0).r;",
    "}",
    "@fragment fn fs(inp:VSOut) -> @location(0) vec4f {",
    "  let dims = u.dims;",
    "  let fp = inp.uv*dims - vec2f(0.5,0.5);",
    "  let ip = vec2i(floor(fp));",
    "  let fr = fp - floor(fp);",
    "  let m = mix(mix(loadM(ip),loadM(ip+vec2i(1,0)),fr.x), mix(loadM(ip+vec2i(0,1)),loadM(ip+vec2i(1,1)),fr.x), fr.y);",
    "  let frag = clamp(1.0 - m/max(u.marginVmax,1e-6), 0.0, 1.0);",
    "  let ic = clamp(vec2i(floor(inp.uv*dims)), vec2i(0,0), vec2i(dims)-vec2i(1,1));",
    "  let cls = i32(textureLoad(texA, ic, 0).r);",
    "  let gt = i32(textureLoad(texG, ic, 0).r);",
    "  let pc = palette(cls);",
    "  let md = i32(round(u.mode));",
    "  var col = vec3f(0.0,0.0,0.0);",
    "  if (md == 0) { let g = select(frag*0.22, frag, frag >= u.threshold); col = inferno(g); }",
    "  else if (md == 1) { col = pc; }",
    "  else if (md == 2) { col = mix(pc, inferno(frag), clamp(u.blend,0.0,1.0)); }",
    "  else { let dis = cls != gt; col = select(pc*0.32, vec3f(1.0,0.16,0.16), dis); }",
    "  if ((md == 1 || md == 2) && frag >= u.threshold) {",
    "    let k = 0.55*(frag-u.threshold)/max(1e-3, 1.0-u.threshold);",
    "    col = mix(col, vec3f(1.0,0.95,0.6), clamp(k,0.0,0.9));",
    "  }",
    "  let isoc = i32(round(u.classIso));",
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
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "uint" } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "uint" } }
      ] });
      var pl = dev.createPipelineLayout({ bindGroupLayouts: [bgl] });
      pipe = dev.createRenderPipeline({
        layout: pl,
        vertex: { module: mod, entryPoint: "vs" },
        fragment: { module: mod, entryPoint: "fs", targets: [{ format: fmt }] },
        primitive: { topology: "triangle-list" }
      });
      uni = dev.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    });
  }
  function ensureTex(W, H) {
    if (texW === W && texH === H && texM) return;
    [texM, texA, texG].forEach(function (t) { if (t && t.destroy) t.destroy(); });
    var TB = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST;
    texM = dev.createTexture({ size: [W, H], format: "r16float", usage: TB });
    texA = dev.createTexture({ size: [W, H], format: "r8uint", usage: TB });
    texG = dev.createTexture({ size: [W, H], format: "r8uint", usage: TB });
    texW = W; texH = H;
  }
  function renderGPU(field) {
    var W = field.W, H = field.H;
    ensureTex(W, H);
    dev.queue.writeTexture({ texture: texM }, _dmBytes, { bytesPerRow: W * 2, rowsPerImage: H }, { width: W, height: H });
    dev.queue.writeTexture({ texture: texA }, _dA, { bytesPerRow: W, rowsPerImage: H }, { width: W, height: H });
    dev.queue.writeTexture({ texture: texG }, _dG, { bytesPerRow: W, rowsPerImage: H }, { width: W, height: H });
    dev.queue.writeBuffer(uni, 0, new Float32Array([W, H, thr, mode, iso, field.vmax, blend, 0]));
    sizeCanvas(W, H);
    var bg = dev.createBindGroup({ layout: bgl, entries: [
      { binding: 0, resource: { buffer: uni } },
      { binding: 1, resource: texM.createView() },
      { binding: 2, resource: texA.createView() },
      { binding: 3, resource: texG.createView() }
    ] });
    var enc = dev.createCommandEncoder();
    var view = gctx.getCurrentTexture().createView();
    var pass = enc.beginRenderPass({ colorAttachments: [{ view: view,
      clearValue: { r: 0.05, g: 0.06, b: 0.08, a: 1 }, loadOp: "clear", storeOp: "store" }] });
    pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.draw(3); pass.end();
    dev.queue.submit([enc.finish()]);
  }

  // ---------- canvas2d fallback path ----------
  var ctx2d = null, off2d = null, off2dCtx = null;
  function ensure2dOff(W, H) {
    if (!off2d) { off2d = document.createElement("canvas"); off2dCtx = off2d.getContext("2d"); }
    if (off2d.width !== W || off2d.height !== H) { off2d.width = W; off2d.height = H; }
  }
  function render2d(field) {
    var W = field.W, H = field.H;
    ensure2dOff(W, H);
    var img = off2dCtx.createImageData(W, H), d = img.data, vmax = Math.max(field.vmax, 1e-6);
    for (var i = 0; i < W * H; i++) {
      var m = _dmF32[i], frag = Math.min(1, Math.max(0, 1 - m / vmax)), cls = _dA[i], gt = _dG[i], rgb;
      if (mode === 0) { rgb = inferno(frag >= thr ? frag : frag * 0.22); }
      else if (mode === 1) { var p1 = PAL[cls] || [128, 128, 128]; rgb = [p1[0] / 255, p1[1] / 255, p1[2] / 255]; }
      else if (mode === 2) {
        var p2 = PAL[cls] || [128, 128, 128], h2 = inferno(frag), b = Math.min(1, Math.max(0, blend));
        rgb = [p2[0] / 255 * (1 - b) + h2[0] * b, p2[1] / 255 * (1 - b) + h2[1] * b, p2[2] / 255 * (1 - b) + h2[2] * b];
      } else {
        var p3 = PAL[cls] || [128, 128, 128];
        rgb = (cls !== gt) ? [1, 0.16, 0.16] : [p3[0] / 255 * 0.32, p3[1] / 255 * 0.32, p3[2] / 255 * 0.32];
      }
      if ((mode === 1 || mode === 2) && frag >= thr) {
        var k = Math.min(0.9, 0.55 * (frag - thr) / Math.max(1e-3, 1 - thr));
        rgb = [rgb[0] * (1 - k) + 0.98 * k, rgb[1] * (1 - k) + 0.95 * k, rgb[2] * (1 - k) + 0.6 * k];
      }
      if (iso >= 0 && cls !== iso) { rgb = [rgb[0] * 0.12, rgb[1] * 0.12, rgb[2] * 0.12]; }
      var o = i * 4;
      d[o] = Math.round(rgb[0] * 255); d[o + 1] = Math.round(rgb[1] * 255);
      d[o + 2] = Math.round(rgb[2] * 255); d[o + 3] = 255;
    }
    off2dCtx.putImageData(img, 0, 0);
    sizeCanvas(W, H);
    ctx2d.imageSmoothingEnabled = true;
    ctx2d.clearRect(0, 0, CANVAS.width, CANVAS.height);
    ctx2d.drawImage(off2d, 0, 0, W, H, 0, 0, CANVAS.width, CANVAS.height);
  }
  function initFallback() {
    // a canvas can hold only one context type; if a webgpu context was already attached, clone
    // the node to get a clean one for 2d.
    var old = $("flowcanvas");
    var n = old.cloneNode(false);
    old.parentNode.replaceChild(n, old);
    CANVAS = n;
    ctx2d = CANVAS.getContext("2d");
  }

  // ---------- UI ----------
  var MODE_LABELS = ["margin heat", "argmax partition", "blend", "disagreement"];
  function setBadge() {
    var b = $("flowbadge"); if (!b) return;
    if (gpuOk && !gpuLost) { b.className = "flowbadge gpu"; b.textContent = "WebGPU"; }
    else { b.className = "flowbadge cpu"; b.textContent = "canvas2d fallback"; }
  }
  function showMsg(t) { var m = $("flowmsg"); if (m) { m.classList.remove("hide"); m.textContent = t; } }
  function hideMsg() { var m = $("flowmsg"); if (m) m.classList.add("hide"); }

  function refreshControls() {
    var F = window.__flow || {}, L = F.latest;
    // pair dropdown
    var ps = $("flowpair");
    if (ps && L && Array.isArray(L.pairs)) {
      var want = L.pairs.map(function (p) { return p.pair_idx; }).sort(function (a, b) { return a - b; });
      var have = Array.prototype.map.call(ps.options, function (o) { return parseInt(o.value, 10); });
      if (want.join(",") !== have.join(",")) {
        ps.innerHTML = "";
        want.forEach(function (pi) {
          var o = document.createElement("option"); o.value = String(pi); o.textContent = "pair " + pi;
          ps.appendChild(o);
        });
        if (want.indexOf(curPair) < 0) curPair = want[0];
        ps.value = String(curPair);
      }
    }
    // legend from classes
    var lg = $("flowlegend");
    if (lg && L && Array.isArray(L.classes) && L.classes.length && lg.dataset.n !== String(L.classes.length)) {
      PAL = L.classes.map(function (c) { return hexToRgb(c.hex); });
      lg.innerHTML = "";
      var all = document.createElement("span"); all.className = "lc" + (iso < 0 ? " on" : "");
      all.dataset.cls = "-1"; all.innerHTML = "<span class='dot' style='background:transparent;border-style:dashed'></span>all";
      lg.appendChild(all);
      L.classes.forEach(function (c) {
        var el = document.createElement("span"); el.className = "lc" + (iso === c.i ? " on" : "");
        el.dataset.cls = String(c.i);
        el.innerHTML = "<span class='dot' style='background:" + c.hex + "'></span>" + c.label;
        lg.appendChild(el);
      });
      lg.dataset.n = String(L.classes.length);
      Array.prototype.forEach.call(lg.querySelectorAll(".lc"), function (el) {
        el.addEventListener("click", function () { setIso(parseInt(el.dataset.cls, 10)); });
      });
    }
    // scrub range
    var sc = $("flowscrub"), F2 = window.__flow || {}, hist = F2.history || [];
    var p0 = (L && L.pairs && L.pairs.length) ? L.pairs.reduce(function (a, b) { return a.pair_idx < b.pair_idx ? a : b; }).pair_idx : 0;
    var scrubOn = (curPair === p0 && hist.length > 1);
    if (sc) {
      sc.max = String(Math.max(0, hist.length - 1));
      sc.disabled = !scrubOn;
      if (scrubIdx < 0) sc.value = String(Math.max(0, hist.length - 1));
    }
    var pb = $("flowplay"); if (pb) pb.disabled = !scrubOn;
    syncLegendActive();
  }
  function syncLegendActive() {
    var lg = $("flowlegend"); if (!lg) return;
    Array.prototype.forEach.call(lg.querySelectorAll(".lc"), function (el) {
      el.classList.toggle("on", parseInt(el.dataset.cls, 10) === iso);
    });
  }
  function setIso(v) {
    iso = v; var s = $("flowiso"); if (s) s.value = String(v);
    var lab = $("flowiso_v"); if (lab) lab.textContent = (v < 0 ? "all classes" : (PAL_LABEL(v)));
    syncLegendActive(); renderNow();
  }
  function PAL_LABEL(v) {
    var L = (window.__flow || {}).latest;
    if (L && L.classes && L.classes[v]) return L.classes[v].label;
    return "class " + v;
  }
  function setScrubUI(idx) {
    var sc = $("flowscrub"); if (sc) sc.value = String(idx);
    var lab = $("flowscrub_v"); var hist = (window.__flow || {}).history || [];
    if (lab) {
      if (!hist.length) { lab.textContent = "live"; return; }
      var live = (idx === hist.length - 1);
      var ep = hist[idx] ? hist[idx].epoch : "?";
      lab.textContent = (live ? "live · ep " + ep : "ep " + ep) + " (" + (idx + 1) + "/" + hist.length + ")";
    }
  }

  function renderNow() {
    var field = currentField();
    if (!field) {
      showMsg((window.__flow || {}).status === "error"
        ? "witness render error — see server log"
        : "waiting for the first checkpoint field… (renders on each new #205 checkpoint)");
      updateStatus(null); return;
    }
    setScrubUI(field.scrubOn ? field.idx : ((window.__flow || {}).history || []).length - 1);
    updateStatus(field);
    if (!flowVisible() || !CANVAS || CANVAS.clientWidth === 0) return; // draw only when shown
    try {
      decodeField(field);
      if (gpuOk && !gpuLost) renderGPU(field); else render2d(field);
      hideMsg();
    } catch (e) {
      // WebGPU hiccup (device lost / driver) -> fall back to 2d for the rest of the session
      if (gpuOk) { gpuOk = false; try { initFallback(); } catch (e2) {} setBadge(); }
      try { decodeField(field); render2d(field); hideMsg(); }
      catch (e3) { showMsg("field render error: " + (e3 && e3.message || e3)); }
    }
  }
  function updateStatus(field) {
    var el = $("flowstatus"); if (!el) return;
    var L = (window.__flow || {}).latest;
    if (!field) { el.textContent = "no field yet"; return; }
    var dims = (field.W && field.H) ? (field.H + "×" + field.W) : "?";
    var path = (gpuOk && !gpuLost) ? "WebGPU" : "canvas2d";
    var hist = (window.__flow || {}).history || [];
    el.textContent = "ep " + (field.epoch != null ? field.epoch : "?") +
      " · d_seg " + fmtSig(field.dseg, 5) + " · pair " + curPair +
      " · " + dims + " · " + path + " · " + hist.length + " frame" + (hist.length === 1 ? "" : "s");
  }

  // ---------- play loop (epoch scrubber animation) ----------
  function playStep(t) {
    if (!playing) { rafId = null; return; }
    if (!lastT) lastT = t;
    playAccum += (t - lastT); lastT = t;
    var hist = (window.__flow || {}).history || [];
    if (hist.length > 1 && playAccum > 380) {
      playAccum = 0;
      var idx = (scrubIdx < 0) ? hist.length - 1 : scrubIdx;
      idx = (idx + 1) % hist.length;
      scrubIdx = idx;
      renderNow();
    }
    rafId = requestAnimationFrame(playStep);
  }
  function togglePlay() {
    var hist = (window.__flow || {}).history || [];
    if (hist.length < 2) return;
    playing = !playing;
    var b = $("flowplay");
    if (b) { b.classList.toggle("on", playing); b.setAttribute("aria-pressed", String(playing));
      b.innerHTML = playing ? "▍▍ pause" : "▶ play"; }
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
      var v = $("flowmode_v"); if (v) v.textContent = MODE_LABELS[mode] || "";
      renderNow();
    });
    var is = $("flowiso");
    if (is) is.addEventListener("change", function () { setIso(parseInt(is.value, 10)); });
    var tr = $("flowthr");
    if (tr) tr.addEventListener("input", function () {
      thr = parseFloat(tr.value); var v = $("flowthr_v"); if (v) v.textContent = thr.toFixed(2); renderNow();
    });
    var bl = $("flowblend");
    if (bl) bl.addEventListener("input", function () {
      blend = parseFloat(bl.value); var v = $("flowblend_v"); if (v) v.textContent = blend.toFixed(2); renderNow();
    });
    var ps = $("flowpair");
    if (ps) ps.addEventListener("change", function () {
      curPair = parseInt(ps.value, 10) || 0; scrubIdx = -1; stopPlay();
      var v = $("flowpair_v"); if (v) v.textContent = String(curPair);
      refreshControls(); renderNow();
    });
    var sc = $("flowscrub");
    if (sc) sc.addEventListener("input", function () {
      scrubIdx = parseInt(sc.value, 10) || 0; stopPlay(); renderNow();
    });
    var pl = $("flowplay");
    if (pl) pl.addEventListener("click", togglePlay);
    window.addEventListener("resize", function () { if (flowVisible()) renderNow(); });
  }

  // ---------- public hooks (called by the main script) ----------
  function activate() {
    if (inited) { refreshControls(); renderNow(); return; }
    inited = true;
    wireControls();
    var force2d = /[?&]flow2d=1/.test(location.search || "");
    var start = (!force2d && navigator.gpu) ? initGPU().then(function () { gpuOk = true; })
      .catch(function () { gpuOk = false; }) : Promise.resolve();
    start.then(function () {
      if (!gpuOk) initFallback();
      setBadge();
      // initial control labels
      var v;
      v = $("flowmode_v"); if (v) v.textContent = MODE_LABELS[mode];
      v = $("flowthr_v"); if (v) v.textContent = thr.toFixed(2);
      v = $("flowblend_v"); if (v) v.textContent = blend.toFixed(2);
      refreshControls();
      renderNow();
    });
  }
  function onData() {
    if (!inited) return;               // not activated yet -> nothing to draw
    refreshControls();
    if (!playing && scrubIdx < 0) renderNow();   // live-follow the newest checkpoint
    else updateStatus(currentField()); // pinned/playing: keep the status fresh, don't yank the view
  }

  window.__flowActivate = activate;
  window.__flowNotify = onData;
})();
