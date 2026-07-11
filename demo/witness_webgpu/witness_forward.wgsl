// witness_forward.wgsl — Kernel B (the level-set witness FORWARD) as a WebGPU compute shader.
//
// Faithful port of tac ... levelset_sdf_argmax_mlx (the trainer's own GPU twin):
//     h   = act(feats @ in_proj.W^T + in_proj.b)
//     for li in 0..N_HIDDEN:
//         pre = (h @ hidden[li].W^T + hidden[li].b) * (1 + film[li,0]) + film[li,1]
//         h   = act(pre)
//     phi = h @ out_sdf.W^T + out_sdf.b
//     partition = argmax_k phi_k        // the class partition = level-set argmax
//
// The FiLM vector (film[li,0..1] per hidden unit) is precomputed per-frame on the host
// (code @ film.W^T + film.b) and uploaded as `film`, so the shader is per-PIXEL work only.
// One invocation per pixel. Activation is hosc: tanh(beta*sin(omega*u)) (or wire / relu).
//
// AUTHORITY: [WebGPU/WebNN demo — NON-AUTHORITY]. WGSL fp32; the numpy-fp32 reference is the
// bit-identical CPU authority; a browser "d_seg" here is a VISUALIZATION, never a contest score.
//
// Dimension constants (IN_FEAT / HIDDEN / N_HIDDEN / N_CLASSES / ACT_KIND / activation params)
// are injected by the host from fixture.meta at pipeline-build time (the {{...}} placeholders).

const IN_FEAT   : u32 = {{IN_FEAT}}u;
const HIDDEN    : u32 = {{HIDDEN}}u;
const N_HIDDEN  : u32 = {{N_HIDDEN}}u;
const N_CLASSES : u32 = {{N_CLASSES}}u;
const ACT_KIND  : u32 = {{ACT_KIND}}u;      // 0 = hosc, 1 = wire, 2 = relu
const A0 : f32 = {{A0}};                     // hosc beta  / wire w0
const A1 : f32 = {{A1}};                     // hosc omega / wire s0

@group(0) @binding(0) var<storage, read>       feats     : array<f32>; // P * IN_FEAT
@group(0) @binding(1) var<storage, read>       in_w      : array<f32>; // HIDDEN * IN_FEAT
@group(0) @binding(2) var<storage, read>       in_b      : array<f32>; // HIDDEN
@group(0) @binding(3) var<storage, read>       hid_w     : array<f32>; // N_HIDDEN * HIDDEN * HIDDEN
@group(0) @binding(4) var<storage, read>       hid_b     : array<f32>; // N_HIDDEN * HIDDEN
@group(0) @binding(5) var<storage, read>       film      : array<f32>; // N_HIDDEN * 2 * HIDDEN  (per frame)
@group(0) @binding(6) var<storage, read>       out_w     : array<f32>; // N_CLASSES * HIDDEN
@group(0) @binding(7) var<storage, read>       out_b     : array<f32>; // N_CLASSES
@group(0) @binding(8) var<storage, read_write> partition : array<u32>; // P
@group(0) @binding(9) var<uniform>             dims      : vec4<u32>;  // (P, _, _, _)

fn act(u : f32) -> f32 {
    if (ACT_KIND == 0u) { return tanh(A0 * sin(A1 * u)); }   // hosc
    if (ACT_KIND == 1u) { return cos(A0 * u) * exp(-((A1 * u) * (A1 * u))); } // wire
    return max(u, 0.0);                                       // relu
}

// Fixed-capacity locals. HIDDEN is small (96); keep two buffers for the ping-pong.
var<private> h  : array<f32, {{HIDDEN}}>;
var<private> hn : array<f32, {{HIDDEN}}>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let pix : u32 = gid.x;
    if (pix >= dims.x) { return; }
    let fbase : u32 = pix * IN_FEAT;

    // layer 0: in_proj (IN_FEAT -> HIDDEN) + act
    for (var o : u32 = 0u; o < HIDDEN; o = o + 1u) {
        var acc : f32 = in_b[o];
        let wbase : u32 = o * IN_FEAT;
        for (var i : u32 = 0u; i < IN_FEAT; i = i + 1u) {
            acc = acc + feats[fbase + i] * in_w[wbase + i];
        }
        h[o] = act(acc);
    }

    // FiLM-modulated hidden stack
    for (var li : u32 = 0u; li < N_HIDDEN; li = li + 1u) {
        let wl : u32 = li * HIDDEN * HIDDEN;
        let bl : u32 = li * HIDDEN;
        let sc : u32 = li * 2u * HIDDEN;             // scale row  (film[li,0])
        let sh : u32 = li * 2u * HIDDEN + HIDDEN;    // shift row  (film[li,1])
        for (var o : u32 = 0u; o < HIDDEN; o = o + 1u) {
            var acc : f32 = hid_b[bl + o];
            let wbase : u32 = wl + o * HIDDEN;
            for (var i : u32 = 0u; i < HIDDEN; i = i + 1u) {
                acc = acc + h[i] * hid_w[wbase + i];
            }
            let scale : f32 = 1.0 + film[sc + o];
            let shift : f32 = film[sh + o];
            hn[o] = act(acc * scale + shift);
        }
        for (var o : u32 = 0u; o < HIDDEN; o = o + 1u) { h[o] = hn[o]; }
    }

    // out_sdf (HIDDEN -> N_CLASSES) + argmax
    var best_k : u32 = 0u;
    var best_v : f32 = -3.4e38;
    for (var k : u32 = 0u; k < N_CLASSES; k = k + 1u) {
        var acc : f32 = out_b[k];
        let wbase : u32 = k * HIDDEN;
        for (var o : u32 = 0u; o < HIDDEN; o = o + 1u) {
            acc = acc + h[o] * out_w[wbase + o];
        }
        if (acc > best_v) { best_v = acc; best_k = k; }
    }
    partition[pix] = best_k;
}
