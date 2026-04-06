#[derive(Debug, Clone)]
pub struct RoiPatch {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
    pub bytes: Vec<u8>,
}

pub fn apply_roi_patch(
    frame: &mut [u8],
    frame_w: usize,
    frame_h: usize,
    channels: usize,
    patch: &RoiPatch,
) -> Result<(), String> {
    if channels == 0 {
        return Err("channels must be > 0".into());
    }
    if patch.x + patch.w > frame_w || patch.y + patch.h > frame_h {
        return Err("patch out of bounds".into());
    }
    let expected = patch.w * patch.h * channels;
    if patch.bytes.len() != expected {
        return Err("patch payload length mismatch".into());
    }

    for row in 0..patch.h {
        let dst_row_start = ((patch.y + row) * frame_w + patch.x) * channels;
        let dst_row_end = dst_row_start + patch.w * channels;
        let src_row_start = row * patch.w * channels;
        let src_row_end = src_row_start + patch.w * channels;
        frame[dst_row_start..dst_row_end].copy_from_slice(&patch.bytes[src_row_start..src_row_end]);
    }

    Ok(())
}
