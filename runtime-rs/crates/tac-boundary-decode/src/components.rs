// SPDX-License-Identifier: MIT OR Apache-2.0
//! Connected components — native port of `tac.boundary_math.partition.connected_components`.
//!
//! Labels the 4-connected components of constant class in the partition `L*`,
//! producing the dense per-pixel `region_of` id map. This is the RAG node
//! structure the MDL region-merge solve operates on (region rasterize/fill at
//! the structural level: pixel → region id).
//!
//! # Bit-for-bit id-ordering contract
//!
//! The Python oracle iterates class `0..n_classes`, runs `scipy.ndimage.label`
//! per class (4-connectivity), and assigns a global contiguous id to each
//! component in the order `scipy.ndimage.label` returns them. Verified
//! empirically: `scipy.ndimage.label` numbers components in **first-pixel
//! raster-scan order**, identical to a raster-scan flood-fill. This port
//! reproduces that: for each class in `0..n_classes`, scan pixels in raster
//! order; on the first unvisited pixel of a component, assign the next global
//! id and flood-fill (4-connectivity). The resulting `region_of` int32 raster
//! is byte-for-byte equal to Python's (`connected_components_v1` golden vector).
//!
//! PAYLOAD CLEANLINESS: this reads an argmax label array from the caller. The
//! only constants are the 4-connectivity neighbour offsets and `n_classes` —
//! structural, not learned/video data.

use crate::{BoundaryDecodeError, Result};

/// Result of [`connected_components`]: the dense per-pixel region id map.
pub struct ConnectedComponents {
    /// `(H*W)` raster of region id per pixel (>= 0, contiguous, `i32`).
    pub region_of: Vec<i32>,
    /// Total number of regions found.
    pub n_regions: usize,
    pub height: usize,
    pub width: usize,
}

impl ConnectedComponents {
    /// Serialize `region_of` as int32 little-endian raster bytes — the exact
    /// byte layout the golden vector pins (`region_of.astype(np.int32).tobytes()`).
    pub fn region_of_i32_le_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.region_of.len() * 4);
        for &v in &self.region_of {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }
}

/// Label 4-connected components of constant class.
///
/// `argmax` is the `(H*W)` raster of `uint8` class labels; each label must be in
/// `[0, n_classes)`. Returns the `region_of` id map with ids assigned per the
/// Python oracle's class-then-first-pixel-raster order.
pub fn connected_components(
    argmax: &[u8],
    height: usize,
    width: usize,
    n_classes: usize,
) -> Result<ConnectedComponents> {
    if argmax.len() != height * width {
        return Err(BoundaryDecodeError::ShapeMismatch {
            expected: height * width,
            got: argmax.len(),
        });
    }
    let n = height * width;
    let mut region_of = vec![-1i32; n];
    let mut next_id: i32 = 0;
    // 4-connectivity neighbour offsets (row, col deltas) — structural constant.
    const NEIGH: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    // Iterate class 0..n_classes (matches the Python per-class loop), and within
    // each class raster-scan for unvisited component seeds.
    let mut stack: Vec<usize> = Vec::new();
    for c in 0..n_classes as i64 {
        for r in 0..height {
            for col in 0..width {
                let idx = r * width + col;
                if argmax[idx] as i64 != c || region_of[idx] != -1 {
                    continue;
                }
                // New component seed: assign id, flood-fill (iterative DFS).
                let id = next_id;
                next_id += 1;
                region_of[idx] = id;
                stack.clear();
                stack.push(idx);
                while let Some(p) = stack.pop() {
                    let pr = p / width;
                    let pc = p % width;
                    for (dr, dc) in NEIGH {
                        let ny = pr as isize + dr;
                        let nx = pc as isize + dc;
                        if ny < 0 || nx < 0 || ny >= height as isize || nx >= width as isize {
                            continue;
                        }
                        let nidx = ny as usize * width + nx as usize;
                        if region_of[nidx] == -1 && argmax[nidx] as i64 == c {
                            region_of[nidx] = id;
                            stack.push(nidx);
                        }
                    }
                }
            }
        }
    }
    // Fail closed if any pixel was unassigned (class id out of range), mirroring
    // the Python oracle's `(region_of < 0).any()` guard.
    if let Some(pos) = region_of.iter().position(|&v| v == -1) {
        return Err(BoundaryDecodeError::ClassOutOfRange {
            value: argmax[pos] as i64,
            n_classes,
        });
    }
    Ok(ConnectedComponents {
        region_of,
        n_regions: next_id as usize,
        height,
        width,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_class_one_region() {
        let a = vec![2u8; 12];
        let cc = connected_components(&a, 3, 4, 5).unwrap();
        assert_eq!(cc.n_regions, 1);
        assert!(cc.region_of.iter().all(|&v| v == 0));
    }

    #[test]
    fn raster_order_id_assignment() {
        // class 0 has two components; ids must follow first-pixel raster order.
        // layout (class ids):
        //   0 0 1 1
        //   0 1 1 1
        //   1 1 0 0
        let a: Vec<u8> = vec![0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0];
        let cc = connected_components(&a, 3, 4, 5).unwrap();
        // first class-0 component (top-left) gets id 0; second (bottom-right) id 1;
        // class-1 single component gets id 2.
        assert_eq!(cc.region_of[0], 0); // (0,0)
        assert_eq!(cc.region_of[10], 1); // (2,2) second class-0 comp
        assert_eq!(cc.region_of[2], 2); // (0,2) class-1
        assert_eq!(cc.n_regions, 3);
    }

    #[test]
    fn class_out_of_range_fails_closed() {
        let a = vec![0u8, 9u8]; // 9 >= n_classes=5 → never assigned
        assert!(connected_components(&a, 1, 2, 5).is_err());
    }
}
