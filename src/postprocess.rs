/// Tracking result
#[derive(Debug, Clone, Copy)]
pub struct TrackingResult {
    pub success: bool,
    pub bbox: [i32; 4], // [x, y, w, h]
    pub score: f32,
}

impl Default for TrackingResult {
    fn default() -> Self {
        Self {
            success: false,
            bbox: [0, 0, 0, 0],
            score: 0.0,
        }
    }
}

/// Create 1D Hanning window (matching OpenCV implementation)
pub fn hann1d(size: usize) -> Vec<f32> {
    let mut window = vec![0.0f32; size];
    let pi = std::f32::consts::PI;

    for i in 0..size {
        window[i] = 0.5 * (1.0 - (2.0 * pi / (size + 1) as f32 * (i + 1) as f32).cos());
    }

    window
}

/// Create 2D Hanning window as flat array
pub fn hann2d(rows: usize, cols: usize) -> Vec<f32> {
    let hann_rows = hann1d(rows);
    let hann_cols = hann1d(cols);

    let mut window = vec![0.0f32; rows * cols];

    for r in 0..rows {
        for c in 0..cols {
            window[r * cols + c] = hann_rows[r] * hann_cols[c];
        }
    }

    window
}

/// Process model outputs
///
/// # Arguments
/// * `conf_map` - Confidence map (256 elements, 16x16)
/// * `size_map` - Size map (512 elements, 2x16x16)
/// * `offset_map` - Offset map (512 elements, 2x16x16)
/// * `hanning` - Hanning window (256 elements, 16x16)
/// * `rect_last` - Previous bounding box [x, y, w, h]
/// * `crop_size` - Crop size in original image pixels
/// * `threshold` - Score threshold
///
/// # Returns
/// * Tracking result with updated bounding box
pub fn process_outputs(
    conf_map: &[f32],
    size_map: &[f32],
    offset_map: &[f32],
    hanning: &[f32],
    rect_last: &mut [i32; 4],
    crop_size: i32,
    threshold: f32,
) -> TrackingResult {
    const SCORE_SIZE: usize = 16;

    // Apply Hanning window
    let mut conf_windowed = vec![0.0f32; SCORE_SIZE * SCORE_SIZE];
    for i in 0..conf_windowed.len() {
        conf_windowed[i] = conf_map[i] * hanning[i];
    }

    // Find max location
    let (max_idx, max_score) = find_max(&conf_windowed);
    let max_loc_y = max_idx / SCORE_SIZE;
    let max_loc_x = max_idx % SCORE_SIZE;

    if max_score >= threshold {
        // Get predictions at max location
        // offset_map layout: [2, 16, 16] -> index = channel * 256 + y * 16 + x
        let offset_x = offset_map[0 * 256 + max_loc_y * SCORE_SIZE + max_loc_x];
        let offset_y = offset_map[1 * 256 + max_loc_y * SCORE_SIZE + max_loc_x];

        // size_map layout: [2, 16, 16]
        let size_w = size_map[0 * 256 + max_loc_y * SCORE_SIZE + max_loc_x];
        let size_h = size_map[1 * 256 + max_loc_y * SCORE_SIZE + max_loc_x];

        // Normalized coordinates [0, 1]
        let cx = (max_loc_x as f32 + offset_x) / SCORE_SIZE as f32;
        let cy = (max_loc_y as f32 + offset_y) / SCORE_SIZE as f32;

        // Update rectangle
        update_rect(rect_last, cx, cy, size_w, size_h, crop_size);

        TrackingResult {
            success: true,
            bbox: *rect_last,
            score: max_score,
        }
    } else {
        TrackingResult {
            success: false,
            bbox: *rect_last,
            score: max_score,
        }
    }
}

/// Find maximum value and its index
fn find_max(arr: &[f32]) -> (usize, f32) {
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;

    for (idx, &val) in arr.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = idx;
        }
    }

    (max_idx, max_val)
}

/// Update rectangle based on predictions (matching OpenCV logic)
fn update_rect(rect: &mut [i32; 4], cx: f32, cy: f32, w: f32, h: f32, crop_size: i32) {
    // Origin of crop in original image
    let x0 = rect[0] + (rect[2] - crop_size) / 2;
    let y0 = rect[1] + (rect[3] - crop_size) / 2;

    // Convert normalized coords to image coords
    let x1 = cx - w / 2.0;
    let y1 = cy - h / 2.0;

    rect[0] = (x1 * crop_size as f32 + x0 as f32).floor() as i32;
    rect[1] = (y1 * crop_size as f32 + y0 as f32).floor() as i32;
    rect[2] = (w * crop_size as f32).floor() as i32;
    rect[3] = (h * crop_size as f32).floor() as i32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann1d() {
        let window = hann1d(16);
        assert_eq!(window.len(), 16);
        // Window should be symmetric
        for i in 0..8 {
            assert!((window[i] - window[15 - i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_hann2d() {
        let window = hann2d(16, 16);
        assert_eq!(window.len(), 256);
    }

    #[test]
    fn test_find_max() {
        let arr = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let (idx, val) = find_max(&arr);
        assert_eq!(idx, 3);
        assert!((val - 0.9).abs() < 1e-6);
    }
}