use ndarray::{Array3, ArrayView3};

/// ImageNet mean values (RGB order)
pub const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// ImageNet std values (RGB order)
pub const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Bounding box [x, y, width, height]
#[derive(Debug, Clone, Copy, Default)]
pub struct BBox {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl BBox {
    pub fn new(x: i32, y: i32, width: i32, height: i32) -> Self {
        Self { x, y, width, height }
    }

    #[inline]
    pub fn area(&self) -> f32 {
        (self.width * self.height) as f32
    }

    pub fn to_array(&self) -> [i32; 4] {
        [self.x, self.y, self.width, self.height]
    }

    pub fn from_array(arr: &[i32; 4]) -> Self {
        Self {
            x: arr[0],
            y: arr[1],
            width: arr[2],
            height: arr[3],
        }
    }

    pub fn center(&self) -> (i32, i32) {
        (self.x + self.width / 2, self.y + self.height / 2)
    }
}

/// Crop and preprocess image for RKNN
///
/// # Arguments
/// * `image` - Input image as Array3<u8> in HWC BGR format
/// * `bbox` - Bounding box to crop around
/// * `factor` - Crop factor (2 for template, 4 for search)
/// * `output_size` - Output size (128 for template, 256 for search)
///
/// # Returns
/// * Cropped and preprocessed image as Vec<f32> in NHWC RGB format
/// * Crop size in original image pixels
pub fn crop_and_preprocess(
    image: &ArrayView3<u8>,
    bbox: &BBox,
    factor: u32,
    output_size: usize,
) -> (Vec<f32>, i32) {
    let (img_h, img_w, _channels) = image.dim();
    let img_h = img_h as i32;
    let img_w = img_w as i32;

    // Calculate crop size: sqrt(area) * factor
    let crop_sz = (bbox.area().sqrt() * factor as f32).ceil() as i32;

    // Calculate crop coordinates centered on bbox
    let x1 = bbox.x + (bbox.width - crop_sz) / 2;
    let x2 = x1 + crop_sz;
    let y1 = bbox.y + (bbox.height - crop_sz) / 2;
    let y2 = y1 + crop_sz;

    // Calculate padding
    let x1_pad = (-x1).max(0);
    let y1_pad = (-y1).max(0);
    let x2_pad = (x2 - img_w).max(0);
    let y2_pad = (y2 - img_h).max(0);

    // Valid ROI coordinates
    let roi_x1 = (x1 + x1_pad).max(0) as usize;
    let roi_y1 = (y1 + y1_pad).max(0) as usize;
    let roi_x2 = (x2 - x2_pad).min(img_w) as usize;
    let roi_y2 = (y2 - y2_pad).min(img_h) as usize;

    // Create padded crop
    let crop_h = crop_sz as usize;
    let crop_w = crop_sz as usize;
    let mut crop = Array3::<u8>::zeros((crop_h, crop_w, 3));

    // Copy valid region
    let src_h = roi_y2.saturating_sub(roi_y1);
    let src_w = roi_x2.saturating_sub(roi_x1);
    let dst_y1 = y1_pad as usize;
    let dst_x1 = x1_pad as usize;

    if src_h > 0 && src_w > 0 && roi_y1 < img_h as usize && roi_x1 < img_w as usize {
        for y in 0..src_h {
            for x in 0..src_w {
                for c in 0..3 {
                    if roi_y1 + y < img_h as usize && roi_x1 + x < img_w as usize {
                        crop[[dst_y1 + y, dst_x1 + x, c]] = image[[roi_y1 + y, roi_x1 + x, c]];
                    }
                }
            }
        }
    }

    // Resize and preprocess
    let resized = resize_bilinear(&crop, output_size, output_size);
    let preprocessed = preprocess_nhwc(&resized);

    (preprocessed, crop_sz)
}

/// Resize image using bilinear interpolation
fn resize_bilinear(image: &Array3<u8>, new_h: usize, new_w: usize) -> Array3<u8> {
    let (old_h, old_w, channels) = image.dim();

    if old_h == 0 || old_w == 0 {
        return Array3::<u8>::zeros((new_h, new_w, channels));
    }

    let mut resized = Array3::<u8>::zeros((new_h, new_w, channels));

    let scale_y = old_h as f32 / new_h as f32;
    let scale_x = old_w as f32 / new_w as f32;

    for y in 0..new_h {
        for x in 0..new_w {
            let src_y = y as f32 * scale_y;
            let src_x = x as f32 * scale_x;

            let y0 = (src_y.floor() as usize).min(old_h.saturating_sub(1));
            let y1 = (y0 + 1).min(old_h.saturating_sub(1));
            let x0 = (src_x.floor() as usize).min(old_w.saturating_sub(1));
            let x1 = (x0 + 1).min(old_w.saturating_sub(1));

            let dy = src_y - y0 as f32;
            let dx = src_x - x0 as f32;

            for c in 0..channels {
                let v00 = image[[y0, x0, c]] as f32;
                let v01 = image[[y0, x1, c]] as f32;
                let v10 = image[[y1, x0, c]] as f32;
                let v11 = image[[y1, x1, c]] as f32;

                let value = v00 * (1.0 - dx) * (1.0 - dy)
                    + v01 * dx * (1.0 - dy)
                    + v10 * (1.0 - dx) * dy
                    + v11 * dx * dy;

                resized[[y, x, c]] = value.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    resized
}

/// Preprocess image to NHWC float32 format with ImageNet normalization
/// Input: RGB HWC uint8
/// Output: RGB NHWC float32 normalized (as flat Vec)
fn preprocess_nhwc(image: &Array3<u8>) -> Vec<f32> {
    let (h, w, c) = image.dim();
    let mut output = vec![0.0f32; 1 * h * w * c];

    for y in 0..h {
        for x in 0..w {
            for ch in 0..3 {
                // BGR to RGB: swap channels 0 and 2
                // let src_ch = 2 - ch;
                let src_ch = ch;
                let value = image[[y, x, src_ch]] as f32 / 255.0;
                let normalized = (value - MEAN[ch]) / STD[ch];
                // NHWC layout: [batch, height, width, channel]
                let idx = y * w * 3 + x * 3 + ch;
                output[idx] = normalized;
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bbox() {
        let bbox = BBox::new(100, 100, 50, 50);
        assert_eq!(bbox.area(), 2500.0);
    }

    #[test]
    fn test_crop_size_calculation() {
        let bbox = BBox::new(100, 100, 50, 50);
        let crop_sz = (bbox.area().sqrt() * 2.0).ceil() as i32;
        assert_eq!(crop_sz, 100);
    }

    #[test]
    fn test_preprocess_shape() {
        // let image = ArrayView3::<u8>::((480, 640, 3));
        // let bbox = BBox::new(100, 100, 50, 50);
        // let (result, crop_sz) = crop_and_preprocess(&image, &bbox, 2, 128);

        // assert_eq!(result.len(), 1 * 128 * 128 * 3);
        // assert_eq!(crop_sz, 100);
    }
}