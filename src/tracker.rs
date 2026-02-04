use ndarray::{ArrayView3};

use crate::postprocess::{hann2d, process_outputs, TrackingResult};
use crate::preprocess::{crop_and_preprocess, BBox};
use crate::rknn::{RknnError, RknnModel};

/// VitTrack configuration
#[derive(Debug, Clone)]
pub struct VitTrackConfig {
    pub template_size: usize,
    pub search_size: usize,
    pub score_size: usize,
    pub template_factor: u32,
    pub search_factor: u32,
    pub score_threshold: f32,
}

impl Default for VitTrackConfig {
    fn default() -> Self {
        Self {
            template_size: 128,
            search_size: 256,
            score_size: 16,
            template_factor: 2,
            search_factor: 4,
            score_threshold: 0.25,
        }
    }
}

/// VitTrack tracker using RKNN
pub struct VitTrack {
    config: VitTrackConfig,
    model: RknnModel,
    hanning: Vec<f32>,
    template: Option<Vec<f32>>,
    rect_last: [i32; 4],
}

impl VitTrack {
    /// Create new VitTrack tracker
    ///
    /// # Arguments
    /// * `model_path` - Path to RKNN model file
    pub fn new<P: AsRef<std::path::Path>>(model_path: P) -> Result<Self, RknnError> {
        Self::with_config(model_path, VitTrackConfig::default())
    }

    /// Create new VitTrack tracker with custom config
    pub fn with_config<P: AsRef<std::path::Path>>(
        model_path: P,
        config: VitTrackConfig,
    ) -> Result<Self, RknnError> {
        let model = RknnModel::load(model_path)?;
        let hanning = hann2d(config.score_size, config.score_size);

        Ok(Self {
            config,
            model,
            hanning,
            template: None,
            rect_last: [0, 0, 0, 0],
        })
    }

    /// Initialize tracker with bounding box
    ///
    /// # Arguments
    /// * `image` - Input image as Array3<u8> in HWC BGR format
    /// * `bbox` - Initial bounding box
    pub fn init(&mut self, image: &ArrayView3<u8>, bbox: BBox) {
        self.rect_last = bbox.to_array();

        let (template, _crop_size) = crop_and_preprocess(
            image,
            &bbox,
            self.config.template_factor,
            self.config.template_size,
        );

        self.template = Some(template);
    }

    /// Initialize tracker with raw bounding box values
    pub fn init_with_rect(&mut self, image: &ArrayView3<u8>, x: i32, y: i32, w: i32, h: i32) {
        self.init(image, BBox::new(x, y, w, h));
    }

    /// Track object in new frame
    ///
    /// # Arguments
    /// * `image` - Input image as Array3<u8> in HWC BGR format
    ///
    /// # Returns
    /// * Tracking result with bounding box and score
    pub fn update(&mut self, image: &ArrayView3<u8>) -> Result<TrackingResult, RknnError> {
        let template = match &self.template {
            Some(t) => t,
            None => {
                return Ok(TrackingResult::default());
            }
        };

        let bbox = BBox::from_array(&self.rect_last);

        let (search, crop_size) = crop_and_preprocess(
            image,
            &bbox,
            self.config.search_factor,
            self.config.search_size,
        );

        // Run RKNN inference
        let outputs = self.model.inference(template, &search)?;

        // Process outputs
        let result = process_outputs(
            &outputs.conf_map,
            &outputs.size_map,
            &outputs.offset_map,
            &self.hanning,
            &mut self.rect_last,
            crop_size,
            self.config.score_threshold,
        );

        Ok(result)
    }

    /// Get current bounding box
    pub fn get_bbox(&self) -> [i32; 4] {
        self.rect_last
    }

    /// Check if tracker is initialized
    pub fn is_initialized(&self) -> bool {
        self.template.is_some()
    }
}