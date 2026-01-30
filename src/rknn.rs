use rknn_rs::prelude::{Rknn, RknnInput, RknnTensorFormat, RknnTensorType};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RknnError {
    #[error("Failed to load model: {0}")]
    LoadError(String),
    #[error("Failed to set inputs: {0}")]
    InputError(String),
    #[error("Failed to run inference: {0}")]
    RunError(String),
    #[error("Failed to get outputs: {0}")]
    OutputError(String),
}

/// RKNN Model outputs for VitTrack
#[derive(Debug)]
pub struct VitTrackOutputs {
    /// Confidence map (1x1x16x16 = 256 elements)
    pub conf_map: Vec<f32>,
    /// Size map (1x2x16x16 = 512 elements)
    pub size_map: Vec<f32>,
    /// Offset map (1x2x16x16 = 512 elements)
    pub offset_map: Vec<f32>,
}

/// RKNN Model wrapper for VitTrack
pub struct RknnModel {
    rknn: Rknn,
}

impl RknnModel {
    /// Load RKNN model from file
    pub fn load<P: AsRef<std::path::Path>>(model_path: P) -> Result<Self, RknnError> {
        let rknn = Rknn::rknn_init(model_path)
            .map_err(|e| RknnError::LoadError(e.to_string()))?;

        Ok(Self { rknn })
    }

    /// Run inference with template and search inputs
    ///
    /// # Arguments
    /// * `template` - Template input as NHWC float32 (1x128x128x3)
    /// * `search` - Search input as NHWC float32 (1x256x256x3)
    ///
    /// # Returns
    /// * VitTrackOutputs containing conf_map, size_map, offset_map
    pub fn inference(
        &self,
        template: &[f32],
        search: &[f32],
    ) -> Result<VitTrackOutputs, RknnError> {
        // Create inputs
        let mut inputs = vec![
            RknnInput {
                index: 0,
                buf: template.to_vec(),
                pass_through: false,
                type_: RknnTensorType::Float32,
                fmt: RknnTensorFormat::NHWC,
            },
            RknnInput {
                index: 1,
                buf: search.to_vec(),
                pass_through: false,
                type_: RknnTensorType::Float32,
                fmt: RknnTensorFormat::NHWC,
            },
        ];

        // Set inputs
        self.rknn
            .inputs_set(&mut inputs)
            .map_err(|e| RknnError::InputError(e.to_string()))?;

        // Run inference
        self.rknn
            .run()
            .map_err(|e| RknnError::RunError(e.to_string()))?;

        // Get outputs (3 outputs for VitTrack)
        let outputs = self.rknn
            .outputs_get::<f32>(3)
            .map_err(|e| RknnError::OutputError(e.to_string()))?;

        // Extract output data
        let conf_map = outputs[0].to_vec();
        let size_map = outputs[1].to_vec();
        let offset_map = outputs[2].to_vec();

        Ok(VitTrackOutputs {
            conf_map,
            size_map,
            offset_map,
        })
    }
}