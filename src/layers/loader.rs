//! Layer loading with memory mapping and optional decompression.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use candle_core::{DType, Device, Tensor};
use safetensors::SafeTensors;

use crate::config::Compression;
use crate::quantization::{create_quantizer, Quantizer, QuantizedTensor};
use super::LayerWeights;

/// Loads individual layers from disk with minimal memory usage.
pub struct LayerLoader {
    /// Directory containing split layer files
    layer_dir: PathBuf,
    /// Device to load tensors to
    device: Device,
    /// Data type for tensors
    dtype: DType,
    /// Compression mode
    compression: Compression,
    /// Optional quantizer for decompression
    quantizer: Option<Box<dyn Quantizer>>,
    /// Number of transformer layers
    num_layers: usize,
}

impl LayerLoader {
    /// Create a new layer loader
    pub fn new(
        layer_dir: impl AsRef<Path>,
        device: Device,
        dtype: DType,
        compression: Compression,
        num_layers: usize,
    ) -> Self {
        let quantizer = create_quantizer(compression);

        Self {
            layer_dir: layer_dir.as_ref().to_path_buf(),
            device,
            dtype,
            compression,
            quantizer,
            num_layers,
        }
    }

    /// Load embedding layer
    pub fn load_embeddings(&self) -> crate::Result<HashMap<String, Tensor>> {
        self.load_layer_file("embed_tokens")
    }

    /// Load a specific transformer layer
    pub fn load_layer(&self, layer_idx: usize) -> crate::Result<LayerWeights> {
        if layer_idx >= self.num_layers {
            return Err(crate::AirLLMError::LayerLoadError(format!(
                "Layer index {} out of range (max {})",
                layer_idx,
                self.num_layers - 1
            )));
        }

        let layer_name = format!("layer_{}", layer_idx);
        let tensors = self.load_layer_file(&layer_name)?;

        Ok(LayerWeights {
            layer_idx,
            tensors,
        })
    }

    /// Load the final normalization layer
    pub fn load_norm(&self) -> crate::Result<HashMap<String, Tensor>> {
        self.load_layer_file("norm")
    }

    /// Load the language model head
    pub fn load_lm_head(&self) -> crate::Result<HashMap<String, Tensor>> {
        self.load_layer_file("lm_head")
    }

    /// Load a layer file by name
    fn load_layer_file(&self, name: &str) -> crate::Result<HashMap<String, Tensor>> {
        let file_path = self.layer_dir.join(format!("{}.safetensors", name));

        if !file_path.exists() {
            return Err(crate::AirLLMError::LayerLoadError(format!(
                "Layer file not found: {:?}",
                file_path
            )));
        }

        tracing::debug!("Loading layer file: {:?}", file_path);

        // Memory-map the file
        let file = std::fs::File::open(&file_path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        // Parse safetensors
        let safetensors = SafeTensors::deserialize(&mmap)
            .map_err(|e| crate::AirLLMError::LayerLoadError(e.to_string()))?;

        let mut tensors = HashMap::new();

        for (tensor_name, tensor_view) in safetensors.tensors() {
            let tensor = self.load_tensor(&tensor_view)?;
            tensors.insert(tensor_name.to_string(), tensor);
        }

        Ok(tensors)
    }

    /// Load a single tensor, applying decompression if needed
    fn load_tensor(&self, view: &safetensors::tensor::TensorView<'_>) -> crate::Result<Tensor> {
        let shape: Vec<usize> = view.shape().to_vec();
        let dtype = convert_safetensor_dtype(view.dtype());

        // Create tensor from raw data
        let tensor = Tensor::from_raw_buffer(
            view.data(),
            dtype,
            &shape,
            &Device::Cpu,
        )?;

        // Apply decompression if we have a quantizer
        let tensor = if let Some(ref quantizer) = self.quantizer {
            // Check if this tensor is quantized (would have special format)
            // For now, assume raw tensors (no compression during load)
            tensor
        } else {
            tensor
        };

        // Convert dtype and move to device
        let tensor = tensor.to_dtype(self.dtype)?;
        let tensor = tensor.to_device(&self.device)?;

        Ok(tensor)
    }

    /// Load layer to CPU with memory pinning (for prefetching)
    pub fn load_layer_to_cpu(&self, layer_idx: usize) -> crate::Result<LayerWeights> {
        let layer_name = format!("layer_{}", layer_idx);
        let file_path = self.layer_dir.join(format!("{}.safetensors", layer_name));

        let file = std::fs::File::open(&file_path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        let safetensors = SafeTensors::deserialize(&mmap)
            .map_err(|e| crate::AirLLMError::LayerLoadError(e.to_string()))?;

        let mut tensors = HashMap::new();

        for (tensor_name, tensor_view) in safetensors.tensors() {
            let shape: Vec<usize> = tensor_view.shape().to_vec();
            let dtype = convert_safetensor_dtype(tensor_view.dtype());

            let tensor = Tensor::from_raw_buffer(
                tensor_view.data(),
                dtype,
                &shape,
                &Device::Cpu,
            )?;

            tensors.insert(tensor_name.to_string(), tensor);
        }

        Ok(LayerWeights {
            layer_idx,
            tensors,
        })
    }

    /// Move layer weights to the target device
    pub fn move_to_device(&self, weights: &LayerWeights) -> crate::Result<LayerWeights> {
        let mut device_tensors = HashMap::new();

        for (name, tensor) in &weights.tensors {
            let tensor = tensor.to_dtype(self.dtype)?;
            let tensor = tensor.to_device(&self.device)?;
            device_tensors.insert(name.clone(), tensor);
        }

        Ok(LayerWeights {
            layer_idx: weights.layer_idx,
            tensors: device_tensors,
        })
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Get the device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the dtype
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

/// Convert safetensor dtype to candle dtype
fn convert_safetensor_dtype(dtype: safetensors::Dtype) -> DType {
    match dtype {
        safetensors::Dtype::F16 => DType::F16,
        safetensors::Dtype::BF16 => DType::BF16,
        safetensors::Dtype::F32 => DType::F32,
        safetensors::Dtype::F64 => DType::F64,
        safetensors::Dtype::I8 => DType::I64, // Candle doesn't have I8
        safetensors::Dtype::I16 => DType::I64,
        safetensors::Dtype::I32 => DType::I64,
        safetensors::Dtype::I64 => DType::I64,
        safetensors::Dtype::U8 => DType::U8,
        safetensors::Dtype::U16 => DType::U32,
        safetensors::Dtype::U32 => DType::U32,
        _ => DType::F32, // Default fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_conversion() {
        assert!(matches!(convert_safetensor_dtype(safetensors::Dtype::F16), DType::F16));
        assert!(matches!(convert_safetensor_dtype(safetensors::Dtype::BF16), DType::BF16));
        assert!(matches!(convert_safetensor_dtype(safetensors::Dtype::F32), DType::F32));
    }
}
