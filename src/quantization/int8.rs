//! 8-bit blockwise quantization.
//!
//! Implements symmetric 8-bit quantization with per-block scaling factors.
//! This provides a good balance between compression (2x) and accuracy.

use candle_core::{DType, Device, Tensor};
use super::{Quantizer, QuantizedTensor, tensor::QuantType};

/// Int8 blockwise quantizer
pub struct Int8Quantizer {
    /// Block size for quantization
    block_size: usize,
}

impl Int8Quantizer {
    /// Create a new Int8 quantizer with default block size
    pub fn new() -> Self {
        Self { block_size: 64 }
    }

    /// Create with custom block size
    pub fn with_block_size(block_size: usize) -> Self {
        assert!(block_size > 0, "Block size must be positive");
        Self { block_size }
    }
}

impl Default for Int8Quantizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Quantizer for Int8Quantizer {
    fn quantize(&self, tensor: &Tensor) -> crate::Result<QuantizedTensor> {
        // Flatten tensor and convert to f32
        let shape: Vec<usize> = tensor.dims().to_vec();
        let flat = tensor.flatten_all()?.to_dtype(DType::F32)?;
        let data: Vec<f32> = flat.to_vec1()?;
        let numel = data.len();

        // Calculate number of blocks
        let num_blocks = (numel + self.block_size - 1) / self.block_size;
        let mut scales = Vec::with_capacity(num_blocks);
        let mut quantized_data = Vec::with_capacity(numel);

        // Process each block
        for block_idx in 0..num_blocks {
            let start = block_idx * self.block_size;
            let end = (start + self.block_size).min(numel);
            let block = &data[start..end];

            // Find absmax for symmetric quantization
            let absmax = block.iter()
                .map(|x| x.abs())
                .fold(0.0f32, f32::max)
                .max(1e-10); // Avoid division by zero

            // Scale factor: absmax / 127 (symmetric int8 range is -127 to 127)
            let scale = absmax / 127.0;
            scales.push(scale);

            // Quantize each value
            for &val in block {
                let quantized = (val / scale).round().clamp(-127.0, 127.0) as i8;
                quantized_data.push(quantized as u8);
            }
        }

        Ok(QuantizedTensor::new(
            quantized_data,
            scales,
            None, // Symmetric quantization doesn't need zero points
            shape,
            DType::F16,
            self.block_size,
            QuantType::Int8,
        ))
    }

    fn dequantize(&self, quantized: &QuantizedTensor) -> crate::Result<Tensor> {
        let numel = quantized.numel();
        let mut output = Vec::with_capacity(numel);

        for (block_idx, scale) in quantized.scales.iter().enumerate() {
            let start = block_idx * quantized.block_size;
            let end = (start + quantized.block_size).min(numel);

            for i in start..end {
                let quantized_val = quantized.data[i] as i8;
                let dequantized = (quantized_val as f32) * scale;
                output.push(dequantized);
            }
        }

        // Convert to tensor with original shape
        let tensor = Tensor::from_vec(output, quantized.shape.as_slice(), &Device::Cpu)?;
        tensor.to_dtype(quantized.dtype).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int8_roundtrip() {
        let quantizer = Int8Quantizer::new();

        // Create a test tensor
        let data = vec![0.1f32, -0.5, 0.8, -0.2, 0.0, 0.3, -0.7, 0.9];
        let tensor = Tensor::from_vec(data.clone(), &[8], &Device::Cpu).unwrap();

        // Quantize and dequantize
        let quantized = quantizer.quantize(&tensor).unwrap();
        let restored = quantizer.dequantize(&quantized).unwrap();

        // Check shape preserved
        assert_eq!(restored.dims(), tensor.dims());

        // Check values are approximately correct
        let restored_data: Vec<f32> = restored.to_dtype(DType::F32).unwrap().to_vec1().unwrap();
        for (orig, rest) in data.iter().zip(restored_data.iter()) {
            let error = (orig - rest).abs();
            // Int8 should have lower error than NF4
            assert!(error < 0.02, "Error too large: {} vs {} (diff: {})", orig, rest, error);
        }
    }

    #[test]
    fn test_int8_compression_ratio() {
        let quantizer = Int8Quantizer::new();

        let data: Vec<f32> = (0..1024).map(|i| (i as f32) / 1024.0 - 0.5).collect();
        let tensor = Tensor::from_vec(data, &[1024], &Device::Cpu).unwrap();

        let quantized = quantizer.quantize(&tensor).unwrap();

        // 8-bit = 1 byte per element, vs 2 bytes for f16
        // Plus scale overhead (4 bytes per 64 elements)
        let ratio = quantized.compression_ratio();
        assert!(ratio < 0.55, "Compression ratio should be ~0.5, got {}", ratio);
    }

    #[test]
    fn test_extreme_values() {
        let quantizer = Int8Quantizer::new();

        // Test with extreme values
        let data = vec![100.0f32, -100.0, 0.001, -0.001];
        let tensor = Tensor::from_vec(data.clone(), &[4], &Device::Cpu).unwrap();

        let quantized = quantizer.quantize(&tensor).unwrap();
        let restored = quantizer.dequantize(&quantized).unwrap();

        let restored_data: Vec<f32> = restored.to_dtype(DType::F32).unwrap().to_vec1().unwrap();

        // Large values should be preserved reasonably well
        assert!((restored_data[0] - 100.0).abs() < 1.0);
        assert!((restored_data[1] + 100.0).abs() < 1.0);
    }
}
