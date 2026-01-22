//! 4-bit NormalFloat (NF4) quantization.
//!
//! NF4 is an information-theoretically optimal data type for normally distributed
//! weights. It provides 4-bit quantization with minimal accuracy loss for LLMs.
//!
//! Reference: QLoRA paper (https://arxiv.org/abs/2305.14314)

use candle_core::{DType, Device, Tensor};
use super::{Quantizer, QuantizedTensor, tensor::QuantType};

/// The 16 NF4 quantization levels (normalized to [-1, 1])
/// These are optimally spaced for normally distributed data
const NF4_QUANT_LEVELS: [f32; 16] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
];

/// NF4 quantizer for 4-bit weight compression
pub struct Nf4Quantizer {
    /// Block size for quantization (number of elements per scale)
    block_size: usize,
}

impl Nf4Quantizer {
    /// Create a new NF4 quantizer with default block size
    pub fn new() -> Self {
        Self { block_size: 64 }
    }

    /// Create with custom block size
    pub fn with_block_size(block_size: usize) -> Self {
        assert!(block_size > 0 && block_size % 2 == 0, "Block size must be positive and even");
        Self { block_size }
    }

    /// Find the closest NF4 level index for a normalized value
    fn quantize_value(normalized: f32) -> u8 {
        let mut best_idx = 0u8;
        let mut best_dist = f32::MAX;

        for (i, &level) in NF4_QUANT_LEVELS.iter().enumerate() {
            let dist = (normalized - level).abs();
            if dist < best_dist {
                best_dist = dist;
                best_idx = i as u8;
            }
        }

        best_idx
    }

    /// Pack two 4-bit values into one byte
    fn pack_nibbles(high: u8, low: u8) -> u8 {
        ((high & 0x0F) << 4) | (low & 0x0F)
    }

    /// Unpack one byte into two 4-bit values
    fn unpack_nibbles(byte: u8) -> (u8, u8) {
        ((byte >> 4) & 0x0F, byte & 0x0F)
    }
}

impl Default for Nf4Quantizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Quantizer for Nf4Quantizer {
    fn quantize(&self, tensor: &Tensor) -> crate::Result<QuantizedTensor> {
        // Flatten tensor and convert to f32
        let shape: Vec<usize> = tensor.dims().to_vec();
        let flat = tensor.flatten_all()?.to_dtype(DType::F32)?;
        let data: Vec<f32> = flat.to_vec1()?;
        let numel = data.len();

        // Calculate number of blocks
        let num_blocks = (numel + self.block_size - 1) / self.block_size;
        let mut scales = Vec::with_capacity(num_blocks);
        let mut quantized_data = Vec::with_capacity((numel + 1) / 2);

        // Process each block
        for block_idx in 0..num_blocks {
            let start = block_idx * self.block_size;
            let end = (start + self.block_size).min(numel);
            let block = &data[start..end];

            // Find absmax for this block
            let absmax = block.iter()
                .map(|x| x.abs())
                .fold(0.0f32, f32::max)
                .max(1e-10); // Avoid division by zero

            scales.push(absmax);

            // Quantize each value in the block
            let mut pending: Option<u8> = None;

            for &val in block {
                let normalized = val / absmax; // Now in [-1, 1]
                let idx = Self::quantize_value(normalized);

                match pending.take() {
                    Some(prev) => {
                        // Pack two values
                        quantized_data.push(Self::pack_nibbles(prev, idx));
                    }
                    None => {
                        pending = Some(idx);
                    }
                }
            }

            // Handle odd block size (pad with zero)
            if let Some(last) = pending {
                quantized_data.push(Self::pack_nibbles(last, 0));
            }
        }

        Ok(QuantizedTensor::new(
            quantized_data,
            scales,
            None, // NF4 doesn't use zero points
            shape,
            DType::F16,
            self.block_size,
            QuantType::Nf4,
        ))
    }

    fn dequantize(&self, quantized: &QuantizedTensor) -> crate::Result<Tensor> {
        let numel = quantized.numel();
        let mut output = Vec::with_capacity(numel);

        let mut byte_idx = 0;
        let mut block_idx = 0;
        let mut in_block = 0;

        while output.len() < numel {
            let scale = quantized.scales[block_idx];

            // Unpack byte
            let (high, low) = Self::unpack_nibbles(quantized.data[byte_idx]);
            byte_idx += 1;

            // First value (high nibble)
            let val1 = NF4_QUANT_LEVELS[high as usize] * scale;
            output.push(val1);
            in_block += 1;

            if in_block >= self.block_size {
                block_idx += 1;
                in_block = 0;
            }

            // Second value (low nibble) if we haven't reached numel
            if output.len() < numel {
                let val2 = NF4_QUANT_LEVELS[low as usize] * scale;
                output.push(val2);
                in_block += 1;

                if in_block >= self.block_size {
                    block_idx += 1;
                    in_block = 0;
                }
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
    fn test_nf4_roundtrip() {
        let quantizer = Nf4Quantizer::new();

        // Create a test tensor
        let data = vec![0.1f32, -0.5, 0.8, -0.2, 0.0, 0.3, -0.7, 0.9];
        let tensor = Tensor::from_vec(data.clone(), &[8], &Device::Cpu).unwrap();

        // Quantize and dequantize
        let quantized = quantizer.quantize(&tensor).unwrap();
        let restored = quantizer.dequantize(&quantized).unwrap();

        // Check shape preserved
        assert_eq!(restored.dims(), tensor.dims());

        // Check values are approximately correct (quantization introduces error)
        let restored_data: Vec<f32> = restored.to_dtype(DType::F32).unwrap().to_vec1().unwrap();
        for (orig, rest) in data.iter().zip(restored_data.iter()) {
            let error = (orig - rest).abs();
            assert!(error < 0.3, "Error too large: {} vs {} (diff: {})", orig, rest, error);
        }
    }

    #[test]
    fn test_compression_ratio() {
        let quantizer = Nf4Quantizer::new();

        let data: Vec<f32> = (0..1024).map(|i| (i as f32) / 1024.0 - 0.5).collect();
        let tensor = Tensor::from_vec(data, &[1024], &Device::Cpu).unwrap();

        let quantized = quantizer.quantize(&tensor).unwrap();

        // 4-bit = 0.5 bytes per element, plus scale overhead
        // Original: 1024 * 2 bytes (f16) = 2048 bytes
        // Compressed: 512 bytes data + 64 bytes scales = 576 bytes
        let ratio = quantized.compression_ratio();
        assert!(ratio < 0.35, "Compression ratio should be ~0.25-0.3, got {}", ratio);
    }
}
