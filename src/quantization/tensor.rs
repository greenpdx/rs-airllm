//! Quantized tensor representation.

use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

/// A quantized tensor that stores compressed weights.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Compressed data bytes
    pub data: Vec<u8>,

    /// Quantization scale factors (per block)
    pub scales: Vec<f32>,

    /// Zero points for asymmetric quantization (optional)
    pub zero_points: Option<Vec<f32>>,

    /// Original tensor shape
    pub shape: Vec<usize>,

    /// Original dtype
    pub dtype: DType,

    /// Block size used for quantization
    pub block_size: usize,

    /// Quantization type
    pub quant_type: QuantType,
}

/// Type of quantization applied
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantType {
    /// 4-bit NormalFloat (NF4)
    Nf4,
    /// 8-bit integer
    Int8,
}

impl QuantizedTensor {
    /// Create a new quantized tensor
    pub fn new(
        data: Vec<u8>,
        scales: Vec<f32>,
        zero_points: Option<Vec<f32>>,
        shape: Vec<usize>,
        dtype: DType,
        block_size: usize,
        quant_type: QuantType,
    ) -> Self {
        Self {
            data,
            scales,
            zero_points,
            shape,
            dtype,
            block_size,
            quant_type,
        }
    }

    /// Get the number of elements in the original tensor
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get compressed size in bytes
    pub fn compressed_size(&self) -> usize {
        self.data.len() + self.scales.len() * 4 + self.zero_points.as_ref().map_or(0, |zp| zp.len() * 4)
    }

    /// Get original size in bytes (assuming f16)
    pub fn original_size(&self) -> usize {
        self.numel() * 2 // f16 = 2 bytes
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        self.compressed_size() as f32 / self.original_size() as f32
    }

    /// Serialize to bytes for storage
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Header: quant_type (1 byte), block_size (4 bytes), shape length (4 bytes)
        bytes.push(match self.quant_type {
            QuantType::Nf4 => 0,
            QuantType::Int8 => 1,
        });
        bytes.extend_from_slice(&(self.block_size as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.shape.len() as u32).to_le_bytes());

        // Shape
        for dim in &self.shape {
            bytes.extend_from_slice(&(*dim as u64).to_le_bytes());
        }

        // Scales length and data
        bytes.extend_from_slice(&(self.scales.len() as u32).to_le_bytes());
        for scale in &self.scales {
            bytes.extend_from_slice(&scale.to_le_bytes());
        }

        // Zero points (optional)
        if let Some(ref zps) = self.zero_points {
            bytes.push(1); // has zero points
            bytes.extend_from_slice(&(zps.len() as u32).to_le_bytes());
            for zp in zps {
                bytes.extend_from_slice(&zp.to_le_bytes());
            }
        } else {
            bytes.push(0); // no zero points
        }

        // Data length and bytes
        bytes.extend_from_slice(&(self.data.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&self.data);

        bytes
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> crate::Result<Self> {
        let mut pos = 0;

        // Header
        let quant_type = match bytes[pos] {
            0 => QuantType::Nf4,
            1 => QuantType::Int8,
            _ => return Err(crate::AirLLMError::QuantizationError("Invalid quant type".to_string())),
        };
        pos += 1;

        let block_size = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        let shape_len = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        // Shape
        let mut shape = Vec::with_capacity(shape_len);
        for _ in 0..shape_len {
            shape.push(u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()) as usize);
            pos += 8;
        }

        // Scales
        let scales_len = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        let mut scales = Vec::with_capacity(scales_len);
        for _ in 0..scales_len {
            scales.push(f32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()));
            pos += 4;
        }

        // Zero points
        let has_zp = bytes[pos] != 0;
        pos += 1;

        let zero_points = if has_zp {
            let zp_len = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;

            let mut zps = Vec::with_capacity(zp_len);
            for _ in 0..zp_len {
                zps.push(f32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()));
                pos += 4;
            }
            Some(zps)
        } else {
            None
        };

        // Data
        let data_len = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;

        let data = bytes[pos..pos + data_len].to_vec();

        Ok(Self {
            data,
            scales,
            zero_points,
            shape,
            dtype: DType::F16, // Default, could be stored in header
            block_size,
            quant_type,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialization_roundtrip() {
        let qt = QuantizedTensor::new(
            vec![1, 2, 3, 4],
            vec![1.0, 2.0],
            Some(vec![0.5, 0.5]),
            vec![2, 2],
            DType::F16,
            64,
            QuantType::Nf4,
        );

        let bytes = qt.to_bytes();
        let restored = QuantizedTensor::from_bytes(&bytes).unwrap();

        assert_eq!(restored.data, qt.data);
        assert_eq!(restored.scales, qt.scales);
        assert_eq!(restored.zero_points, qt.zero_points);
        assert_eq!(restored.shape, qt.shape);
        assert_eq!(restored.block_size, qt.block_size);
        assert_eq!(restored.quant_type, qt.quant_type);
    }
}
