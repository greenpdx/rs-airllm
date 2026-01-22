//! Quantization module for 4-bit and 8-bit weight compression.
//!
//! Implements quantization schemes compatible with bitsandbytes:
//! - NF4 (4-bit NormalFloat) for aggressive compression
//! - Int8 blockwise for moderate compression with better accuracy

mod nf4;
mod int8;
mod tensor;

pub use nf4::Nf4Quantizer;
pub use int8::Int8Quantizer;
pub use tensor::QuantizedTensor;

use crate::config::Compression;

/// Trait for quantization implementations
pub trait Quantizer: Send + Sync {
    /// Quantize a tensor to compressed format
    fn quantize(&self, tensor: &candle_core::Tensor) -> crate::Result<QuantizedTensor>;

    /// Dequantize back to full precision
    fn dequantize(&self, quantized: &QuantizedTensor) -> crate::Result<candle_core::Tensor>;
}

/// Create a quantizer based on compression mode
pub fn create_quantizer(compression: Compression) -> Option<Box<dyn Quantizer>> {
    match compression {
        Compression::None => None,
        Compression::Nf4 => Some(Box::new(Nf4Quantizer::new())),
        Compression::Int8 => Some(Box::new(Int8Quantizer::new())),
    }
}
