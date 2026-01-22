//! Device and data type utilities.

use candle_core::{Device as CandleDevice, DType as CandleDType};

/// Re-export candle Device for convenience
pub type Device = CandleDevice;

/// Re-export candle DType for convenience
pub type DType = CandleDType;

/// Get the best available device (CUDA > Metal > CPU)
pub fn get_device(device_id: Option<usize>) -> crate::Result<Device> {
    #[cfg(feature = "cuda")]
    {
        let id = device_id.unwrap_or(0);
        match CandleDevice::new_cuda(id) {
            Ok(device) => {
                tracing::info!("Using CUDA device {}", id);
                return Ok(device);
            }
            Err(e) => {
                tracing::warn!("CUDA not available: {}", e);
            }
        }
    }

    #[cfg(feature = "metal")]
    {
        match CandleDevice::new_metal(device_id.unwrap_or(0)) {
            Ok(device) => {
                tracing::info!("Using Metal device");
                return Ok(device);
            }
            Err(e) => {
                tracing::warn!("Metal not available: {}", e);
            }
        }
    }

    tracing::info!("Using CPU device");
    Ok(CandleDevice::Cpu)
}

/// Parse dtype string to candle DType
pub fn parse_dtype(dtype_str: &str) -> CandleDType {
    match dtype_str.to_lowercase().as_str() {
        "f32" | "float32" | "float" => CandleDType::F32,
        "f16" | "float16" | "half" => CandleDType::F16,
        "bf16" | "bfloat16" => CandleDType::BF16,
        _ => {
            tracing::warn!("Unknown dtype '{}', defaulting to F16", dtype_str);
            CandleDType::F16
        }
    }
}

/// Check if a device is CPU
pub fn is_cpu(device: &Device) -> bool {
    matches!(device, Device::Cpu)
}

/// Check if a device is CUDA
#[cfg(feature = "cuda")]
pub fn is_cuda(device: &Device) -> bool {
    matches!(device, Device::Cuda(_))
}

#[cfg(not(feature = "cuda"))]
pub fn is_cuda(_device: &Device) -> bool {
    false
}
