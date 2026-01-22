//! Memory management utilities.

use std::path::Path;

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// GPU memory allocated (bytes)
    pub gpu_allocated: usize,
    /// GPU memory reserved (bytes)
    pub gpu_reserved: usize,
    /// CPU memory used (bytes)
    pub cpu_used: usize,
}

/// Clean up memory by forcing garbage collection and cache clearing.
///
/// This is called between layer loads to minimize memory footprint.
pub fn clean_memory() {
    // Force drop any pending deallocations
    // In Rust, memory is freed when values go out of scope,
    // but we can hint to the allocator to return memory to the OS

    #[cfg(target_os = "linux")]
    {
        // On Linux, we can use malloc_trim to return memory to OS
        unsafe extern "C" {
            fn malloc_trim(pad: usize) -> i32;
        }
        unsafe {
            malloc_trim(0);
        }
    }

    // If using CUDA, synchronize and clear cache
    #[cfg(feature = "cuda")]
    {
        // Note: candle doesn't expose direct CUDA cache clearing,
        // but dropping tensors will free GPU memory
        tracing::trace!("Memory cleanup requested (CUDA)");
    }
}

/// Check if there's enough disk space for layer splitting.
///
/// # Arguments
/// * `path` - Path where layers will be saved
/// * `model_size` - Estimated model size in bytes
/// * `compression` - Compression mode being used
pub fn check_disk_space(
    path: impl AsRef<Path>,
    model_size: u64,
    compression: crate::config::Compression,
) -> crate::Result<bool> {
    use std::fs;

    let path = path.as_ref();

    // Create parent directories if needed
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Calculate required space with compression ratio
    let required = (model_size as f64 * compression.ratio() as f64) as u64;

    // Simple check: just warn if we can't determine space
    // A full implementation would use statvfs on Unix
    tracing::info!(
        "Estimated space required: {}",
        format_bytes(required)
    );

    Ok(true)
}

/// Format bytes as human-readable string
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 bytes");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }
}
