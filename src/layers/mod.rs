//! Layer management for memory-efficient inference.
//!
//! This module provides:
//! - `LayerSplitter`: Split a full model into individual layer files
//! - `LayerLoader`: Load individual layers with optional decompression
//! - `Prefetcher`: Async prefetching to overlap I/O with computation

mod splitter;
mod loader;
mod prefetch;

pub use splitter::LayerSplitter;
pub use loader::LayerLoader;
pub use prefetch::{Prefetcher, SyncPrefetcher};

/// Layer weight data loaded from disk
pub struct LayerWeights {
    /// Layer index
    pub layer_idx: usize,
    /// Weight tensors keyed by parameter name
    pub tensors: std::collections::HashMap<String, candle_core::Tensor>,
}

impl std::fmt::Debug for LayerWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerWeights")
            .field("layer_idx", &self.layer_idx)
            .field("num_tensors", &self.tensors.len())
            .finish()
    }
}
