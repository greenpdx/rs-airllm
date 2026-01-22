//! Key-value cache for efficient autoregressive generation.

use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;

/// Cache for key and value tensors across generation steps.
///
/// During generation, we only need to compute attention for new tokens,
/// reusing cached K and V from previous tokens.
pub struct KvCache {
    /// Cached key tensors per layer
    keys: HashMap<usize, Tensor>,
    /// Cached value tensors per layer
    values: HashMap<usize, Tensor>,
    /// Maximum sequence length
    max_seq_len: usize,
    /// Current sequence length (how many tokens cached)
    current_len: usize,
    /// Number of layers
    num_layers: usize,
    /// Number of KV heads
    num_kv_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Device
    device: Device,
    /// Data type
    dtype: DType,
}

impl KvCache {
    /// Create a new KV cache
    pub fn new(
        max_seq_len: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        device: Device,
        dtype: DType,
    ) -> Self {
        Self {
            keys: HashMap::new(),
            values: HashMap::new(),
            max_seq_len,
            current_len: 0,
            num_layers,
            num_kv_heads,
            head_dim,
            device,
            dtype,
        }
    }

    /// Update cache with new key and value tensors for a layer
    ///
    /// # Arguments
    /// * `layer_idx` - The layer index
    /// * `new_key` - New key tensor [batch, num_kv_heads, seq_len, head_dim]
    /// * `new_value` - New value tensor [batch, num_kv_heads, seq_len, head_dim]
    ///
    /// # Returns
    /// The full key and value tensors including cached values
    pub fn update(
        &mut self,
        layer_idx: usize,
        new_key: &Tensor,
        new_value: &Tensor,
    ) -> crate::Result<(Tensor, Tensor)> {
        let new_seq_len = new_key.dims()[2];

        let (full_key, full_value) = if let (Some(cached_k), Some(cached_v)) =
            (self.keys.get(&layer_idx), self.values.get(&layer_idx))
        {
            // Concatenate with cached values
            let full_key = Tensor::cat(&[cached_k, new_key], 2)?;
            let full_value = Tensor::cat(&[cached_v, new_value], 2)?;
            (full_key, full_value)
        } else {
            // First update, just use new values
            (new_key.clone(), new_value.clone())
        };

        // Store updated cache
        self.keys.insert(layer_idx, full_key.clone());
        self.values.insert(layer_idx, full_value.clone());

        // Update current length (assuming all layers have same length)
        self.current_len = full_key.dims()[2];

        Ok((full_key, full_value))
    }

    /// Get cached key and value for a layer
    pub fn get(&self, layer_idx: usize) -> Option<(&Tensor, &Tensor)> {
        match (self.keys.get(&layer_idx), self.values.get(&layer_idx)) {
            (Some(k), Some(v)) => Some((k, v)),
            _ => None,
        }
    }

    /// Get current sequence length
    pub fn current_len(&self) -> usize {
        self.current_len
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.current_len == 0
    }

    /// Check if cache is full
    pub fn is_full(&self) -> bool {
        self.current_len >= self.max_seq_len
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.current_len = 0;
    }

    /// Trim cache to keep only the last n tokens
    pub fn trim(&mut self, keep_last: usize) -> crate::Result<()> {
        if keep_last >= self.current_len {
            return Ok(());
        }

        let trim_start = self.current_len - keep_last;

        for (_, tensor) in self.keys.iter_mut() {
            *tensor = tensor.narrow(2, trim_start, keep_last)?;
        }

        for (_, tensor) in self.values.iter_mut() {
            *tensor = tensor.narrow(2, trim_start, keep_last)?;
        }

        self.current_len = keep_last;
        Ok(())
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let mut total = 0;

        for tensor in self.keys.values() {
            total += tensor.elem_count() * tensor.dtype().size_in_bytes();
        }

        for tensor in self.values.values() {
            total += tensor.elem_count() * tensor.dtype().size_in_bytes();
        }

        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_basic() {
        let cache = KvCache::new(1024, 32, 8, 128, Device::Cpu, DType::F16);
        assert!(cache.is_empty());
        assert!(!cache.is_full());
    }
}
