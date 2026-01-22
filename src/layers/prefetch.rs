//! Async layer prefetching to overlap I/O with computation.
//!
//! The prefetcher runs a background task that loads the next layer from disk
//! while the GPU is processing the current layer. This hides I/O latency.

use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, Mutex};

use super::{LayerLoader, LayerWeights};

/// Request to prefetch a layer
#[derive(Debug)]
struct PrefetchRequest {
    layer_idx: usize,
    response: oneshot::Sender<crate::Result<LayerWeights>>,
}

/// Async layer prefetcher
pub struct Prefetcher {
    /// Channel to send prefetch requests
    request_tx: mpsc::Sender<PrefetchRequest>,
    /// Currently prefetched layer (if any)
    prefetched: Arc<Mutex<Option<LayerWeights>>>,
    /// Layer loader reference
    loader: Arc<LayerLoader>,
}

impl Prefetcher {
    /// Create a new prefetcher with the given layer loader
    pub fn new(loader: Arc<LayerLoader>) -> Self {
        let (request_tx, request_rx) = mpsc::channel::<PrefetchRequest>(2);
        let prefetched = Arc::new(Mutex::new(None));

        // Spawn background prefetch task
        let loader_clone = loader.clone();
        tokio::spawn(Self::prefetch_task(loader_clone, request_rx));

        Self {
            request_tx,
            prefetched,
            loader,
        }
    }

    /// Background task that handles prefetch requests
    async fn prefetch_task(
        loader: Arc<LayerLoader>,
        mut request_rx: mpsc::Receiver<PrefetchRequest>,
    ) {
        while let Some(request) = request_rx.recv().await {
            tracing::debug!("Prefetching layer {}", request.layer_idx);

            // Load layer to CPU (don't move to GPU yet)
            let result = loader.load_layer_to_cpu(request.layer_idx);

            // Send result back
            let _ = request.response.send(result);
        }
    }

    /// Request prefetching of a layer (non-blocking)
    pub async fn prefetch(&self, layer_idx: usize) -> crate::Result<()> {
        let (response_tx, _response_rx) = oneshot::channel();

        self.request_tx
            .send(PrefetchRequest {
                layer_idx,
                response: response_tx,
            })
            .await
            .map_err(|e| {
                crate::AirLLMError::LayerLoadError(format!("Prefetch channel error: {}", e))
            })?;

        Ok(())
    }

    /// Get a layer, using prefetched data if available
    pub async fn get_layer(&self, layer_idx: usize) -> crate::Result<LayerWeights> {
        // Check if we have this layer prefetched
        {
            let mut prefetched = self.prefetched.lock().await;
            if let Some(ref weights) = *prefetched {
                if weights.layer_idx == layer_idx {
                    let weights = prefetched.take().unwrap();
                    return self.loader.move_to_device(&weights);
                }
            }
        }

        // Not prefetched, load directly
        self.loader.load_layer(layer_idx)
    }

    /// Get layer with prefetch of next layer
    pub async fn get_layer_and_prefetch_next(
        &self,
        layer_idx: usize,
    ) -> crate::Result<LayerWeights> {
        // Start prefetching next layer
        let next_layer = layer_idx + 1;
        if next_layer < self.loader.num_layers() {
            // Fire off prefetch request (don't await)
            let _ = self.prefetch(next_layer).await;
        }

        // Get current layer
        self.get_layer(layer_idx).await
    }
}

/// Synchronous prefetcher for non-async contexts
pub struct SyncPrefetcher {
    loader: Arc<LayerLoader>,
    /// Handle for the prefetch thread
    prefetch_handle: Option<std::thread::JoinHandle<crate::Result<LayerWeights>>>,
    /// Prefetched layer index
    prefetched_idx: Option<usize>,
}

impl SyncPrefetcher {
    /// Create a new synchronous prefetcher
    pub fn new(loader: Arc<LayerLoader>) -> Self {
        Self {
            loader,
            prefetch_handle: None,
            prefetched_idx: None,
        }
    }

    /// Start prefetching a layer in the background
    pub fn start_prefetch(&mut self, layer_idx: usize) {
        // If there's an existing prefetch, wait for it
        if let Some(handle) = self.prefetch_handle.take() {
            let _ = handle.join();
        }

        let loader = self.loader.clone();
        self.prefetched_idx = Some(layer_idx);

        self.prefetch_handle = Some(std::thread::spawn(move || {
            loader.load_layer_to_cpu(layer_idx)
        }));
    }

    /// Get a layer, using prefetched data if available
    pub fn get_layer(&mut self, layer_idx: usize) -> crate::Result<LayerWeights> {
        // Check if we have this layer prefetched
        if self.prefetched_idx == Some(layer_idx) {
            if let Some(handle) = self.prefetch_handle.take() {
                self.prefetched_idx = None;
                let cpu_weights = handle.join().map_err(|_| {
                    crate::AirLLMError::LayerLoadError("Prefetch thread panicked".to_string())
                })??;
                return self.loader.move_to_device(&cpu_weights);
            }
        }

        // Not prefetched or different layer, load directly
        self.loader.load_layer(layer_idx)
    }

    /// Get layer and start prefetching the next one
    pub fn get_layer_and_prefetch_next(&mut self, layer_idx: usize) -> crate::Result<LayerWeights> {
        let weights = self.get_layer(layer_idx)?;

        // Start prefetching next layer
        let next_layer = layer_idx + 1;
        if next_layer < self.loader.num_layers() {
            self.start_prefetch(next_layer);
        }

        Ok(weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use crate::config::Compression;

    // Note: Tests require actual layer files to run
    // These are integration tests that would need a test fixture

    #[test]
    fn test_sync_prefetcher_creation() {
        let loader = Arc::new(LayerLoader::new(
            "/tmp/nonexistent",
            Device::Cpu,
            candle_core::DType::F16,
            Compression::None,
            32,
        ));

        let _prefetcher = SyncPrefetcher::new(loader);
    }
}
