//! # rs-airllm
//!
//! Memory-efficient LLM inference library that enables running 70B+ parameter
//! models on GPUs with limited VRAM (4-8GB).
//!
//! ## Core Concept
//!
//! AirLLM achieves low memory usage by:
//! - **Layer-by-layer processing**: Load one transformer layer at a time
//! - **Prefetching**: Load next layer while GPU processes current layer
//! - **Immediate cleanup**: Free layer memory after processing
//! - **Quantization**: 4-bit/8-bit compression to reduce disk I/O
//!
//! ## Example
//!
//! ```ignore
//! use rs_airllm::{AutoModel, GenerationConfig, Compression};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let mut model = AutoModel::from_pretrained(
//!         "meta-llama/Llama-2-70b-hf",
//!         None,
//!         None,
//!         Compression::None,
//!         None,
//!     ).await?;
//!     // Use model.generate() with input tensor...
//!     Ok(())
//! }
//! ```

pub mod config;
pub mod generation;
pub mod layers;
pub mod model;
pub mod quantization;
pub mod utils;

// Re-exports for convenience
pub use config::{ModelConfig, Compression};
pub use generation::{GenerationConfig, KvCache, Sampler};
pub use layers::{LayerLoader, LayerSplitter, Prefetcher};
pub use model::{AutoModel, AirLLMModel, LlamaModel};
pub use quantization::{QuantizedTensor, Nf4Quantizer, Int8Quantizer};
pub use utils::{Device, DType};

/// Error types for the library
#[derive(thiserror::Error, Debug)]
pub enum AirLLMError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Unsupported model architecture: {0}")]
    UnsupportedArchitecture(String),

    #[error("Layer loading failed: {0}")]
    LayerLoadError(String),

    #[error("Quantization error: {0}")]
    QuantizationError(String),

    #[error("Generation error: {0}")]
    GenerationError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Candle error: {0}")]
    CandleError(#[from] candle_core::Error),

    #[error("JSON parse error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Tokenizer error: {0}")]
    TokenizerError(String),

    #[error("Download error: {0}")]
    DownloadError(String),
}

pub type Result<T> = std::result::Result<T, AirLLMError>;
