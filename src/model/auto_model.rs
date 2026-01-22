//! Auto model detection and loading.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device};

use crate::config::{Compression, ModelArchitecture, ModelConfig};
use crate::utils::{download_model, ensure_model_files, get_device};
use super::base::AirLLMModel;
use super::llama::LlamaModel;

/// Automatic model loader that detects architecture and loads appropriate implementation.
pub struct AutoModel;

impl AutoModel {
    /// Load a model from a local path or HuggingFace Hub
    ///
    /// # Arguments
    /// * `model_id` - Either a local path or HuggingFace model ID (e.g., "meta-llama/Llama-2-7b-hf")
    /// * `device` - Device to run on (None for auto-detect)
    /// * `dtype` - Data type (None for F16)
    /// * `compression` - Compression mode for weights
    /// * `hf_token` - Optional HuggingFace token for private models
    ///
    /// # Returns
    /// A boxed model implementing the AirLLMModel trait
    pub async fn from_pretrained(
        model_id: &str,
        device: Option<Device>,
        dtype: Option<DType>,
        compression: Compression,
        hf_token: Option<&str>,
    ) -> crate::Result<Box<dyn AirLLMModel>> {
        let device = device.unwrap_or_else(|| get_device(None).unwrap_or(Device::Cpu));
        let dtype = dtype.unwrap_or(DType::F16);

        // Determine if local path or HF model ID
        let model_path = Self::resolve_model_path(model_id, hf_token).await?;

        // Load config to detect architecture
        let config = ModelConfig::from_file(model_path.join("config.json"))?;
        let architecture = config.architecture();

        tracing::info!("Detected architecture: {:?}", architecture);

        // Create appropriate model
        let model: Box<dyn AirLLMModel> = match architecture {
            ModelArchitecture::Llama
            | ModelArchitecture::Llama2
            | ModelArchitecture::Llama3
            | ModelArchitecture::Mistral
            | ModelArchitecture::Mixtral => {
                Box::new(LlamaModel::new(&model_path, device, dtype, compression)?)
            }
            ModelArchitecture::Unknown(arch) => {
                tracing::warn!(
                    "Unknown architecture '{}', attempting to load as Llama",
                    arch
                );
                Box::new(LlamaModel::new(&model_path, device, dtype, compression)?)
            }
            _ => {
                // For now, try Llama for unsupported architectures
                tracing::warn!(
                    "Architecture {:?} not yet supported, attempting Llama fallback",
                    architecture
                );
                Box::new(LlamaModel::new(&model_path, device, dtype, compression)?)
            }
        };

        Ok(model)
    }

    /// Synchronous version of from_pretrained for non-async contexts
    pub fn from_pretrained_sync(
        model_id: &str,
        device: Option<Device>,
        dtype: Option<DType>,
        compression: Compression,
        hf_token: Option<&str>,
    ) -> crate::Result<Box<dyn AirLLMModel>> {
        // Create a runtime for the async download
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| crate::AirLLMError::IoError(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            )))?;

        rt.block_on(Self::from_pretrained(model_id, device, dtype, compression, hf_token))
    }

    /// Resolve model path (download if necessary)
    async fn resolve_model_path(model_id: &str, hf_token: Option<&str>) -> crate::Result<PathBuf> {
        let path = Path::new(model_id);

        // Check if it's a local path
        if path.exists() && path.is_dir() {
            ensure_model_files(path).await?;
            return Ok(path.to_path_buf());
        }

        // Otherwise, treat as HuggingFace model ID and download
        let cache_dir = Self::get_cache_dir()?;
        let model_path = download_model(model_id, &cache_dir, hf_token).await?;

        Ok(model_path)
    }

    /// Get the cache directory for downloaded models
    fn get_cache_dir() -> crate::Result<PathBuf> {
        // Check environment variable first
        if let Ok(cache) = std::env::var("AIRLLM_CACHE") {
            return Ok(PathBuf::from(cache));
        }

        // Fall back to HF cache location
        if let Ok(cache) = std::env::var("HF_HOME") {
            return Ok(PathBuf::from(cache).join("hub"));
        }

        // Default to ~/.cache/airllm
        let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE"))
            .map_err(|_| crate::AirLLMError::ConfigError("Could not determine home directory".to_string()))?;

        Ok(PathBuf::from(home).join(".cache").join("airllm"))
    }

    /// Get detected architecture for a model
    pub fn detect_architecture(model_path: impl AsRef<Path>) -> crate::Result<ModelArchitecture> {
        let config = ModelConfig::from_file(model_path.as_ref().join("config.json"))?;
        Ok(config.architecture())
    }
}

/// Builder for model loading with configuration options
pub struct ModelBuilder {
    model_id: String,
    device: Option<Device>,
    dtype: Option<DType>,
    compression: Compression,
    hf_token: Option<String>,
    layer_path: Option<PathBuf>,
}

impl ModelBuilder {
    /// Create a new model builder
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            device: None,
            dtype: None,
            compression: Compression::None,
            hf_token: None,
            layer_path: None,
        }
    }

    /// Set the device
    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Set the data type
    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// Set compression mode
    pub fn compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    /// Set HuggingFace token
    pub fn hf_token(mut self, token: impl Into<String>) -> Self {
        self.hf_token = Some(token.into());
        self
    }

    /// Set custom layer path
    pub fn layer_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.layer_path = Some(path.into());
        self
    }

    /// Build the model (async)
    pub async fn build(self) -> crate::Result<Box<dyn AirLLMModel>> {
        AutoModel::from_pretrained(
            &self.model_id,
            self.device,
            self.dtype,
            self.compression,
            self.hf_token.as_deref(),
        )
        .await
    }

    /// Build the model (sync)
    pub fn build_sync(self) -> crate::Result<Box<dyn AirLLMModel>> {
        AutoModel::from_pretrained_sync(
            &self.model_id,
            self.device,
            self.dtype,
            self.compression,
            self.hf_token.as_deref(),
        )
    }
}
