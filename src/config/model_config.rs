//! Model configuration structures for various LLM architectures.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Compression mode for model weights
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Compression {
    /// No compression (full precision)
    #[default]
    None,
    /// 4-bit NF4 quantization
    Nf4,
    /// 8-bit blockwise quantization
    Int8,
}

impl Compression {
    /// Returns the compression ratio (compressed / original size)
    pub fn ratio(&self) -> f32 {
        match self {
            Compression::None => 1.0,
            Compression::Nf4 => 0.25,  // 4-bit vs 16-bit = 1/4
            Compression::Int8 => 0.5,   // 8-bit vs 16-bit = 1/2
        }
    }
}

/// Supported model architectures
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelArchitecture {
    Llama,
    Llama2,
    Llama3,
    Mistral,
    Mixtral,
    Qwen,
    Qwen2,
    ChatGLM,
    Baichuan,
    InternLM,
    Unknown(String),
}

impl ModelArchitecture {
    /// Detect architecture from config.json architectures field
    pub fn from_architectures(architectures: &[String]) -> Self {
        if architectures.is_empty() {
            return ModelArchitecture::Unknown("empty".to_string());
        }

        let arch = &architectures[0];

        if arch.contains("Qwen2") {
            ModelArchitecture::Qwen2
        } else if arch.contains("Qwen") || arch.contains("QWen") {
            ModelArchitecture::Qwen
        } else if arch.contains("Baichuan") {
            ModelArchitecture::Baichuan
        } else if arch.contains("ChatGLM") {
            ModelArchitecture::ChatGLM
        } else if arch.contains("InternLM") {
            ModelArchitecture::InternLM
        } else if arch.contains("Mistral") {
            ModelArchitecture::Mistral
        } else if arch.contains("Mixtral") {
            ModelArchitecture::Mixtral
        } else if arch.contains("Llama") {
            // Try to distinguish Llama versions
            ModelArchitecture::Llama
        } else {
            ModelArchitecture::Unknown(arch.clone())
        }
    }
}

/// Base model configuration loaded from config.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model architecture type(s)
    #[serde(default)]
    pub architectures: Vec<String>,

    /// Hidden size (embedding dimension)
    pub hidden_size: usize,

    /// Intermediate size in MLP layers
    pub intermediate_size: usize,

    /// Number of attention heads
    pub num_attention_heads: usize,

    /// Number of key-value heads (for GQA)
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    /// Number of transformer layers
    pub num_hidden_layers: usize,

    /// RMS norm epsilon
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,

    /// Vocabulary size
    pub vocab_size: usize,

    /// Maximum sequence length
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    /// Rope theta for positional encoding
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,

    /// Beginning of sequence token ID
    #[serde(default = "default_bos_token_id")]
    pub bos_token_id: u32,

    /// End of sequence token ID
    #[serde(default = "default_eos_token_id")]
    pub eos_token_id: u32,

    /// Pad token ID
    #[serde(default)]
    pub pad_token_id: Option<u32>,

    /// Torch dtype hint
    #[serde(default)]
    pub torch_dtype: Option<String>,

    /// Tie word embeddings
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_rms_norm_eps() -> f64 {
    1e-5
}

fn default_max_position_embeddings() -> usize {
    4096
}

fn default_rope_theta() -> f64 {
    10000.0
}

fn default_bos_token_id() -> u32 {
    1
}

fn default_eos_token_id() -> u32 {
    2
}

impl ModelConfig {
    /// Load configuration from a config.json file
    pub fn from_file(path: impl AsRef<Path>) -> crate::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: ModelConfig = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Get the detected model architecture
    pub fn architecture(&self) -> ModelArchitecture {
        ModelArchitecture::from_architectures(&self.architectures)
    }

    /// Get number of KV heads (defaults to num_attention_heads for MHA)
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Check if using grouped query attention
    pub fn is_gqa(&self) -> bool {
        self.num_kv_heads() != self.num_attention_heads
    }
}

/// Llama-specific configuration with additional fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaConfig {
    #[serde(flatten)]
    pub base: ModelConfig,

    /// Use bias in attention
    #[serde(default)]
    pub attention_bias: bool,

    /// Use bias in MLP
    #[serde(default)]
    pub mlp_bias: bool,

    /// Rope scaling configuration
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
}

/// RoPE scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    #[serde(rename = "type")]
    pub scaling_type: String,
    pub factor: f64,
}

impl LlamaConfig {
    pub fn from_file(path: impl AsRef<Path>) -> crate::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: LlamaConfig = serde_json::from_str(&content)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_detection() {
        assert_eq!(
            ModelArchitecture::from_architectures(&["LlamaForCausalLM".to_string()]),
            ModelArchitecture::Llama
        );
        assert_eq!(
            ModelArchitecture::from_architectures(&["MistralForCausalLM".to_string()]),
            ModelArchitecture::Mistral
        );
        assert_eq!(
            ModelArchitecture::from_architectures(&["Qwen2ForCausalLM".to_string()]),
            ModelArchitecture::Qwen2
        );
    }

    #[test]
    fn test_compression_ratio() {
        assert_eq!(Compression::None.ratio(), 1.0);
        assert_eq!(Compression::Nf4.ratio(), 0.25);
        assert_eq!(Compression::Int8.ratio(), 0.5);
    }
}
