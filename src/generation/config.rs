//! Generation configuration.

use serde::{Deserialize, Serialize};

/// Configuration for text generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate
    pub max_new_tokens: usize,

    /// Temperature for sampling (higher = more random)
    pub temperature: f32,

    /// Top-p (nucleus) sampling threshold
    pub top_p: f32,

    /// Top-k sampling (0 = disabled)
    pub top_k: usize,

    /// Repetition penalty (1.0 = no penalty)
    pub repetition_penalty: f32,

    /// End of sequence token ID
    pub eos_token_id: u32,

    /// Beginning of sequence token ID
    pub bos_token_id: u32,

    /// Pad token ID
    pub pad_token_id: u32,

    /// Whether to use sampling or greedy decoding
    pub do_sample: bool,

    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            repetition_penalty: 1.1,
            eos_token_id: 2,
            bos_token_id: 1,
            pad_token_id: 0,
            do_sample: true,
            seed: None,
        }
    }
}

impl GenerationConfig {
    /// Create a greedy decoding configuration
    pub fn greedy() -> Self {
        Self {
            do_sample: false,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            ..Default::default()
        }
    }

    /// Create a creative sampling configuration
    pub fn creative() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 0,
            ..Default::default()
        }
    }

    /// Create a precise/factual configuration
    pub fn precise() -> Self {
        Self {
            temperature: 0.3,
            top_p: 0.85,
            top_k: 40,
            ..Default::default()
        }
    }

    /// Set max new tokens
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_new_tokens = max_tokens;
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Set EOS token ID
    pub fn with_eos_token_id(mut self, eos: u32) -> Self {
        self.eos_token_id = eos;
        self
    }
}
