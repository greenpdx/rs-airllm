//! Base trait for AirLLM models.

use std::collections::HashMap;
use candle_core::Tensor;

use crate::generation::GenerationConfig;

/// Trait for memory-efficient LLM models.
///
/// Implementations process models layer-by-layer to minimize memory usage.
pub trait AirLLMModel: Send {
    /// Get the model's vocabulary size
    fn vocab_size(&self) -> usize;

    /// Get the number of transformer layers
    fn num_layers(&self) -> usize;

    /// Get the hidden dimension
    fn hidden_size(&self) -> usize;

    /// Get the maximum sequence length
    fn max_seq_len(&self) -> usize;

    /// Process embeddings for input tokens
    fn embed(&mut self, input_ids: &Tensor) -> crate::Result<Tensor>;

    /// Process a single transformer layer
    fn forward_layer(
        &mut self,
        layer_idx: usize,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
    ) -> crate::Result<Tensor>;

    /// Apply final normalization
    fn final_norm(&mut self, hidden_states: &Tensor) -> crate::Result<Tensor>;

    /// Project hidden states to vocabulary logits
    fn lm_head(&mut self, hidden_states: &Tensor) -> crate::Result<Tensor>;

    /// Full forward pass through all layers
    fn forward(
        &mut self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> crate::Result<Tensor> {
        let seq_len = input_ids.dims()[1];

        // Create position IDs [0, 1, 2, ..., seq_len-1]
        let position_ids = Tensor::arange(0u32, seq_len as u32, input_ids.device())?
            .unsqueeze(0)?;

        // Embed tokens
        let mut hidden_states = self.embed(input_ids)?;

        // Process each layer
        for layer_idx in 0..self.num_layers() {
            hidden_states = self.forward_layer(
                layer_idx,
                &hidden_states,
                attention_mask,
                Some(&position_ids),
            )?;
        }

        // Final normalization
        hidden_states = self.final_norm(&hidden_states)?;

        // Get logits for last token only (for generation)
        let last_hidden = hidden_states.narrow(1, seq_len - 1, 1)?;
        let logits = self.lm_head(&last_hidden)?;

        Ok(logits)
    }

    /// Generate text from a prompt
    fn generate(
        &mut self,
        input_ids: &Tensor,
        config: &GenerationConfig,
    ) -> crate::Result<Vec<u32>> {
        use crate::generation::Sampler;

        let mut tokens: Vec<u32> = input_ids.squeeze(0)?.to_vec1()?;
        let sampler = Sampler::new(config.clone());

        for _ in 0..config.max_new_tokens {
            // Build input tensor from current tokens
            let input = Tensor::from_vec(
                tokens.clone(),
                &[1, tokens.len()],
                input_ids.device(),
            )?;

            // Forward pass
            let logits = self.forward(&input, None)?;

            // Sample next token
            let next_token = sampler.sample(&logits)?;

            // Check for EOS
            if next_token == config.eos_token_id {
                break;
            }

            tokens.push(next_token);

            // Check max length
            if tokens.len() >= self.max_seq_len() {
                break;
            }
        }

        Ok(tokens)
    }
}

/// Layer names for different model architectures
#[derive(Debug, Clone)]
pub struct LayerNames {
    pub embed_tokens: &'static str,
    pub layers_prefix: &'static str,
    pub self_attn: &'static str,
    pub q_proj: &'static str,
    pub k_proj: &'static str,
    pub v_proj: &'static str,
    pub o_proj: &'static str,
    pub mlp: &'static str,
    pub gate_proj: &'static str,
    pub up_proj: &'static str,
    pub down_proj: &'static str,
    pub input_layernorm: &'static str,
    pub post_attention_layernorm: &'static str,
    pub norm: &'static str,
    pub lm_head: &'static str,
}

impl LayerNames {
    /// Get layer names for Llama architecture
    pub fn llama() -> Self {
        Self {
            embed_tokens: "model.embed_tokens",
            layers_prefix: "model.layers",
            self_attn: "self_attn",
            q_proj: "q_proj",
            k_proj: "k_proj",
            v_proj: "v_proj",
            o_proj: "o_proj",
            mlp: "mlp",
            gate_proj: "gate_proj",
            up_proj: "up_proj",
            down_proj: "down_proj",
            input_layernorm: "input_layernorm",
            post_attention_layernorm: "post_attention_layernorm",
            norm: "model.norm",
            lm_head: "lm_head",
        }
    }

    /// Get layer names for Mistral architecture (same as Llama)
    pub fn mistral() -> Self {
        Self::llama()
    }

    /// Get layer names for Qwen architecture
    pub fn qwen() -> Self {
        Self {
            embed_tokens: "transformer.wte",
            layers_prefix: "transformer.h",
            self_attn: "attn",
            q_proj: "c_attn", // Combined QKV in Qwen
            k_proj: "c_attn",
            v_proj: "c_attn",
            o_proj: "c_proj",
            mlp: "mlp",
            gate_proj: "w1",
            up_proj: "w2",
            down_proj: "c_proj",
            input_layernorm: "ln_1",
            post_attention_layernorm: "ln_2",
            norm: "transformer.ln_f",
            lm_head: "lm_head",
        }
    }
}
