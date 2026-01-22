//! Llama model implementation for memory-efficient inference.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{Embedding, Linear, Module, RmsNorm, VarBuilder};

use crate::config::{Compression, LlamaConfig, ModelConfig};
use crate::layers::{LayerLoader, LayerWeights, SyncPrefetcher};
use crate::generation::KvCache;
use crate::utils::clean_memory;
use super::base::{AirLLMModel, LayerNames};

/// Llama model with layer-by-layer loading
pub struct LlamaModel {
    /// Model configuration
    config: ModelConfig,
    /// Layer loader
    loader: Arc<LayerLoader>,
    /// Prefetcher for async layer loading
    prefetcher: SyncPrefetcher,
    /// Device for computation
    device: Device,
    /// Data type
    dtype: DType,
    /// Embedding weights (kept in memory)
    embed_tokens: Option<Tensor>,
    /// Final norm weights
    norm_weight: Option<Tensor>,
    /// LM head weights
    lm_head_weight: Option<Tensor>,
    /// Key-value cache
    kv_cache: Option<KvCache>,
    /// Layer naming convention
    layer_names: LayerNames,
    /// RoPE frequency cache
    rope_cache: Option<(Tensor, Tensor)>,
}

impl LlamaModel {
    /// Create a new Llama model from a path
    pub fn new(
        model_path: impl AsRef<Path>,
        device: Device,
        dtype: DType,
        compression: Compression,
    ) -> crate::Result<Self> {
        let model_path = model_path.as_ref();

        // Load config
        let config = ModelConfig::from_file(model_path.join("config.json"))?;

        // Determine layer directory (split layers or original)
        let layer_dir = model_path.join("layers");
        let layer_dir = if layer_dir.exists() {
            layer_dir
        } else {
            model_path.to_path_buf()
        };

        let loader = Arc::new(LayerLoader::new(
            &layer_dir,
            device.clone(),
            dtype,
            compression,
            config.num_hidden_layers,
        ));

        let prefetcher = SyncPrefetcher::new(loader.clone());

        let mut model = Self {
            config,
            loader,
            prefetcher,
            device,
            dtype,
            embed_tokens: None,
            norm_weight: None,
            lm_head_weight: None,
            kv_cache: None,
            layer_names: LayerNames::llama(),
            rope_cache: None,
        };

        // Load persistent weights (embeddings, norm, lm_head)
        model.load_persistent_weights()?;

        // Initialize RoPE cache
        model.init_rope_cache()?;

        Ok(model)
    }

    /// Load weights that stay in memory throughout inference
    fn load_persistent_weights(&mut self) -> crate::Result<()> {
        tracing::info!("Loading persistent weights");

        // Load embeddings
        let embed_tensors = self.loader.load_embeddings()?;
        if let Some(weight) = embed_tensors.get("model.embed_tokens.weight")
            .or_else(|| embed_tensors.get("weight"))
        {
            self.embed_tokens = Some(weight.clone());
        }

        // Load final norm
        let norm_tensors = self.loader.load_norm()?;
        if let Some(weight) = norm_tensors.get("model.norm.weight")
            .or_else(|| norm_tensors.get("weight"))
        {
            self.norm_weight = Some(weight.clone());
        }

        // Load LM head
        let lm_head_tensors = self.loader.load_lm_head()?;
        if let Some(weight) = lm_head_tensors.get("lm_head.weight")
            .or_else(|| lm_head_tensors.get("weight"))
        {
            self.lm_head_weight = Some(weight.clone());
        }

        Ok(())
    }

    /// Initialize rotary position embedding cache
    fn init_rope_cache(&mut self) -> crate::Result<()> {
        let head_dim = self.config.head_dim();
        let max_seq_len = self.config.max_position_embeddings;
        let theta = self.config.rope_theta;

        // Compute inverse frequencies
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / (theta as f32).powf(i as f32 / head_dim as f32))
            .collect();

        let inv_freq = Tensor::from_vec(inv_freq, &[head_dim / 2], &self.device)?
            .to_dtype(self.dtype)?;

        // Compute position indices
        let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let positions = Tensor::from_vec(positions, &[max_seq_len], &self.device)?
            .to_dtype(self.dtype)?;

        // Outer product: [seq_len] x [head_dim/2] -> [seq_len, head_dim/2]
        let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;

        // Duplicate for complex representation: [seq_len, head_dim]
        let freqs = Tensor::cat(&[&freqs, &freqs], 1)?;

        // Compute cos and sin
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        self.rope_cache = Some((cos, sin));

        Ok(())
    }

    /// Apply rotary position embeddings
    fn apply_rope(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_ids: &Tensor,
    ) -> crate::Result<(Tensor, Tensor)> {
        let (cos, sin) = self.rope_cache.as_ref().ok_or_else(|| {
            crate::AirLLMError::ConfigError("RoPE cache not initialized".to_string())
        })?;

        // Get positions for this sequence
        let seq_len = q.dims()[2];
        let cos = cos.narrow(0, 0, seq_len)?;
        let sin = sin.narrow(0, 0, seq_len)?;

        // Apply rotary embeddings
        let q_embed = self.rotate_half(q, &cos, &sin)?;
        let k_embed = self.rotate_half(k, &cos, &sin)?;

        Ok((q_embed, k_embed))
    }

    /// Rotate half of the hidden dims
    fn rotate_half(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> crate::Result<Tensor> {
        let dims = x.dims();
        let half_dim = dims[dims.len() - 1] / 2;

        // Split into two halves
        let x1 = x.narrow(D::Minus1, 0, half_dim)?;
        let x2 = x.narrow(D::Minus1, half_dim, half_dim)?;

        // Rotate: [x1, x2] -> [x1 * cos - x2 * sin, x2 * cos + x1 * sin]
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?; // Add batch and head dims
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        let rotated_x1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let rotated_x2 = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;

        Tensor::cat(&[&rotated_x1, &rotated_x2], D::Minus1).map_err(Into::into)
    }

    /// RMS normalization
    fn rms_norm(&self, x: &Tensor, weight: &Tensor) -> crate::Result<Tensor> {
        let eps = self.config.rms_norm_eps as f32;

        // Compute RMS
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let rms = (variance + eps as f64)?.sqrt()?;

        // Normalize and scale
        let normalized = x.broadcast_div(&rms)?;
        normalized.broadcast_mul(weight).map_err(Into::into)
    }

    /// Process attention for a layer
    fn forward_attention(
        &self,
        hidden_states: &Tensor,
        weights: &LayerWeights,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
    ) -> crate::Result<Tensor> {
        let batch_size = hidden_states.dims()[0];
        let seq_len = hidden_states.dims()[1];
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads();
        let head_dim = self.config.head_dim();

        // Get projection weights
        let q_weight = weights.tensors.get("self_attn.q_proj.weight")
            .ok_or_else(|| crate::AirLLMError::LayerLoadError("Missing q_proj weight".to_string()))?;
        let k_weight = weights.tensors.get("self_attn.k_proj.weight")
            .ok_or_else(|| crate::AirLLMError::LayerLoadError("Missing k_proj weight".to_string()))?;
        let v_weight = weights.tensors.get("self_attn.v_proj.weight")
            .ok_or_else(|| crate::AirLLMError::LayerLoadError("Missing v_proj weight".to_string()))?;
        let o_weight = weights.tensors.get("self_attn.o_proj.weight")
            .ok_or_else(|| crate::AirLLMError::LayerLoadError("Missing o_proj weight".to_string()))?;

        // Project Q, K, V
        let q = hidden_states.matmul(&q_weight.t()?)?;
        let k = hidden_states.matmul(&k_weight.t()?)?;
        let v = hidden_states.matmul(&v_weight.t()?)?;

        // Reshape for multi-head attention
        let q = q.reshape(&[batch_size, seq_len, num_heads, head_dim])?
            .transpose(1, 2)?; // [batch, heads, seq, head_dim]
        let k = k.reshape(&[batch_size, seq_len, num_kv_heads, head_dim])?
            .transpose(1, 2)?;
        let v = v.reshape(&[batch_size, seq_len, num_kv_heads, head_dim])?
            .transpose(1, 2)?;

        // Apply RoPE
        let (q, k) = self.apply_rope(&q, &k, position_ids)?;

        // Repeat KV heads if using GQA
        let (k, v) = if num_kv_heads != num_heads {
            let n_rep = num_heads / num_kv_heads;
            let k = k.repeat(&[1, n_rep, 1, 1])?;
            let v = v.repeat(&[1, n_rep, 1, 1])?;
            (k, v)
        } else {
            (k, v)
        };

        // Compute attention scores
        let scale = (head_dim as f64).sqrt();
        let attn_weights = q.matmul(&k.transpose(2, 3)?)? / scale;

        // Apply causal mask
        let attn_weights = if let Some(mask) = attention_mask {
            (attn_weights + mask)?
        } else {
            // Create causal mask
            let mask = Self::create_causal_mask(seq_len, &self.device, self.dtype)?;
            (attn_weights + mask)?
        };

        // Softmax
        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;

        // Apply attention to values
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape(&[batch_size, seq_len, num_heads * head_dim])?;

        // Output projection
        attn_output.matmul(&o_weight.t()?).map_err(Into::into)
    }

    /// Create a causal attention mask
    fn create_causal_mask(seq_len: usize, device: &Device, dtype: DType) -> crate::Result<Tensor> {
        // Create lower triangular mask manually
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    mask_data[i * seq_len + j] = -1e10; // Masked positions
                }
            }
        }

        Tensor::from_vec(mask_data, &[seq_len, seq_len], device)?
            .to_dtype(dtype)
            .map_err(Into::into)
    }

    /// Process MLP for a layer
    fn forward_mlp(
        &self,
        hidden_states: &Tensor,
        weights: &LayerWeights,
    ) -> crate::Result<Tensor> {
        // Get MLP weights
        let gate_weight = weights.tensors.get("mlp.gate_proj.weight")
            .ok_or_else(|| crate::AirLLMError::LayerLoadError("Missing gate_proj weight".to_string()))?;
        let up_weight = weights.tensors.get("mlp.up_proj.weight")
            .ok_or_else(|| crate::AirLLMError::LayerLoadError("Missing up_proj weight".to_string()))?;
        let down_weight = weights.tensors.get("mlp.down_proj.weight")
            .ok_or_else(|| crate::AirLLMError::LayerLoadError("Missing down_proj weight".to_string()))?;

        // SwiGLU activation: down(silu(gate(x)) * up(x))
        let gate = hidden_states.matmul(&gate_weight.t()?)?;
        let up = hidden_states.matmul(&up_weight.t()?)?;

        // SiLU activation on gate
        let gate_activated = candle_nn::ops::silu(&gate)?;

        // Element-wise multiplication
        let hidden = (gate_activated * up)?;

        // Down projection
        hidden.matmul(&down_weight.t()?).map_err(Into::into)
    }
}

impl AirLLMModel for LlamaModel {
    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    fn max_seq_len(&self) -> usize {
        self.config.max_position_embeddings
    }

    fn embed(&mut self, input_ids: &Tensor) -> crate::Result<Tensor> {
        let embed_weight = self.embed_tokens.as_ref()
            .ok_or_else(|| crate::AirLLMError::LayerLoadError("Embeddings not loaded".to_string()))?;

        // Embedding lookup
        let embeddings = candle_nn::Embedding::new(embed_weight.clone(), self.config.hidden_size);
        embeddings.forward(input_ids).map_err(Into::into)
    }

    fn forward_layer(
        &mut self,
        layer_idx: usize,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
    ) -> crate::Result<Tensor> {
        // Load layer weights (with prefetching)
        let weights = self.prefetcher.get_layer_and_prefetch_next(layer_idx)?;

        // Get norm weights
        let input_norm_weight = weights.tensors.get("input_layernorm.weight")
            .ok_or_else(|| crate::AirLLMError::LayerLoadError("Missing input_layernorm weight".to_string()))?;
        let post_attn_norm_weight = weights.tensors.get("post_attention_layernorm.weight")
            .ok_or_else(|| crate::AirLLMError::LayerLoadError("Missing post_attention_layernorm weight".to_string()))?;

        // Pre-attention norm
        let normed = self.rms_norm(hidden_states, input_norm_weight)?;

        // Self-attention with residual
        let position_ids = position_ids.ok_or_else(|| {
            crate::AirLLMError::ConfigError("Position IDs required".to_string())
        })?;
        let attn_output = self.forward_attention(&normed, &weights, attention_mask, position_ids)?;
        let hidden_states = (hidden_states + attn_output)?;

        // Post-attention norm
        let normed = self.rms_norm(&hidden_states, post_attn_norm_weight)?;

        // MLP with residual
        let mlp_output = self.forward_mlp(&normed, &weights)?;
        let output = (hidden_states + mlp_output)?;

        // Clean up layer weights
        drop(weights);
        clean_memory();

        Ok(output)
    }

    fn final_norm(&mut self, hidden_states: &Tensor) -> crate::Result<Tensor> {
        let norm_weight = self.norm_weight.as_ref()
            .ok_or_else(|| crate::AirLLMError::LayerLoadError("Final norm not loaded".to_string()))?;

        self.rms_norm(hidden_states, norm_weight)
    }

    fn lm_head(&mut self, hidden_states: &Tensor) -> crate::Result<Tensor> {
        let lm_head_weight = self.lm_head_weight.as_ref()
            .ok_or_else(|| crate::AirLLMError::LayerLoadError("LM head not loaded".to_string()))?;

        hidden_states.matmul(&lm_head_weight.t()?).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_names() {
        let names = LayerNames::llama();
        assert_eq!(names.embed_tokens, "model.embed_tokens");
        assert_eq!(names.q_proj, "q_proj");
    }
}
