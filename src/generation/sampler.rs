//! Token sampling strategies for text generation.

use candle_core::{DType, Device, Tensor, D};
use rand::prelude::*;

use super::GenerationConfig;

/// Token sampler with various sampling strategies.
pub struct Sampler {
    config: GenerationConfig,
    rng: StdRng,
}

impl Sampler {
    /// Create a new sampler with the given configuration
    pub fn new(config: GenerationConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };

        Self { config, rng }
    }

    /// Sample a token from logits
    ///
    /// # Arguments
    /// * `logits` - Tensor of shape [batch, vocab_size] or [batch, 1, vocab_size]
    ///
    /// # Returns
    /// The sampled token ID
    pub fn sample(&self, logits: &Tensor) -> crate::Result<u32> {
        // Squeeze to [batch, vocab_size] or [vocab_size]
        let logits = if logits.dims().len() == 3 {
            logits.squeeze(1)?
        } else {
            logits.clone()
        };

        // Get last batch item if batched
        let logits = if logits.dims().len() == 2 {
            logits.get(0)?
        } else {
            logits
        };

        // Apply temperature
        let logits = if self.config.temperature != 1.0 {
            (logits / self.config.temperature as f64)?
        } else {
            logits
        };

        if !self.config.do_sample {
            // Greedy decoding - just take argmax
            return self.argmax(&logits);
        }

        // Apply top-k filtering
        let logits = if self.config.top_k > 0 {
            self.top_k_filter(&logits, self.config.top_k)?
        } else {
            logits
        };

        // Apply top-p (nucleus) filtering
        let logits = if self.config.top_p < 1.0 {
            self.top_p_filter(&logits, self.config.top_p)?
        } else {
            logits
        };

        // Convert to probabilities
        let probs = candle_nn::ops::softmax(&logits, D::Minus1)?;

        // Sample from distribution
        self.sample_from_probs(&probs)
    }

    /// Greedy argmax selection
    fn argmax(&self, logits: &Tensor) -> crate::Result<u32> {
        let logits_vec: Vec<f32> = logits.to_dtype(DType::F32)?.to_vec1()?;

        let (idx, _) = logits_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .ok_or_else(|| crate::AirLLMError::GenerationError("Empty logits".to_string()))?;

        Ok(idx as u32)
    }

    /// Apply top-k filtering
    fn top_k_filter(&self, logits: &Tensor, k: usize) -> crate::Result<Tensor> {
        let vocab_size = logits.dims()[0];
        let k = k.min(vocab_size);

        // Get top-k values and indices
        let logits_vec: Vec<f32> = logits.to_dtype(DType::F32)?.to_vec1()?;

        let mut indexed: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Find threshold (k-th largest value)
        let threshold = indexed[k - 1].1;

        // Mask values below threshold
        let mask: Vec<f32> = logits_vec
            .iter()
            .map(|&v| if v >= threshold { v } else { f32::NEG_INFINITY })
            .collect();

        Tensor::from_vec(mask, logits.dims(), logits.device())
            .map_err(Into::into)
    }

    /// Apply top-p (nucleus) filtering
    fn top_p_filter(&self, logits: &Tensor, p: f32) -> crate::Result<Tensor> {
        // Convert to probabilities for cumsum
        let probs = candle_nn::ops::softmax(logits, D::Minus1)?;
        let probs_vec: Vec<f32> = probs.to_dtype(DType::F32)?.to_vec1()?;

        // Sort by probability descending
        let mut indexed: Vec<(usize, f32)> = probs_vec.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Find cumulative probability cutoff
        let mut cumsum = 0.0;
        let mut cutoff_idx = indexed.len();

        for (i, (_, prob)) in indexed.iter().enumerate() {
            cumsum += prob;
            if cumsum > p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Get the threshold probability
        let threshold = if cutoff_idx < indexed.len() {
            indexed[cutoff_idx].1
        } else {
            0.0
        };

        // Mask values below threshold
        let logits_vec: Vec<f32> = logits.to_dtype(DType::F32)?.to_vec1()?;
        let mask: Vec<f32> = logits_vec
            .iter()
            .zip(probs_vec.iter())
            .map(|(&l, &p)| if p >= threshold { l } else { f32::NEG_INFINITY })
            .collect();

        Tensor::from_vec(mask, logits.dims(), logits.device())
            .map_err(Into::into)
    }

    /// Sample from probability distribution
    fn sample_from_probs(&self, probs: &Tensor) -> crate::Result<u32> {
        let probs_vec: Vec<f32> = probs.to_dtype(DType::F32)?.to_vec1()?;

        // Create a mutable copy of rng for sampling
        let mut rng = self.rng.clone();
        let sample: f32 = rng.random();

        // Cumulative sampling
        let mut cumsum = 0.0;
        for (idx, &prob) in probs_vec.iter().enumerate() {
            cumsum += prob;
            if sample <= cumsum {
                return Ok(idx as u32);
            }
        }

        // Fallback to last token (shouldn't happen with proper probs)
        Ok((probs_vec.len() - 1) as u32)
    }

    /// Apply repetition penalty to logits
    pub fn apply_repetition_penalty(
        &self,
        logits: &Tensor,
        generated_tokens: &[u32],
    ) -> crate::Result<Tensor> {
        if self.config.repetition_penalty == 1.0 || generated_tokens.is_empty() {
            return Ok(logits.clone());
        }

        let mut logits_vec: Vec<f32> = logits.to_dtype(DType::F32)?.to_vec1()?;
        let penalty = self.config.repetition_penalty;

        for &token in generated_tokens {
            let idx = token as usize;
            if idx < logits_vec.len() {
                if logits_vec[idx] > 0.0 {
                    logits_vec[idx] /= penalty;
                } else {
                    logits_vec[idx] *= penalty;
                }
            }
        }

        Tensor::from_vec(logits_vec, logits.dims(), logits.device())
            .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_sampling() {
        let config = GenerationConfig::greedy();
        let sampler = Sampler::new(config);

        // Create logits with clear maximum
        let logits = Tensor::from_vec(
            vec![0.1f32, 0.2, 0.9, 0.1, 0.05],
            &[5],
            &Device::Cpu,
        ).unwrap();

        let token = sampler.sample(&logits).unwrap();
        assert_eq!(token, 2); // Index of 0.9
    }

    #[test]
    fn test_temperature_scaling() {
        let mut config = GenerationConfig::default();
        config.temperature = 0.5;
        config.do_sample = false;

        let sampler = Sampler::new(config);

        // Should still pick the max with greedy
        let logits = Tensor::from_vec(
            vec![0.1f32, 0.5, 0.3],
            &[3],
            &Device::Cpu,
        ).unwrap();

        let token = sampler.sample(&logits).unwrap();
        assert_eq!(token, 1);
    }
}
