//! Layer splitting utility to convert full models to per-layer shards.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use safetensors::SafeTensors;
use candle_core::{Device, Tensor};
use indicatif::{ProgressBar, ProgressStyle};

use crate::config::Compression;
use crate::quantization::{create_quantizer, QuantizedTensor};

/// Splits a model into individual layer files for memory-efficient loading.
pub struct LayerSplitter {
    /// Source model directory
    model_path: PathBuf,
    /// Output directory for split layers
    output_path: PathBuf,
    /// Compression mode
    compression: Compression,
    /// Number of transformer layers
    num_layers: usize,
}

impl LayerSplitter {
    /// Create a new layer splitter
    pub fn new(
        model_path: impl AsRef<Path>,
        output_path: impl AsRef<Path>,
        compression: Compression,
        num_layers: usize,
    ) -> Self {
        Self {
            model_path: model_path.as_ref().to_path_buf(),
            output_path: output_path.as_ref().to_path_buf(),
            compression,
            num_layers,
        }
    }

    /// Check if layers are already split at the output path
    pub fn is_already_split(&self) -> bool {
        let first_layer = self.output_path.join("layer_0.safetensors");
        let last_layer = self.output_path.join(format!("layer_{}.safetensors", self.num_layers - 1));
        let embed = self.output_path.join("embed_tokens.safetensors");
        let lm_head = self.output_path.join("lm_head.safetensors");

        first_layer.exists() && last_layer.exists() && embed.exists() && lm_head.exists()
    }

    /// Split the model into individual layer files
    pub fn split(&self) -> crate::Result<()> {
        std::fs::create_dir_all(&self.output_path)?;

        if self.is_already_split() {
            tracing::info!("Layers already split at {:?}", self.output_path);
            return Ok(());
        }

        tracing::info!("Splitting model from {:?} to {:?}", self.model_path, self.output_path);

        // Find all safetensor files
        let safetensor_files = self.find_safetensor_files()?;
        if safetensor_files.is_empty() {
            return Err(crate::AirLLMError::ModelNotFound(
                "No safetensor files found".to_string(),
            ));
        }

        // Build layer mapping from all shards
        let mut layer_tensors: HashMap<String, HashMap<String, Vec<u8>>> = HashMap::new();

        let pb = ProgressBar::new(safetensor_files.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} Processing shards")
                .unwrap(),
        );

        for shard_path in &safetensor_files {
            self.process_shard(shard_path, &mut layer_tensors)?;
            pb.inc(1);
        }
        pb.finish();

        // Write individual layer files
        self.write_layer_files(&layer_tensors)?;

        tracing::info!("Model split complete");
        Ok(())
    }

    fn find_safetensor_files(&self) -> crate::Result<Vec<PathBuf>> {
        let mut files = Vec::new();

        for entry in std::fs::read_dir(&self.model_path)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "safetensors") {
                files.push(path);
            }
        }

        // Sort for consistent ordering
        files.sort();
        Ok(files)
    }

    fn process_shard(
        &self,
        shard_path: &Path,
        layer_tensors: &mut HashMap<String, HashMap<String, Vec<u8>>>,
    ) -> crate::Result<()> {
        let file = std::fs::File::open(shard_path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let tensors = SafeTensors::deserialize(&mmap)
            .map_err(|e| crate::AirLLMError::LayerLoadError(e.to_string()))?;

        for (name, tensor_view) in tensors.tensors() {
            let layer_name = self.classify_tensor_name(&name);

            layer_tensors
                .entry(layer_name)
                .or_insert_with(HashMap::new)
                .insert(name.to_string(), tensor_view.data().to_vec());
        }

        Ok(())
    }

    fn classify_tensor_name(&self, name: &str) -> String {
        // Common patterns for LLM architectures
        // model.embed_tokens.weight -> embed_tokens
        // model.layers.0.xxx -> layer_0
        // model.norm.weight -> norm
        // lm_head.weight -> lm_head

        if name.contains("embed_tokens") || name.contains("wte") {
            "embed_tokens".to_string()
        } else if name.contains("lm_head") || name.contains("output") && name.contains("weight") {
            "lm_head".to_string()
        } else if name.contains("layers.") || name.contains("h.") {
            // Extract layer number
            let parts: Vec<&str> = name.split('.').collect();
            for (i, part) in parts.iter().enumerate() {
                if *part == "layers" || *part == "h" {
                    if let Some(num_str) = parts.get(i + 1) {
                        if let Ok(num) = num_str.parse::<usize>() {
                            return format!("layer_{}", num);
                        }
                    }
                }
            }
            "unknown".to_string()
        } else if name.contains("norm") || name.contains("ln_f") {
            "norm".to_string()
        } else {
            "other".to_string()
        }
    }

    fn write_layer_files(
        &self,
        layer_tensors: &HashMap<String, HashMap<String, Vec<u8>>>,
    ) -> crate::Result<()> {
        let quantizer = create_quantizer(self.compression);

        let pb = ProgressBar::new(layer_tensors.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} Writing layers")
                .unwrap(),
        );

        for (layer_name, tensors) in layer_tensors {
            let output_file = self.output_path.join(format!("{}.safetensors", layer_name));

            // For now, just copy the raw tensor data
            // In a full implementation, we would:
            // 1. Deserialize tensors properly
            // 2. Apply quantization if enabled
            // 3. Write to new safetensor file

            let mut tensor_data: HashMap<String, safetensors::tensor::TensorView<'_>> = HashMap::new();

            // Simplified: write without quantization for now
            // Full implementation would need proper tensor handling
            if quantizer.is_some() {
                tracing::debug!("Quantization would be applied to {}", layer_name);
            }

            // Write raw tensors to file
            self.write_safetensors(&output_file, tensors)?;

            pb.inc(1);
        }

        pb.finish();
        Ok(())
    }

    fn write_safetensors(
        &self,
        path: &Path,
        tensors: &HashMap<String, Vec<u8>>,
    ) -> crate::Result<()> {
        // This is a simplified version - a full implementation would
        // properly serialize with tensor metadata

        // For now, we'll use safetensors::serialize
        use std::io::Write;

        let mut file = std::fs::File::create(path)?;

        // Create a simple format: JSON header + raw data
        // In production, use safetensors::serialize properly
        let metadata: HashMap<String, serde_json::Value> = tensors
            .iter()
            .map(|(name, data)| {
                (
                    name.clone(),
                    serde_json::json!({
                        "size": data.len()
                    }),
                )
            })
            .collect();

        let header = serde_json::to_vec(&metadata)?;
        let header_size = header.len() as u64;

        file.write_all(&header_size.to_le_bytes())?;
        file.write_all(&header)?;

        for (_, data) in tensors {
            file.write_all(data)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_tensor_name() {
        let splitter = LayerSplitter::new(".", ".", Compression::None, 32);

        assert_eq!(
            splitter.classify_tensor_name("model.embed_tokens.weight"),
            "embed_tokens"
        );
        assert_eq!(
            splitter.classify_tensor_name("model.layers.5.self_attn.q_proj.weight"),
            "layer_5"
        );
        assert_eq!(
            splitter.classify_tensor_name("lm_head.weight"),
            "lm_head"
        );
        assert_eq!(
            splitter.classify_tensor_name("model.norm.weight"),
            "norm"
        );
    }
}
