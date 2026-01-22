//! Model download utilities for HuggingFace Hub.

use std::path::{Path, PathBuf};
use indicatif::{ProgressBar, ProgressStyle};

/// Download a model from HuggingFace Hub
///
/// # Arguments
/// * `model_id` - HuggingFace model ID (e.g., "meta-llama/Llama-2-7b-hf")
/// * `cache_dir` - Local cache directory
/// * `token` - Optional HuggingFace token for private models
pub async fn download_model(
    model_id: &str,
    cache_dir: impl AsRef<Path>,
    token: Option<&str>,
) -> crate::Result<PathBuf> {
    let cache_dir = cache_dir.as_ref();
    let model_dir = cache_dir.join(model_id.replace('/', "--"));

    // Check if already downloaded
    if model_dir.exists() && model_dir.join("config.json").exists() {
        tracing::info!("Model already cached at {:?}", model_dir);
        return Ok(model_dir);
    }

    std::fs::create_dir_all(&model_dir)?;

    tracing::info!("Downloading model {} to {:?}", model_id, model_dir);

    // Files to download
    let required_files = vec![
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ];

    let client = reqwest::Client::new();
    let base_url = format!("https://huggingface.co/{}/resolve/main", model_id);

    for file in required_files {
        download_file(&client, &base_url, file, &model_dir, token).await?;
    }

    // Download model weights (safetensors or pytorch)
    download_weights(&client, model_id, &model_dir, token).await?;

    Ok(model_dir)
}

/// Ensure all required model files exist
pub async fn ensure_model_files(model_path: impl AsRef<Path>) -> crate::Result<()> {
    let path = model_path.as_ref();

    if !path.exists() {
        return Err(crate::AirLLMError::ModelNotFound(
            path.display().to_string(),
        ));
    }

    let config_path = path.join("config.json");
    if !config_path.exists() {
        return Err(crate::AirLLMError::ConfigError(
            "config.json not found".to_string(),
        ));
    }

    Ok(())
}

async fn download_file(
    client: &reqwest::Client,
    base_url: &str,
    filename: &str,
    dest_dir: &Path,
    token: Option<&str>,
) -> crate::Result<()> {
    let url = format!("{}/{}", base_url, filename);
    let dest_path = dest_dir.join(filename);

    if dest_path.exists() {
        tracing::debug!("File {} already exists, skipping", filename);
        return Ok(());
    }

    tracing::info!("Downloading {}", filename);

    let mut request = client.get(&url);
    if let Some(token) = token {
        request = request.header("Authorization", format!("Bearer {}", token));
    }

    let response = request.send().await.map_err(|e| {
        crate::AirLLMError::DownloadError(format!("Failed to download {}: {}", filename, e))
    })?;

    if !response.status().is_success() {
        return Err(crate::AirLLMError::DownloadError(format!(
            "Failed to download {}: HTTP {}",
            filename,
            response.status()
        )));
    }

    let content = response.bytes().await.map_err(|e| {
        crate::AirLLMError::DownloadError(format!("Failed to read {}: {}", filename, e))
    })?;

    std::fs::write(&dest_path, &content)?;
    tracing::debug!("Downloaded {} ({} bytes)", filename, content.len());

    Ok(())
}

async fn download_weights(
    client: &reqwest::Client,
    model_id: &str,
    dest_dir: &Path,
    token: Option<&str>,
) -> crate::Result<()> {
    // First, try to get the file list from the model page
    let api_url = format!("https://huggingface.co/api/models/{}", model_id);

    let mut request = client.get(&api_url);
    if let Some(token) = token {
        request = request.header("Authorization", format!("Bearer {}", token));
    }

    let response = request.send().await.map_err(|e| {
        crate::AirLLMError::DownloadError(format!("Failed to fetch model info: {}", e))
    })?;

    if !response.status().is_success() {
        return Err(crate::AirLLMError::DownloadError(format!(
            "Failed to fetch model info: HTTP {}",
            response.status()
        )));
    }

    let model_info: serde_json::Value = response.json().await.map_err(|e| {
        crate::AirLLMError::DownloadError(format!("Failed to parse model info: {}", e))
    })?;

    // Find safetensor or bin files
    let siblings = model_info["siblings"]
        .as_array()
        .ok_or_else(|| crate::AirLLMError::DownloadError("No files found in model".to_string()))?;

    let weight_files: Vec<&str> = siblings
        .iter()
        .filter_map(|s| s["rfilename"].as_str())
        .filter(|name| name.ends_with(".safetensors") || name.ends_with(".bin"))
        .collect();

    if weight_files.is_empty() {
        return Err(crate::AirLLMError::DownloadError(
            "No weight files found".to_string(),
        ));
    }

    // Prefer safetensors over bin
    let safetensors: Vec<_> = weight_files
        .iter()
        .filter(|f| f.ends_with(".safetensors"))
        .collect();

    let files_to_download = if !safetensors.is_empty() {
        safetensors.into_iter().map(|s| *s).collect::<Vec<_>>()
    } else {
        weight_files
    };

    let base_url = format!("https://huggingface.co/{}/resolve/main", model_id);

    // Download with progress bar
    let pb = ProgressBar::new(files_to_download.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );

    for filename in files_to_download {
        pb.set_message(filename.to_string());
        download_large_file(client, &base_url, filename, dest_dir, token).await?;
        pb.inc(1);
    }

    pb.finish_with_message("Download complete");

    Ok(())
}

async fn download_large_file(
    client: &reqwest::Client,
    base_url: &str,
    filename: &str,
    dest_dir: &Path,
    token: Option<&str>,
) -> crate::Result<()> {
    let url = format!("{}/{}", base_url, filename);
    let dest_path = dest_dir.join(filename);

    if dest_path.exists() {
        tracing::debug!("File {} already exists, skipping", filename);
        return Ok(());
    }

    let mut request = client.get(&url);
    if let Some(token) = token {
        request = request.header("Authorization", format!("Bearer {}", token));
    }

    let response = request.send().await.map_err(|e| {
        crate::AirLLMError::DownloadError(format!("Failed to download {}: {}", filename, e))
    })?;

    if !response.status().is_success() {
        return Err(crate::AirLLMError::DownloadError(format!(
            "Failed to download {}: HTTP {}",
            filename,
            response.status()
        )));
    }

    let total_size = response.content_length().unwrap_or(0);
    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} {bytes}/{total_bytes} ({eta})")
            .unwrap(),
    );

    // Download entire file at once
    let bytes = response.bytes().await.map_err(|e| {
        crate::AirLLMError::DownloadError(format!("Download error: {}", e))
    })?;

    pb.inc(bytes.len() as u64);
    pb.finish();

    // Write to file
    tokio::fs::write(&dest_path, &bytes).await?;

    tracing::info!("Downloaded {} ({} bytes)", filename, bytes.len());
    Ok(())
}
