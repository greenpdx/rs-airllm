//! AirLLM CLI - Memory-efficient LLM inference
//!
//! Run 70B+ parameter models on limited GPU memory by loading layers one at a time.

use std::io::{self, Write};
use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use tokenizers::Tokenizer;

use rs_airllm::{
    AutoModel, GenerationConfig, Compression,
    config::ModelConfig,
    layers::LayerSplitter,
    utils::get_device,
};

#[derive(Parser)]
#[command(name = "airllm")]
#[command(author, version, about = "Memory-efficient LLM inference", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate text from a prompt
    Generate {
        /// Model path or HuggingFace model ID
        #[arg(short, long)]
        model: String,

        /// Input prompt
        #[arg(short, long)]
        prompt: String,

        /// Maximum new tokens to generate
        #[arg(long, default_value = "256")]
        max_tokens: usize,

        /// Sampling temperature (0.0 = greedy)
        #[arg(long, default_value = "0.7")]
        temperature: f32,

        /// Top-p sampling threshold
        #[arg(long, default_value = "0.9")]
        top_p: f32,

        /// Top-k sampling (0 = disabled)
        #[arg(long, default_value = "50")]
        top_k: usize,

        /// Compression mode for weights
        #[arg(long, value_enum, default_value = "none")]
        compression: CompressionMode,

        /// HuggingFace token for private models
        #[arg(long, env = "HF_TOKEN")]
        hf_token: Option<String>,

        /// GPU device ID (auto-detect if not specified)
        #[arg(long)]
        device: Option<usize>,

        /// Stream output tokens as they're generated
        #[arg(long, default_value = "true")]
        stream: bool,
    },

    /// Split a model into layer shards for efficient loading
    Split {
        /// Source model path
        #[arg(short, long)]
        model: String,

        /// Output directory for split layers
        #[arg(short, long)]
        output: PathBuf,

        /// Compression mode
        #[arg(long, value_enum, default_value = "none")]
        compression: CompressionMode,
    },

    /// Show information about a model
    Info {
        /// Model path
        #[arg(short, long)]
        model: String,
    },

    /// Interactive chat mode
    Chat {
        /// Model path or HuggingFace model ID
        #[arg(short, long)]
        model: String,

        /// System prompt
        #[arg(long)]
        system: Option<String>,

        /// Compression mode
        #[arg(long, value_enum, default_value = "none")]
        compression: CompressionMode,

        /// HuggingFace token
        #[arg(long, env = "HF_TOKEN")]
        hf_token: Option<String>,
    },
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum CompressionMode {
    None,
    Nf4,
    Int8,
}

impl From<CompressionMode> for Compression {
    fn from(mode: CompressionMode) -> Self {
        match mode {
            CompressionMode::None => Compression::None,
            CompressionMode::Nf4 => Compression::Nf4,
            CompressionMode::Int8 => Compression::Int8,
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("rs_airllm=info".parse()?)
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            model,
            prompt,
            max_tokens,
            temperature,
            top_p,
            top_k,
            compression,
            hf_token,
            device,
            stream,
        } => {
            generate(
                &model,
                &prompt,
                max_tokens,
                temperature,
                top_p,
                top_k,
                compression.into(),
                hf_token.as_deref(),
                device,
                stream,
            )
            .await?;
        }

        Commands::Split {
            model,
            output,
            compression,
        } => {
            split_model(&model, &output, compression.into())?;
        }

        Commands::Info { model } => {
            show_info(&model)?;
        }

        Commands::Chat {
            model,
            system,
            compression,
            hf_token,
        } => {
            chat(&model, system.as_deref(), compression.into(), hf_token.as_deref()).await?;
        }
    }

    Ok(())
}

async fn generate(
    model_path: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    compression: Compression,
    hf_token: Option<&str>,
    device_id: Option<usize>,
    stream: bool,
) -> anyhow::Result<()> {
    println!("Loading model: {}", model_path);

    let device = get_device(device_id)?;
    println!("Using device: {:?}", device);

    // Load model
    let mut model = AutoModel::from_pretrained(
        model_path,
        Some(device.clone()),
        None,
        compression,
        hf_token,
    )
    .await?;

    // Load tokenizer
    let tokenizer_path = if std::path::Path::new(model_path).exists() {
        std::path::Path::new(model_path).join("tokenizer.json")
    } else {
        // For HF models, tokenizer should be downloaded with model
        let cache_dir = std::env::var("AIRLLM_CACHE")
            .or_else(|_| std::env::var("HF_HOME").map(|h| format!("{}/hub", h)))
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").unwrap_or_default();
                format!("{}/.cache/airllm", home)
            });
        std::path::Path::new(&cache_dir)
            .join(model_path.replace('/', "--"))
            .join("tokenizer.json")
    };

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Tokenize prompt
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

    let input_ids: Vec<u32> = encoding.get_ids().to_vec();
    println!("Prompt tokens: {}", input_ids.len());

    // Create input tensor
    let input_tensor = candle_core::Tensor::from_vec(
        input_ids.clone(),
        &[1, input_ids.len()],
        &device,
    )?;

    // Configure generation
    let config = GenerationConfig {
        max_new_tokens: max_tokens,
        temperature,
        top_p,
        top_k,
        do_sample: temperature > 0.0,
        ..Default::default()
    };

    println!("\nGenerating...\n");

    // Generate
    let pb = if !stream {
        let pb = ProgressBar::new(max_tokens as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len}")
                .unwrap(),
        );
        Some(pb)
    } else {
        None
    };

    let output_tokens = model.generate(&input_tensor, &config)?;

    if let Some(pb) = pb {
        pb.finish_and_clear();
    }

    // Decode output
    let output_text = tokenizer
        .decode(&output_tokens, true)
        .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;

    println!("{}", output_text);
    println!("\n---");
    println!("Generated {} tokens", output_tokens.len() - input_ids.len());

    Ok(())
}

fn split_model(model_path: &str, output: &PathBuf, compression: Compression) -> anyhow::Result<()> {
    println!("Splitting model: {}", model_path);
    println!("Output directory: {:?}", output);
    println!("Compression: {:?}", compression);

    // Load config to get layer count
    let config = ModelConfig::from_file(
        std::path::Path::new(model_path).join("config.json"),
    )?;

    println!("Model has {} layers", config.num_hidden_layers);

    let splitter = LayerSplitter::new(model_path, output, compression, config.num_hidden_layers);

    if splitter.is_already_split() {
        println!("Model already split at {:?}", output);
        return Ok(());
    }

    splitter.split()?;

    println!("Model split complete!");
    Ok(())
}

fn show_info(model_path: &str) -> anyhow::Result<()> {
    let config = ModelConfig::from_file(
        std::path::Path::new(model_path).join("config.json"),
    )?;

    println!("Model Information");
    println!("=================");
    println!("Architecture: {:?}", config.architecture());
    println!("Vocab size: {}", config.vocab_size);
    println!("Hidden size: {}", config.hidden_size);
    println!("Intermediate size: {}", config.intermediate_size);
    println!("Num layers: {}", config.num_hidden_layers);
    println!("Num attention heads: {}", config.num_attention_heads);
    println!("Num KV heads: {}", config.num_kv_heads());
    println!("Head dim: {}", config.head_dim());
    println!("Max position embeddings: {}", config.max_position_embeddings);
    println!("RoPE theta: {}", config.rope_theta);
    println!("Using GQA: {}", config.is_gqa());

    // Estimate memory
    let params_per_layer = config.hidden_size * config.hidden_size * 4  // QKV + O projections
        + config.hidden_size * config.intermediate_size * 3;  // MLP

    let total_params = config.vocab_size * config.hidden_size  // embeddings
        + params_per_layer * config.num_hidden_layers
        + config.hidden_size * config.vocab_size;  // lm_head

    println!("\nEstimated Parameters");
    println!("====================");
    println!("Per layer: ~{:.1}M", params_per_layer as f64 / 1e6);
    println!("Total: ~{:.1}B", total_params as f64 / 1e9);
    println!("Memory (FP16): ~{:.1} GB", (total_params * 2) as f64 / 1e9);
    println!("Memory (INT8): ~{:.1} GB", total_params as f64 / 1e9);
    println!("Memory (NF4): ~{:.1} GB", (total_params / 2) as f64 / 1e9);

    Ok(())
}

async fn chat(
    model_path: &str,
    system_prompt: Option<&str>,
    compression: Compression,
    hf_token: Option<&str>,
) -> anyhow::Result<()> {
    println!("Loading model for chat: {}", model_path);

    let device = get_device(None)?;
    println!("Using device: {:?}", device);

    let mut model = AutoModel::from_pretrained(
        model_path,
        Some(device.clone()),
        None,
        compression,
        hf_token,
    )
    .await?;

    // Load tokenizer
    let tokenizer_path = std::path::Path::new(model_path).join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    let config = GenerationConfig {
        max_new_tokens: 512,
        temperature: 0.7,
        top_p: 0.9,
        do_sample: true,
        ..Default::default()
    };

    println!("\nChat mode started. Type 'exit' or 'quit' to end.\n");

    if let Some(system) = system_prompt {
        println!("System: {}\n", system);
    }

    let mut conversation = String::new();
    if let Some(system) = system_prompt {
        conversation.push_str(&format!("System: {}\n\n", system));
    }

    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
            println!("Goodbye!");
            break;
        }

        if input.is_empty() {
            continue;
        }

        conversation.push_str(&format!("User: {}\nAssistant:", input));

        // Tokenize
        let encoding = tokenizer
            .encode(conversation.as_str(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let input_ids: Vec<u32> = encoding.get_ids().to_vec();

        // Check length
        if input_ids.len() >= model.max_seq_len() - config.max_new_tokens {
            println!("(Context too long, truncating...)");
            // Simple truncation - keep recent context
            conversation = conversation
                .split('\n')
                .rev()
                .take(10)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .join("\n");
            continue;
        }

        let input_tensor = candle_core::Tensor::from_vec(
            input_ids.clone(),
            &[1, input_ids.len()],
            &device,
        )?;

        // Generate response
        let output_tokens = model.generate(&input_tensor, &config)?;

        // Decode only new tokens
        let new_tokens = &output_tokens[input_ids.len()..];
        let response = tokenizer
            .decode(new_tokens, true)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;

        println!("Assistant: {}\n", response.trim());

        // Add response to conversation
        conversation.push_str(&format!(" {}\n\n", response.trim()));
    }

    Ok(())
}
