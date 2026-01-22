# rs-airllm

Memory-efficient LLM inference in Rust. Run 70B+ parameter models on GPUs with limited VRAM (4-8GB).

## How It Works

rs-airllm achieves low memory usage through:

- **Layer-by-layer processing** - Load one transformer layer at a time instead of the entire model
- **Prefetching** - Load the next layer while the GPU processes the current one
- **Immediate cleanup** - Free layer memory immediately after processing
- **Quantization** - Optional 4-bit (NF4) or 8-bit compression to reduce memory and disk I/O

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/rs-airllm
cd rs-airllm

# Build (CPU only)
cargo build --release

# Build with CUDA support
cargo build --release --features cuda

# Build with Metal support (macOS)
cargo build --release --features metal
```

## Downloading Models

### Automatic Download

Models are automatically downloaded from HuggingFace Hub when you specify a model ID:

```bash
airllm generate --model meta-llama/Llama-2-7b-hf --prompt "Hello"
```

The following files are downloaded:
- `config.json` - Model configuration
- `tokenizer.json` - Tokenizer data
- `tokenizer_config.json` - Tokenizer settings
- `*.safetensors` - Model weights (preferred) or `*.bin` files

### Private/Gated Models

For private or gated models (like LLaMA), you need a HuggingFace token:

1. Create an account at [huggingface.co](https://huggingface.co)
2. Accept the model's license agreement on its model page
3. Generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Provide the token via environment variable or CLI flag:

```bash
# Via environment variable
export HF_TOKEN=hf_your_token_here
airllm generate --model meta-llama/Llama-2-70b-hf --prompt "Hello"

# Via CLI flag
airllm generate --model meta-llama/Llama-2-70b-hf --prompt "Hello" --hf-token hf_your_token_here
```

### Cache Location

Downloaded models are cached locally. The cache directory is determined by (in order of priority):

1. `AIRLLM_CACHE` environment variable
2. `$HF_HOME/hub` if `HF_HOME` is set
3. `~/.cache/airllm` (default)

To use a custom cache location:

```bash
export AIRLLM_CACHE=/path/to/cache
```

### Using Local Models

You can also use locally downloaded models by providing the path:

```bash
# If you already have a model downloaded
airllm generate --model /path/to/local/model --prompt "Hello"

# Show info about a local model
airllm info --model /path/to/local/model
```

### Pre-downloading Models

To download a model without running inference, use the `split` command which will download and optionally compress the model:

```bash
airllm split \
  --model meta-llama/Llama-2-70b-hf \
  --output ./models/llama-70b-nf4 \
  --compression nf4
```

## CLI Usage

### Generate Text

```bash
airllm generate \
  --model meta-llama/Llama-2-70b-hf \
  --prompt "Explain quantum computing in simple terms" \
  --max-tokens 256 \
  --temperature 0.7
```

### Interactive Chat

```bash
airllm chat \
  --model meta-llama/Llama-2-70b-hf \
  --system "You are a helpful assistant"
```

### Split Model for Faster Loading

Pre-split models into layer shards for faster subsequent loads:

```bash
airllm split \
  --model /path/to/model \
  --output /path/to/split-model \
  --compression nf4
```

### Show Model Info

```bash
airllm info --model /path/to/model
```

## Library Usage

```rust
use rs_airllm::{AutoModel, GenerationConfig, Compression};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load model with NF4 quantization
    let mut model = AutoModel::from_pretrained(
        "meta-llama/Llama-2-70b-hf",
        None,  // auto-detect device
        None,  // default dtype
        Compression::Nf4,
        Some("your-hf-token"),
    ).await?;

    // Configure generation
    let config = GenerationConfig {
        max_new_tokens: 256,
        temperature: 0.7,
        top_p: 0.9,
        top_k: 50,
        do_sample: true,
        ..Default::default()
    };

    // Generate (input_tensor is your tokenized prompt)
    let output_tokens = model.generate(&input_tensor, &config)?;

    Ok(())
}
```

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace token for private models |
| `AIRLLM_CACHE` | Cache directory for downloaded models |
| `HF_HOME` | Alternative cache location (uses `$HF_HOME/hub`) |
| `RUST_LOG` | Logging level (e.g., `rs_airllm=debug`) |

### Compression Modes

| Mode | Memory Savings | Quality |
|------|----------------|---------|
| `none` | Baseline | Full precision |
| `int8` | ~50% | Minimal loss |
| `nf4` | ~75% | Some quality loss |

## Supported Models

Currently supports LLaMA-style architectures:
- LLaMA 2 (7B, 13B, 70B)
- LLaMA 3
- Code Llama
- Other LLaMA-compatible models

## Requirements

- Rust 1.85+ (2024 edition)
- For CUDA: CUDA toolkit and compatible NVIDIA GPU
- For Metal: macOS with Apple Silicon or AMD GPU

## License

MIT
