//! Model implementations for memory-efficient inference.

mod auto_model;
mod base;
mod llama;

pub use auto_model::AutoModel;
pub use base::AirLLMModel;
pub use llama::LlamaModel;
