use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use web_rwkv::model::{EmbedDevice};

use crate::ReloadRequest;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub model: Model,
    pub lora: Vec<Lora>,
    pub tokenizer: Tokenizer,
    pub adapter: AdapterOption
}

impl From<Config> for ReloadRequest {
    fn from(value: Config) -> Self {
        let Config {
            model:
            Model {
                path: model_path,
                quant: _,
                turbo,
                token_chunk_size,
                head_chunk_size,
                state_chunk_size,
                max_runtime_batch,
                max_batch,
                embed_device
            },
            lora,
            tokenizer: Tokenizer {
                path: tokenizer_path,
            },
            adapter,
            ..
        } = value;
        Self {
            model: model_path,
            lora,
            strategy: "fp16".into(),
            turbo,
            token_chunk_size,
            head_chunk_size,
            state_chunk_size,
            max_runtime_batch,
            max_batch,
            embed_device,
            tokenizer_path,
            adapter,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Model {
    /// Path to the model.
    pub path: PathBuf,
    /// Specify layers that needs to be quantized.
    pub quant: usize,
    /// Whether to use alternative GEMM kernel to speed-up long prompts.
    pub turbo: bool,
    /// Maximum tokens to be processed in parallel at once.
    pub token_chunk_size: usize,
    /// The chunk size for each split of the head matrix.
    pub head_chunk_size: usize,
    /// The chunk size of layers in model state.
    pub state_chunk_size: usize,
    /// Maximum number of batches that are active at once.
    pub max_runtime_batch: usize,
    /// Number of states that are cached on GPU.
    pub max_batch: usize,
    /// Device to put the embed tensor.
    pub embed_device: EmbedDevice,
}

impl Default for Model {
    fn default() -> Self {
        Self {
            path: Default::default(),
            quant: Default::default(),
            turbo: true,
            token_chunk_size: 128,
            head_chunk_size: 8192,
            state_chunk_size: 4,
            max_runtime_batch: 8,
            max_batch: 16,
            embed_device: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Lora {
    /// Path to the LoRA.
    pub path: PathBuf,
    /// Blend factor.
    pub alpha: f32,
}

impl Default for Lora {
    fn default() -> Self {
        Self {
            path: Default::default(),
            alpha: 1.0,
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Tokenizer {
    pub path: PathBuf,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub enum AdapterOption {
    #[default]
    Auto,
    Economical,
    Manual(usize),
}
