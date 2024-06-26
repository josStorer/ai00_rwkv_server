use std::env;
use std::path::PathBuf;

use derivative::Derivative;
use serde::{Deserialize, Serialize};
use web_rwkv::runtime::model::EmbedDevice;

use crate::run::StateId;

#[derive(Debug, Clone, Derivative, Serialize, Deserialize)]
#[derivative(Default)]
#[serde(default)]
pub struct Model {
    /// Path to the folder containing all models.
    #[derivative(Default(value = "\"models\".into()"))]
    #[serde(alias = "model_path")]
    pub path: PathBuf,
    /// Name of the model.
    #[serde(alias = "model_name")]
    pub name: PathBuf,
    /// Specify layers that needs to be quantized.
    pub strategy: String,
    /// Maximum tokens to be processed in parallel at once.
    #[derivative(Default(value = "32"))]
    pub token_chunk_size: usize,
    /// Number of states that are cached on GPU.
    #[derivative(Default(value = "8"))]
    pub max_batch: usize,
    /// Device to put the embed tensor.
    pub embed_device: EmbedDevice,
}

#[derive(Debug, Clone, Derivative, Serialize, Deserialize)]
#[derivative(Default)]
#[serde(default)]
pub struct Lora {
    /// Path to the LoRA.
    pub path: PathBuf,
    /// Blend factor.
    #[derivative(Default(value = "1.0"))]
    pub alpha: f32,
}

/// State-tuned initial state.
#[derive(Debug, Clone, Derivative, Serialize, Deserialize)]
#[derivative(Default)]
#[serde(default)]
pub struct State {
    /// Path to the initial state.
    pub path: PathBuf,
    /// A UUID for this state.
    #[serde(default = "StateId::new")]
    pub id: StateId,
    /// If this state should be loaded on startup.
    pub default: bool,
}

#[derive(Debug, Derivative, Clone, Serialize, Deserialize)]
#[derivative(Default)]
#[serde(default)]
pub struct Tokenizer {
    #[derivative(Default(value = "env::current_exe().unwrap().parent().unwrap().join(\"assets/rwkv_vocab_v20230424.json\")"))]
    pub path: PathBuf,
}

#[derive(Debug, Derivative, Clone, Serialize, Deserialize)]
#[derivative(Default)]
#[serde(default)]
pub struct BnfOption {
    /// Enable the cache that accelerates the expansion of certain short schemas.
    #[derivative(Default(value = "true"))]
    pub enable_bytes_cache: bool,
    /// The initial nonterminal of the BNF schemas.
    #[derivative(Default(value = "\"start\".into()"))]
    pub start_nonterminal: String,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub enum Precision {
    #[default]
    Fp16,
    Fp32,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub enum AdapterOption {
    #[default]
    Auto,
    Economical,
    Manual(usize),
}
