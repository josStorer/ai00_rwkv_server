use std::sync::Arc;

use salvo::oapi::ToSchema;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

pub use chat::chat_completions;
pub use completion::completions;
pub use embedding::embeddings;
pub use info::models;

use crate::sampler::{
    mirostat::{MirostatParams, MirostatSampler},
    nucleus::{NucleusParams, NucleusSampler},
    Sampler,
};

pub mod chat;
pub mod completion;
pub mod embedding;
pub mod info;

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum SamplerParams {
    Mirostat(MirostatParams),
    Nucleus(NucleusParams),
}

impl Default for SamplerParams {
    fn default() -> Self {
        Self::Nucleus(Default::default())
    }
}

impl From<SamplerParams> for Arc<RwLock<dyn Sampler + Send + Sync>> {
    fn from(value: SamplerParams) -> Self {
        match value {
            SamplerParams::Mirostat(params) => Arc::new(RwLock::new(MirostatSampler::new(params))),
            SamplerParams::Nucleus(params) => Arc::new(RwLock::new(NucleusSampler::new(params))),
        }
    }
}
