use std::{
    collections::HashMap,
    env,
    fs::{File},
    io::{BufReader, Read},
    net::{Ipv4Addr, SocketAddr},
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use anyhow::Result;
use axum::{
    routing::{get, post},
    Router,
};
use clap::Parser;
use config::{AdapterOption};
use flume::{Receiver, Sender};
use itertools::Itertools;
use memmap2::Mmap;
use run::RuntimeUntyped;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, RwLock};
use tower_http::{cors::CorsLayer};
use web_rwkv::{
    context::{Context, ContextBuilder, Instance},
    model::{
        loader::Loader, Lora, LoraBlend, Model, ModelBuilder, ModelInfo, ModelState,
        ModelVersion, StateBuilder,
    },
    tokenizer::Tokenizer,
    wgpu::{Backends, PowerPreference},
};
use web_rwkv::model::Quant;

use crate::{
    run::{GenerateContext, Runtime, SlotResult, Tokens},
    sampler::Sampler,
};

mod api;
mod config;
mod oai;
mod run;
mod sampler;
mod utils;
mod root;

pub const MAX_TOKENS: usize = 4096;
pub const STATE_CHUNK_SIZE: usize = 4;

#[derive(Debug)]
pub enum Token {
    Start,
    Token(String),
    Stop(FinishReason, TokenCounter),
    Embed(Vec<f32>),
    Done,
}

#[derive(Debug, Default, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// API returned complete model output.
    Stop,
    /// Incomplete model output due to max_tokens parameter or token limit.
    Length,
    /// Omitted content due to a flag from our content filters.
    ContentFilter,
    /// API response still in progress or incomplete.
    #[default]
    Null,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Array<T> {
    #[default]
    None,
    Item(T),
    Vec(Vec<T>),
}

impl<T> From<Array<T>> for Vec<T>
    where
        T: std::fmt::Debug + Clone + Serialize,
{
    fn from(value: Array<T>) -> Self {
        match value {
            Array::None => vec![],
            Array::Item(item) => vec![item],
            Array::Vec(vec) => vec,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ThreadRequest {
    Adapter(Sender<AdapterList>),
    Info(Sender<RuntimeInfo>),
    Generate {
        request: GenerateRequest,
        tokenizer: Arc<Tokenizer>,
        sender: Sender<Token>,
    },
    Reload {
        request: ReloadRequest,
        sender: Option<Sender<bool>>,
    },
    Unload,
}

#[derive(Default)]
pub enum Environment<'a> {
    Loaded {
        runtime: RuntimeUntyped<'a>,
        reload: ReloadRequest,
    },
    #[default]
    None,
}

impl Environment<'_> {
    pub async fn enqueue(&self, context: GenerateContext) -> Vec<GenerateContext> {
        let mut queue = vec![];
        match self {
            Environment::Loaded { runtime, .. } => match runtime.queue(context).await {
                SlotResult::Success(batch) => log::info!("queued task at slot {batch}"),
                SlotResult::Fault(batch) => log::info!("swapped task at slot {batch}"),
                SlotResult::Failure(context) => {
                    log::info!("failed to queue task");
                    queue.push(*context);
                }
                SlotResult::Error => log::error!("empty task is not queued"),
            },
            Environment::None => queue.push(context),
        };
        queue
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeInfo {
    pub reload: ReloadRequest,
    pub model: ModelInfo,
    pub tokenizer: Arc<Tokenizer>,
}

#[derive(Debug, Default, Clone)]
pub struct AdapterList(pub Vec<String>);

#[derive(Debug, Default, Clone)]
pub struct GenerateRequest {
    /// The prompt for the model.
    pub prompt: String,
    /// All text the model output earlier.
    pub model_text: String,
    /// Output token limit.
    pub max_tokens: usize,
    /// Stop indicators.
    pub stop: Vec<String>,
    /// Sampler parameters.
    pub sampler: Sampler,
    /// Bias added to tokens before sampling.
    pub logit_bias: HashMap<u16, f32>,
    /// Whether this is an embedding request.
    pub embed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ReloadRequest {
    /// Path to the model.
    pub model: PathBuf,
    /// List of LoRA blended on the model.
    pub lora: Vec<config::Lora>,
    /// Specify layers that needs to be quantized.
    pub strategy: String,
    /// Whether to use alternative GEMM kernel to speed-up long prompts.
    pub turbo: bool,
    /// Maximum tokens to be processed in parallel at once.
    pub token_chunk_size: usize,
    /// The chunk size for each split of the head matrix.
    pub head_chunk_size: usize,
    /// Maximum number of batches that are active at once.
    pub max_runtime_batch: usize,
    /// Number of states that are cached on GPU.
    pub max_batch: usize,
    /// the (reversed) number of layer at which the output is as embedding.
    pub embed_layer: usize,
    /// Path to the tokenizer.
    pub tokenizer_path: PathBuf,
    /// Adapter selection.
    pub adapter: AdapterOption,
}

impl Default for ReloadRequest {
    fn default() -> Self {
        Self {
            model: Default::default(),
            lora: Default::default(),
            strategy: Default::default(),
            turbo: true,
            token_chunk_size: 32,
            head_chunk_size: 8192,
            max_runtime_batch: 8,
            max_batch: 16,
            embed_layer: 2,
            tokenizer_path: env::current_exe()
                .unwrap()
                .parent()
                .unwrap()
                .join("assets/rwkv_vocab_v20230424.json"),
            adapter: AdapterOption::Auto,
        }
    }
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct TokenCounter {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Clone)]
pub struct ThreadState(pub Sender<ThreadRequest>);

fn list_adapters() -> AdapterList {
    let backends = Backends::VULKAN | Backends::METAL;
    let instance = Instance::new();
    let list = instance
        .enumerate_adapters(backends)
        .map(|adapter| adapter.get_info())
        .map(|info| format!("{} ({:?})", info.name, info.backend))
        .collect();
    AdapterList(list)
}

async fn create_context(adapter: AdapterOption, info: &ModelInfo) -> Result<Context> {
    let backends = Backends::VULKAN | Backends::METAL;
    let instance = Instance::new();
    let adapter = match adapter {
        AdapterOption::Auto => instance.adapter(PowerPreference::HighPerformance).await,
        AdapterOption::Economical => instance.adapter(PowerPreference::LowPower).await,
        AdapterOption::Manual(selection) => instance.select_adapter(backends, selection),
    }?;

    let limits = web_rwkv::wgpu::Limits {
        max_storage_buffer_binding_size: info.max_buffer_size() as u32,
        ..Default::default()
    };
    let context = ContextBuilder::new(adapter)
        .with_default_pipelines()
        .with_limits(limits)
        .build()
        .await?;
    Ok(context)
}

fn load_tokenizer(path: impl AsRef<Path>) -> Result<Tokenizer> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents)?;
    Ok(Tokenizer::new(&contents)?)
}

fn load_model<M, S>(context: &Context, request: ReloadRequest, data: &[u8]) -> Result<(M, S)>
    where
        S: ModelState,
        M: Model<ModelState = S>,
{
    let ReloadRequest {
        strategy,
        lora,
        token_chunk_size,
        turbo,
        ..
    } = request;
    let quant_type = if strategy.contains("i8") {
        Quant::Int8
    } else if strategy.contains("i4") {
        Quant::NF4
    } else {
        Quant::None
    };

    let layer = strategy
        .split_whitespace()
        .flat_map(|s| s.split(','))
        .find_map(|s| s.strip_prefix("layer").and_then(|n| n.parse::<usize>().ok()))
        .unwrap_or(26);
    let quant = (0..layer).map(|layer| (layer, quant_type)).collect();

    let lora: Vec<Lora> = lora
        .into_iter()
        .map(|lora| -> Result<Lora> {
            let file = File::open(&lora.path)?;
            let data = unsafe { Mmap::map(&file) }?.to_vec();
            let blend = LoraBlend::full(lora.alpha);
            Ok(Lora { data, blend })
        })
        .try_collect()?;

    let model = ModelBuilder::new(context, data)
        .with_quant(quant)
        .with_turbo(turbo)
        .with_token_chunk_size(token_chunk_size);
    let model: M = lora
        .into_iter()
        .fold(model, |acc, x| acc.add_lora(x))
        .build()?;

    let state: S = StateBuilder::new(context, model.info())
        .with_max_batch(request.max_batch)
        .with_chunk_size(STATE_CHUNK_SIZE)
        .build();
    Ok((model, state))
}

#[tokio::main]
async fn model_route(receiver: Receiver<ThreadRequest>) -> Result<()> {
    let env: Arc<RwLock<Environment>> = Default::default();
    let queue: Arc<Mutex<Vec<GenerateContext>>> = Default::default();

    let sender = {
        let (sender, receiver) = flume::unbounded();
        let env = env.clone();
        tokio::task::spawn_blocking(move || run::run(receiver, env));
        sender
    };

    let dequeue = {
        let env = env.clone();
        let queue = queue.clone();
        let sender = sender.clone();

        async move {
            loop {
                let mut queue = queue.lock().await;
                let mut temp = vec![];
                for context in queue.drain(..) {
                    temp.append(&mut env.read().await.enqueue(context).await);
                    let _ = sender.send(());
                }
                std::mem::swap(&mut *queue, &mut temp);
                drop(queue);

                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }
    };
    tokio::spawn(dequeue);

    loop {
        let listen = async {
            match receiver.recv_async().await? {
                ThreadRequest::Adapter(sender) => {
                    let _ = sender.send(list_adapters());
                }
                ThreadRequest::Info(sender) => {
                    let env = env.clone();
                    let task = async move {
                        let env = &(*env.read().await);
                        if let Environment::Loaded { runtime, reload } = env {
                            let reload = reload.clone();
                            let model = runtime.info().clone();
                            let tokenizer = runtime.tokenizer();
                            let _ = sender.send(RuntimeInfo {
                                reload,
                                model,
                                tokenizer,
                            });
                        }
                    };
                    tokio::spawn(task);
                }
                ThreadRequest::Reload {
                    request,
                    sender: reload_sender,
                } => {
                    let callback = move |result: bool| {
                        if let Some(sender) = reload_sender {
                            let _ = sender.send(result);
                        }
                    };
                    let sender = sender.clone();
                    let env = env.clone();
                    let reload = async move {
                        let sender = sender.clone();
                        let max_runtime_batch = request.max_runtime_batch;
                        let embed_layer = request.embed_layer;

                        let file = File::open(&request.model)?;
                        let data = unsafe { Mmap::map(&file)? };
                        let info = Loader::info(&data)?;
                        log::info!("{:#?}", info);

                        let context = create_context(request.adapter, &info).await?;
                        let tokenizer = load_tokenizer(&request.tokenizer_path)?;
                        log::info!("{:#?}", context.adapter.get_info());

                        let mut env = env.write().await;
                        *env = Environment::None;

                        let runtime = match info.version {
                            ModelVersion::V4 => {
                                let (model, state) = load_model(&context, request.clone(), &data)?;
                                RuntimeUntyped::V4(Runtime::new(
                                    tokenizer,
                                    model,
                                    state,
                                    max_runtime_batch,
                                    embed_layer,
                                ))
                            }
                            ModelVersion::V5 => {
                                let (model, state) = load_model(&context, request.clone(), &data)?;
                                RuntimeUntyped::V5(Runtime::new(
                                    tokenizer,
                                    model,
                                    state,
                                    max_runtime_batch,
                                    embed_layer,
                                ))
                            }
                            ModelVersion::V6 => {
                                let (model, state) = load_model(&context, request.clone(), &data)?;
                                RuntimeUntyped::V6(Runtime::new(
                                    tokenizer,
                                    model,
                                    state,
                                    max_runtime_batch,
                                    embed_layer,
                                ))
                            }
                        };
                        let reload = request;
                        *env = Environment::Loaded { runtime, reload };

                        let _ = sender.send(());
                        anyhow::Ok(())
                    };

                    let reload = async move {
                        match reload.await {
                            Ok(_) => {
                                callback(true);
                                log::info!("model reloaded")
                            }
                            Err(err) => {
                                callback(false);
                                log::error!("reload model failed: {}", err);
                            }
                        }
                    };
                    tokio::spawn(reload);
                }
                ThreadRequest::Unload => {
                    let env = env.clone();
                    let unload = async move {
                        let mut env = env.write().await;
                        *env = Environment::None;
                        log::info!("model unloaded");
                    };
                    tokio::spawn(unload);
                }
                ThreadRequest::Generate {
                    request,
                    tokenizer,
                    sender: token_sender,
                } => {
                    let tokens = Tokens(tokenizer.encode(request.prompt.as_bytes())?);
                    let model_tokens = Tokens(tokenizer.encode(request.model_text.as_bytes())?);
                    let mut penalties = HashMap::new();
                    for (index, token) in model_tokens.iter().rev().enumerate() {
                        let ap = request.sampler.presence_penalty;
                        let af = request.sampler.frequency_penalty;
                        let ad = request.sampler.penalty_decay;
                        let mut penalty = penalties.remove(token).unwrap_or(ap);
                        penalty += af * ad.powf(index as f32);
                        penalties.insert(*token, penalty);
                    }

                    let context = GenerateContext {
                        prompt_tokens: tokens.to_vec(),
                        prefix: Default::default(),
                        suffix: tokens,
                        penalties,
                        model_text: Default::default(),
                        output_buffer: Default::default(),
                        model_tokens: Default::default(),
                        request,
                        sender: token_sender,
                    };

                    let env = env.clone();
                    let queue = queue.clone();
                    let sender = sender.clone();
                    let task = async move {
                        let mut queue = queue.lock().await;
                        queue.append(&mut env.read().await.enqueue(context).await);
                        let _ = sender.send(());
                    };
                    tokio::spawn(task);
                }
            };
            anyhow::Ok(())
        };

        if let Err(err) = listen.await {
            log::error!("{err}");
        }
    }
}

pub async fn try_request_info(sender: Sender<ThreadRequest>) -> Result<RuntimeInfo> {
    let (info_sender, info_receiver) = flume::unbounded();
    let _ = sender.send(ThreadRequest::Info(info_sender));
    let info = info_receiver.recv_async().await?;
    Ok(info)
}

pub async fn request_info(sender: Sender<ThreadRequest>, sleep: Duration) -> RuntimeInfo {
    loop {
        if let Ok(info) = try_request_info(sender.clone()).await {
            break info;
        }
        tokio::time::sleep(sleep).await;
    }
}

pub async fn request_info_stream(
    sender: Sender<ThreadRequest>,
    info_sender: Sender<RuntimeInfo>,
    sleep: Duration,
) {
    loop {
        if let Ok(info) = try_request_info(sender.clone()).await {
            if info_sender.send(info).is_err() {
                break;
            }
        }
        tokio::time::sleep(sleep).await;
    }
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, short)]
    ip: Option<Ipv4Addr>,
    #[arg(long, short, default_value_t = 8000)]
    port: u16,
}

#[tokio::main]
async fn main() {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("ai00_server", log::LevelFilter::Trace)
        .init()
        .unwrap();

    let args = Args::parse();
    let (sender, receiver) = flume::unbounded::<ThreadRequest>();
    tokio::task::spawn_blocking(move || model_route(receiver));

    let app = Router::new()
        .route("/", get(root::read_root))
        .route("/exit", post(root::exit))
        .route("/switch-model", post(api::load))
        .route("/unload-model", get(api::unload))
        .route("/models", get(oai::models))
        .route("/v1/models", get(oai::models))
        .route("/completions", post(oai::completions))
        .route("/v1/completions", post(oai::completions))
        .route("/chat/completions", post(oai::chat_completions))
        .route("/v1/chat/completions", post(oai::chat_completions))
        .route("/embeddings", post(oai::embeddings))
        .route("/v1/embeddings", post(oai::embeddings))
        .layer(CorsLayer::permissive())
        .with_state(ThreadState(sender));

    let addr = SocketAddr::from((args.ip.unwrap_or(Ipv4Addr::new(0, 0, 0, 0)), args.port));
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    log::info!("server started at {addr}");
    axum::serve(listener, app).await.unwrap();
}
