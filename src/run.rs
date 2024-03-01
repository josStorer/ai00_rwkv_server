use std::{
    borrow::Borrow,
    cmp::Ordering,
    collections::{HashMap, HashSet},
    convert::Infallible,
    future::Future,
    pin::Pin,
    sync::Arc,
    time::Instant,
};

use anyhow::Result;
use flume::{Receiver, Sender};
use itertools::Itertools;
use qp_trie::Trie;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use tokio::sync::{Mutex, RwLock};
use web_rwkv::{
    model::{
        BackedState, Build, Model, ModelInfo, ModelInput, ModelOutput, ModelState, StateBuilder,
    },
    tokenizer::Tokenizer,
};

use crate::{Environment, FinishReason, GenerateRequest, Token, TokenCounter};

const PENALTY_FREE_LIST: [&str; 5] = ["\n", ",", ".", "\u{002c}", "\u{002f}"];
pub const PROMPT_CACHE_TOKENS: usize = 32;

#[derive(Debug)]
pub enum SlotResult {
    /// There is an idle slot ready to be picked up.
    Success(usize),
    /// An idle slot is swapped.
    Fault(usize),
    /// There is no idle slot left.
    Failure(Box<GenerateContext>),
    /// An error occurred.
    Error,
}

#[derive(Debug)]
enum SlotState {
    /// The slot might be either picked up or swapped.
    Idle(Tokens, Instant),
    /// The slot is locked and is waiting for processing.
    Wait(Box<GenerateContext>),
    /// The slot is currently under processing.
    Busy,
}

impl Default for SlotState {
    fn default() -> Self {
        Self::Idle(Default::default(), Instant::now())
    }
}

#[derive(Debug, PartialEq, Eq)]
enum SlotChoice {
    Continue(usize, usize),
    Back(usize),
    Empty(usize),
}

impl std::cmp::Ord for SlotChoice {
    fn cmp(&self, other: &Self) -> Ordering {
        // priority: continue > empty > back
        use SlotChoice::{Back, Continue, Empty};
        match (self, other) {
            (Continue(_, x), Continue(_, y)) => x.cmp(y),
            (Continue(_, _), _) => Ordering::Greater,
            (_, Continue(_, _)) => Ordering::Less,
            (Empty(_), Empty(_)) => Ordering::Equal,
            (Empty(_), Back(_)) => Ordering::Greater,
            (Back(_), Empty(_)) => Ordering::Less,
            (Back(_), Back(_)) => Ordering::Equal,
        }
    }
}

impl std::cmp::PartialOrd for SlotChoice {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, Default)]
pub enum Payload {
    #[default]
    Empty,
    Busy(GenerateContext),
    Done(GenerateContext),
}

impl Payload {
    /// Takes out the value if `self` is [`Payload::Done`], and reset `self` to [`Payload::Empty`].
    pub fn take(&mut self) -> Option<GenerateContext> {
        match std::mem::take(self) {
            Payload::Done(context) => Some(context),
            payload => {
                *self = payload;
                None
            }
        }
    }

    /// Set `self` to [`Payload::Done`] if `self` is [`Payload::Busy`].
    pub fn finalize(&mut self) {
        *self = match std::mem::take(self) {
            Payload::Busy(context) => Payload::Done(context),
            payload => payload,
        }
    }

    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }
}

#[repr(transparent)]
#[derive(Debug, Default, Clone)]
pub struct Tokens(pub Vec<u16>);

impl std::ops::Deref for Tokens {
    type Target = TokenSlice;

    fn deref(&self) -> &Self::Target {
        self.0.as_token_slice()
    }
}

impl Borrow<[u8]> for Tokens {
    fn borrow(&self) -> &[u8] {
        bytemuck::cast_slice(&self.0)
    }
}

impl Borrow<[u16]> for Tokens {
    fn borrow(&self) -> &[u16] {
        &self.0
    }
}

impl Borrow<TokenSlice> for Tokens {
    fn borrow(&self) -> &TokenSlice {
        self.0[..].as_token_slice()
    }
}

impl qp_trie::Break for Tokens {
    type Split = TokenSlice;

    fn empty<'a>() -> &'a Self::Split {
        Default::default()
    }

    fn find_break(&self, loc: usize) -> &Self::Split {
        self.0[..loc >> 1].as_token_slice()
    }
}

#[repr(transparent)]
pub struct TokenSlice([u16]);

impl std::ops::Deref for TokenSlice {
    type Target = [u16];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Borrow<[u8]> for TokenSlice {
    fn borrow(&self) -> &[u8] {
        bytemuck::cast_slice(&self.0)
    }
}

impl Default for &TokenSlice {
    fn default() -> Self {
        <&[u16]>::default().as_token_slice()
    }
}

pub trait AsTokenSlice {
    fn as_token_slice(&self) -> &TokenSlice;
}

impl AsTokenSlice for [u16] {
    fn as_token_slice(&self) -> &TokenSlice {
        let ptr = self as *const [u16] as *const TokenSlice;
        unsafe { &*ptr }
    }
}

#[derive(Debug, Clone)]
pub struct GenerateContext {
    /// Tokens that are provided at first.
    pub prompt_tokens: Vec<u16>,
    /// Whether the prompt has already been processed and cached.
    pub prompt_cached: bool,
    /// Tokens that have been computed and cached.
    pub prefix: Tokens,
    /// Tokens to be computed.
    pub suffix: Tokens,
    /// The accumulated penalties for model-output tokens.
    pub penalties: HashMap<u16, f32>,
    /// Texts that are output by the model.
    pub model_text: Vec<u8>,
    /// Model may output partial utf-8. This makes sure the output is always valid.
    pub buffer: Vec<u8>,
    /// Tokens that are output by the model.
    pub model_tokens: Vec<u16>,
    /// Generate request provided by the caller.
    pub request: GenerateRequest,
    /// To send back generated tokens.
    pub sender: Sender<Token>,
}

pub trait Runner {
    fn info(&self) -> &ModelInfo;
    fn num_batch(&self) -> usize;
    fn tokenizer(&self) -> Arc<Tokenizer>;

    /// Queue an inference task.
    fn queue(
        &self,
        context: GenerateContext,
    ) -> Pin<Box<dyn Future<Output = SlotResult> + Send + '_>>;

    /// Note: only called on the process thread.
    fn process<'a>(
        &'a self,
        payloads: &'a mut [Payload],
    ) -> Pin<Box<dyn Future<Output =Result<()>> + 'a>>;
}

#[derive(Debug)]
pub struct Runtime<M, S, B>
    where
        B: BackedState,
        S: ModelState<BackedState = B>,
        M: Model<State = S>,
        StateBuilder: Build<B, Error = Infallible>,
{
    model: M,
    state: S,
    tokenizer: Arc<Tokenizer>,
    slots: Mutex<Vec<SlotState>>,
    backed: Mutex<Trie<Tokens, Arc<B>>>,
    max_runtime_batch: usize,
    state_chunk_size: usize,
    _penalty_free_tokens: HashSet<u16>,
}

impl<M, S, B> Runtime<M, S, B>
    where
        B: BackedState,
        S: ModelState<BackedState = B>,
        M: Model<State = S>,
        StateBuilder: Build<B, Error = Infallible>,
{
    pub fn new(
        tokenizer: Tokenizer,
        model: M,
        state: S,
        max_runtime_batch: usize,
        state_chunk_size: usize,
    ) -> Self {
        let slots = (0..state.num_batch())
            .map(|_| SlotState::default())
            .collect();
        let _penalty_free_tokens = (0..u16::MAX)
            .filter(|&token| {
                let word = tokenizer.decode(&[token]).unwrap_or_default();
                let word = String::from_utf8_lossy(&word).into_owned();
                PENALTY_FREE_LIST.iter().any(|x| word.contains(x))
            })
            .collect();

        Self {
            model,
            state,
            tokenizer: Arc::new(tokenizer),
            slots: Mutex::new(slots),
            backed: Mutex::new(Trie::new()),
            max_runtime_batch,
            state_chunk_size,
            _penalty_free_tokens,
        }
    }

    /// Search for the longest common prefix in the memory cache and checkout the state from that point.
    /// Should there be a cache miss, an initial state is returned.
    async fn checkout(&self, tokens: &[u16], batch: usize) -> (Vec<u16>, Arc<B>) {
        let mut cache = self.backed.lock().await;
        let prefix = cache.longest_common_prefix(tokens.as_token_slice());
        let len = (1..=prefix.len())
            .rev()
            .find(|len| cache.contains_key(prefix[0..*len].as_token_slice()))
            .unwrap_or_default();
        log::info!("slot {} checks out backed cache of length {}", batch, len);

        let prefix = prefix[0..len].to_vec();
        let reload = match cache.remove(prefix[..].as_token_slice()) {
            Some(reload) => reload,
            None => {
                let context = self.model.context();
                let info = self.model.info();
                let backed = StateBuilder::new(context, info)
                    .with_chunk_size(self.state_chunk_size)
                    .build()
                    .unwrap();
                Arc::new(backed)
            }
        };
        if len > 0 {
            let key = Tokens(prefix.clone());
            cache.insert(key, reload.clone());
        }
        (prefix, reload)
    }

    /// Queue an inference task.
    async fn queue(&self, context: GenerateContext) -> SlotResult {
        let mut slots = self.slots.lock().await;

        // we must ensure that there is at least one token as the suffix, otherwise the whole slot will loop forever as there is no input
        let (last, tokens) = match [context.prefix, context.suffix].concat().split_last() {
            Some((last, tokens)) => (*last, tokens.to_vec()),
            None => return SlotResult::Error,
        };

        let choice = slots
            .iter()
            .enumerate()
            .filter_map(|(batch, slot)| match slot {
                SlotState::Idle(content, time) => {
                    let delta = time.elapsed().as_millis();
                    match (content.is_empty(), tokens.starts_with(content)) {
                        (true, _) => Some((SlotChoice::Empty(batch), delta)),
                        (false, true) => Some((SlotChoice::Continue(batch, content.len()), delta)),
                        (false, false) => Some((SlotChoice::Back(batch), delta)),
                    }
                }
                _ => None,
            })
            .max_by(|lhs, rhs| lhs.0.cmp(&rhs.0).then(lhs.1.cmp(&rhs.1)));

        match choice {
            None => SlotResult::Failure(
                GenerateContext {
                    prefix: Default::default(),
                    suffix: Tokens([tokens, vec![last]].concat()),
                    ..context
                }
                    .into(),
            ),
            Some((SlotChoice::Back(batch), _)) => {
                log::info!("start at non-empty slot {}", batch);
                let (prefix, reload) = self.checkout(&tokens, batch).await;

                let tokens = [tokens, vec![last]].concat();
                let len = prefix.len();
                let mut state = SlotState::Wait(
                    GenerateContext {
                        prefix: Tokens(tokens[..len].to_vec()),
                        suffix: Tokens(tokens[len..].to_vec()),
                        ..context
                    }
                        .into(),
                );

                std::mem::swap(&mut state, &mut slots[batch]);
                match state {
                    SlotState::Idle(_, _) => {
                        // let backed = self.state.back_batch(batch).await.unwarp();
                        // cache.insert(content, backed.into());
                        self.state.load_batch(&reload, batch).unwrap();
                        SlotResult::Fault(batch)
                    }
                    _ => unreachable!(),
                }
            }
            Some((SlotChoice::Empty(batch), _)) => {
                log::info!("start at empty slot {}", batch);
                let (prefix, reload) = self.checkout(&tokens, batch).await;

                let tokens = [tokens, vec![last]].concat();
                let len = prefix.len();
                let state = SlotState::Wait(
                    GenerateContext {
                        prefix: Tokens(tokens[..len].to_vec()),
                        suffix: Tokens(tokens[len..].to_vec()),
                        ..context
                    }
                        .into(),
                );
                slots[batch] = state;

                self.state.load_batch(&reload, batch).expect("load state");
                SlotResult::Fault(batch)
            }
            Some((SlotChoice::Continue(batch, len), _)) => {
                log::info!("continue at slot {}", batch);
                let tokens = [tokens, vec![last]].concat();
                let state = SlotState::Wait(
                    GenerateContext {
                        prefix: Tokens(tokens[..len].to_vec()),
                        suffix: Tokens(tokens[len..].to_vec()),
                        ..context
                    }
                        .into(),
                );
                slots[batch] = state;
                SlotResult::Success(batch)
            }
        }
    }

    /// This critical section synchronizes `slots` and fills `payloads`.
    async fn prepare(&self, payloads: &mut [Payload]) {
        let mut slots = self.slots.lock().await;
        let mut cache = self.backed.lock().await;

        // sync payloads and slots: kill dead payloads
        for (slot, payload) in slots.iter().zip_eq(payloads.iter_mut()) {
            if !(payload.is_empty() || matches!(slot, SlotState::Busy)) {
                log::warn!("payload should either be empty or slot should be busy");
                *payload = Payload::Empty;
            }
        }

        // reset all finished slots to idle
        for (batch, payload) in payloads.iter_mut().enumerate() {
            let Some(context) = payload.take() else {
                continue;
            };

            let backed = self.state.back_batch(batch).await.unwrap();

            if context.request.embed {
                let embed_layer = context
                    .request
                    .embed_layer
                    .clamp(0, self.model.info().num_layer - 1);
                let embed = backed.embed(0, embed_layer);
                let _ = context.sender.send(Token::Embed(embed));
            }

            cache.insert(context.prefix.clone(), backed.into());
            log::info!("backed slot {} of length {}", batch, context.prefix.len());

            assert!(matches!(slots[batch], SlotState::Busy));
            slots[batch] = SlotState::Idle(context.prefix, Instant::now());
        }

        // take data from some waiting slots
        let occupancy = payloads
            .iter()
            .filter(|x| matches!(x, Payload::Busy(_)))
            .count();
        let remain = self.max_runtime_batch - self.max_runtime_batch.min(occupancy);
        let batches = slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| matches!(slot, SlotState::Wait(_)))
            .take(remain)
            .map(|(batch, _)| batch)
            .collect_vec();
        for batch in batches {
            let mut slot = SlotState::Busy;
            std::mem::swap(&mut slots[batch], &mut slot);
            match slot {
                SlotState::Wait(context) => {
                    let _ = context.sender.send(Token::Start);
                    assert!(matches!(payloads[batch], Payload::Empty));
                    payloads[batch] = Payload::Busy(*context);
                }
                _ => unreachable!(),
            };
        }
    }

    async fn process(&self, payloads: &mut [Payload]) -> Result<()> {
        self.prepare(payloads).await;

        let mut inputs = payloads
            .iter()
            .map(|payload| match payload {
                Payload::Busy(context) => context.suffix.0.clone(),
                _ => vec![],
            })
            .map(|tokens| ModelInput {
                tokens,
                ..Default::default()
            })
            .collect_vec();

        // run the model until there is at least one slot finished
        let occupancy = payloads
            .iter()
            .filter(|x| matches!(x, Payload::Busy(_)))
            .count();
        let outputs = match occupancy {
            0 => vec![ModelOutput::None; payloads.len()],
            _ => loop {
                let output = self.model.run(&mut inputs, &self.state).await?;
                if output.iter().any(ModelOutput::is_some) {
                    break output;
                }
            },
        };
        // let penalty_free_tokens = &self._penalty_free_tokens;
        let outputs = payloads
            .par_iter()
            .zip_eq(outputs.into_par_iter())
            .map(|(payload, output)| match payload {
                Payload::Busy(context) => match output {
                    ModelOutput::None => None,
                    ModelOutput::Last(data) => Some(data),
                    ModelOutput::Full(data) => Some(data.into_iter().last()?),
                }
                    .map(|mut data| {
                        context
                            .penalties
                            .iter()
                            // .filter(|(token, _)| !penalty_free_tokens.contains(token))
                            .for_each(|(token, penalty)| data[*token as usize] -= penalty);
                        context
                            .request
                            .logit_bias
                            .iter()
                            .for_each(|(token, bias)| data[*token as usize] += *bias);
                        data
                    }),
                _ => None,
            })
            .map(|x| match x {
                Some(data) => ModelOutput::Last(data),
                None => ModelOutput::None,
            })
            .collect();

        // compute probabilities
        let outputs = match occupancy {
            0 => vec![ModelOutput::None; payloads.len()],
            _ => self.model.softmax(outputs).await?,
        };

        // sample tokens
        let outputs: Vec<_> = payloads
            .par_iter()
            .zip_eq(outputs.into_par_iter())
            .map(|(payload, outputs)| match payload {
                Payload::Busy(context) => match outputs {
                    ModelOutput::None => None,
                    ModelOutput::Last(data) => Some(context.request.sampler.sample(data)),
                    ModelOutput::Full(_) => unreachable!(),
                },
                _ => None,
            })
            .collect();

        for (batch, payload, token, input) in payloads
            .iter_mut()
            .zip_eq(outputs.into_iter().zip_eq(inputs.into_iter()))
            .enumerate()
            .map(|(i, (x, (y, z)))| (i, x, y, z))
        {
            let Payload::Busy(context) = payload else {
                continue;
            };

            let prefix = std::mem::take(&mut context.prefix);
            let suffix = std::mem::take(&mut context.suffix);
            let model_tokens = [prefix.0, suffix.0].concat();

            // compute new prefix and suffix using the current remaining tokens
            assert!(model_tokens.len() >= input.tokens.len());
            let len = model_tokens.len() - input.tokens.len();
            context.prefix = Tokens(model_tokens[..len].to_vec());
            context.suffix = Tokens(model_tokens[len..].to_vec());
            context
                .penalties
                .iter_mut()
                .for_each(|(_, penalty)| *penalty *= context.request.sampler.penalty_decay);

            let Some(token) = token else {
                continue;
            };

            // cache the prompt if it is too long.
            if !context.prompt_cached && context.prompt_tokens.len() > PROMPT_CACHE_TOKENS {
                let mut cache = self.backed.lock().await;
                let backed = self.state.back_batch(batch).await.unwrap();

                cache.insert(context.prefix.clone(), backed.into());
                context.prompt_cached = true;

                log::info!(
                    "backed prompt of slot {} of length {}",
                    batch,
                    context.prefix.len()
                );
            }

            assert_eq!(context.suffix.len(), 0);
            context.suffix.0.push(token);

            let penalty = match context.penalties.get(&token) {
                Some(penalty) => penalty + context.request.sampler.frequency_penalty,
                None => context.request.sampler.presence_penalty,
            };
            context.penalties.insert(token, penalty);

            let mut word = self.tokenizer.decode(&[token])?;
            context.model_text.append(&mut word.clone());
            context.buffer.append(&mut word);
            context.model_tokens.push(token);

            let count_tokens = || {
                let prompt_tokens = context.prompt_tokens.len();
                let completion_tokens = context.model_tokens.len();
                let total_tokens = prompt_tokens + completion_tokens;
                TokenCounter {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                }
            };

            let mut done = false;
            let mut finish = |reason| {
                let _ = context.sender.send(Token::Stop(reason, count_tokens()));
                let _ = context.sender.send(Token::Done);
                done = true;
            };

            // here we detect if there is a stop word in our buffer
            let ((head, tail), stop_matched) = context
                .request
                .stop
                .iter()
                .map(|stop| {
                    let stop = stop.as_bytes();
                    let mut index_safe = 0;
                    let mut index_unsafe = 0;
                    while index_unsafe < context.buffer.len() {
                        // the maximum match of the current stop string
                        let index_stop = index_unsafe - index_safe;
                        if index_stop >= stop.len() {
                            // we have a total match
                            return (index_safe, true);
                        }

                        let output = context.buffer[index_unsafe];
                        let stop = stop[index_stop];

                        index_unsafe += 1;
                        if output != stop {
                            index_safe = index_unsafe;
                        }
                    }
                    (index_safe, index_unsafe - index_safe >= stop.len())
                })
                .min_by(|x, y| match (x.1, y.1) {
                    (true, false) => Ordering::Less,
                    (false, true) => Ordering::Greater,
                    _ => x.0.cmp(&y.0),
                })
                .map(|(mid, matched)| (context.buffer.split_at(mid), matched))
                .unwrap_or(((&context.buffer[..], &[]), false));

            if context.sender.is_disconnected() {
                done = true;
            } else if stop_matched {
                let output = String::from_utf8_lossy(head);
                let _ = context.sender.send(Token::Token(output.into()));
                finish(FinishReason::Stop);
            } else if context.model_tokens.len() >= context.request.max_tokens {
                finish(FinishReason::Length);
            } else if let Ok(word) = String::from_utf8(head.to_vec()) {
                let _ = context.sender.send(Token::Token(word));
                context.buffer = tail.to_vec();
            }

            done.then(|| payload.finalize());
        }

        Ok(())
    }
}

impl<M, S, B> Runner for Runtime<M, S, B>
    where
        B: BackedState + Send + Sync,
        S: ModelState<BackedState = B> + Send + Sync,
        M: Model<State = S> + Send + Sync,
        StateBuilder: Build<B, Error = Infallible>,
{
    #[inline]
    fn info(&self) -> &ModelInfo {
        self.model.info()
    }

    #[inline]
    fn num_batch(&self) -> usize {
        self.state.num_batch()
    }

    #[inline]
    fn tokenizer(&self) -> Arc<Tokenizer> {
        self.tokenizer.clone()
    }

    #[inline]
    fn queue(
        &self,
        context: GenerateContext,
    ) -> Pin<Box<dyn Future<Output = SlotResult> + Send + '_>> {
        Box::pin(self.queue(context))
    }

    #[inline]
    fn process<'a>(
        &'a self,
        payloads: &'a mut [Payload],
    ) -> Pin<Box<dyn Future<Output =Result<()>> + 'a>> {
        Box::pin(self.process(payloads))
    }
}

#[tokio::main]
pub async fn run(receiver: Receiver<()>, env: Arc<RwLock<Environment>>) {
    while let Ok(()) = receiver.recv_async().await {
        if let Environment::Loaded { runtime, .. } = &*env.read().await {
            let mut payloads = vec![Payload::default(); runtime.num_batch()];
            'run: loop {
                if let Err(err) = runtime.process(&mut payloads).await {
                    log::error!("{}", err);
                    break 'run;
                }
                if payloads.iter().all(Payload::is_empty) {
                    break 'run;
                }
            }
        }
    }
}
