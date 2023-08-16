use std::path::PathBuf;
use axum::{Json};
use axum::extract::State;
use axum::http::{StatusCode};
use axum::response::{IntoResponse};
use serde::{Deserialize};
use web_rwkv::{LayerFlags, Quantization};
use crate::{create_environment, load_model, model_task, ThreadState};

#[derive(Debug, Deserialize)]
pub struct SwitchModelRequest {
    model: String,
    strategy: String,
}

pub async fn switch_model(
    State(ThreadState { tokenizer, receiver, .. }): State<ThreadState>,
    Json(request): Json<SwitchModelRequest>,
) -> impl IntoResponse {
    let model_path = PathBuf::from(request.model);
    let env = match create_environment(Some(0)).await {
        Ok(env) => env,
        Err(err) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("Error: {}", err)),
    };

    {
        let quantization = if request.strategy.contains("i8") {
            let mut layers = LayerFlags::empty();
            (0..32).for_each(|layer| layers |= LayerFlags::from_layer(layer as u64));
            Quantization::Int8(layers)
        } else {
            Quantization::None
        };

        let model = match load_model(env.clone(), model_path, quantization) {
            Ok(model) => model,
            Err(err) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("Error: {}", err)),
        };
        let tokenizer = tokenizer.clone();
        std::thread::spawn(move || model_task(model, tokenizer, receiver));
    }

    (
        StatusCode::OK,
        format!("success"),
    )
}
