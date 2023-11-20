use axum::{
    extract::State,
    Json,
};
use serde::Serialize;
use web_rwkv::model::ModelInfo;

use crate::{
    try_request_info, ReloadRequest, ThreadRequest,
    ThreadState,
};

#[derive(Debug, Clone, Serialize)]
pub struct InfoResponse {
    reload: ReloadRequest,
    model: ModelInfo,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum LoadResponse {
    Ok,
    Err,
}

/// `/api/models/load`.
pub async fn load(
    State(ThreadState(sender)): State<ThreadState>,
    Json(request): Json<ReloadRequest>,
) -> Json<LoadResponse> {
    let (result_sender, result_receiver) = flume::unbounded();
    let _ = sender.send(ThreadRequest::Reload {
        request,
        sender: Some(result_sender),
    });
    match result_receiver.recv_async().await.unwrap() {
        true => Json(LoadResponse::Ok),
        false => Json(LoadResponse::Err),
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum UnloadResponse {
    Ok,
}

/// `/api/models/unload`.
pub async fn unload(State(ThreadState(sender)): State<ThreadState>) -> Json<UnloadResponse> {
    let _ = sender.send(ThreadRequest::Unload);
    while try_request_info(sender.clone()).await.is_ok() {}
    Json(UnloadResponse::Ok)
}
