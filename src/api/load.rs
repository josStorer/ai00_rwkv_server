use std::time::Duration;

use axum::{
    extract::State,
    Json,
};
use serde::Serialize;
use web_rwkv::model::ModelInfo;

use crate::{
    request_info, try_request_info, ReloadRequest, RuntimeInfo, ThreadRequest,
    ThreadState,
};

#[derive(Debug, Clone, Serialize)]
pub struct LoadResponse {
    reload: ReloadRequest,
    model: ModelInfo,
}

/// `/api/models/info`.
pub async fn info(State(ThreadState(sender)): State<ThreadState>) -> Json<LoadResponse> {
    let RuntimeInfo { reload, model, .. } = request_info(sender, Duration::from_millis(500)).await;
    Json(LoadResponse { reload, model })
}

/// `/api/models/load`.
pub async fn load(
    State(ThreadState(sender)): State<ThreadState>,
    Json(request): Json<ReloadRequest>,
) -> Json<LoadResponse> {
    let _ = sender.send(ThreadRequest::Reload(request));
    info(State(ThreadState(sender))).await
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
