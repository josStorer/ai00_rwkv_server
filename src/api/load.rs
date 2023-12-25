use axum::{
    extract::State,
    http::StatusCode,
    Json,
};
use serde::Serialize;
use web_rwkv::model::ModelInfo;

use crate::{
    utils::{try_request_info},
    ReloadRequest, ThreadRequest, ThreadState,
};

#[derive(Debug, Clone, Serialize)]
pub struct InfoResponse {
    reload: ReloadRequest,
    model: ModelInfo,
}

/// `/api/models/load`.
pub async fn load(
    State(ThreadState(sender)): State<ThreadState>,
    Json(request): Json<ReloadRequest>,
) -> StatusCode {
    let (result_sender, result_receiver) = flume::unbounded();
    let _ = sender.send(ThreadRequest::Reload {
        request,
        sender: Some(result_sender),
    });
    match result_receiver.recv_async().await.unwrap() {
        true => StatusCode::OK,
        false => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// `/api/models/unload`.
pub async fn unload(State(ThreadState(sender)): State<ThreadState>) -> StatusCode {
    let _ = sender.send(ThreadRequest::Unload);
    while try_request_info(sender.clone()).await.is_ok() {}
    StatusCode::OK
}
