use salvo::prelude::*;
use serde::Serialize;
use web_rwkv::model::ModelInfo;

use crate::{
    build_path,
    middleware::{ReloadRequest, SaveRequest, ThreadRequest, ThreadState},
};

use super::*;

#[derive(Debug, Clone, Serialize)]
pub struct InfoResponse {
    reload: ReloadRequest,
    model: ModelInfo,
}

/// `/api/models/load`.
#[handler]
pub async fn load(depot: &mut Depot, req: &mut Request) -> StatusCode {
    let ThreadState { sender, model } = depot.obtain::<ThreadState>().unwrap();
    let (result_sender, result_receiver) = flume::unbounded();
    let mut request: ReloadRequest = req.parse_body().await.unwrap();

    // make sure that we are not visiting un-permitted path.
    request.model = match build_path(model, &request.model) {
        Ok(path) => path,
        Err(_) => return StatusCode::NOT_FOUND,
    };
    for lora in request.lora.iter_mut() {
        lora.path = match build_path(model, &lora.path) {
            Ok(path) => path,
            Err(_) => return StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    let _ = sender.send(ThreadRequest::Reload {
        request: Box::new(request),
        sender: Some(result_sender),
    });
    match result_receiver.recv_async().await.unwrap() {
        true => StatusCode::OK,
        false => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// `/api/models/save`.
#[handler]
pub async fn save(depot: &mut Depot, req: &mut Request) -> StatusCode {
    let ThreadState { sender, model } = depot.obtain::<ThreadState>().unwrap();
    let (result_sender, result_receiver) = flume::unbounded();
    let mut request: SaveRequest = req.parse_body().await.unwrap();

    // make sure that we are not visiting un-permitted path.
    request.model_path = match build_path(model, &request.model_path) {
        Ok(path) => path,
        Err(_) => return StatusCode::NOT_FOUND,
    };

    let _ = sender.send(ThreadRequest::Save {
        request,
        sender: result_sender,
    });
    match result_receiver.recv_async().await.unwrap() {
        true => StatusCode::OK,
        false => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// `/api/models/unload`.
#[handler]
pub async fn unload(depot: &mut Depot) -> StatusCode {
    let ThreadState { sender, .. } = depot.obtain::<ThreadState>().unwrap();
    let _ = sender.send(ThreadRequest::Unload);
    while try_request_info(sender.clone()).await.is_ok() {}
    StatusCode::OK
}
