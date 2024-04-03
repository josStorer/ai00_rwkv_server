use std::time::Duration;

use anyhow::Result;
use flume::Sender;

pub use model::{load, save, unload};

use crate::middleware::{RuntimeInfo, ThreadRequest};

pub mod model;
pub mod oai;

pub async fn try_request_info(sender: Sender<ThreadRequest>) -> Result<RuntimeInfo> {
    let (info_sender, info_receiver) = flume::unbounded();
    let _ = sender.send(ThreadRequest::Info(info_sender));
    let _info = info_receiver.recv_async().await?;
    Ok(_info)
}

pub async fn request_info(sender: Sender<ThreadRequest>, sleep: Duration) -> RuntimeInfo {
    loop {
        if let Ok(_info) = try_request_info(sender.clone()).await {
            break _info;
        }
        tokio::time::sleep(sleep).await;
    }
}
