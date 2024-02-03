use std::time::Duration;

use anyhow::Result;
use flume::Sender;

use crate::{RuntimeInfo, ThreadRequest};

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
