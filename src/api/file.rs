use std::{
    fs::{File, Metadata},
    io::{BufReader, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use anyhow::Result;
use itertools::Itertools;
use salvo::{macros::Extractible, prelude::*};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::ThreadState;

fn compute_sha(path: impl AsRef<Path>, meta: &Metadata) -> Result<String> {
    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::new(file);
    let mut buffer = Vec::new();

    if meta.len() > 10_000_000 {
        let segment_size = meta.len() / 10;
        for i in 0..10 {
            let mut segment_buffer = vec![0; 1_000_000];
            reader.seek(SeekFrom::Start(i * segment_size))?;
            reader.read_exact(&mut segment_buffer)?;
            buffer.extend(segment_buffer);
        }
    } else {
        reader.read_to_end(&mut buffer)?;
    }

    let mut sha = Sha256::new();
    sha.update(&buffer);
    let result = sha.finalize();

    Ok(format!("{:x}", result))
}

#[derive(Debug, Clone, Deserialize, Extractible)]
#[salvo(extract(default_source(from = "body")))]
pub struct FileInfoRequest {
    path: PathBuf,
    #[serde(default)]
    is_sha: bool,
}

#[derive(Debug, Clone, Serialize, ToSchema, ToResponse)]
pub struct FileInfo {
    name: String,
    size: u64,
    sha: String,
}

async fn dir_inner(
    _depot: &mut Depot,
    Json(request): Json<FileInfoRequest>,
) -> Result<(StatusCode, Vec<FileInfo>), StatusCode> {
    match std::fs::read_dir(request.path) {
        Ok(path) => {
            let files = path
                .filter_map(|x| x.ok())
                .filter(|x| x.path().is_file())
                .filter_map(|x| {
                    let path = x.path();
                    let meta = x.metadata().ok()?;

                    let name = x.file_name().to_string_lossy().into();
                    let sha = request
                        .is_sha
                        .then(|| compute_sha(path, &meta).ok())
                        .flatten()
                        .unwrap_or_default();

                    Some(FileInfo {
                        name,
                        size: meta.len(),
                        sha,
                    })
                })
                .collect_vec();
            Ok((StatusCode::OK, files))
        }
        Err(err) => {
            log::error!("failed to read directory: {}", err);
            Err(StatusCode::NOT_FOUND)
        }
    }
}

/// `/api/models/list`.
#[handler]
pub async fn models(depot: &mut Depot, res: &mut Response) {
    let ThreadState { path, .. } = depot.obtain::<ThreadState>().unwrap();
    let request = FileInfoRequest {
        path: path.clone(),
        is_sha: true,
    };
    match dir_inner(depot, Json(request)).await {
        Ok((status, files)) => {
            res.status_code(status);
            res.render(Json(files));
        }
        Err(status) => {
            res.status_code(status);
            res.render("ERROR");
        }
    }
}
