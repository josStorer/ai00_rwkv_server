use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::{Path, PathBuf},
};

use anyhow::{bail, Result};
use clap::{CommandFactory, Parser};
use salvo::{
    cors::{AllowHeaders, AllowOrigin, Cors},
    http::Method
    ,
    logging::Logger,
    prelude::*,
    Router
    ,
};
use serde::{Deserialize, Serialize};

use middleware::ThreadRequest;

use crate::middleware::{model_route, ThreadState};

mod api;
mod config;
mod middleware;
mod run;
mod sampler;
mod root;

pub fn build_path(path: impl AsRef<Path>, name: impl AsRef<Path>) -> Result<PathBuf> {
    let permitted = path.as_ref();
    let name = name.as_ref();
    if name.ancestors().any(|p| p.ends_with(Path::new(".."))) {
        bail!("cannot have \"..\" in names");
    }
    let path = match name.is_absolute() || name.starts_with(permitted) {
        true => name.into(),
        false => permitted.join(name),
    };
    match path.starts_with(permitted) {
        true => Ok(path),
        false => bail!("path not permitted"),
    }
}

pub fn check_path_permitted(path: impl AsRef<Path>, permitted: &[&str]) -> Result<()> {
    let current_path = std::env::current_dir()?;
    for sub in permitted {
        let permitted = current_path.join(sub).canonicalize()?;
        let path = path.as_ref().canonicalize()?;
        if path.starts_with(permitted) {
            return Ok(());
        }
    }
    bail!("path not permitted");
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JwtClaims {
    pub sid: String,
    pub exp: i64,
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(long, short)]
    ip: Option<IpAddr>,
    #[arg(long, short)]
    port: Option<u16>,
}

#[tokio::main]
async fn main() {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("ai00_server", log::LevelFilter::Info)
        .with_module_level("web_rwkv", log::LevelFilter::Info)
        .init()
        .unwrap();

    let args = Args::parse();
    let (sender, receiver) = flume::unbounded::<ThreadRequest>();
    tokio::task::spawn_blocking(move || model_route(receiver));

    let cors = Cors::new()
        .allow_origin(AllowOrigin::any())
        .allow_methods(vec![Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(AllowHeaders::any())
        .into_handler();

    let api_router = Router::new().get(root::read_root)
        .push(Router::with_path("exit").post(root::exit))
        .push(Router::with_path("/save-model").post(api::save))
        .push(Router::with_path("/switch-model").post(api::load))
        .push(Router::with_path("/unload-model").get(api::unload))
        .push(Router::with_path("/models").get(api::oai::models))
        .push(Router::with_path("/v1/models").get(api::oai::models))
        .push(Router::with_path("/completions").post(api::oai::completions))
        .push(Router::with_path("/v1/completions").post(api::oai::completions))
        .push(Router::with_path("/chat/completions").post(api::oai::chat_completions))
        .push(Router::with_path("/v1/chat/completions").post(api::oai::chat_completions))
        .push(Router::with_path("/embeddings").post(api::oai::embeddings))
        .push(Router::with_path("/v1/embeddings").post(api::oai::embeddings));

    let app = Router::new()
        //.hoop(CorsLayer::permissive())
        .hoop(Logger::new())
        .hoop(
            affix::inject(ThreadState {
                sender,
                model: Default::default(),
            })
        )
        .push(api_router);

    let cmd = Args::command();
    let version = cmd.get_version().unwrap_or("0.0.1");
    let bin_name = cmd.get_bin_name().unwrap_or("ai00_server");

    let doc = OpenApi::new(bin_name, version).merge_router(&app);

    let app = app
        .push(doc.into_router("/api-doc/openapi.json"))
        .push(SwaggerUi::new("/api-doc/openapi.json").into_router("docs"));
    let service = Service::new(app).hoop(cors);
    let ip_addr = args.ip.unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST));
    let (ipv4_addr, _ipv6_addr) = match ip_addr {
        IpAddr::V4(addr) => (addr, None),
        IpAddr::V6(addr) => (Ipv4Addr::UNSPECIFIED, Some(addr)),
    };
    let port = args.port.unwrap_or(8000);
    let addr = SocketAddr::new(IpAddr::V4(ipv4_addr), port);
    log::info!("server started at {addr} without tls");
    let acceptor = TcpListener::new(addr).bind().await;
    Server::new(acceptor).serve(service).await;
}
