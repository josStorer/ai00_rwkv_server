use salvo::handler;

#[handler]
pub async fn read_root() -> &'static str {
    "{\"Hello\":\"World!\"}"
}

#[handler]
pub async fn exit() {
    std::process::exit(0);
}