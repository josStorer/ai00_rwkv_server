pub async fn read_root() -> &'static str {
    "{\"Hello\":\"World!\"}"
}

pub async fn exit() {
    std::process::exit(0);
}
