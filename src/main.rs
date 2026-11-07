//! fm-shim: a small proxy in front of Apple's `fm serve` that rewrites the
//! request shapes it rejects. See `rewrite.rs` for what changes.

mod rewrite;

use std::convert::Infallible;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use http_body_util::{BodyExt, Full, StreamBody, combinators::BoxBody};
use hyper::body::{Frame, Incoming};
use hyper::header::CONTENT_TYPE;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::client::legacy::{Client, connect::HttpConnector};
use hyper_util::rt::{TokioExecutor, TokioIo};
use tokio::net::TcpListener;
use tokio_stream::wrappers::ReceiverStream;

type Body = BoxBody<Bytes, Infallible>;

const USAGE: &str = "\
fm-shim: proxy for Apple's `fm serve` that rewrites requests it rejects

usage: fm-shim [--listen PORT] [--upstream PORT] [--log FILE] [--verbose]

  --listen PORT     port to listen on, 127.0.0.1 only (default 1976)
  --upstream PORT   port `fm serve` listens on, 127.0.0.1 (default 1977)
  --log FILE        append the request log to FILE (default: stderr)
  --verbose         also log each individual rewrite
";

/// Request headers not forwarded upstream (hop-by-hop, or recomputed).
const DROP_REQUEST_HEADERS: [&str; 6] = [
    "host",
    "content-length",
    "connection",
    "accept-encoding",
    "transfer-encoding",
    "keep-alive",
];
/// Response headers not copied back (hop-by-hop; hyper re-frames the body).
const DROP_RESPONSE_HEADERS: [&str; 4] = [
    "transfer-encoding",
    "content-length",
    "connection",
    "keep-alive",
];
/// How long to wait for `fm serve` to start answering.
const UPSTREAM_TIMEOUT: Duration = Duration::from_secs(600);

struct Config {
    listen: u16,
    upstream: u16,
    log: Option<String>,
    verbose: bool,
}

fn parse_args() -> Result<Config, String> {
    let mut cfg = Config {
        listen: 1976,
        upstream: 1977,
        log: None,
        verbose: false,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let (flag, inline) = match arg.split_once('=') {
            Some((f, v)) => (f.to_string(), Some(v.to_string())),
            None => (arg, None),
        };
        let mut value = |name: &str| {
            inline
                .clone()
                .or_else(|| args.next())
                .ok_or(format!("{name} needs a value"))
        };
        match flag.as_str() {
            "--listen" => {
                cfg.listen = value("--listen")?
                    .parse()
                    .map_err(|_| "--listen: not a port")?
            }
            "--upstream" => {
                cfg.upstream = value("--upstream")?
                    .parse()
                    .map_err(|_| "--upstream: not a port")?
            }
            "--log" => cfg.log = Some(value("--log")?),
            "--verbose" | "-v" => cfg.verbose = true,
            "--help" | "-h" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            "--version" | "-V" => {
                println!("fm-shim {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(cfg)
}

struct Logger(Mutex<Box<dyn Write + Send>>);

impl Logger {
    fn line(&self, msg: &str) {
        let ts = chrono::Local::now().format("%H:%M:%S");
        let mut out = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let _ = writeln!(out, "{ts} {msg}");
        let _ = out.flush();
    }
}

struct State {
    upstream: u16,
    verbose: bool,
    log: Logger,
    client: Client<HttpConnector, Full<Bytes>>,
}

fn error_response(status: StatusCode, message: &str) -> Response<Body> {
    let body =
        serde_json::json!({"error": {"message": message, "type": "fm_shim_error"}}).to_string();
    Response::builder()
        .status(status)
        .header(CONTENT_TYPE, "application/json")
        .body(Full::new(Bytes::from(body)).boxed())
        .expect("static response parts are valid")
}

async fn handle(req: Request<Incoming>, st: Arc<State>) -> Result<Response<Body>, Infallible> {
    let (parts, body) = req.into_parts();
    let body = match body.collect().await {
        Ok(c) => c.to_bytes(),
        Err(e) => {
            return Ok(error_response(
                StatusCode::BAD_REQUEST,
                &format!("reading request body: {e}"),
            ));
        }
    };
    let is_chat = parts.uri.path().ends_with("/chat/completions");
    let mut model: Option<String> = None;

    let body = if is_chat && !body.is_empty() {
        let r = rewrite::rewrite(&body);
        model = r.model.clone();
        if let Some(summary) = &r.summary {
            st.log.line(summary);
        }
        if st.verbose {
            for note in &r.notes {
                st.log.line(&format!("  rewrite: {note}"));
            }
        }
        Bytes::from(r.body)
    } else {
        body
    };

    let path = parts.uri.path_and_query().map_or("/", |p| p.as_str());
    let mut up = Request::builder()
        .method(parts.method)
        .uri(format!("http://127.0.0.1:{}{path}", st.upstream));
    for (k, v) in &parts.headers {
        if !DROP_REQUEST_HEADERS.contains(&k.as_str()) {
            up = up.header(k, v);
        }
    }
    let up = match up.body(Full::new(body)) {
        Ok(r) => r,
        Err(e) => {
            return Ok(error_response(
                StatusCode::BAD_REQUEST,
                &format!("building upstream request: {e}"),
            ));
        }
    };

    let t0 = Instant::now();
    let resp = match tokio::time::timeout(UPSTREAM_TIMEOUT, st.client.request(up)).await {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            let msg = format!("fm serve on 127.0.0.1:{} unreachable: {e}", st.upstream);
            st.log.line(&format!("  -> 502 {msg}"));
            return Ok(error_response(StatusCode::BAD_GATEWAY, &msg));
        }
        Err(_) => {
            let msg = format!(
                "fm serve did not answer within {}s",
                UPSTREAM_TIMEOUT.as_secs()
            );
            st.log.line(&format!("  -> 504 {msg}"));
            return Ok(error_response(StatusCode::GATEWAY_TIMEOUT, &msg));
        }
    };

    let status = resp.status();
    let (rparts, mut rbody) = resp.into_parts();
    let is_sse = rparts
        .headers
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("text/event-stream"));

    // A non-streamed context overflow arrives as a 500: answer with OpenAI's 400 instead.
    if is_chat && status == StatusCode::INTERNAL_SERVER_ERROR && !is_sse {
        let body = rbody
            .collect()
            .await
            .map(|c| c.to_bytes())
            .unwrap_or_default();
        if let Some(new) = rewrite::rewrite_overflow_error(&body, model.as_deref()) {
            st.log.line(&format!(
                "  -> 500 context overflow, sent as 400 context_length_exceeded total={:.1}s",
                t0.elapsed().as_secs_f64()
            ));
            return Ok(Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .header(CONTENT_TYPE, "application/json")
                .body(Full::new(Bytes::from(new)).boxed())
                .expect("static response parts are valid"));
        }
        let head = String::from_utf8_lossy(&body[..body.len().min(300)]).into_owned();
        st.log.line(&format!(
            "  -> 500 total={:.1}s bytes={} {head}",
            t0.elapsed().as_secs_f64(),
            body.len()
        ));
        let mut out = Response::builder().status(status);
        for (k, v) in &rparts.headers {
            if !DROP_RESPONSE_HEADERS.contains(&k.as_str()) {
                out = out.header(k, v);
            }
        }
        return Ok(out.body(Full::new(body).boxed()).unwrap_or_else(|e| {
            error_response(StatusCode::BAD_GATEWAY, &format!("building response: {e}"))
        }));
    }

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Frame<Bytes>, Infallible>>(16);
    let rewrite_stream = is_chat && is_sse;

    // Stream the upstream body through, then log one line about it. Event streams are
    // forwarded line by line so a context-overflow error event can be rewritten.
    tokio::spawn(async move {
        let mut first_byte = None;
        let mut total = 0usize;
        let mut head = Vec::new();
        let mut pending: Vec<u8> = Vec::new();
        let mut overflow = false;
        while let Some(frame) = rbody.frame().await {
            let frame = match frame {
                Ok(f) => f,
                Err(e) => {
                    st.log.line(&format!("  upstream body error: {e}"));
                    break;
                }
            };
            let frame = match frame.data_ref() {
                Some(data) => {
                    first_byte.get_or_insert_with(|| t0.elapsed().as_secs_f64());
                    total += data.len();
                    if status.as_u16() >= 400 && head.len() < 500 {
                        head.extend_from_slice(data);
                    }
                    if !rewrite_stream {
                        frame
                    } else {
                        pending.extend_from_slice(data);
                        let Some(end) = pending.iter().rposition(|&b| b == b'\n') else {
                            continue; // no complete line yet
                        };
                        let rest = pending.split_off(end + 1);
                        let ready = std::mem::replace(&mut pending, rest);
                        let (out, changed) = rewrite::rewrite_sse_lines(&ready, model.as_deref());
                        overflow |= changed;
                        Frame::data(Bytes::from(out))
                    }
                }
                None => frame,
            };
            if tx.send(Ok(frame)).await.is_err() {
                break; // client went away
            }
        }
        if !pending.is_empty() {
            let (out, changed) = rewrite::rewrite_sse_lines(&pending, model.as_deref());
            overflow |= changed;
            let _ = tx.send(Ok(Frame::data(Bytes::from(out)))).await;
        }
        if overflow {
            st.log
                .line("  stream reported a context overflow, sent as context_length_exceeded");
        }
        if is_chat {
            let head = String::from_utf8_lossy(&head[..head.len().min(300)]).into_owned();
            st.log.line(&format!(
                "  -> {} first_byte={:.1}s total={:.1}s bytes={total} {head}",
                status.as_u16(),
                first_byte.unwrap_or(0.0),
                t0.elapsed().as_secs_f64()
            ));
        }
    });

    let mut out = Response::builder().status(status);
    for (k, v) in &rparts.headers {
        if !DROP_RESPONSE_HEADERS.contains(&k.as_str()) {
            out = out.header(k, v);
        }
    }
    Ok(out
        .body(StreamBody::new(ReceiverStream::new(rx)).boxed())
        .unwrap_or_else(|e| {
            error_response(StatusCode::BAD_GATEWAY, &format!("building response: {e}"))
        }))
}

#[tokio::main]
async fn main() {
    let cfg = match parse_args() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("fm-shim: {e}\n\n{USAGE}");
            std::process::exit(2);
        }
    };
    let sink: Box<dyn Write + Send> = match &cfg.log {
        Some(path) => match OpenOptions::new().create(true).append(true).open(path) {
            Ok(f) => Box::new(f),
            Err(e) => {
                eprintln!("fm-shim: can't open log {path}: {e}");
                std::process::exit(1);
            }
        },
        None => Box::new(std::io::stderr()),
    };
    let listener = match TcpListener::bind(("127.0.0.1", cfg.listen)).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("fm-shim: can't listen on 127.0.0.1:{}: {e}", cfg.listen);
            std::process::exit(1);
        }
    };
    let st = Arc::new(State {
        upstream: cfg.upstream,
        verbose: cfg.verbose,
        log: Logger(Mutex::new(sink)),
        client: Client::builder(TokioExecutor::new()).build_http(),
    });
    st.log.line(&format!(
        "fm-shim {} listening on 127.0.0.1:{} -> 127.0.0.1:{}",
        env!("CARGO_PKG_VERSION"),
        cfg.listen,
        cfg.upstream
    ));

    loop {
        let stream = match listener.accept().await {
            Ok((s, _)) => s,
            Err(e) => {
                st.log.line(&format!("accept failed: {e}"));
                continue;
            }
        };
        let st = st.clone();
        tokio::spawn(async move {
            let svc = service_fn(move |req| handle(req, st.clone()));
            // Errors here are client-side disconnects; nothing useful to do.
            let _ = http1::Builder::new()
                .serve_connection(TokioIo::new(stream), svc)
                .await;
        });
    }
}
