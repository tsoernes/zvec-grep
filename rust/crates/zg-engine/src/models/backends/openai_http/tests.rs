use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::{Duration, SystemTime},
};

use async_trait::async_trait;
use serde_json::{Value, json};

use super::model::{
    HttpRequest, HttpResponse, OpenAiHttpClient, OpenAiHttpEmbeddingModel, parse_retry_after,
    parse_vectors,
};
use crate::{
    EngineError,
    domain::{
        Content,
        model::{EmbeddingPurpose, ModelConfig, OpenAiHttpEmbeddingConfig},
    },
    models::spi::{EmbeddingModel, EmbeddingOptions, ModelError},
};

#[derive(Default)]
struct FakeHttp {
    replies: Mutex<VecDeque<Result<HttpResponse, ModelError>>>,
    requests: Mutex<Vec<CapturedRequest>>,
    delay: Mutex<Option<Duration>>,
}

#[derive(Clone, Debug)]
struct CapturedRequest {
    endpoint: String,
    bearer_token: Option<String>,
    body: Value,
    timeout: Duration,
}

#[async_trait]
impl OpenAiHttpClient for FakeHttp {
    async fn post(&self, request: HttpRequest) -> Result<HttpResponse, ModelError> {
        self.requests
            .lock()
            .expect("requests mutex")
            .push(CapturedRequest {
                endpoint: request.endpoint,
                bearer_token: request.bearer_token,
                body: request.body,
                timeout: request.timeout,
            });
        let delay = { *self.delay.lock().expect("delay mutex") };
        if let Some(delay) = delay {
            tokio::time::sleep(delay).await;
        }
        self.replies
            .lock()
            .expect("replies mutex")
            .pop_front()
            .unwrap_or_else(|| Err(ModelError::internal("fake HTTP response queue is empty")))
    }
}

fn config() -> OpenAiHttpEmbeddingConfig {
    serde_json::from_value(json!({
        "id": "fixture",
        "model": "uncataloged-test-model",
        "dimension": 2,
        "maxRetries": 0,
        "retryBaseMs": 0,
        "retryMaxMs": 0,
        "retryWindowMs": 1_000,
        "timeoutMs": 500
    }))
    .expect("descriptor")
}

fn model_with(config: OpenAiHttpEmbeddingConfig, http: Arc<FakeHttp>) -> OpenAiHttpEmbeddingModel {
    model_with_endpoint(config, http, "http://127.0.0.1:4141/v1/embeddings")
}

fn model_with_endpoint(
    config: OpenAiHttpEmbeddingConfig,
    http: Arc<FakeHttp>,
    endpoint: &str,
) -> OpenAiHttpEmbeddingModel {
    let api_key = config
        .api_key_env
        .as_ref()
        .map(|_| "synthetic-token".to_owned());
    OpenAiHttpEmbeddingModel::with_http(
        &config.reference(),
        config.clone(),
        ModelConfig {
            api_key,
            endpoint: Some(endpoint.to_owned()),
            http_embedding: Some(config),
            ..ModelConfig::default()
        },
        http,
    )
    .expect("model")
}

fn ok_response(data: &Value) -> HttpResponse {
    HttpResponse {
        status: 200,
        retry_after: None,
        body: serde_json::to_vec(&json!({"data": data})).expect("json"),
    }
}

fn text(value: &str) -> Vec<Vec<Content>> {
    vec![vec![Content::Text(value.to_owned())]]
}

#[tokio::test]
async fn uses_openai_request_shape_omits_unconfigured_optional_fields_and_restores_indices() {
    let http = Arc::new(FakeHttp::default());
    http.replies
        .lock()
        .expect("replies")
        .push_back(Ok(ok_response(&json!([
            {"index": 1, "embedding": [3.0, 4.0]},
            {"index": 0, "embedding": [1.0, 2.0]}
        ]))));
    let model = model_with(config(), http.clone());
    let result = model
        .embed(
            &[text("first").remove(0), text("second").remove(0)],
            EmbeddingOptions::default(),
        )
        .await
        .expect("embedding");
    assert_eq!(result.vectors, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let requests = http.requests.lock().expect("requests");
    let request = &requests[0];
    assert_eq!(request.endpoint, "http://127.0.0.1:4141/v1/embeddings");
    assert_eq!(request.timeout, Duration::from_millis(500));
    assert_eq!(request.body["model"], "uncataloged-test-model");
    assert_eq!(request.body["input"], json!(["first", "second"]));
    assert!(request.body.get("dimensions").is_none());
    assert!(request.body.get("encoding_format").is_none());
    assert_eq!(request.bearer_token, None);
}

#[tokio::test]
async fn sends_only_configured_optional_fields_prefix_and_env_resolved_bearer_token() {
    let mut descriptor = config();
    descriptor.send_dimensions = true;
    descriptor.encoding_format = Some("float".to_owned());
    descriptor.query_prefix = Some("query: ".to_owned());
    descriptor.api_key_env = Some("SYNTHETIC_TOKEN".to_owned());
    let http = Arc::new(FakeHttp::default());
    http.replies
        .lock()
        .expect("replies")
        .push_back(Ok(ok_response(&json!([
            {"index": 0, "embedding": [1.0, 2.0]}
        ]))));
    let model = model_with_endpoint(
        descriptor,
        http.clone(),
        "https://embedding.example/v1/embeddings",
    );
    model
        .embed(
            &text("sample"),
            EmbeddingOptions {
                purpose: EmbeddingPurpose::Query,
                ..EmbeddingOptions::default()
            },
        )
        .await
        .expect("embedding");
    let requests = http.requests.lock().expect("requests");
    assert_eq!(
        requests[0].endpoint,
        "https://embedding.example/v1/embeddings"
    );
    assert_eq!(requests[0].body["input"], json!(["query: sample"]));
    assert_eq!(requests[0].body["dimensions"], 2);
    assert_eq!(requests[0].body["encoding_format"], "float");
    assert_eq!(requests[0].bearer_token.as_deref(), Some("synthetic-token"));
    assert!(
        !serde_json::to_string(&requests[0].body)
            .expect("body json")
            .contains("synthetic-token")
    );
}

#[tokio::test]
async fn opt_in_purpose_field_distinguishes_query_and_document_requests() {
    let mut descriptor = config();
    descriptor.purpose_field = Some("input_type".to_owned());
    let http = Arc::new(FakeHttp::default());
    http.replies.lock().expect("replies").extend([
        Ok(ok_response(&json!([{"index": 0, "embedding": [1.0, 2.0]}]))),
        Ok(ok_response(&json!([{"index": 0, "embedding": [1.0, 2.0]}]))),
    ]);
    let model = model_with(descriptor, http.clone());
    model
        .embed(
            &text("document fixture"),
            EmbeddingOptions {
                purpose: EmbeddingPurpose::Document,
                ..EmbeddingOptions::default()
            },
        )
        .await
        .expect("document embedding");
    model
        .embed(
            &text("query fixture"),
            EmbeddingOptions {
                purpose: EmbeddingPurpose::Query,
                ..EmbeddingOptions::default()
            },
        )
        .await
        .expect("query embedding");
    let requests = http.requests.lock().expect("requests");
    assert_eq!(requests[0].body["input_type"], "document");
    assert_eq!(requests[1].body["input_type"], "query");
}

#[tokio::test]
async fn single_input_limit_rejects_larger_batch_before_http() {
    let mut descriptor = config();
    descriptor.max_batch_size = 1;
    let http = Arc::new(FakeHttp::default());
    let model = model_with(descriptor, http.clone());
    let error = model
        .embed(
            &[text("a").remove(0), text("b").remove(0)],
            EmbeddingOptions::default(),
        )
        .await
        .expect_err("limit");
    assert_eq!(error.code(), EngineError::INVALID_ARGUMENT);
    assert!(http.requests.lock().expect("requests").is_empty());
}

#[tokio::test]
async fn retries_rate_limit_and_honors_retry_after() {
    let mut descriptor = config();
    descriptor.max_retries = 1;
    descriptor.retry_window_ms = 1_000;
    let http = Arc::new(FakeHttp::default());
    http.replies.lock().expect("replies").extend([
        Ok(HttpResponse {
            status: 429,
            retry_after: Some("0".to_owned()),
            body: Vec::new(),
        }),
        Ok(ok_response(&json!([{"index": 0, "embedding": [1.0, 2.0]}]))),
    ]);
    let model = model_with(descriptor, http.clone());
    assert_eq!(
        model
            .embed(&text("retry-safe"), EmbeddingOptions::default())
            .await
            .expect("retried")
            .vectors,
        vec![vec![1.0, 2.0]]
    );
    assert_eq!(http.requests.lock().expect("requests").len(), 2);
}

#[tokio::test]
async fn retries_http_408_within_the_configured_window() {
    let mut descriptor = config();
    descriptor.max_retries = 1;
    descriptor.retry_window_ms = 1_000;
    let http = Arc::new(FakeHttp::default());
    http.replies.lock().expect("replies").extend([
        Ok(HttpResponse {
            status: 408,
            retry_after: Some("0".to_owned()),
            body: Vec::new(),
        }),
        Ok(ok_response(&json!([{"index": 0, "embedding": [1.0, 2.0]}]))),
    ]);
    let model = model_with(descriptor, http.clone());
    model
        .embed(&text("408-retry"), EmbeddingOptions::default())
        .await
        .expect("retry");
    assert_eq!(http.requests.lock().expect("requests").len(), 2);
}

#[tokio::test]
async fn does_not_retry_auth_failures() {
    let mut descriptor = config();
    descriptor.max_retries = 3;
    let http = Arc::new(FakeHttp::default());
    http.replies
        .lock()
        .expect("replies")
        .push_back(Ok(HttpResponse {
            status: 401,
            retry_after: None,
            body: Vec::new(),
        }));
    let model = model_with(descriptor, http.clone());
    let error = model
        .embed(&text("no-fallback"), EmbeddingOptions::default())
        .await
        .expect_err("401");
    assert_eq!(error.code(), EngineError::PERMISSION_DENIED);
    assert!(error.to_string().contains("HTTP 401"));
    assert_eq!(http.requests.lock().expect("requests").len(), 1);
}

#[tokio::test]
async fn enforces_timeout_and_cancellation_during_http_request() {
    let mut descriptor = config();
    descriptor.timeout_ms = 10;
    descriptor.retry_window_ms = 20;
    let http = Arc::new(FakeHttp::default());
    *http.delay.lock().expect("delay") = Some(Duration::from_millis(100));
    let model = model_with(descriptor, http.clone());
    let error = model
        .embed(&text("timeout"), EmbeddingOptions::default())
        .await
        .expect_err("timeout");
    assert_eq!(error.code(), EngineError::DEADLINE_EXCEEDED);

    let mut descriptor = config();
    descriptor.timeout_ms = 1_000;
    let http = Arc::new(FakeHttp::default());
    *http.delay.lock().expect("delay") = Some(Duration::from_secs(5));
    let model = Arc::new(model_with(descriptor, http));
    let token = tokio_util::sync::CancellationToken::new();
    let cancel = token.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(10)).await;
        cancel.cancel();
    });
    let error = model
        .embed(
            &text("cancel"),
            EmbeddingOptions {
                signal: Some(token),
                ..EmbeddingOptions::default()
            },
        )
        .await
        .expect_err("cancel");
    assert_eq!(error.code(), EngineError::CANCELLED);
}

#[tokio::test]
async fn input_budget_is_total_and_retry_after_cannot_exceed_retry_window() {
    let mut descriptor = config();
    descriptor.max_input_bytes = 4;
    let http = Arc::new(FakeHttp::default());
    http.replies.lock().expect("replies").extend([
        Ok(ok_response(&json!([{"index": 0, "embedding": [1.0, 2.0]}]))),
        Ok(ok_response(&json!([{"index": 0, "embedding": [3.0, 4.0]}]))),
    ]);
    let model = model_with(descriptor.clone(), http.clone());
    let result = model
        .embed(
            &[text("1234").remove(0), text("5678").remove(0)],
            EmbeddingOptions::default(),
        )
        .await
        .expect("split by byte budget");
    assert_eq!(result.vectors, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    assert_eq!(http.requests.lock().expect("requests").len(), 2);
    let error = model
        .embed(&text("12345"), EmbeddingOptions::default())
        .await
        .expect_err("single input exceeds budget");
    assert_eq!(error.code(), EngineError::INVALID_ARGUMENT);
    assert_eq!(http.requests.lock().expect("requests").len(), 2);

    let mut retry_descriptor = config();
    retry_descriptor.max_retries = 2;
    retry_descriptor.retry_window_ms = 25;
    let retry_http = Arc::new(FakeHttp::default());
    retry_http
        .replies
        .lock()
        .expect("replies")
        .push_back(Ok(HttpResponse {
            status: 429,
            retry_after: Some("5".to_owned()),
            body: Vec::new(),
        }));
    let retry_model = model_with(retry_descriptor, retry_http.clone());
    let error = retry_model
        .embed(&text("x"), EmbeddingOptions::default())
        .await
        .expect_err("bounded Retry-After");
    assert!(error.to_string().contains("HTTP 429"));
    assert_eq!(retry_http.requests.lock().expect("requests").len(), 1);
}

#[test]
fn rejects_implicit_or_inline_authentication() {
    let mut descriptor = config();
    descriptor.api_key_env = Some("SYNTHETIC_TOKEN".to_owned());
    let error = OpenAiHttpEmbeddingModel::with_http(
        &descriptor.reference(),
        descriptor.clone(),
        ModelConfig {
            endpoint: Some("https://embedding.example/v1/embeddings".to_owned()),
            http_embedding: Some(descriptor.clone()),
            ..ModelConfig::default()
        },
        Arc::new(FakeHttp::default()),
    )
    .err()
    .expect("missing named environment value");
    assert_eq!(error.code(), EngineError::PERMISSION_DENIED);

    descriptor.api_key_env = None;
    let error = OpenAiHttpEmbeddingModel::with_http(
        &descriptor.reference(),
        descriptor.clone(),
        ModelConfig {
            api_key: Some("must-not-be-used".to_owned()),
            endpoint: Some("https://embedding.example/v1/embeddings".to_owned()),
            http_embedding: Some(descriptor),
            ..ModelConfig::default()
        },
        Arc::new(FakeHttp::default()),
    )
    .err()
    .expect("inline fallback key");
    assert_eq!(error.code(), EngineError::INVALID_ARGUMENT);
}

#[test]
fn validates_http_date_retry_after_and_vector_shapes() {
    let future = httpdate::fmt_http_date(SystemTime::now() + Duration::from_secs(3));
    assert!(parse_retry_after(&future).is_some_and(|delay| delay >= Duration::from_secs(1)));
    assert_eq!(parse_retry_after("0"), Some(Duration::ZERO));
    assert_eq!(parse_retry_after("not-a-date"), None);

    let model = model_with(config(), Arc::new(FakeHttp::default()));
    let info = model.info();
    for invalid in [
        json!({"data": [{"index": 0, "embedding": [1.0]}]}),
        json!({"data": [{"index": 0, "embedding": [1.0, 2.0]}, {"index": 0, "embedding": [3.0, 4.0]}]}),
        json!({"data": [{"index": 2, "embedding": [1.0, 2.0]}]}),
        json!({"data": [{"embedding": [1.0, 2.0]}]}),
        json!({"data": [{"index": 0, "embedding": [1.0e40, 2.0]}]}),
    ] {
        assert!(parse_vectors(info, &invalid, 1).is_err());
    }
    let valid = json!({"data": [{"index": 0, "embedding": [1.0, 2.0]}]});
    assert_eq!(
        parse_vectors(info, &valid, 1).expect("valid vector"),
        vec![vec![1.0, 2.0]]
    );
}
