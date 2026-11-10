use std::{
    sync::Arc,
    time::{Duration, SystemTime},
};

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::{
    EngineError,
    domain::{
        Content, ContentKind,
        model::{
            EmbeddingMetric, EmbeddingModelInfo, EmbeddingPurpose, EmbeddingResult, ModelConfig,
            ModelInfo, OpenAiHttpEmbeddingConfig,
        },
    },
    models::spi::{
        EmbeddingConcurrencyDefaults, EmbeddingModel, EmbeddingOptions, EmbeddingTraceHeaders,
        ModelError, input_text, validate_inputs, validate_result,
    },
};

const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

pub(crate) struct OpenAiHttpEmbeddingModel {
    config: OpenAiHttpEmbeddingConfig,
    info: EmbeddingModelInfo,
    endpoint: String,
    api_key: Option<String>,
    http: Arc<dyn OpenAiHttpClient>,
}

impl OpenAiHttpEmbeddingModel {
    pub(crate) fn new(
        reference: &str,
        config: OpenAiHttpEmbeddingConfig,
        options: ModelConfig,
    ) -> Result<Self, ModelError> {
        Self::with_http(
            reference,
            config,
            options,
            Arc::new(ReqwestOpenAiHttpClient::new()?),
        )
    }

    pub(super) fn with_http(
        reference: &str,
        config: OpenAiHttpEmbeddingConfig,
        options: ModelConfig,
        http: Arc<dyn OpenAiHttpClient>,
    ) -> Result<Self, ModelError> {
        validate_config(reference, &config, &options)?;
        if options.http_embedding.as_ref() != Some(&config) {
            return Err(ModelError::invalid_argument(
                "OpenAI-compatible embedding runtime options do not match the descriptor",
            ));
        }
        let metric = match config.metric.as_str() {
            "cosine" => EmbeddingMetric::Cosine,
            "dot_product" => EmbeddingMetric::DotProduct,
            "euclidean" => EmbeddingMetric::Euclidean,
            _ => {
                return Err(ModelError::invalid_argument(
                    "HTTP embedding metric must be cosine, dot_product, or euclidean",
                ));
            }
        };
        let model_name = config
            .reference()
            .strip_prefix("openai-http/")
            .expect("descriptor reference has provider prefix")
            .to_owned();
        let model = ModelInfo::new(
            "openai-http",
            model_name,
            [ContentKind::Text, ContentKind::Code],
            config.max_batch_size,
        )
        .map_err(|error| ModelError::invalid_argument(error.to_string()))?;
        let info = EmbeddingModelInfo::new(model, config.dimension, metric, None, None)
            .map_err(|error| ModelError::invalid_argument(error.to_string()))?;
        let endpoint = options.endpoint.ok_or_else(|| {
            ModelError::invalid_argument("OpenAI-compatible embedding requires an endpoint")
        })?;
        Ok(Self {
            config,
            info,
            endpoint,
            api_key: options.api_key,
            http,
        })
    }

    async fn embed_inputs(
        &self,
        inputs: &[Vec<Content>],
        options: EmbeddingOptions,
    ) -> Result<EmbeddingResult, ModelError> {
        validate_inputs(&self.info, inputs)?;
        let prefix = match options.purpose {
            EmbeddingPurpose::Document => self.config.document_prefix.as_deref(),
            EmbeddingPurpose::Query => self.config.query_prefix.as_deref(),
        }
        .unwrap_or_default();
        let texts = inputs
            .iter()
            .map(|input| {
                let text = input_text(input)?;
                Ok(format!("{prefix}{text}"))
            })
            .collect::<Result<Vec<_>, ModelError>>()?;
        if let Some((input_index, text)) = texts
            .iter()
            .enumerate()
            .find(|(_, text)| text.len() > self.config.max_input_bytes)
        {
            return Err(ModelError::new(
                EngineError::INVALID_ARGUMENT,
                "HTTP embedding input exceeds configured byte budget",
                Some(format!(
                    "model={} inputIndex={input_index} inputBytes={} maxInputBytes={}",
                    self.info.model().reference(),
                    text.len(),
                    self.config.max_input_bytes
                )),
            ));
        }
        let signal = options.signal;
        let trace_headers = options.trace_headers;
        let mut vectors = Vec::with_capacity(inputs.len());
        let mut start = 0;
        while start < texts.len() {
            let mut end = start;
            let mut request_bytes = 0usize;
            while end < texts.len() && end - start < self.config.max_batch_size {
                let next_bytes = request_bytes.saturating_add(texts[end].len());
                if end > start && next_bytes > self.config.max_input_bytes {
                    break;
                }
                request_bytes = next_bytes;
                end += 1;
            }
            let batch = &texts[start..end];
            let mut body = json!({"model": self.config.model, "input": batch});
            let object = body.as_object_mut().expect("constructed JSON object");
            if self.config.send_dimensions {
                object.insert("dimensions".to_owned(), json!(self.config.dimension));
            }
            if let Some(encoding_format) = &self.config.encoding_format {
                object.insert("encoding_format".to_owned(), json!(encoding_format));
            }
            if let Some(purpose_field) = &self.config.purpose_field {
                let purpose = match options.purpose {
                    EmbeddingPurpose::Document => "document",
                    EmbeddingPurpose::Query => "query",
                };
                object.insert(purpose_field.clone(), json!(purpose));
            }
            let response = self
                .send_with_retry(body, signal.clone(), trace_headers.clone())
                .await?;
            let value: Value = serde_json::from_slice(&response.body).map_err(|_| {
                ModelError::new(
                    EngineError::INTERNAL,
                    "OpenAI-compatible embedding response was not valid JSON",
                    Some(format!(
                        "model={} status={}",
                        self.info.model().reference(),
                        response.status
                    )),
                )
                .shared()
            })?;
            vectors.extend(parse_vectors(&self.info, &value, batch.len())?);
            start = end;
        }
        let result = EmbeddingResult {
            vectors,
            truncated: Vec::new(),
        };
        validate_result(&self.info, inputs.len(), &result)?;
        Ok(result)
    }

    #[allow(
        clippy::too_many_lines,
        reason = "cancellation, retry deadlines, and Retry-After share one bounded request loop"
    )]
    async fn send_with_retry(
        &self,
        body: Value,
        signal: Option<CancellationToken>,
        trace_headers: Option<EmbeddingTraceHeaders>,
    ) -> Result<HttpResponse, ModelError> {
        let started = tokio::time::Instant::now();
        let retry_window = Duration::from_millis(self.config.retry_window_ms);
        let max_attempts = self.config.max_retries.saturating_add(1);
        let mut last_status = None;
        let mut last_error = None;
        let mut last_error_code = EngineError::INTERNAL;
        for attempt in 0..max_attempts {
            if signal.as_ref().is_some_and(CancellationToken::is_cancelled) {
                return Err(ModelError::cancelled(
                    "HTTP embedding request was cancelled",
                ));
            }
            let remaining = retry_window.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                break;
            }
            let timeout = Duration::from_millis(self.config.timeout_ms).min(remaining);
            let request = HttpRequest {
                endpoint: self.endpoint.clone(),
                bearer_token: self.api_key.clone(),
                body: body.clone(),
                trace_headers: trace_headers.clone(),
                timeout,
            };
            let response_timeout = request.timeout;
            let response = if let Some(token) = signal.as_ref() {
                tokio::select! {
                    result = tokio::time::timeout(response_timeout, self.http.post(request)) => match result {
                        Ok(result) => result,
                        Err(_) => Err(request_timeout_error(&self.info, response_timeout)),
                    },
                    () = token.cancelled() => return Err(ModelError::cancelled("HTTP embedding request was cancelled")),
                }
            } else {
                match tokio::time::timeout(response_timeout, self.http.post(request)).await {
                    Ok(result) => result,
                    Err(_) => Err(request_timeout_error(&self.info, response_timeout)),
                }
            };
            match response {
                Ok(response) if (200..300).contains(&response.status) => return Ok(response),
                Ok(response) => {
                    let retryable = response.status == 408
                        || response.status == 429
                        || (500..600).contains(&response.status);
                    last_status = Some(response.status);
                    let retry_after = response.retry_after.as_deref().and_then(parse_retry_after);
                    if !retryable || attempt + 1 >= max_attempts {
                        return Err(http_status_error(
                            &self.info,
                            response.status,
                            attempt + 1,
                            response.retry_after.as_deref(),
                        ));
                    }
                    let fallback = exponential_backoff(&self.config, attempt);
                    let delay = retry_after.unwrap_or(fallback);
                    if delay >= retry_window.saturating_sub(started.elapsed()) {
                        return Err(http_status_error(
                            &self.info,
                            response.status,
                            attempt + 1,
                            response.retry_after.as_deref(),
                        ));
                    }
                    sleep_or_cancel(delay, signal.as_ref()).await?;
                }
                Err(error) => {
                    let retryable = error.is_retryable();
                    last_error_code = error.code();
                    last_error = Some(error.to_string());
                    if !retryable || attempt + 1 >= max_attempts {
                        return Err(transport_failure(
                            &self.info,
                            attempt + 1,
                            last_error.as_deref(),
                            last_error_code,
                        ));
                    }
                    let delay = exponential_backoff(&self.config, attempt);
                    if delay >= retry_window.saturating_sub(started.elapsed()) {
                        return Err(transport_failure(
                            &self.info,
                            attempt + 1,
                            last_error.as_deref(),
                            last_error_code,
                        ));
                    }
                    sleep_or_cancel(delay, signal.as_ref()).await?;
                }
            }
        }
        if let Some(status) = last_status {
            Err(http_status_error(&self.info, status, max_attempts, None))
        } else {
            Err(transport_failure(
                &self.info,
                max_attempts,
                last_error.as_deref(),
                last_error_code,
            ))
        }
    }
}

#[async_trait]
impl EmbeddingModel for OpenAiHttpEmbeddingModel {
    fn info(&self) -> &EmbeddingModelInfo {
        &self.info
    }

    fn concurrency_defaults(&self) -> EmbeddingConcurrencyDefaults {
        EmbeddingConcurrencyDefaults {
            initial: self.config.max_concurrency,
            maximum: self.config.max_concurrency,
        }
    }

    async fn embed(
        &self,
        inputs: &[Vec<Content>],
        options: EmbeddingOptions,
    ) -> Result<EmbeddingResult, ModelError> {
        self.embed_inputs(inputs, options).await
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "the full descriptor must be validated before any HTTP request or credential use"
)]
fn validate_config(
    reference: &str,
    config: &OpenAiHttpEmbeddingConfig,
    options: &ModelConfig,
) -> Result<(), ModelError> {
    let expected_reference = config.reference();
    if reference != expected_reference {
        return Err(ModelError::invalid_argument(
            "HTTP embedding reference must match descriptor id",
        ));
    }
    if config.id.is_empty()
        || !config
            .id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(ModelError::invalid_argument(
            "HTTP embedding descriptor id must be a non-empty path-safe name",
        ));
    }
    if config.model.trim().is_empty() || config.model.chars().any(char::is_control) {
        return Err(ModelError::invalid_argument(
            "HTTP embedding server model id must be non-empty and contain no control characters",
        ));
    }
    if config.dimension == 0
        || config.max_batch_size == 0
        || config.max_input_bytes == 0
        || config.max_concurrency == 0
        || config.timeout_ms == 0
    {
        return Err(ModelError::invalid_argument(
            "HTTP embedding dimensions, limits, concurrency, and timeout must be greater than zero",
        ));
    }
    if config.dimension > 65_536
        || config.max_batch_size > 4096
        || config.max_input_bytes > 64 * 1024 * 1024
        || config.max_concurrency > 64
        || config.timeout_ms > 3_600_000
    {
        return Err(ModelError::invalid_argument(
            "HTTP embedding batch, input, or concurrency limit exceeds the supported maximum",
        ));
    }
    if config.max_retries > 20
        || config.retry_window_ms == 0
        || config.retry_window_ms > 3_600_000
        || config.retry_base_ms > config.retry_max_ms
        || config.retry_max_ms > 300_000
    {
        return Err(ModelError::invalid_argument(
            "HTTP embedding retry settings are outside supported bounds",
        ));
    }
    if !matches!(
        config.metric.as_str(),
        "cosine" | "dot_product" | "euclidean"
    ) {
        return Err(ModelError::invalid_argument(
            "HTTP embedding metric must be cosine, dot_product, or euclidean",
        ));
    }
    if config
        .query_prefix
        .as_ref()
        .is_some_and(|prefix| prefix.len() > config.max_input_bytes)
        || config
            .document_prefix
            .as_ref()
            .is_some_and(|prefix| prefix.len() > config.max_input_bytes)
    {
        return Err(ModelError::invalid_argument(
            "HTTP embedding prefix exceeds the configured input byte budget",
        ));
    }
    if let Some(field) = config.purpose_field.as_deref() {
        let mut bytes = field.bytes();
        let valid_first = bytes
            .next()
            .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_');
        let valid_rest = bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_');
        if !valid_first
            || !valid_rest
            || field.len() > 64
            || matches!(field, "model" | "input" | "dimensions" | "encoding_format")
        {
            return Err(ModelError::invalid_argument(
                "purposeField must be a non-reserved JSON field name",
            ));
        }
    }
    if config
        .encoding_format
        .as_deref()
        .is_some_and(|format| format != "float")
    {
        return Err(ModelError::invalid_argument(
            "HTTP embedding encodingFormat currently supports only float",
        ));
    }
    if let Some(name) = config.api_key_env.as_deref() {
        let mut bytes = name.bytes();
        let valid_first = bytes
            .next()
            .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_');
        if !valid_first || !bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_') {
            return Err(ModelError::invalid_argument(
                "apiKeyEnv must be a non-empty environment-variable name",
            ));
        }
        if options
            .api_key
            .as_deref()
            .is_none_or(|value| value.trim().is_empty())
        {
            return Err(ModelError::new(
                EngineError::PERMISSION_DENIED,
                "HTTP embedding Bearer token environment variable is unavailable",
                Some(format!("apiKeyEnv={name}")),
            ));
        }
    } else if options.api_key.is_some() {
        return Err(ModelError::invalid_argument(
            "HTTP embedding authentication requires apiKeyEnv; inline or fallback API keys are not accepted",
        ));
    }
    Ok(())
}

pub(super) fn parse_vectors(
    info: &EmbeddingModelInfo,
    body: &Value,
    input_count: usize,
) -> Result<Vec<Vec<f32>>, ModelError> {
    let data = body.get("data").and_then(Value::as_array).ok_or_else(|| {
        ModelError::new(
            EngineError::INTERNAL,
            "OpenAI-compatible embedding response did not contain a data array",
            Some(format!("model={}", info.model().reference())),
        )
        .shared()
    })?;
    if data.len() != input_count {
        return Err(ModelError::new(
            EngineError::INTERNAL,
            "OpenAI-compatible embedding response contained the wrong number of vectors",
            Some(format!(
                "model={} inputCount={input_count} vectorCount={}",
                info.model().reference(),
                data.len()
            )),
        )
        .shared());
    }
    let mut vectors = vec![None; input_count];
    for item in data {
        let object = item
            .as_object()
            .ok_or_else(|| invalid_response(info, "embedding item was not an object"))?;
        let index = object.get("index").and_then(Value::as_u64).ok_or_else(|| {
            invalid_response(info, "embedding item omitted a valid integer index")
        })?;
        let index = usize::try_from(index)
            .map_err(|_| invalid_response(info, "embedding index is out of range"))?;
        if index >= input_count {
            return Err(invalid_response(
                info,
                "embedding index is outside the input batch",
            ));
        }
        if vectors[index].is_some() {
            return Err(invalid_response(
                info,
                "embedding response contains a duplicate index",
            ));
        }
        let values = object
            .get("embedding")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid_response(info, "embedding vector was not an array"))?;
        if values.len() != info.dimension() {
            return Err(ModelError::new(
                EngineError::INTERNAL,
                "OpenAI-compatible embedding vector had the wrong dimension",
                Some(format!(
                    "model={} expectedDimension={} actualDimension={}",
                    info.model().reference(),
                    info.dimension(),
                    values.len()
                )),
            )
            .shared());
        }
        let vector = values
            .iter()
            .map(|value| {
                let number = value.as_f64().ok_or_else(|| {
                    invalid_response(info, "embedding vector contained a non-numeric value")
                })?;
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "finite f64 values are range-checked before required f32 vector conversion"
                )]
                let converted = number as f32;
                if !number.is_finite() || !converted.is_finite() {
                    return Err(invalid_response(
                        info,
                        "embedding vector contained a non-finite value",
                    ));
                }
                Ok(converted)
            })
            .collect::<Result<Vec<_>, _>>()?;
        vectors[index] = Some(vector);
    }
    vectors
        .into_iter()
        .map(|vector| {
            vector
                .ok_or_else(|| invalid_response(info, "embedding response omitted an input index"))
        })
        .collect()
}

fn invalid_response(info: &EmbeddingModelInfo, message: &str) -> ModelError {
    ModelError::new(
        EngineError::INTERNAL,
        format!("Invalid OpenAI-compatible embedding response: {message}"),
        Some(format!("model={}", info.model().reference())),
    )
    .shared()
}

fn http_status_error(
    info: &EmbeddingModelInfo,
    status: u16,
    attempts: u32,
    retry_after: Option<&str>,
) -> ModelError {
    let code = match status {
        401 | 403 => EngineError::PERMISSION_DENIED,
        404 => EngineError::NOT_FOUND,
        408 | 504 => EngineError::DEADLINE_EXCEEDED,
        413 | 422 => EngineError::INVALID_ARGUMENT,
        429 => EngineError::RESOURCE_BUSY,
        _ => EngineError::INTERNAL,
    };
    let retry_after_ms = retry_after
        .and_then(parse_retry_after)
        .map(|delay| delay.as_millis());
    let context = format!(
        "model={} status={status} attempts={attempts}{}",
        info.model().reference(),
        retry_after_ms
            .map(|value| format!(" retryAfterMs={value}"))
            .unwrap_or_default()
    );
    ModelError::new(
        code,
        format!("OpenAI-compatible embedding server returned HTTP {status}"),
        Some(context),
    )
    .shared()
}

fn request_timeout_error(info: &EmbeddingModelInfo, timeout: Duration) -> ModelError {
    ModelError::new(
        EngineError::DEADLINE_EXCEEDED,
        "OpenAI-compatible embedding request timed out",
        Some(format!(
            "model={} timeoutMs={}",
            info.model().reference(),
            timeout.as_millis()
        )),
    )
    .transient(None)
}

fn transport_failure(
    info: &EmbeddingModelInfo,
    attempts: u32,
    cause: Option<&str>,
    code: &'static str,
) -> ModelError {
    let context = format!("model={} attempts={attempts}", info.model().reference());
    let mut error = ModelError::new(
        code,
        "OpenAI-compatible embedding transport failed after bounded retries",
        Some(context),
    )
    .shared();
    if let Some(cause) = cause {
        error = error.with_cause(cause);
    }
    error
}

fn exponential_backoff(config: &OpenAiHttpEmbeddingConfig, retry_index: u32) -> Duration {
    let multiplier = 1_u64.checked_shl(retry_index.min(31)).unwrap_or(u64::MAX);
    Duration::from_millis(
        config
            .retry_base_ms
            .saturating_mul(multiplier)
            .min(config.retry_max_ms),
    )
}

pub(super) fn parse_retry_after(value: &str) -> Option<Duration> {
    if let Ok(seconds) = value.trim().parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    let target = httpdate::parse_http_date(value.trim()).ok()?;
    Some(
        target
            .duration_since(SystemTime::now())
            .unwrap_or(Duration::ZERO),
    )
}

async fn sleep_or_cancel(
    delay: Duration,
    signal: Option<&CancellationToken>,
) -> Result<(), ModelError> {
    if let Some(signal) = signal {
        tokio::select! {
            () = tokio::time::sleep(delay) => Ok(()),
            () = signal.cancelled() => Err(ModelError::cancelled("HTTP embedding retry was cancelled")),
        }
    } else {
        tokio::time::sleep(delay).await;
        Ok(())
    }
}

pub(super) struct HttpRequest {
    pub(super) endpoint: String,
    pub(super) bearer_token: Option<String>,
    pub(super) body: Value,
    pub(super) trace_headers: Option<EmbeddingTraceHeaders>,
    pub(super) timeout: Duration,
}

pub(super) struct HttpResponse {
    pub(super) status: u16,
    pub(super) retry_after: Option<String>,
    pub(super) body: Vec<u8>,
}

#[async_trait]
pub(super) trait OpenAiHttpClient: Send + Sync {
    async fn post(&self, request: HttpRequest) -> Result<HttpResponse, ModelError>;
}

struct ReqwestOpenAiHttpClient {
    client: reqwest::Client,
}

impl ReqwestOpenAiHttpClient {
    fn new() -> Result<Self, ModelError> {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .build()
            .map_err(|error| {
                ModelError::internal("Unable to initialize OpenAI-compatible HTTP client")
                    .with_cause(error)
            })?;
        Ok(Self { client })
    }
}

#[async_trait]
impl OpenAiHttpClient for ReqwestOpenAiHttpClient {
    async fn post(&self, request: HttpRequest) -> Result<HttpResponse, ModelError> {
        let endpoint = request.endpoint;
        let request_body = serde_json::to_vec(&request.body).map_err(|error| {
            ModelError::internal("Unable to serialize OpenAI-compatible embedding request")
                .with_cause(error)
        })?;
        let mut builder = self
            .client
            .post(&endpoint)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .timeout(request.timeout)
            .body(request_body);
        if let Some(token) = request.bearer_token.as_deref() {
            builder = builder.bearer_auth(token);
        }
        if let Some(headers) = request.trace_headers {
            builder = builder.header("traceparent", headers.traceparent);
            if let Some(value) = headers.tracestate {
                builder = builder.header("tracestate", value);
            }
            if let Some(value) = headers.baggage {
                builder = builder.header("baggage", value);
            }
        }
        let response = builder.send().await.map_err(|error| {
            let retryable =
                error.is_timeout() || error.is_connect() || error.is_request() || error.is_body();
            let timed_out = error.is_timeout();
            let wrapped = ModelError::new(
                if timed_out {
                    EngineError::DEADLINE_EXCEEDED
                } else {
                    EngineError::INTERNAL
                },
                "OpenAI-compatible embedding HTTP request failed",
                Some(format!(
                    "endpointHost={} timeoutMs={}",
                    endpoint_host(&endpoint),
                    request.timeout.as_millis()
                )),
            )
            .with_cause(error);
            if retryable {
                wrapped.transient(None)
            } else {
                wrapped.shared()
            }
        })?;
        let status = response.status().as_u16();
        let retry_after = response
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let mut body = Vec::new();
        let mut response = response;
        while let Some(chunk) = response.chunk().await.map_err(|error| {
            ModelError::new(
                EngineError::INTERNAL,
                "Failed to read OpenAI-compatible embedding response",
                Some(format!("endpointHost={}", endpoint_host(&endpoint))),
            )
            .with_cause(error)
            .transient(None)
        })? {
            if body.len().saturating_add(chunk.len()) > MAX_RESPONSE_BYTES {
                return Err(ModelError::new(
                    EngineError::RESOURCE_BUSY,
                    "OpenAI-compatible embedding response exceeded the size limit",
                    Some(format!("maxResponseBytes={MAX_RESPONSE_BYTES}")),
                )
                .shared());
            }
            body.extend_from_slice(&chunk);
        }
        Ok(HttpResponse {
            status,
            retry_after,
            body,
        })
    }
}

fn endpoint_host(endpoint: &str) -> String {
    reqwest::Url::parse(endpoint)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned))
        .unwrap_or_else(|| "invalid".to_owned())
}
