use serde::{Deserialize, Serialize};

/// Runtime descriptor for an uncataloged OpenAI-compatible text embedding API.
/// Secrets are referenced by environment-variable name, never stored here.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OpenAiHttpEmbeddingConfig {
    /// Stable local identity used in the model reference `openai-http/<id>`.
    pub id: String,
    /// Model identifier sent to the embedding server.
    pub model: String,
    /// Expected vector dimension. Responses are validated against this value.
    pub dimension: usize,
    /// `cosine`, `dot_product`, or `euclidean`.
    #[serde(default = "default_metric")]
    pub metric: String,
    /// Maximum inputs sent in one request. Set to 1 for single-input servers.
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,
    /// Maximum total UTF-8 bytes of input texts in one request after prefixing.
    #[serde(default = "default_max_input_bytes")]
    pub max_input_bytes: usize,
    /// Maximum concurrent HTTP requests admitted by the runtime.
    #[serde(default = "default_max_concurrency")]
    pub max_concurrency: usize,
    /// Per-request timeout in milliseconds.
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
    /// Number of retries after the initial attempt for retryable failures.
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    /// Initial fallback backoff; doubles on successive failures.
    #[serde(default = "default_retry_base_ms")]
    pub retry_base_ms: u64,
    /// Maximum exponential fallback backoff.
    #[serde(default = "default_retry_max_ms")]
    pub retry_max_ms: u64,
    /// Total retry window including server-provided Retry-After delays.
    #[serde(default = "default_retry_window_ms")]
    pub retry_window_ms: u64,
    /// Optional environment variable containing a Bearer token. Omitted means no auth.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    /// Optional text prepended to query inputs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_prefix: Option<String>,
    /// Optional text prepended to document inputs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document_prefix: Option<String>,
    /// Optional request-field name receiving `query` or `document` purpose.
    /// Unset for standard OpenAI-compatible servers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub purpose_field: Option<String>,
    /// Send `dimensions` only when the configured server supports it.
    #[serde(default)]
    pub send_dimensions: bool,
    /// Optional `encoding_format` request field; currently only `float` is supported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
}

impl OpenAiHttpEmbeddingConfig {
    /// A stable reference that changes whenever the vector space changes.
    #[must_use]
    pub fn reference(&self) -> String {
        use sha2::{Digest, Sha256};
        fn field(hasher: &mut Sha256, value: &[u8]) {
            hasher.update(u64::try_from(value.len()).unwrap_or(u64::MAX).to_le_bytes());
            hasher.update(value);
        }
        fn optional_field(hasher: &mut Sha256, value: Option<&str>) {
            match value {
                Some(value) => {
                    hasher.update([1]);
                    field(hasher, value.as_bytes());
                }
                None => hasher.update([0]),
            }
        }
        let mut hasher = Sha256::new();
        hasher.update(if self.purpose_field.is_some() {
            b"zvec-grep-openai-http-vector-space-v2\0".as_slice()
        } else {
            b"zvec-grep-openai-http-vector-space-v1\0".as_slice()
        });
        field(&mut hasher, self.id.as_bytes());
        field(&mut hasher, self.model.as_bytes());
        let dimension = u64::try_from(self.dimension).unwrap_or(u64::MAX);
        field(&mut hasher, &dimension.to_le_bytes());
        field(&mut hasher, self.metric.as_bytes());
        optional_field(&mut hasher, self.query_prefix.as_deref());
        optional_field(&mut hasher, self.document_prefix.as_deref());
        if self.purpose_field.is_some() {
            optional_field(&mut hasher, self.purpose_field.as_deref());
        }
        hasher.update([u8::from(self.send_dimensions)]);
        optional_field(&mut hasher, self.encoding_format.as_deref());
        let digest = hasher.finalize();
        format!("openai-http/{}-{}", self.id, hex::encode(&digest[..8]))
    }

    pub(crate) fn same_embedding_space(&self, other: &Self) -> bool {
        self.model == other.model
            && self.dimension == other.dimension
            && self.metric == other.metric
            && self.query_prefix == other.query_prefix
            && self.document_prefix == other.document_prefix
            && self.purpose_field == other.purpose_field
            && self.send_dimensions == other.send_dimensions
            && self.encoding_format == other.encoding_format
    }
}

fn default_metric() -> String {
    "cosine".to_owned()
}
fn default_max_batch_size() -> usize {
    64
}
fn default_max_input_bytes() -> usize {
    1_048_576
}
fn default_max_concurrency() -> usize {
    4
}
fn default_timeout_ms() -> u64 {
    60_000
}
fn default_max_retries() -> u32 {
    3
}
fn default_retry_base_ms() -> u64 {
    250
}
fn default_retry_max_ms() -> u64 {
    5_000
}
fn default_retry_window_ms() -> u64 {
    30_000
}

#[cfg(test)]
mod tests {
    use super::OpenAiHttpEmbeddingConfig;
    use serde_json::json;

    #[test]
    fn purpose_field_uses_new_fingerprint_without_changing_legacy_references() {
        let mut descriptor: OpenAiHttpEmbeddingConfig = serde_json::from_value(json!({
            "id": "fixture-model",
            "model": "served-model",
            "dimension": 3
        }))
        .expect("descriptor");
        assert_eq!(
            descriptor.reference(),
            "openai-http/fixture-model-71a58080c0d46699"
        );
        descriptor.purpose_field = Some("input_type".to_owned());
        assert_eq!(
            descriptor.reference(),
            "openai-http/fixture-model-dd295a8e490abe20"
        );
    }
}
