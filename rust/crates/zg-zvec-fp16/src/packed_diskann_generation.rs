//! Safe Rust adapter for the immutable packed-DiskANN generation C ABI.
//!
//! This API builds and opens an immutable sidecar generation. The caller must
//! atomically pin its trust bundle together with the source-row generation.

use std::{ffi::CString, marker::PhantomData, path::Path, ptr::NonNull, rc::Rc};

use sha2::{Digest, Sha256};

use crate::{packed_sign_index::Metric, packed_sign_scale};

const ABI_VERSION: u32 = 1;
const TRUST_MAGIC: &[u8; 8] = b"ZGV1TRST";
const TRUST_ENCODED_LEN: usize = 8 + 4 + 4 + 4 + 8 + 16 + 8 * 32;
const MAX_DIMENSIONS: usize = packed_sign_scale::MAX_DIMENSIONS;

fn take_array<const N: usize>(
    bytes: &[u8],
    cursor: &mut usize,
) -> Result<[u8; N], GenerationError> {
    let end = (*cursor).checked_add(N).ok_or(GenerationError::Trust)?;
    let source = bytes.get(*cursor..end).ok_or(GenerationError::Trust)?;
    let mut value = [0; N];
    value.copy_from_slice(source);
    *cursor = end;
    Ok(value)
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct NativeTrust {
    struct_size: u32,
    version: u32,
    dimension: u32,
    metric: u32,
    count: u64,
    generation: [u8; 16],
    identity_digest: [u8; 32],
    codebook_digest: [u8; 32],
    node_hashes_digest: [u8; 32],
    generation_digest: [u8; 32],
    header_digest: [u8; 32],
    keys_digest: [u8; 32],
    key_mapping_digest: [u8; 32],
    entrypoints_digest: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NativeCandidate {
    key: u64,
    score: f64,
}

#[repr(C)]
struct NativeLimits {
    struct_size: u32,
    version: u32,
    max_documents: u64,
    max_input_bytes: u64,
    max_build_bytes: u64,
    max_codebook_bytes: u64,
    max_generation_bytes: u64,
    max_query_scratch_bytes: u64,
    max_beam: u64,
    max_degree: u32,
    build_list_size: u32,
    build_threads: u32,
    reserved: u32,
}

#[repr(C)]
struct NativeGeneration {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn zvec_binary_diskann_generation_create_v1(
        staging_directory: *const std::ffi::c_char,
        dimension: u32,
        metric: u32,
        keys: *const u64,
        count: usize,
        records: *const u8,
        records_size: usize,
        generation: *const u8,
        identity_digest: *const u8,
        limits: *const NativeLimits,
        trust_out: *mut NativeTrust,
        native_error_out: *mut i32,
    ) -> u32;
    fn zvec_binary_diskann_generation_open_v1(
        generation_directory: *const std::ffi::c_char,
        trusted: *const NativeTrust,
        keys: *const u64,
        count: usize,
        limits: *const NativeLimits,
        out: *mut *mut NativeGeneration,
        native_error_out: *mut i32,
    ) -> u32;
    fn zvec_binary_diskann_generation_search_v1(
        generation: *const NativeGeneration,
        query: *const f32,
        query_dimensions: usize,
        limit: usize,
        beam: usize,
        out: *mut NativeCandidate,
        capacity: usize,
        written: *mut usize,
        native_error_out: *mut i32,
    ) -> u32;
    fn zvec_binary_diskann_generation_destroy_v1(generation: *mut NativeGeneration);
}

/// Hard logical limits passed into native create/open/search operations.
/// They are not a substitute for process or cgroup memory limits.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_documents: usize,
    pub max_input_bytes: usize,
    pub max_build_bytes: usize,
    pub max_codebook_bytes: usize,
    pub max_generation_bytes: usize,
    pub max_query_scratch_bytes: usize,
    pub max_beam: usize,
    pub max_degree: u32,
    pub build_list_size: u32,
    pub build_threads: u32,
}

impl Limits {
    fn to_native(self) -> Result<NativeLimits, GenerationError> {
        let convert = |value: usize| u64::try_from(value).map_err(|_| GenerationError::Limit);
        if self.max_documents == 0
            || self.max_documents > u32::MAX as usize
            || self.max_input_bytes == 0
            || self.max_build_bytes == 0
            || self.max_codebook_bytes == 0
            || self.max_generation_bytes == 0
            || self.max_query_scratch_bytes == 0
            || self.max_beam == 0
            || self.max_degree == 0
            || self.max_degree > 1024
            || self.build_list_size == 0
            || self.build_list_size > 100_000
            || self.build_threads == 0
            || self.build_threads > 64
        {
            return Err(GenerationError::Limit);
        }
        Ok(NativeLimits {
            struct_size: u32::try_from(std::mem::size_of::<NativeLimits>())
                .map_err(|_| GenerationError::Limit)?,
            version: ABI_VERSION,
            max_documents: convert(self.max_documents)?,
            max_input_bytes: convert(self.max_input_bytes)?,
            max_build_bytes: convert(self.max_build_bytes)?,
            max_codebook_bytes: convert(self.max_codebook_bytes)?,
            max_generation_bytes: convert(self.max_generation_bytes)?,
            max_query_scratch_bytes: convert(self.max_query_scratch_bytes)?,
            max_beam: convert(self.max_beam)?,
            max_degree: self.max_degree,
            build_list_size: self.build_list_size,
            build_threads: self.build_threads,
            reserved: 0,
        })
    }
}

/// Native trust material that must be pinned in the outer workspace manifest.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Trust(NativeTrust);

impl Trust {
    /// Encode a fixed-width manifest value. The bundle is trusted only because
    /// the outer workspace manifest commits it with the same storage generation.
    #[must_use]
    pub fn to_bytes(self) -> Vec<u8> {
        let value = self.0;
        let mut bytes = Vec::with_capacity(TRUST_ENCODED_LEN);
        bytes.extend_from_slice(TRUST_MAGIC);
        bytes.extend_from_slice(&value.version.to_le_bytes());
        bytes.extend_from_slice(&value.dimension.to_le_bytes());
        bytes.extend_from_slice(&value.metric.to_le_bytes());
        bytes.extend_from_slice(&value.count.to_le_bytes());
        bytes.extend_from_slice(&value.generation);
        for digest in [
            &value.identity_digest,
            &value.codebook_digest,
            &value.node_hashes_digest,
            &value.generation_digest,
            &value.header_digest,
            &value.keys_digest,
            &value.key_mapping_digest,
            &value.entrypoints_digest,
        ] {
            bytes.extend_from_slice(digest);
        }
        bytes
    }

    /// Parse and validate a manifest-pinned trust bundle.
    ///
    /// # Errors
    /// Returns [`GenerationError::Trust`] for malformed, unsupported, or
    /// invalid fixed-width metadata.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, GenerationError> {
        if bytes.len() != TRUST_ENCODED_LEN || &bytes[..8] != TRUST_MAGIC {
            return Err(GenerationError::Trust);
        }
        let mut cursor = 8;
        let value = NativeTrust {
            struct_size: u32::try_from(std::mem::size_of::<NativeTrust>())
                .map_err(|_| GenerationError::Trust)?,
            version: u32::from_le_bytes(take_array(bytes, &mut cursor)?),
            dimension: u32::from_le_bytes(take_array(bytes, &mut cursor)?),
            metric: u32::from_le_bytes(take_array(bytes, &mut cursor)?),
            count: u64::from_le_bytes(take_array(bytes, &mut cursor)?),
            generation: take_array(bytes, &mut cursor)?,
            identity_digest: take_array(bytes, &mut cursor)?,
            codebook_digest: take_array(bytes, &mut cursor)?,
            node_hashes_digest: take_array(bytes, &mut cursor)?,
            generation_digest: take_array(bytes, &mut cursor)?,
            header_digest: take_array(bytes, &mut cursor)?,
            keys_digest: take_array(bytes, &mut cursor)?,
            key_mapping_digest: take_array(bytes, &mut cursor)?,
            entrypoints_digest: take_array(bytes, &mut cursor)?,
        };
        if cursor != TRUST_ENCODED_LEN
            || value.version != ABI_VERSION
            || !usize::try_from(value.dimension)
                .is_ok_and(|dimension| (1..=MAX_DIMENSIONS).contains(&dimension))
            || !(1..=3).contains(&value.metric)
            || value.count == 0
            || value.generation.iter().all(|byte| *byte == 0)
            || [
                &value.identity_digest,
                &value.codebook_digest,
                &value.node_hashes_digest,
                &value.generation_digest,
                &value.header_digest,
                &value.keys_digest,
                &value.key_mapping_digest,
                &value.entrypoints_digest,
            ]
            .into_iter()
            .any(|digest| digest.iter().all(|byte| *byte == 0))
        {
            return Err(GenerationError::Trust);
        }
        Ok(Self(value))
    }

    /// Validate trust against generation metadata and the canonical identity digest.
    ///
    /// # Errors
    /// Returns [`GenerationError::Trust`] for metadata mismatches and
    /// [`GenerationError::Limit`] when a count cannot be represented.
    pub fn validate_identity(
        &self,
        dimension: usize,
        metric: Metric,
        count: usize,
        generation: [u8; 16],
        identity_digest: [u8; 32],
    ) -> Result<(), GenerationError> {
        self.validate_generation(dimension, metric, count, generation)?;
        if self.0.identity_digest != identity_digest {
            return Err(GenerationError::Trust);
        }
        Ok(())
    }

    /// Validate trust against the generation's dimension, metric, count, and ID.
    ///
    /// # Errors
    /// Returns [`GenerationError::Trust`] for metadata mismatches and
    /// [`GenerationError::Limit`] when a count cannot be represented.
    pub fn validate_generation(
        &self,
        dimension: usize,
        metric: Metric,
        count: usize,
        generation: [u8; 16],
    ) -> Result<(), GenerationError> {
        let expected_count = u64::try_from(count).map_err(|_| GenerationError::Limit)?;
        if usize::try_from(self.0.dimension).ok() != Some(dimension)
            || self.0.metric != metric_number(metric)
            || self.0.count != expected_count
            || self.0.generation != generation
        {
            return Err(GenerationError::Trust);
        }
        Ok(())
    }
}

/// Opaque native generation handle. It deliberately remains `!Send`/`!Sync`;
/// use the owner-thread worker when sharing queries across Rust tasks.
pub struct GenerationHandle {
    raw: NonNull<NativeGeneration>,
    trust: Trust,
    limits: Limits,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl Drop for GenerationHandle {
    fn drop(&mut self) {
        // SAFETY: this wrapper uniquely owns the native handle and no C ABI call
        // is in flight because the handle is confined to its owner thread.
        unsafe { zvec_binary_diskann_generation_destroy_v1(self.raw.as_ptr()) };
    }
}

/// One graph candidate from the native ANN beam.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    pub key: u64,
    pub score: f64,
}

/// Fail-closed errors returned by the packed-generation ABI adapter.
#[derive(Debug)]
pub enum GenerationError {
    InvalidInput,
    Path,
    Limit,
    Trust,
    Index(crate::packed_sign_index::IndexError),
    Native { status: u32, native_error: i32 },
}

impl From<crate::packed_sign_index::IndexError> for GenerationError {
    fn from(error: crate::packed_sign_index::IndexError) -> Self {
        Self::Index(error)
    }
}

/// SHA-256 used for the identity digest independently pinned in the workspace.
///
/// # Errors
/// Returns [`GenerationError::Limit`] if the input is empty or over budget.
pub fn identity_sha256(bytes: &[u8], max_bytes: usize) -> Result<[u8; 32], GenerationError> {
    if bytes.is_empty() || bytes.len() > max_bytes {
        return Err(GenerationError::Limit);
    }
    let digest = Sha256::digest(bytes);
    Ok(digest.into())
}

/// Hash the identity map and packed vector records using the native V1 domain
/// separator. This digest binds the candidate graph's record order to the
/// caller's canonical IDs; it is distinct from the sidecar-only hash above.
///
/// # Errors
/// Returns [`GenerationError::Limit`] for empty, oversized, or unrepresentable
/// inputs.
pub fn identity_record_sha256(
    identity: &[u8],
    records: &[u8],
    max_bytes: usize,
) -> Result<[u8; 32], GenerationError> {
    const DOMAIN: &[u8; 12] = b"ZGV1-IDREC\0\x01";
    const LENGTH_BYTES: usize = 2 * std::mem::size_of::<u64>();
    let total = DOMAIN
        .len()
        .checked_add(LENGTH_BYTES)
        .and_then(|bytes| bytes.checked_add(identity.len()))
        .and_then(|bytes| bytes.checked_add(records.len()))
        .ok_or(GenerationError::Limit)?;
    if identity.is_empty() || records.is_empty() || total > max_bytes {
        return Err(GenerationError::Limit);
    }
    let identity_len = u64::try_from(identity.len()).map_err(|_| GenerationError::Limit)?;
    let records_len = u64::try_from(records.len()).map_err(|_| GenerationError::Limit)?;
    let mut digest = Sha256::new();
    digest.update(DOMAIN);
    digest.update(identity_len.to_le_bytes());
    digest.update(identity);
    digest.update(records_len.to_le_bytes());
    digest.update(records);
    Ok(digest.finalize().into())
}

/// Inputs required to create a fresh immutable graph generation.
#[derive(Clone, Copy)]
pub struct CreateParams<'a> {
    /// New, absolute staging directory for native graph files.
    pub staging_directory: &'a Path,
    /// Number of dimensions in each source vector.
    pub dimension: usize,
    /// Similarity metric used by native construction and search.
    pub metric: Metric,
    /// Canonical one-based keys in record order.
    pub keys: &'a [u64],
    /// Packed sign records corresponding to `keys`.
    pub records: &'a [u8],
    /// Nonzero immutable generation identifier.
    pub generation: [u8; 16],
    /// Independently computed identity-record digest.
    pub identity_digest: [u8; 32],
    /// Logical resource limits for native construction.
    pub limits: Limits,
}

/// Create a new graph in a fresh staging directory.
///
/// # Errors
/// Returns an error for invalid generation inputs, resource limits, trust
/// validation, path conversion, or native graph construction failures.
pub fn create_v1(params: &CreateParams<'_>) -> Result<Trust, GenerationError> {
    let CreateParams {
        staging_directory,
        dimension,
        metric,
        keys,
        records,
        generation,
        identity_digest,
        limits,
    } = *params;
    if !staging_directory.is_absolute()
        || generation.iter().all(|byte| *byte == 0)
        || identity_digest.iter().all(|byte| *byte == 0)
        || !(1..=MAX_DIMENSIONS).contains(&dimension)
        || keys.is_empty()
        || keys.len() > limits.max_documents
        || u32::try_from(keys.len()).is_err()
    {
        return Err(GenerationError::InvalidInput);
    }
    let record_width = dimension
        .div_ceil(32)
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(4))
        .ok_or(GenerationError::Limit)?;
    let key_bytes = keys
        .len()
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or(GenerationError::Limit)?;
    let input_bytes = records
        .len()
        .checked_add(key_bytes)
        .ok_or(GenerationError::Limit)?;
    if records.len()
        != keys
            .len()
            .checked_mul(record_width)
            .ok_or(GenerationError::Limit)?
        || input_bytes > limits.max_input_bytes
        || keys.iter().enumerate().any(|(i, key)| {
            u64::try_from(i).ok().and_then(|value| value.checked_add(1)) != Some(*key)
        })
    {
        return Err(GenerationError::InvalidInput);
    }
    let dimension_usize = dimension;
    let dimension = u32::try_from(dimension_usize).map_err(|_| GenerationError::Limit)?;
    let native_limits = limits.to_native()?;
    let path = native_path(staging_directory)?;
    let mut trust = NativeTrust::default();
    let mut native_error = 0;
    // SAFETY: all pointed-to slices and fixed-size outputs remain live for the
    // entire synchronous call; the native API copies input and does not retain pointers.
    let status = unsafe {
        zvec_binary_diskann_generation_create_v1(
            path.as_ptr(),
            dimension,
            metric_number(metric),
            keys.as_ptr(),
            keys.len(),
            records.as_ptr(),
            records.len(),
            generation.as_ptr(),
            identity_digest.as_ptr(),
            &raw const native_limits,
            &raw mut trust,
            &raw mut native_error,
        )
    };
    if status != 0 {
        return Err(GenerationError::Native {
            status,
            native_error,
        });
    }
    let trust = Trust(trust);
    let trust = Trust::from_bytes(&trust.to_bytes())?;
    trust.validate_identity(
        dimension_usize,
        metric,
        keys.len(),
        generation,
        identity_digest,
    )?;
    Ok(trust)
}

/// Open a graph after validating its independently pinned identity and trust.
///
/// # Errors
/// Returns an error for invalid keys, trust metadata, limits, paths, or native
/// graph verification failures.
pub fn open_v1(
    generation_directory: &Path,
    trust: Trust,
    keys: &[u64],
    limits: Limits,
) -> Result<GenerationHandle, GenerationError> {
    if !generation_directory.is_absolute()
        || keys.is_empty()
        || keys.len() > limits.max_documents
        || u32::try_from(keys.len()).is_err()
        || keys
            .len()
            .checked_mul(std::mem::size_of::<u64>())
            .is_none_or(|bytes| bytes > limits.max_input_bytes)
        || keys.iter().enumerate().any(|(i, key)| {
            u64::try_from(i).ok().and_then(|value| value.checked_add(1)) != Some(*key)
        })
        || trust.0.count != u64::try_from(keys.len()).map_err(|_| GenerationError::Limit)?
    {
        return Err(GenerationError::InvalidInput);
    }
    let native_limits = limits.to_native()?;
    let path = native_path(generation_directory)?;
    let mut raw = std::ptr::null_mut();
    let mut native_error = 0;
    // SAFETY: the path, trust, key array and limits remain live during the call.
    let status = unsafe {
        zvec_binary_diskann_generation_open_v1(
            path.as_ptr(),
            &raw const trust.0,
            keys.as_ptr(),
            keys.len(),
            &raw const native_limits,
            &raw mut raw,
            &raw mut native_error,
        )
    };
    if status != 0 {
        return Err(GenerationError::Native {
            status,
            native_error,
        });
    }
    let raw = NonNull::new(raw).ok_or(GenerationError::Trust)?;
    Ok(GenerationHandle {
        raw,
        trust,
        limits,
        _not_send_or_sync: PhantomData,
    })
}

impl GenerationHandle {
    #[must_use]
    pub fn trust(&self) -> Trust {
        self.trust
    }

    /// Search an immutable graph and return validated native candidates.
    ///
    /// # Errors
    /// Returns an error for invalid query dimensions or values, a mismatched
    /// metric, an exceeded resource bound, or a native search failure.
    pub fn search_v1(
        &self,
        query: &[f32],
        metric: Metric,
        limit: usize,
        beam: usize,
    ) -> Result<Vec<Candidate>, GenerationError> {
        let dimension =
            usize::try_from(self.trust.0.dimension).map_err(|_| GenerationError::Limit)?;
        packed_sign_scale::validate_query(query, dimension, metric == Metric::Cosine)
            .map_err(crate::packed_sign_index::IndexError::from)?;
        if limit == 0 {
            return Ok(Vec::new());
        }
        let query_bytes = dimension
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(GenerationError::Limit)?;
        let count = usize::try_from(self.trust.0.count).map_err(|_| GenerationError::Limit)?;
        if self.trust.0.metric != metric_number(metric) || beam == 0 {
            return Err(GenerationError::Limit);
        }
        let effective_limit = limit.min(count);
        if effective_limit > self.limits.max_beam {
            return Err(GenerationError::Limit);
        }
        let effective_beam = beam
            .min(count)
            .min(self.limits.max_beam)
            .max(effective_limit);
        let candidate_bytes = effective_beam
            .checked_mul(2 * std::mem::size_of::<NativeCandidate>())
            .ok_or(GenerationError::Limit)?;
        let query_working_bytes = query_bytes
            .checked_add(candidate_bytes)
            .ok_or(GenerationError::Limit)?;
        if query_working_bytes > self.limits.max_query_scratch_bytes {
            return Err(GenerationError::Limit);
        }
        let mut native_candidates = Vec::new();
        native_candidates
            .try_reserve_exact(effective_beam)
            .map_err(|_| GenerationError::Limit)?;
        native_candidates.resize(effective_beam, NativeCandidate::default());
        let mut written = 0usize;
        let mut native_error = 0;
        // SAFETY: the handle is live and confined to this owner thread; query,
        // output and count storage remain valid for the duration of the call.
        let status = unsafe {
            zvec_binary_diskann_generation_search_v1(
                self.raw.as_ptr(),
                query.as_ptr(),
                query.len(),
                effective_limit,
                effective_beam,
                native_candidates.as_mut_ptr(),
                native_candidates.len(),
                &raw mut written,
                &raw mut native_error,
            )
        };
        if status != 0 {
            return Err(GenerationError::Native {
                status,
                native_error,
            });
        }
        if written > effective_beam {
            return Err(GenerationError::Trust);
        }
        let mut candidates = Vec::new();
        candidates
            .try_reserve_exact(written)
            .map_err(|_| GenerationError::Limit)?;
        let mut seen = std::collections::HashSet::with_capacity(written);
        for candidate in &native_candidates[..written] {
            if candidate.key == 0
                || candidate.key > self.trust.0.count
                || !candidate.score.is_finite()
                || !seen.insert(candidate.key)
            {
                return Err(GenerationError::Trust);
            }
            candidates.push(Candidate {
                key: candidate.key,
                score: candidate.score,
            });
        }
        Ok(candidates)
    }
}

fn native_path(path: &Path) -> Result<CString, GenerationError> {
    CString::new(path.as_os_str().as_encoded_bytes()).map_err(|_| GenerationError::Path)
}

fn metric_number(metric: Metric) -> u32 {
    match metric {
        Metric::Cosine => 1,
        Metric::DotProduct => 2,
        Metric::Euclidean => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ABI_VERSION, GenerationError, Limits, NativeTrust, TRUST_ENCODED_LEN, Trust,
        identity_record_sha256, identity_sha256,
    };

    fn limits() -> Limits {
        Limits {
            max_documents: 10,
            max_input_bytes: 4096,
            max_build_bytes: 4096,
            max_codebook_bytes: 4096,
            max_generation_bytes: 4096,
            max_query_scratch_bytes: 4096,
            max_beam: 16,
            max_degree: 4,
            build_list_size: 16,
            build_threads: 2,
        }
    }

    fn trust() -> Trust {
        Trust(NativeTrust {
            struct_size: u32::try_from(std::mem::size_of::<NativeTrust>()).expect("struct size"),
            version: ABI_VERSION,
            dimension: 2,
            metric: 1,
            count: 2,
            generation: [0x42; 16],
            identity_digest: [1; 32],
            codebook_digest: [2; 32],
            node_hashes_digest: [3; 32],
            generation_digest: [4; 32],
            header_digest: [5; 32],
            keys_digest: [6; 32],
            key_mapping_digest: [7; 32],
            entrypoints_digest: [8; 32],
        })
    }

    #[test]
    fn trust_bundle_is_strict_and_fixed_width() {
        let expected = trust();
        let bytes = expected.to_bytes();
        assert_eq!(bytes.len(), TRUST_ENCODED_LEN);
        assert_eq!(Trust::from_bytes(&bytes).expect("valid trust"), expected);
        assert!(matches!(
            Trust::from_bytes(&bytes[..bytes.len() - 1]),
            Err(GenerationError::Trust)
        ));
        let mut bad = bytes;
        bad[0] ^= 0x80;
        assert!(matches!(
            Trust::from_bytes(&bad),
            Err(GenerationError::Trust)
        ));
    }

    #[test]
    fn identity_record_digest_is_domain_separated_and_bounded() {
        let identity = b"id-sidecar";
        let records = b"packed-records";
        let sidecar_digest = identity_sha256(identity, 1024).expect("sidecar hash");
        let row_digest = identity_record_sha256(identity, records, 1024).expect("row hash");
        assert_ne!(sidecar_digest, row_digest);
        assert!(matches!(
            identity_record_sha256(identity, records, 1),
            Err(GenerationError::Limit)
        ));
    }

    #[test]
    fn native_limits_reject_zero_and_oversized_build_settings() {
        let mut invalid = limits();
        invalid.max_beam = 0;
        assert!(matches!(invalid.to_native(), Err(GenerationError::Limit)));
        let mut invalid = limits();
        invalid.build_threads = 65;
        assert!(matches!(invalid.to_native(), Err(GenerationError::Limit)));
    }
}
