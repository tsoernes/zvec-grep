//! Deterministic identity mapping and exact scoring for experimental V1 data.
//!
//! This module builds no `DiskANN` graph and does not publish an index. A native
//! facade must bind its ordered keys and packed records to a verified immutable
//! generation. Its ANN candidates are approximate; exact reranking cannot
//! recover a correct document omitted by native graph navigation.

use crate::packed_sign_scale::{self, ValidationError};

/// Existing zg score convention: higher is better.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Metric {
    /// Full FP32 query cosine against the reconstructed sign vector.
    Cosine,
    /// Full FP32 query dot product against signs and document scale.
    DotProduct,
    /// Negative Euclidean distance against signs and document scale.
    Euclidean,
}

/// A document's stable identity, packed signs and reconstruction scale.
#[derive(Clone, Copy, Debug)]
pub struct Document<'a> {
    /// Stable source fragment identity. The caller retains its provenance.
    pub id: &'a str,
    /// LSB-first signs without 32-bit alignment padding.
    pub signs: &'a [u8],
    /// Finite, nonnegative per-document scale.
    pub scale: f32,
}

/// Explicit validation failures before a native call or result publication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndexError {
    /// Sign, dimension, scale or query validation failed.
    Record(ValidationError),
    /// At least one document identity is empty.
    EmptyId,
    /// Two records claim the same document identity.
    DuplicateId,
    /// The generation identifier is all zero.
    InvalidGeneration,
    /// There is no graph entrypoint for an empty generation.
    EmptyGeneration,
    /// A numeric key is invalid for this exact generation.
    InvalidKey,
    /// The candidate list repeats a key.
    DuplicateCandidate,
    /// Input size overflows or exceeds the caller's soft working-byte budget.
    Budget,
}

impl From<ValidationError> for IndexError {
    fn from(error: ValidationError) -> Self {
        Self::Record(error)
    }
}

/// Immutable record array and stable lexical identity map for one generation.
#[derive(Debug)]
pub struct Generation {
    dimension: usize,
    id_bytes: [u8; 16],
    ids: Vec<String>,
    records: Vec<u8>,
    record_size: usize,
}

/// Private lexical identity state retained after a verified native fixture
/// takes ownership of the only resident packed-code copy.
#[cfg(feature = "experimental-sign-scale-v1")]
pub(crate) struct GenerationIdentity {
    dimension: usize,
    id_bytes: [u8; 16],
    ids: Vec<String>,
}

#[cfg(feature = "experimental-sign-scale-v1")]
impl GenerationIdentity {
    pub(crate) fn dimension(&self) -> usize {
        self.dimension
    }

    pub(crate) fn len(&self) -> usize {
        self.ids.len()
    }

    pub(crate) fn id_for_key(&self, key: u64) -> Result<&str, IndexError> {
        let index = usize::try_from(key.checked_sub(1).ok_or(IndexError::InvalidKey)?)
            .map_err(|_| IndexError::InvalidKey)?;
        self.ids
            .get(index)
            .map(String::as_str)
            .ok_or(IndexError::InvalidKey)
    }

    pub(crate) fn generation(&self) -> [u8; 16] {
        self.id_bytes
    }

    pub(crate) fn keys(&self, max_working_bytes: usize) -> Result<Vec<u64>, IndexError> {
        let bytes = self
            .ids
            .len()
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or(IndexError::Budget)?;
        if bytes > max_working_bytes {
            return Err(IndexError::Budget);
        }
        let mut keys = Vec::new();
        keys.try_reserve_exact(self.ids.len())
            .map_err(|_| IndexError::Budget)?;
        for index in 0..self.ids.len() {
            keys.push(
                u64::try_from(index)
                    .ok()
                    .and_then(|value| value.checked_add(1))
                    .ok_or(IndexError::Budget)?,
            );
        }
        Ok(keys)
    }

    /// Parse the canonical ID mapping without loading vector records.
    ///
    /// The caller must verify the encoded bytes against a digest pinned in the
    /// outer generation manifest before using the returned mapping. This
    /// parser validates structure and canonical key/ID ordering only.
    pub(crate) fn from_bytes(
        bytes: &[u8],
        max_bytes: usize,
        max_documents: usize,
    ) -> Result<Self, IndexError> {
        const HEADER_SIZE: usize = 8 + 16 + 4 + 8;
        if bytes.len() > max_bytes {
            return Err(IndexError::Budget);
        }
        if bytes.len() < HEADER_SIZE || &bytes[..8] != b"ZGIDV1\0\0" {
            return Err(IndexError::InvalidKey);
        }
        let mut id_bytes = [0; 16];
        id_bytes.copy_from_slice(&bytes[8..24]);
        if id_bytes.iter().all(|byte| *byte == 0) {
            return Err(IndexError::InvalidGeneration);
        }
        let dimension = usize::try_from(u32::from_le_bytes(
            bytes[24..28]
                .try_into()
                .map_err(|_| IndexError::InvalidKey)?,
        ))
        .map_err(|_| IndexError::Budget)?;
        if !(1..=packed_sign_scale::MAX_DIMENSIONS).contains(&dimension) {
            return Err(IndexError::Record(ValidationError::Dimension));
        }
        let count = usize::try_from(u64::from_le_bytes(
            bytes[28..36]
                .try_into()
                .map_err(|_| IndexError::InvalidKey)?,
        ))
        .map_err(|_| IndexError::Budget)?;
        if count == 0 {
            return Err(IndexError::EmptyGeneration);
        }
        if count > max_documents || u32::try_from(count).is_err() {
            return Err(IndexError::Budget);
        }
        const ENTRY_HEADER_SIZE: usize = 2 * std::mem::size_of::<u64>();
        const MIN_ID_SIZE: usize = 1;
        let minimum_entry_bytes = count
            .checked_mul(ENTRY_HEADER_SIZE + MIN_ID_SIZE)
            .ok_or(IndexError::Budget)?;
        let remaining_bytes = bytes
            .len()
            .checked_sub(HEADER_SIZE)
            .ok_or(IndexError::InvalidKey)?;
        if minimum_entry_bytes > remaining_bytes {
            return Err(IndexError::InvalidKey);
        }
        let mut ids = Vec::new();
        ids.try_reserve_exact(count)
            .map_err(|_| IndexError::Budget)?;
        let mut cursor = HEADER_SIZE;
        for index in 0..count {
            if bytes.len().saturating_sub(cursor) < 16 {
                return Err(IndexError::InvalidKey);
            }
            let key = u64::from_le_bytes(
                bytes[cursor..cursor + 8]
                    .try_into()
                    .map_err(|_| IndexError::InvalidKey)?,
            );
            cursor += 8;
            let id_len = usize::try_from(u64::from_le_bytes(
                bytes[cursor..cursor + 8]
                    .try_into()
                    .map_err(|_| IndexError::InvalidKey)?,
            ))
            .map_err(|_| IndexError::Budget)?;
            cursor += 8;
            if key
                != u64::try_from(index)
                    .ok()
                    .and_then(|value| value.checked_add(1))
                    .ok_or(IndexError::Budget)?
            {
                return Err(IndexError::InvalidKey);
            }
            if id_len == 0 {
                return Err(IndexError::EmptyId);
            }
            let end = cursor.checked_add(id_len).ok_or(IndexError::Budget)?;
            let id_bytes = bytes.get(cursor..end).ok_or(IndexError::InvalidKey)?;
            let id_text = std::str::from_utf8(id_bytes).map_err(|_| IndexError::InvalidKey)?;
            if ids
                .last()
                .is_some_and(|previous: &String| previous.as_str() >= id_text)
            {
                return Err(IndexError::DuplicateId);
            }
            let mut id = String::new();
            id.try_reserve_exact(id_len)
                .map_err(|_| IndexError::Budget)?;
            id.push_str(id_text);
            ids.push(id);
            cursor = end;
        }
        if cursor != bytes.len() {
            return Err(IndexError::InvalidKey);
        }
        Ok(Self {
            dimension,
            id_bytes,
            ids,
        })
    }
}

/// One result after exact score recomputation and lexical identity tie-breaking.
#[derive(Debug, PartialEq)]
pub struct RankedHit<'a> {
    /// Stable source identity; lifetime is tied to the immutable generation.
    pub id: &'a str,
    /// A V1 native key, valid only within this generation.
    pub key: u64,
    /// Exact `f64` score over the reconstructed binary-sign document vector.
    pub score: f64,
}

impl Generation {
    /// Keep only identity after native open has verified the same generation.
    /// Dropping the Rust record array here removes a duplicate logical copy;
    /// it does not guarantee that the allocator returns pages to the OS.
    #[cfg(all(test, feature = "experimental-sign-scale-v1"))]
    pub(crate) fn into_identity(self) -> GenerationIdentity {
        let Self {
            dimension,
            id_bytes,
            ids,
            records,
            ..
        } = self;
        drop(records);
        GenerationIdentity {
            dimension,
            id_bytes,
            ids,
        }
    }

    /// Build a deterministic lexical map; V1 keys are `1..=N`.
    ///
    /// `max_working_bytes` bounds a conservative estimate for owned records,
    /// identity bytes and allocation bookkeeping, not allocator/thread
    /// overhead. Use a process/cgroup limit as the hard memory bound. No
    /// existing source identities or records are mutated.
    ///
    /// # Errors
    /// Rejects invalid records, empty/duplicate IDs, zero generation IDs,
    /// arithmetic overflow and inputs over the explicit budget.
    pub fn prepare(
        dimension: usize,
        generation: [u8; 16],
        documents: &[Document<'_>],
        max_working_bytes: usize,
    ) -> Result<Self, IndexError> {
        if generation.iter().all(|byte| *byte == 0) {
            return Err(IndexError::InvalidGeneration);
        }
        if documents.is_empty() {
            return Err(IndexError::EmptyGeneration);
        }
        if u32::try_from(documents.len()).is_err() {
            return Err(IndexError::Budget);
        }
        let sign_bytes = if (1..=packed_sign_scale::MAX_DIMENSIONS).contains(&dimension) {
            dimension.div_ceil(8)
        } else {
            return Err(IndexError::Record(ValidationError::Dimension));
        };
        let record_size = dimension.div_ceil(32) * 4 + 4;
        let mut estimated =
            documents
                .len()
                .checked_mul(record_size)
                .and_then(|bytes| {
                    bytes.checked_add(documents.len().checked_mul(
                        std::mem::size_of::<String>() + std::mem::size_of::<usize>(),
                    )?)
                })
                .and_then(|bytes| bytes.checked_add(record_size))
                .ok_or(IndexError::Budget)?;
        for document in documents {
            if document.id.is_empty() {
                return Err(IndexError::EmptyId);
            }
            estimated = estimated
                .checked_add(document.id.len())
                .ok_or(IndexError::Budget)?;
            if document.signs.len() != sign_bytes {
                return Err(IndexError::Record(ValidationError::Length));
            }
        }
        if estimated > max_working_bytes {
            return Err(IndexError::Budget);
        }

        let mut order = Vec::new();
        order
            .try_reserve_exact(documents.len())
            .map_err(|_| IndexError::Budget)?;
        order.extend(0..documents.len());
        order.sort_unstable_by(|left, right| documents[*left].id.cmp(documents[*right].id));
        if order
            .windows(2)
            .any(|pair| documents[pair[0]].id == documents[pair[1]].id)
        {
            return Err(IndexError::DuplicateId);
        }
        let mut ids = Vec::new();
        ids.try_reserve_exact(documents.len())
            .map_err(|_| IndexError::Budget)?;
        let mut records = Vec::new();
        records
            .try_reserve_exact(documents.len() * record_size)
            .map_err(|_| IndexError::Budget)?;
        for index in order {
            let document = documents[index];
            let record = packed_sign_scale::encode(document.signs, document.scale, dimension)?;
            let mut id = String::new();
            id.try_reserve_exact(document.id.len())
                .map_err(|_| IndexError::Budget)?;
            id.push_str(document.id);
            ids.push(id);
            records.extend_from_slice(&record);
        }
        Ok(Self {
            dimension,
            id_bytes: generation,
            ids,
            records,
            record_size,
        })
    }

    /// Logical embedding dimension.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Caller-supplied nonzero immutable-generation identity.
    #[must_use]
    pub fn generation(&self) -> [u8; 16] {
        self.id_bytes
    }

    /// Packed records in the same lexical order as [`Self::ids`].
    #[must_use]
    pub fn records(&self) -> &[u8] {
        &self.records
    }

    /// Stable source identities in lexical order; do not infer geometry or ownership.
    #[must_use]
    pub fn ids(&self) -> &[String] {
        &self.ids
    }

    /// Ordered numeric keys corresponding to [`Self::ids`] and records.
    ///
    /// # Errors
    /// Returns [`IndexError::Budget`] if the requested allocation fails or
    /// exceeds `max_working_bytes`.
    pub fn keys(&self, max_working_bytes: usize) -> Result<Vec<u64>, IndexError> {
        let bytes = self
            .ids
            .len()
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or(IndexError::Budget)?;
        if bytes > max_working_bytes {
            return Err(IndexError::Budget);
        }
        let mut keys = Vec::new();
        keys.try_reserve_exact(self.ids.len())
            .map_err(|_| IndexError::Budget)?;
        for index in 0..self.ids.len() {
            keys.push(
                u64::try_from(index)
                    .ok()
                    .and_then(|value| value.checked_add(1))
                    .ok_or(IndexError::Budget)?,
            );
        }
        Ok(keys)
    }

    /// Canonical identity bytes for a caller to hash into its trusted manifest.
    ///
    /// This does not compute or authenticate a digest. The trusted manifest
    /// must bind these bytes, native codebook/graph digests and generation
    /// atomically; trusting a colocated mutable file alone is insufficient.
    ///
    /// # Errors
    /// Returns [`IndexError::Budget`] if identity lengths overflow or exceed
    /// `max_working_bytes`.
    pub fn identity_bytes(&self, max_working_bytes: usize) -> Result<Vec<u8>, IndexError> {
        let mut total = 8usize + 16 + 4 + 8;
        for id in &self.ids {
            total = total
                .checked_add(8 + 8)
                .and_then(|bytes| bytes.checked_add(id.len()))
                .ok_or(IndexError::Budget)?;
        }
        if total > max_working_bytes {
            return Err(IndexError::Budget);
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(total)
            .map_err(|_| IndexError::Budget)?;
        bytes.extend_from_slice(b"ZGIDV1\0\0");
        bytes.extend_from_slice(&self.id_bytes);
        bytes.extend_from_slice(
            &u32::try_from(self.dimension)
                .map_err(|_| IndexError::Budget)?
                .to_le_bytes(),
        );
        bytes.extend_from_slice(
            &u64::try_from(self.ids.len())
                .map_err(|_| IndexError::Budget)?
                .to_le_bytes(),
        );
        for (index, id) in self.ids.iter().enumerate() {
            let key = u64::try_from(index)
                .ok()
                .and_then(|value| value.checked_add(1))
                .ok_or(IndexError::Budget)?;
            bytes.extend_from_slice(&key.to_le_bytes());
            bytes.extend_from_slice(
                &u64::try_from(id.len())
                    .map_err(|_| IndexError::Budget)?
                    .to_le_bytes(),
            );
            bytes.extend_from_slice(id.as_bytes());
        }
        Ok(bytes)
    }

    fn record(&self, key: u64) -> Result<(&str, &[u8]), IndexError> {
        let index = usize::try_from(key.checked_sub(1).ok_or(IndexError::InvalidKey)?)
            .map_err(|_| IndexError::InvalidKey)?;
        let id = self.ids.get(index).ok_or(IndexError::InvalidKey)?;
        let start = index
            .checked_mul(self.record_size)
            .ok_or(IndexError::InvalidKey)?;
        let end = start
            .checked_add(self.record_size)
            .ok_or(IndexError::InvalidKey)?;
        Ok((
            id,
            self.records.get(start..end).ok_or(IndexError::InvalidKey)?,
        ))
    }

    /// Exactly rerank a native candidate set against the full FP32 query.
    ///
    /// # Errors
    /// Rejects malformed queries, out-of-generation keys and duplicate keys;
    /// no partial results are returned. This does not guarantee ANN recall.
    pub fn rerank<'a>(
        &'a self,
        query: &[f32],
        metric: Metric,
        candidates: &[u64],
        limit: usize,
        max_working_bytes: usize,
    ) -> Result<Vec<RankedHit<'a>>, IndexError> {
        packed_sign_scale::validate_query(query, self.dimension, metric == Metric::Cosine)?;
        let hit_capacity = limit.min(candidates.len());
        let estimated = candidates
            .len()
            .checked_mul(std::mem::size_of::<u64>())
            .and_then(|bytes| {
                bytes.checked_add(hit_capacity.checked_mul(std::mem::size_of::<RankedHit<'_>>())?)
            })
            .ok_or(IndexError::Budget)?;
        if estimated > max_working_bytes {
            return Err(IndexError::Budget);
        }
        let mut ordered = Vec::new();
        ordered
            .try_reserve_exact(candidates.len())
            .map_err(|_| IndexError::Budget)?;
        ordered.extend_from_slice(candidates);
        ordered.sort_unstable();
        if ordered.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(IndexError::DuplicateCandidate);
        }
        let mut hits = Vec::new();
        hits.try_reserve_exact(hit_capacity)
            .map_err(|_| IndexError::Budget)?;
        for key in ordered {
            let (id, record) = self.record(key)?;
            let score = score(query, record, self.dimension, metric)?;
            keep_best(&mut hits, RankedHit { id, key, score }, limit);
        }
        sort_hits(&mut hits);
        Ok(hits)
    }

    /// Exact full scan for a filtered query when native filter support is absent.
    ///
    /// `include` must implement the already-authorized caller's filter; this
    /// method neither widens a workspace nor defines ownership from geometry.
    ///
    /// # Errors
    /// Returns an error if the full FP32 query or a stored record is invalid.
    pub fn filtered_exact<'a, F>(
        &'a self,
        query: &[f32],
        metric: Metric,
        limit: usize,
        max_working_bytes: usize,
        mut include: F,
    ) -> Result<Vec<RankedHit<'a>>, IndexError>
    where
        F: FnMut(&str) -> bool,
    {
        packed_sign_scale::validate_query(query, self.dimension, metric == Metric::Cosine)?;
        let hit_capacity = limit.min(self.ids.len());
        let estimated = hit_capacity
            .checked_mul(std::mem::size_of::<RankedHit<'_>>())
            .ok_or(IndexError::Budget)?;
        if estimated > max_working_bytes {
            return Err(IndexError::Budget);
        }
        let mut hits = Vec::new();
        hits.try_reserve_exact(hit_capacity)
            .map_err(|_| IndexError::Budget)?;
        for (index, id) in self.ids.iter().enumerate() {
            if include(id) {
                let key = u64::try_from(index)
                    .ok()
                    .and_then(|value| value.checked_add(1))
                    .ok_or(IndexError::Budget)?;
                let (_, record) = self.record(key)?;
                keep_best(
                    &mut hits,
                    RankedHit {
                        id,
                        key,
                        score: score(query, record, self.dimension, metric)?,
                    },
                    limit,
                );
            }
        }
        sort_hits(&mut hits);
        Ok(hits)
    }
}

fn score(
    query: &[f32],
    record: &[u8],
    dimension: usize,
    metric: Metric,
) -> Result<f64, IndexError> {
    let view = packed_sign_scale::decode(record, dimension)?;
    if metric == Metric::Euclidean {
        // Avoid catastrophic cancellation in the expanded norm identity.
        let scale = f64::from(view.scale);
        let mut squared = 0.0_f64;
        for (index, &value) in query.iter().enumerate() {
            let positive = view.signs[index / 8] & (1u8 << (index % 8)) != 0;
            let document = if positive { scale } else { -scale };
            let difference = f64::from(value) - document;
            squared += difference * difference;
        }
        let result = -squared.sqrt();
        return Ok(if result == 0.0 { 0.0 } else { result });
    }
    let mut dot = 0.0f64;
    let mut query_norm = 0.0f64;
    for (index, &value) in query.iter().enumerate() {
        let value = f64::from(value);
        query_norm += value * value;
        let positive = view.signs[index / 8] & (1u8 << (index % 8)) != 0;
        dot += if positive { value } else { -value };
    }
    let dimensions = f64::from(u32::try_from(dimension).map_err(|_| IndexError::Budget)?);
    let scale = f64::from(view.scale);
    let result = match metric {
        Metric::Cosine if query_norm == 0.0 || scale == 0.0 => 0.0,
        Metric::Cosine => dot / (query_norm.sqrt() * dimensions.sqrt()),
        Metric::DotProduct => dot * scale,
        Metric::Euclidean => unreachable!("handled above"),
    };
    Ok(if result == 0.0 { 0.0 } else { result })
}

pub(crate) fn sort_hits(hits: &mut [RankedHit<'_>]) {
    hits.sort_unstable_by(compare_hits);
}

fn compare_hits(left: &RankedHit<'_>, right: &RankedHit<'_>) -> std::cmp::Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| left.id.cmp(right.id))
}

pub(crate) fn keep_best<'a>(hits: &mut Vec<RankedHit<'a>>, candidate: RankedHit<'a>, limit: usize) {
    if limit == 0 {
        return;
    }
    if hits.len() < limit {
        hits.push(candidate);
        return;
    }
    if let Some((index, worst)) = hits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| compare_hits(left, right))
        && compare_hits(&candidate, worst).is_lt()
    {
        hits[index] = candidate;
    }
}

#[cfg(test)]
mod tests {
    use super::{Document, Generation, IndexError, Metric};
    use crate::packed_sign_scale::ValidationError;

    const BUDGET: usize = 1_000_000;

    fn example() -> Generation {
        Generation::prepare(
            2,
            [7; 16],
            &[
                Document {
                    id: "light",
                    signs: &[0b10],
                    scale: 1.0,
                },
                Document {
                    id: "heavy",
                    signs: &[0b01],
                    scale: 1.0,
                },
                Document {
                    id: "zero",
                    signs: &[0b01],
                    scale: 0.0,
                },
            ],
            BUDGET,
        )
        .expect("prepare synthetic generation")
    }

    #[test]
    fn mapping_is_canonical_and_generation_bound() {
        let first = example();
        assert_eq!(first.dimension(), 2);
        assert_eq!(first.generation(), [7; 16]);
        assert_eq!(first.keys(BUDGET).expect("keys"), [1, 2, 3]);
        assert_eq!(first.keys(0), Err(IndexError::Budget));
        let second = Generation::prepare(
            2,
            [7; 16],
            &[
                Document {
                    id: "zero",
                    signs: &[0b01],
                    scale: 0.0,
                },
                Document {
                    id: "heavy",
                    signs: &[0b01],
                    scale: 1.0,
                },
                Document {
                    id: "light",
                    signs: &[0b10],
                    scale: 1.0,
                },
            ],
            BUDGET,
        )
        .expect("reorder inputs");
        assert_eq!(first.ids(), ["heavy", "light", "zero"]);
        assert_eq!(first.records(), second.records());
        assert_eq!(
            first.identity_bytes(BUDGET).expect("first identity"),
            second.identity_bytes(BUDGET).expect("second identity")
        );
        assert_ne!(
            first.identity_bytes(BUDGET).expect("identity"),
            Generation::prepare(
                2,
                [8; 16],
                &[
                    Document {
                        id: "light",
                        signs: &[0b10],
                        scale: 1.0
                    },
                    Document {
                        id: "heavy",
                        signs: &[0b01],
                        scale: 1.0
                    },
                    Document {
                        id: "zero",
                        signs: &[0b01],
                        scale: 0.0
                    },
                ],
                BUDGET,
            )
            .expect("other generation")
            .identity_bytes(BUDGET)
            .expect("other identity")
        );
        assert_eq!(first.identity_bytes(1), Err(IndexError::Budget));
    }

    #[test]
    #[cfg(feature = "experimental-sign-scale-v1")]
    fn persisted_identity_reopens_without_vector_records() {
        let generation = example();
        let bytes = generation.identity_bytes(BUDGET).expect("encode identity");
        let identity = super::GenerationIdentity::from_bytes(&bytes, BUDGET, 3)
            .expect("parse persisted identity");
        assert_eq!(identity.dimension(), 2);
        assert_eq!(identity.generation(), [7; 16]);
        assert_eq!(identity.len(), 3);
        assert_eq!(identity.id_for_key(1), Ok("heavy"));
        assert_eq!(identity.id_for_key(3), Ok("zero"));

        let mut truncated = bytes.clone();
        truncated.pop();
        assert!(matches!(
            super::GenerationIdentity::from_bytes(&truncated, BUDGET, 3),
            Err(IndexError::InvalidKey)
        ));
        let mut bad_key = bytes;
        bad_key[36..44].copy_from_slice(&2_u64.to_le_bytes());
        assert!(matches!(
            super::GenerationIdentity::from_bytes(&bad_key, BUDGET, 3),
            Err(IndexError::InvalidKey)
        ));
    }

    #[test]
    #[cfg(feature = "experimental-sign-scale-v1")]
    fn persisted_identity_checks_row_and_byte_limits_before_allocation() {
        const HEADER_SIZE: usize = 8 + 16 + 4 + 8;
        let mut bytes = vec![0_u8; HEADER_SIZE];
        bytes[..8].copy_from_slice(b"ZGIDV1\0\0");
        bytes[8..24].copy_from_slice(&[1; 16]);
        bytes[24..28].copy_from_slice(&1_u32.to_le_bytes());
        bytes[28..36].copy_from_slice(&2_u64.to_le_bytes());

        assert!(matches!(
            GenerationIdentity::from_bytes(&bytes, HEADER_SIZE, 1),
            Err(IndexError::Budget)
        ));
        assert!(matches!(
            GenerationIdentity::from_bytes(&bytes, HEADER_SIZE, 2),
            Err(IndexError::InvalidKey)
        ));
    }

    #[test]
    fn weighted_float_query_breaks_hamming_tie_and_uses_scale() {
        let generation = example();
        // heavy/light each differ from the sign of [10, 1] in one bit.
        // Hamming ties them; the full query must prefer heavy.
        let hits = generation
            .rerank(&[10.0, 1.0], Metric::Cosine, &[2, 1, 3], 3, BUDGET)
            .expect("rerank");
        assert_eq!(
            hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
            ["heavy", "zero", "light"]
        );
        let expected = 9.0 / (101.0f64.sqrt() * 2.0f64.sqrt());
        assert!((hits[0].score - expected).abs() < 1e-12);
        assert_eq!(hits[1].score.to_bits(), 0.0_f64.to_bits());

        let dot = generation
            .rerank(&[10.0, 1.0], Metric::DotProduct, &[1, 3], 2, BUDGET)
            .expect("dot rerank");
        assert_eq!(dot[0].score.to_bits(), 9.0_f64.to_bits());
        assert_eq!(dot[1].score.to_bits(), 0.0_f64.to_bits());
        let l2 = generation
            .rerank(&[10.0, 1.0], Metric::Euclidean, &[1, 3], 2, BUDGET)
            .expect("L2 rerank");
        assert!((l2[0].score + 85.0f64.sqrt()).abs() < 1e-12);
        assert!((l2[1].score + 101.0f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn ties_are_lexical_and_filtered_fallback_scans_all_records() {
        let generation = Generation::prepare(
            2,
            [1; 16],
            &[
                Document {
                    id: "z",
                    signs: &[0b11],
                    scale: 1.0,
                },
                Document {
                    id: "a",
                    signs: &[0b11],
                    scale: 1.0,
                },
                Document {
                    id: "m",
                    signs: &[0b00],
                    scale: 1.0,
                },
            ],
            BUDGET,
        )
        .expect("prepare");
        let hits = generation
            .rerank(&[1.0, 0.5], Metric::Cosine, &[3, 1, 2], 2, BUDGET)
            .expect("ties");
        assert_eq!(
            hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
            ["a", "z"]
        );
        // Pretend the approximate candidate set omitted `m`; the explicit
        // filtered fallback still finds it by scanning the entire generation.
        let filtered = generation
            .filtered_exact(&[1.0, 0.5], Metric::Cosine, 1, BUDGET, |id| id == "m")
            .expect("filtered fallback");
        assert_eq!(filtered.iter().map(|hit| hit.id).collect::<Vec<_>>(), ["m"]);
    }

    #[test]
    fn near_equal_euclidean_and_signed_zero_ties_are_correct() {
        let signs = vec![0xff; 20_000 / 8];
        let next = f32::from_bits(1.0_f32.to_bits() + 1);
        let mut query = vec![1.0_f32; 20_000];
        query[0] = next;
        let generation = Generation::prepare(
            20_000,
            [8; 16],
            &[
                Document {
                    id: "near",
                    signs: &signs,
                    scale: 1.0,
                },
                Document {
                    id: "far",
                    signs: &signs,
                    scale: next,
                },
            ],
            BUDGET,
        )
        .expect("prepare");
        let hits = generation
            .rerank(&query, Metric::Euclidean, &[1, 2], 2, BUDGET)
            .expect("Euclidean rerank");
        assert_eq!(hits[0].id, "near");
        assert_eq!(
            hits[0].score.to_bits(),
            (-(f64::from(next) - 1.0)).to_bits()
        );
        assert!(hits[0].score > hits[1].score);

        let zero = Generation::prepare(
            1,
            [9; 16],
            &[
                Document {
                    id: "a",
                    signs: &[0],
                    scale: 0.0,
                },
                Document {
                    id: "z",
                    signs: &[1],
                    scale: 0.0,
                },
            ],
            BUDGET,
        )
        .expect("prepare zero vectors");
        for hits in [
            zero.rerank(&[1.0], Metric::DotProduct, &[2, 1], 2, BUDGET)
                .expect("rerank"),
            zero.filtered_exact(&[1.0], Metric::DotProduct, 2, BUDGET, |_| true)
                .expect("scan"),
        ] {
            assert_eq!(
                hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
                ["a", "z"]
            );
            assert!(
                hits.iter()
                    .all(|hit| hit.score.to_bits() == 0.0_f64.to_bits())
            );
        }
    }

    #[test]
    fn invalid_inputs_fail_without_partial_results() {
        let generation = example();
        assert_eq!(
            generation.rerank(&[0.0, 0.0], Metric::Cosine, &[1], 1, BUDGET),
            Err(IndexError::Record(ValidationError::ZeroQuery))
        );
        assert_eq!(
            generation.rerank(&[f32::NAN, 1.0], Metric::Cosine, &[1], 1, BUDGET),
            Err(IndexError::Record(ValidationError::NonFiniteQuery))
        );
        assert_eq!(
            generation.rerank(&[1.0, 1.0], Metric::Cosine, &[1, 1], 2, BUDGET),
            Err(IndexError::DuplicateCandidate)
        );
        assert_eq!(
            generation.rerank(&[1.0, 1.0], Metric::Cosine, &[1, 99], 2, BUDGET),
            Err(IndexError::InvalidKey)
        );
        assert_eq!(
            generation.rerank(&[1.0, 1.0], Metric::Cosine, &[1], 1, 0),
            Err(IndexError::Budget)
        );
        assert_eq!(
            generation.filtered_exact(&[1.0, 1.0], Metric::Cosine, 1, 0, |_| true),
            Err(IndexError::Budget)
        );
        assert!(matches!(
            Generation::prepare(2, [0; 16], &[], BUDGET),
            Err(IndexError::InvalidGeneration)
        ));
        assert!(matches!(
            Generation::prepare(2, [1; 16], &[], BUDGET),
            Err(IndexError::EmptyGeneration)
        ));
        assert!(matches!(
            Generation::prepare(
                2,
                [1; 16],
                &[Document {
                    id: "",
                    signs: &[1],
                    scale: 1.0
                }],
                BUDGET
            ),
            Err(IndexError::EmptyId)
        ));
        assert!(matches!(
            Generation::prepare(
                2,
                [1; 16],
                &[
                    Document {
                        id: "x",
                        signs: &[1],
                        scale: 1.0
                    },
                    Document {
                        id: "x",
                        signs: &[2],
                        scale: 1.0
                    }
                ],
                BUDGET
            ),
            Err(IndexError::DuplicateId)
        ));
    }

    #[cfg(feature = "experimental-sign-scale-v1")]
    #[test]
    fn generation_identity_keeps_mapping_after_record_array_is_dropped() {
        let generation = Generation::prepare(
            1,
            [9; 16],
            &[
                Document {
                    id: "a",
                    signs: &[1],
                    scale: 1.0,
                },
                Document {
                    id: "b",
                    signs: &[0],
                    scale: 1.0,
                },
            ],
            BUDGET,
        )
        .expect("generation");
        let identity = generation.into_identity();

        assert_eq!(identity.generation(), [9; 16]);
        assert_eq!(identity.len(), 2);
        assert_eq!(identity.id_for_key(1).expect("first key"), "a");
        assert_eq!(identity.id_for_key(2).expect("second key"), "b");
        assert_eq!(identity.keys(BUDGET).expect("keys"), vec![1, 2]);
        assert_eq!(identity.keys(0), Err(IndexError::Budget));
    }
}
