//! Search projections: one zvec collection per embedding model, with FTS and FP16 vectors.

use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    path::Path,
};

#[cfg(feature = "experimental-sign-scale-v1")]
use std::{fs, sync::Arc};

use zvec_rust::{
    Collection, CollectionSchema, DataType, Doc, FieldSchema, Fts, IndexParams, MetricType,
    SearchQuery,
};

#[cfg(feature = "experimental-sign-scale-v1")]
use zg_zvec_fp16::packed_v1_worker::BinaryDiskAnnWorker;

use super::{
    path::{file_membership_doc, file_membership_schema},
    types::{
        IndexedFragment, StoragePathFilter, StorageSearchFilter, StorageSearchHit,
        StorageSearchPath,
    },
    zvec::{
        WRITE_BATCH, collect_entity_ids, corrupt, doc_key, fetch_map, native, open_collection,
        quote, scalar, string_field, u32_field, write_docs,
    },
};
use crate::{
    EngineError, EngineResult,
    domain::{
        BinaryDiskAnnManifest, CodeMetadata, DirectoryId, Entity, EntityFragment, EntityId,
        EntityMetadata, FTS_CONFIG, FileId, FileRecord, FragmentId, IndexField, SourcePath,
        model::{EmbeddingMetric, EmbeddingModelInfo, VectorIndexQuantization},
    },
    utils::sha256_hex_parts,
};

const MAX_TOP_K: usize = 100_000;
const BINARY_SCAN_BUCKETS: u32 = u16::MAX as u32 + 1;
const BINARY_SCAN_TARGET_HITS: usize = 5_000;
const BINARY_SCAN_MAX_INITIAL_RANGES: usize = 256;
const BINARY_SCAN_PAGE_SIZE: usize = 10_000;
#[cfg(feature = "experimental-sign-scale-v1")]
const BINARY_DISKANN_MAX_BEAM: usize = 8192;

pub(super) struct Fragments {
    indexes: BTreeMap<String, Collection>,
    embeddings: BTreeMap<String, EmbeddingModelInfo>,
    #[cfg(feature = "experimental-sign-scale-v1")]
    binary_diskann: Option<(String, Arc<BinaryDiskAnnWorker>)>,
}

impl Fragments {
    #[cfg(test)]
    pub(super) fn open(
        root: &Path,
        embeddings: &[EmbeddingModelInfo],
        read_only: bool,
    ) -> EngineResult<Self> {
        Self::open_with_binary_diskann_manifest(root, embeddings, read_only, None)
    }

    pub(super) fn open_with_binary_diskann_manifest(
        root: &Path,
        embeddings: &[EmbeddingModelInfo],
        read_only: bool,
        binary_diskann_manifest: Option<&BinaryDiskAnnManifest>,
    ) -> EngineResult<Self> {
        let mut indexes = BTreeMap::new();
        let mut stored_embeddings = BTreeMap::new();
        for embedding in embeddings {
            let dimension = u32::try_from(embedding.dimension())
                .ok()
                .filter(|value| (1..=20_000).contains(value))
                .ok_or_else(|| {
                    EngineError::invalid_argument(
                        "storage embedding dimension must be between 1 and 20,000",
                    )
                })?;
            let metric = match embedding.metric() {
                EmbeddingMetric::Cosine => MetricType::Cosine,
                EmbeddingMetric::DotProduct => MetricType::Ip,
                EmbeddingMetric::Euclidean => MetricType::L2,
            };
            let name = fragment_collection_name(embedding);
            let collection = open_collection(
                &root.join(&name),
                &fragments_schema(dimension, metric, embedding.vector_index_quantization())?,
                read_only,
            )?;
            let model_reference = embedding.model().reference();
            if indexes
                .insert(model_reference.clone(), collection)
                .is_some()
            {
                return Err(EngineError::invalid_argument("duplicate embedding model"));
            }
            stored_embeddings.insert(model_reference, embedding.clone());
        }

        #[cfg(feature = "experimental-sign-scale-v1")]
        let binary_diskann = if let Some(manifest) = binary_diskann_manifest {
            if !read_only {
                return Err(EngineError::unsupported(
                    "a published binary DiskANN generation is immutable; rebuild it before writes",
                ));
            }
            let (model, worker) =
                Self::open_binary_diskann_worker(root, &indexes, &stored_embeddings, manifest)?;
            Some((model, Arc::new(worker)))
        } else {
            None
        };

        #[cfg(not(feature = "experimental-sign-scale-v1"))]
        if binary_diskann_manifest.is_some() {
            return Err(EngineError::unsupported(
                "this build does not include binary DiskANN search support",
            ));
        }

        Ok(Self {
            indexes,
            embeddings: stored_embeddings,
            #[cfg(feature = "experimental-sign-scale-v1")]
            binary_diskann,
        })
    }

    #[cfg(feature = "experimental-sign-scale-v1")]
    fn open_binary_diskann_worker(
        root: &Path,
        indexes: &BTreeMap<String, Collection>,
        embeddings: &BTreeMap<String, EmbeddingModelInfo>,
        metadata: &BinaryDiskAnnManifest,
    ) -> EngineResult<(String, BinaryDiskAnnWorker)> {
        const MAX_IDENTITY_BYTES: usize = 256 * 1024 * 1024;
        let mut candidates = embeddings.iter().filter(|(_, embedding)| {
            embedding.vector_index_quantization() == VectorIndexQuantization::BinarySignDiskAnnV1
        });
        let (model, embedding) = candidates.next().ok_or_else(|| {
            EngineError::storage_failure("binaryDiskAnn metadata has no V1 embedding model")
        })?;
        if candidates.next().is_some() {
            return Err(EngineError::unsupported(
                "one binary DiskANN manifest currently supports one V1 embedding model",
            ));
        }
        if u32::try_from(embedding.dimension()).ok() != Some(metadata.dimension()) {
            return Err(EngineError::storage_failure(
                "binaryDiskAnn dimension does not match the embedding model",
            ));
        }
        let index = indexes
            .get(model)
            .ok_or_else(|| EngineError::storage_failure("binaryDiskAnn collection is missing"))?;
        let count = native(index.stats(), "inspect binary DiskANN collection")?.doc_count;
        if count != metadata.count() {
            return Err(EngineError::storage_failure(
                "binaryDiskAnn count does not match its fragment collection",
            ));
        }

        let canonical_root = fs::canonicalize(root)
            .map_err(|error| EngineError::from_io("locate binary DiskANN root", &error))?;
        let graph_path = root.join(metadata.graph_path());
        let identity_path = root.join(metadata.identity_path());
        let graph_metadata = fs::symlink_metadata(&graph_path)
            .map_err(|error| EngineError::from_io("inspect binary DiskANN graph", &error))?;
        if graph_metadata.file_type().is_symlink() || !graph_metadata.is_dir() {
            return Err(EngineError::storage_failure(
                "binary DiskANN graph is not a regular directory",
            ));
        }
        let graph_path = fs::canonicalize(&graph_path)
            .map_err(|error| EngineError::from_io("resolve binary DiskANN graph", &error))?;
        if !graph_path.starts_with(&canonical_root) {
            return Err(EngineError::storage_failure(
                "binary DiskANN graph path escapes the workspace index",
            ));
        }
        let identity_metadata = fs::symlink_metadata(&identity_path)
            .map_err(|error| EngineError::from_io("inspect binary DiskANN identity", &error))?;
        if identity_metadata.file_type().is_symlink()
            || !identity_metadata.is_file()
            || identity_metadata.len() > MAX_IDENTITY_BYTES as u64
        {
            return Err(EngineError::storage_failure(
                "binary DiskANN identity is not a bounded regular file",
            ));
        }
        let identity_path = fs::canonicalize(&identity_path)
            .map_err(|error| EngineError::from_io("resolve binary DiskANN identity", &error))?;
        if !identity_path.starts_with(&canonical_root) {
            return Err(EngineError::storage_failure(
                "binary DiskANN identity path escapes the workspace index",
            ));
        }
        let identity_bytes = fs::read(&identity_path)
            .map_err(|error| EngineError::from_io("read binary DiskANN identity", &error))?;
        if identity_bytes.len() as u64 != identity_metadata.len() {
            return Err(EngineError::storage_failure(
                "binary DiskANN identity changed while being opened",
            ));
        }
        let trust = zg_zvec_fp16::packed_diskann_generation::Trust::from_bytes(metadata.trust())
            .map_err(|error| {
                EngineError::storage_failure(format!(
                    "invalid binary DiskANN trust bundle: {error:?}"
                ))
            })?;
        let metric = match embedding.metric() {
            EmbeddingMetric::Cosine => zg_zvec_fp16::packed_sign_index::Metric::Cosine,
            EmbeddingMetric::DotProduct => zg_zvec_fp16::packed_sign_index::Metric::DotProduct,
            EmbeddingMetric::Euclidean => zg_zvec_fp16::packed_sign_index::Metric::Euclidean,
        };
        let limits = zg_zvec_fp16::packed_diskann_generation::Limits {
            max_documents: 100_000,
            max_input_bytes: MAX_IDENTITY_BYTES,
            max_build_bytes: 512 * 1024 * 1024,
            max_codebook_bytes: 256 * 1024 * 1024,
            max_generation_bytes: 512 * 1024 * 1024,
            max_query_scratch_bytes: 16 * 1024 * 1024,
            max_beam: BINARY_DISKANN_MAX_BEAM,
            max_degree: 64,
            build_list_size: 100,
            build_threads: 2,
        };
        let worker = BinaryDiskAnnWorker::open(
            &graph_path,
            trust,
            &identity_bytes,
            metadata.identity_sha256(),
            metric,
            limits,
        )
        .map_err(|error| {
            EngineError::storage_failure(format!("open binary DiskANN graph: {error:?}"))
        })?;
        Ok((model.clone(), worker))
    }

    /// Encode the entire replacement before any stored rows are changed.
    pub(super) fn prepare(
        file: &FileRecord,
        entities: &[Entity],
        entries: &[IndexedFragment],
        directories: &[DirectoryId],
        embeddings: &BTreeMap<String, EmbeddingModelInfo>,
    ) -> EngineResult<BTreeMap<String, Vec<Doc>>> {
        let owners = entities
            .iter()
            .map(|entity| {
                let fields = entity
                    .metadata()
                    .map(|metadata| {
                        let value = serde_json::to_value(metadata)
                            .map_err(|error| corrupt(format!("encode entity metadata: {error}")))?;
                        encode_metadata_fields(&value)
                    })
                    .transpose()?
                    .unwrap_or_default();
                Ok((entity.id(), (entity, fields)))
            })
            .collect::<EngineResult<HashMap<_, _>>>()?;
        let fragments = entities
            .iter()
            .flat_map(Entity::fragments)
            .map(|fragment| (&fragment.id, fragment))
            .collect::<HashMap<_, _>>();
        let mut projections: BTreeMap<String, Vec<Doc>> = BTreeMap::new();
        for entry in entries {
            let (owner, fields) = owners.get(&entry.entity_id).ok_or_else(|| {
                EngineError::invalid_argument("search projection references a missing entity")
            })?;
            let fragment = fragments.get(&entry.fragment_id).ok_or_else(|| {
                EngineError::invalid_argument("search projection references a missing fragment")
            })?;
            let embedding = embeddings.get(&entry.model).ok_or_else(|| {
                EngineError::invalid_argument("search projection references an unknown model")
            })?;
            if entry.vector.len() != embedding.dimension() {
                return Err(EngineError::invalid_argument(format!(
                    "embedding vector has {} values; model {} requires {}",
                    entry.vector.len(),
                    entry.model,
                    embedding.dimension()
                )));
            }
            let mut doc = fragment_doc(owner, fragment, file, directories, fields)?;
            native(
                doc.add_string("text", &index_text(&entry.fts_text)),
                "encode searchable text",
            )?;
            match embedding.vector_index_quantization() {
                VectorIndexQuantization::Fp16 => native(
                    zg_zvec_fp16::add_embedding(&mut doc, "embedding", &entry.vector),
                    "encode embedding vector",
                )?,
                VectorIndexQuantization::BinarySign
                | VectorIndexQuantization::BinarySignDiskAnnV1 => {
                    let (packed, scale) = encode_sign_vector(&entry.vector)?;
                    native(
                        doc.add_binary("embedding", &packed),
                        "encode packed sign embedding",
                    )?;
                    native(
                        doc.add_f32("embedding_scale", scale),
                        "encode embedding scale",
                    )?;
                    native(
                        doc.add_u32(
                            "vector_scan_bucket",
                            vector_scan_bucket(entry.fragment_id.as_str())?,
                        ),
                        "encode vector scan bucket",
                    )?;
                }
            }
            projections
                .entry(entry.model.clone())
                .or_default()
                .push(doc);
        }
        Ok(projections)
    }

    pub(super) fn write(&self, projections: &BTreeMap<String, Vec<Doc>>) -> EngineResult<()> {
        for (model, docs) in projections {
            write_docs(self.index(model)?, docs, "write model fragments")?;
        }
        Ok(())
    }

    pub(super) fn embedding_models(&self) -> &BTreeMap<String, EmbeddingModelInfo> {
        &self.embeddings
    }

    pub(super) fn validate_ownership(
        &self,
        entries: &[IndexedFragment],
        file_id: FileId,
    ) -> EngineResult<()> {
        let keys = entries
            .iter()
            .map(|entry| entry.fragment_id.as_str().to_owned())
            .collect::<Vec<_>>();
        for index in self.indexes.values() {
            reject_foreign_ids(index, &keys, file_id)?;
        }
        Ok(())
    }

    pub(super) fn delete_file(&self, file_id: FileId) -> EngineResult<()> {
        let filter = format!("file_id = {}", file_id.get());
        for index in self.indexes.values() {
            native(index.delete_by_filter(&filter), "delete source fragments")?;
        }
        Ok(())
    }

    /// Count live search projections without loading text or embedding vectors.
    pub(super) fn count(&self) -> EngineResult<u64> {
        let mut count = 0_u64;
        for index in self.indexes.values() {
            let iterator = native(
                index.iter_with_options(Some(&["document_id"]), false),
                "iterate fragment identities",
            )?;
            for doc in iterator {
                native(doc, "read fragment identity")?;
                count = count
                    .checked_add(1)
                    .ok_or_else(|| EngineError::storage_failure("fragment count exceeds u64"))?;
            }
        }
        Ok(count)
    }

    /// Read only the identities of live projections for a diagnostic's watched files.
    pub(super) fn projected_fragments(&self, keys: &[&str]) -> EngineResult<HashSet<String>> {
        let mut projected = HashSet::new();
        for index in self.indexes.values() {
            for batch in keys.chunks(WRITE_BATCH) {
                for doc in native(
                    index.fetch_with_options(batch, Some(&["document_id"]), false),
                    "fetch diagnostic fragment identities",
                )? {
                    projected.insert(string_field(&doc, "document_id")?);
                }
            }
        }
        Ok(projected)
    }

    pub(super) fn flush(&self) -> EngineResult<()> {
        for index in self.indexes.values() {
            native(index.flush(), "flush fragment collection")?;
        }
        Ok(())
    }

    pub(super) fn needs_optimize(&self) -> EngineResult<bool> {
        for index in self.indexes.values() {
            let stats = native(index.stats(), "inspect fragment indexes")?;
            if stats.indexes.iter().any(|stat| stat.completeness < 1.0) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(super) fn optimize(&self) -> EngineResult<()> {
        for index in self.indexes.values() {
            native(index.optimize(), "optimize fragment collection")?;
        }
        Ok(())
    }

    pub(super) fn search_fts(
        &self,
        query: &str,
        limit: usize,
        filter: Option<&str>,
    ) -> EngineResult<Vec<StorageSearchHit>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let mut fts = native(Fts::new(), "create full-text request")?;
        native(
            fts.set_match_string(&index_text(query)),
            "set full-text query",
        )?;
        let mut request = native(
            SearchQuery::fts("text", &fts, top_k(limit)?),
            "create full-text query",
        )?;
        configure_query(
            &mut request,
            filter,
            &["document_id", "entity_id", "file_id"],
        )?;
        let mut hits = Vec::new();
        for index in self.indexes.values() {
            for (rank, doc) in native(index.query(&request), "search full-text index")?
                .into_iter()
                .enumerate()
            {
                let mut hit = decode_search_hit(&doc, StorageSearchPath::Fts)?;
                // BM25 is computed against each table's corpus; merge independent rankings.
                if self.indexes.len() > 1 {
                    let rank = u32::try_from(rank + 1)
                        .map_err(|_| corrupt("FTS rank exceeds query limit"))?;
                    hit.score = 1.0 / (60.0 + f64::from(rank));
                }
                hits.push(hit);
            }
        }
        hits.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.document_id.cmp(&b.document_id))
        });
        hits.truncate(limit);
        Ok(hits)
    }

    /// Collect all matching entities across model tables, independent of fragment count.
    /// The existing hashed symbol-name projection preserves exact case and literals.
    pub(super) fn find_symbol_entity_ids(
        &self,
        name: &str,
        path_filter: Option<&str>,
    ) -> EngineResult<Vec<EntityId>> {
        let mut filter = metadata_in_filter(CodeMetadata::SYMBOL_NAME, std::iter::once(name));
        if let Some(path_filter) = path_filter {
            filter.push_str(" AND (");
            filter.push_str(path_filter);
            filter.push(')');
        }
        let mut ids = BTreeSet::new();
        for index in self.indexes.values() {
            ids.extend(collect_entity_ids(index, &filter, |_| Ok(()))?);
        }
        Ok(ids.into_iter().map(EntityId::from_string).collect())
    }

    pub(super) fn search_vector(
        &self,
        model: &str,
        vector: &[f32],
        limit: usize,
        filter: Option<&str>,
    ) -> EngineResult<Vec<StorageSearchHit>> {
        let index = self.index(model)?;
        if limit == 0 {
            return Ok(Vec::new());
        }
        let requested_top_k = top_k(limit)?;
        let embedding = self.embeddings.get(model).ok_or_else(|| {
            EngineError::invalid_argument(format!("unknown embedding model {model:?}"))
        })?;
        match embedding.vector_index_quantization() {
            VectorIndexQuantization::BinarySign => {
                return Self::search_binary_sign(index, embedding, vector, limit, filter);
            }
            VectorIndexQuantization::BinarySignDiskAnnV1 => {
                #[cfg(feature = "experimental-sign-scale-v1")]
                {
                    let (worker_model, worker) = self.binary_diskann.as_ref().ok_or_else(|| {
                        EngineError::unsupported(
                            "binary DiskANN graph is not open for this workspace",
                        )
                    })?;
                    if worker_model != model {
                        return Err(EngineError::storage_failure(
                            "binary DiskANN worker model does not match the query model",
                        ));
                    }
                    return Self::search_binary_diskann(
                        index, embedding, worker, vector, limit, filter,
                    );
                }
                #[cfg(not(feature = "experimental-sign-scale-v1"))]
                {
                    return Err(EngineError::unsupported(
                        "this build does not include binary DiskANN search support",
                    ));
                }
            }
            VectorIndexQuantization::Fp16 => {}
        }
        let mut request = native(
            zg_zvec_fp16::embedding_query("embedding", vector, requested_top_k),
            "create vector query",
        )?;
        configure_query(
            &mut request,
            filter,
            &["document_id", "entity_id", "file_id"],
        )?;
        native(index.query(&request), "search vector index")?
            .into_iter()
            .map(|doc| decode_search_hit(&doc, StorageSearchPath::Vector))
            .collect()
    }

    fn search_binary_sign(
        index: &Collection,
        embedding: &EmbeddingModelInfo,
        query: &[f32],
        limit: usize,
        filter: Option<&str>,
    ) -> EngineResult<Vec<StorageSearchHit>> {
        if query.len() != embedding.dimension() || query.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::invalid_argument(
                "binary-sign query has an invalid dimension or non-finite value",
            ));
        }
        if embedding.metric() == EmbeddingMetric::Cosine && query.iter().all(|value| *value == 0.0)
        {
            return Err(EngineError::invalid_argument(
                "cosine query vector must not be zero",
            ));
        }

        let stats = native(index.stats(), "inspect binary-vector collection")?;
        let document_count = usize::try_from(stats.doc_count)
            .map_err(|_| EngineError::storage_failure("binary vector count exceeds usize"))?;
        if document_count == 0 {
            return Ok(Vec::new());
        }
        let initial_ranges = document_count
            .div_ceil(BINARY_SCAN_TARGET_HITS)
            .clamp(1, BINARY_SCAN_MAX_INITIAL_RANGES);
        let bucket_width = usize::try_from(BINARY_SCAN_BUCKETS)
            .expect("bucket count fits usize")
            .div_ceil(initial_ranges);
        let mut hits = Vec::with_capacity(limit.min(document_count));
        for start in (0..BINARY_SCAN_BUCKETS).step_by(bucket_width) {
            let end = start
                .saturating_add(u32::try_from(bucket_width).unwrap_or(u32::MAX))
                .min(BINARY_SCAN_BUCKETS);
            Self::scan_binary_sign_range(
                index, embedding, query, limit, filter, start, end, &mut hits,
            )?;
        }
        sort_search_hits(&mut hits);
        hits.truncate(limit);
        Ok(hits)
    }

    #[cfg(feature = "experimental-sign-scale-v1")]
    fn search_binary_diskann(
        index: &Collection,
        embedding: &EmbeddingModelInfo,
        worker: &BinaryDiskAnnWorker,
        query: &[f32],
        limit: usize,
        filter: Option<&str>,
    ) -> EngineResult<Vec<StorageSearchHit>> {
        if filter.is_some() {
            // Native filtered ANN is not integrated yet. Preserve exact existing
            // filter semantics rather than returning candidates outside scope.
            return Self::search_binary_sign(index, embedding, query, limit, filter);
        }
        if query.len() != embedding.dimension() || query.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::invalid_argument(
                "binary DiskANN query has an invalid dimension or non-finite value",
            ));
        }
        if embedding.metric() == EmbeddingMetric::Cosine && query.iter().all(|value| *value == 0.0)
        {
            return Err(EngineError::invalid_argument(
                "cosine query vector must not be zero",
            ));
        }
        let candidate_limit = limit.min(worker.document_count());
        if candidate_limit == 0 {
            return Ok(Vec::new());
        }
        if candidate_limit > BINARY_DISKANN_MAX_BEAM {
            return Err(EngineError::invalid_argument(format!(
                "binary DiskANN top-k exceeds the beam bound of {BINARY_DISKANN_MAX_BEAM}"
            )));
        }
        let beam = candidate_limit
            .saturating_mul(4)
            .max(64)
            .min(BINARY_DISKANN_MAX_BEAM);
        let hits = worker
            .search(query, candidate_limit, beam)
            .map_err(|error| {
                EngineError::storage_failure(format!("binary DiskANN search: {error:?}"))
            })?;
        if hits.is_empty() {
            return Ok(Vec::new());
        }
        let mut scores = HashMap::with_capacity(hits.len());
        for hit in &hits {
            if !hit.score.is_finite() || scores.insert(hit.id.clone(), hit.score).is_some() {
                return Err(corrupt(
                    "binary DiskANN returned an invalid or duplicate candidate",
                ));
            }
        }
        let filter = in_filter("document_id", hits.iter().map(|hit| hit.id.as_str()));
        let mut request = native(
            SearchQuery::scalar(
                i32::try_from(hits.len())
                    .map_err(|_| EngineError::invalid_argument("candidate count exceeds i32"))?,
            ),
            "create binary DiskANN hydration query",
        )?;
        configure_query(
            &mut request,
            Some(&filter),
            &["document_id", "entity_id", "file_id"],
        )?;
        let documents = native(index.query(&request), "hydrate binary DiskANN candidates")?;
        let mut results = Vec::with_capacity(documents.len());
        for document in documents {
            let id = string_field(&document, "document_id")?;
            let score = scores
                .remove(&id)
                .ok_or_else(|| corrupt("binary DiskANN candidate is absent from fragment rows"))?;
            results.push(decode_search_hit_with_score(
                &document,
                StorageSearchPath::Vector,
                score,
            )?);
        }
        if !scores.is_empty() {
            return Err(corrupt("binary DiskANN candidates could not be hydrated"));
        }
        sort_search_hits(&mut results);
        results.truncate(limit);
        Ok(results)
    }

    #[allow(clippy::too_many_arguments)]
    fn scan_binary_sign_range(
        index: &Collection,
        embedding: &EmbeddingModelInfo,
        query: &[f32],
        limit: usize,
        filter: Option<&str>,
        bucket_start: u32,
        bucket_end: u32,
        top_hits: &mut Vec<StorageSearchHit>,
    ) -> EngineResult<()> {
        let bucket_filter =
            format!("vector_scan_bucket >= {bucket_start} AND vector_scan_bucket < {bucket_end}");
        let filter = filter.map_or_else(
            || bucket_filter.clone(),
            |filter| format!("({filter}) AND ({bucket_filter})"),
        );
        let mut request = native(
            SearchQuery::scalar(
                i32::try_from(BINARY_SCAN_PAGE_SIZE + 1)
                    .map_err(|_| EngineError::internal("binary scan page size exceeds i32"))?,
            ),
            "create binary-vector scan query",
        )?;
        configure_query(
            &mut request,
            Some(&filter),
            &[
                "document_id",
                "entity_id",
                "file_id",
                "embedding",
                "embedding_scale",
            ],
        )?;
        let documents = native(index.query(&request), "scan packed sign vectors")?;
        if documents.len() > BINARY_SCAN_PAGE_SIZE {
            if bucket_end.saturating_sub(bucket_start) <= 1 {
                return Err(EngineError::unsupported(
                    "binary vector scan bucket exceeds its bounded page size",
                ));
            }
            let midpoint = bucket_start + (bucket_end - bucket_start) / 2;
            Self::scan_binary_sign_range(
                index,
                embedding,
                query,
                limit,
                Some(&filter),
                bucket_start,
                midpoint,
                top_hits,
            )?;
            return Self::scan_binary_sign_range(
                index,
                embedding,
                query,
                limit,
                Some(&filter),
                midpoint,
                bucket_end,
                top_hits,
            );
        }

        let mut range_hits = Vec::with_capacity(documents.len());
        for document in documents {
            let packed = native(document.get_binary("embedding"), "read packed sign vector")?
                .ok_or_else(|| corrupt("binary vector document is missing its packed embedding"))?;
            let scale = native(
                document.get_f32("embedding_scale"),
                "read sign-vector scale",
            )?
            .ok_or_else(|| corrupt("binary vector document is missing its embedding scale"))?;
            let score = binary_sign_score(query, &packed, scale, embedding.metric())?;
            range_hits.push(decode_search_hit_with_score(
                &document,
                StorageSearchPath::Vector,
                score,
            )?);
        }
        sort_search_hits(&mut range_hits);
        range_hits.truncate(limit);
        top_hits.extend(range_hits);
        sort_search_hits(top_hits);
        top_hits.truncate(limit);
        Ok(())
    }

    fn index(&self, model: &str) -> EngineResult<&Collection> {
        self.indexes.get(model).ok_or_else(|| {
            EngineError::invalid_argument(format!("unknown embedding model {model:?}"))
        })
    }

    #[cfg(test)]
    pub(super) fn collection(&self, model: &str) -> EngineResult<&Collection> {
        self.index(model)
    }
}

fn primary_key(namespace: &str, value: &str) -> String {
    sha256_hex_parts([namespace.as_bytes(), b"\0", value.as_bytes()])
}

pub(super) fn fragment_collection_name(embedding: &EmbeddingModelInfo) -> String {
    format!(
        "fragments_{:.4}",
        primary_key(
            "space",
            &format!(
                "{}\0{}\0{}\0{:?}",
                embedding.model().provider(),
                embedding.model().name(),
                embedding.dimension(),
                embedding.metric()
            )
        )
    )
}

fn identity_schema(name: &str) -> EngineResult<CollectionSchema> {
    let mut schema = native(CollectionSchema::new(name), "create collection schema")?;
    scalar(&mut schema, "file_id", DataType::Uint32, false, true)?;
    scalar(&mut schema, "entity_id", DataType::String, false, true)?;
    scalar(&mut schema, "document_id", DataType::String, false, false)?;
    Ok(schema)
}

fn retrieval_schema(name: &str) -> EngineResult<CollectionSchema> {
    let mut schema = identity_schema(name)?;
    file_membership_schema(&mut schema)?;
    for field in EntityMetadata::index_schema() {
        match field {
            IndexField::String(name) => {
                scalar(&mut schema, name, DataType::String, true, true)?;
            }
        }
    }
    Ok(schema)
}

fn fragments_schema(
    dimension: u32,
    metric: MetricType,
    vector_index_quantization: VectorIndexQuantization,
) -> EngineResult<CollectionSchema> {
    let mut schema = retrieval_schema("fragments")?;
    let mut text = native(
        FieldSchema::new("text", DataType::String, false, 0),
        "define full-text field",
    )?;
    native(
        text.set_index_params(&native(
            IndexParams::fts(Some(FTS_CONFIG.tokenizer), Some(FTS_CONFIG.filters), None),
            "define full-text index",
        )?),
        "attach full-text index",
    )?;
    native(schema.add_field(&text), "add full-text field")?;

    match vector_index_quantization {
        VectorIndexQuantization::Fp16 => {
            let mut vector = native(
                FieldSchema::new("embedding", DataType::VectorFp16, false, dimension),
                "define vector field",
            )?;
            native(
                vector.set_index_params(&native(
                    IndexParams::hnsw(metric, 50, 500),
                    "define vector index",
                )?),
                "attach vector index",
            )?;
            native(schema.add_field(&vector), "add vector field")?;
        }
        VectorIndexQuantization::BinarySign | VectorIndexQuantization::BinarySignDiskAnnV1 => {
            let embedding = native(
                FieldSchema::new("embedding", DataType::Binary, false, 0),
                "define packed sign field",
            )?;
            native(schema.add_field(&embedding), "add packed sign field")?;

            let scale = native(
                FieldSchema::new("embedding_scale", DataType::Float, false, 0),
                "define sign-vector scale field",
            )?;
            native(schema.add_field(&scale), "add sign-vector scale field")?;

            let mut bucket = native(
                FieldSchema::new("vector_scan_bucket", DataType::Uint32, false, 0),
                "define vector scan bucket field",
            )?;
            native(
                bucket.set_index_params(&native(
                    IndexParams::invert(true, false),
                    "define vector scan bucket index",
                )?),
                "attach vector scan bucket index",
            )?;
            native(schema.add_field(&bucket), "add vector scan bucket field")?;
        }
    }
    Ok(schema)
}

fn encode_sign_vector(vector: &[f32]) -> EngineResult<(Vec<u8>, f32)> {
    if vector.is_empty() || vector.iter().any(|value| !value.is_finite()) {
        return Err(EngineError::invalid_argument(
            "binary-sign document vector must be non-empty and finite",
        ));
    }
    let mut packed = vec![0_u8; vector.len().div_ceil(u8::BITS as usize)];
    let mut absolute_sum = 0.0_f64;
    for (dimension, value) in vector.iter().enumerate() {
        absolute_sum += f64::from(value.abs());
        if *value >= 0.0 {
            packed[dimension / u8::BITS as usize] |= 1_u8 << (dimension % u8::BITS as usize);
        }
    }
    let scale = (absolute_sum / vector.len() as f64) as f32;
    Ok((packed, scale))
}

fn vector_scan_bucket(fragment_id: &str) -> EngineResult<u32> {
    let hash = sha256_hex_parts([b"vector-scan-bucket", fragment_id.as_bytes()]);
    let prefix = hash
        .get(..4)
        .ok_or_else(|| EngineError::internal("binary vector bucket hash is too short"))?;
    u16::from_str_radix(prefix, 16)
        .map(u32::from)
        .map_err(|error| EngineError::internal(format!("decode vector scan bucket: {error}")))
}

fn binary_sign_score(
    query: &[f32],
    packed: &[u8],
    scale: f32,
    metric: EmbeddingMetric,
) -> EngineResult<f64> {
    if !scale.is_finite() || scale < 0.0 {
        return Err(corrupt("binary embedding scale is invalid"));
    }
    let expected_bytes = query.len().div_ceil(u8::BITS as usize);
    if packed.len() != expected_bytes {
        return Err(corrupt(format!(
            "packed sign embedding has {} bytes; expected {expected_bytes}",
            packed.len()
        )));
    }
    let dimensions = f64::from(
        u32::try_from(query.len())
            .map_err(|_| EngineError::invalid_argument("binary query dimension exceeds u32"))?,
    );
    if metric == EmbeddingMetric::Euclidean {
        // Compute differences directly: the expanded norm identity loses
        // small distances between nearly equal high-dimensional vectors.
        let magnitude = f64::from(scale);
        let mut squared = 0.0_f64;
        for (dimension, value) in query.iter().enumerate() {
            if !value.is_finite() {
                return Err(EngineError::invalid_argument(
                    "binary-sign query contains a non-finite value",
                ));
            }
            let positive = packed[dimension / u8::BITS as usize]
                & (1_u8 << (dimension % u8::BITS as usize))
                != 0;
            let document = if positive { magnitude } else { -magnitude };
            let difference = f64::from(*value) - document;
            squared += difference * difference;
        }
        let result = -squared.sqrt();
        return Ok(if result == 0.0 { 0.0 } else { result });
    }

    let mut dot = 0.0_f64;
    let mut query_norm_squared = 0.0_f64;
    for (dimension, value) in query.iter().enumerate() {
        if !value.is_finite() {
            return Err(EngineError::invalid_argument(
                "binary-sign query contains a non-finite value",
            ));
        }
        let value = f64::from(*value);
        query_norm_squared += value * value;
        let positive =
            packed[dimension / u8::BITS as usize] & (1_u8 << (dimension % u8::BITS as usize)) != 0;
        dot += value * if positive { 1.0 } else { -1.0 };
    }
    let result = match metric {
        EmbeddingMetric::Cosine => {
            if query_norm_squared == 0.0 || scale == 0.0 {
                0.0
            } else {
                dot / (query_norm_squared.sqrt() * dimensions.sqrt())
            }
        }
        EmbeddingMetric::DotProduct => dot * f64::from(scale),
        EmbeddingMetric::Euclidean => unreachable!("handled above"),
    };
    Ok(if result == 0.0 { 0.0 } else { result })
}

fn sort_search_hits(hits: &mut [StorageSearchHit]) {
    hits.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.document_id.cmp(&right.document_id))
    });
}

fn identity_doc(entity: &Entity, fragment: &EntityFragment) -> EngineResult<Doc> {
    let mut doc = native(Doc::new(), "create fragment record")?;
    doc.set_pk(fragment.id.as_str());
    native(
        doc.add_u32("file_id", entity.file_id().get()),
        "encode source identity",
    )?;
    native(
        doc.add_string("entity_id", entity.id().as_str()),
        "encode entity identity",
    )?;
    native(
        doc.add_string("document_id", fragment.id.as_str()),
        "encode document identity",
    )?;
    Ok(doc)
}

fn fragment_doc(
    entity: &Entity,
    fragment: &EntityFragment,
    file: &FileRecord,
    directories: &[DirectoryId],
    fields: &[(IndexField, String)],
) -> EngineResult<Doc> {
    let mut doc = identity_doc(entity, fragment)?;
    file_membership_doc(&mut doc, file, directories)?;
    for (field, value) in fields {
        native(
            doc.add_string(field.name(), value),
            "encode metadata index field",
        )?;
    }
    Ok(doc)
}

fn decode_search_hit(doc: &Doc, path: StorageSearchPath) -> EngineResult<StorageSearchHit> {
    decode_search_hit_with_score(doc, path, f64::from(doc.get_score()))
}

fn decode_search_hit_with_score(
    doc: &Doc,
    path: StorageSearchPath,
    score: f64,
) -> EngineResult<StorageSearchHit> {
    let document_id = FragmentId::from_string(string_field(doc, "document_id")?);
    let entity_id = EntityId::from_string(string_field(doc, "entity_id")?);
    if doc_key(doc)? != document_id.as_str() {
        return Err(corrupt(
            "search document identity differs from its primary key",
        ));
    }
    Ok(StorageSearchHit {
        document_id: document_id.as_str().to_owned(),
        entity_id,
        file_id: FileId::new(u32_field(doc, "file_id")?),
        path,
        score,
    })
}

fn configure_query(
    query: &mut SearchQuery,
    filter: Option<&str>,
    fields: &[&str],
) -> EngineResult<()> {
    native(
        query.set_include_vector(false),
        "omit vectors from search output",
    )?;
    native(
        query.set_output_fields(fields),
        "select search output fields",
    )?;
    if let Some(filter) = filter {
        native(query.set_filter(filter), "set search filter")?;
    }
    Ok(())
}

pub(super) fn build_filter(
    filter: Option<&StorageSearchFilter>,
    resolve_directory: &impl Fn(&SourcePath) -> EngineResult<Option<DirectoryId>>,
) -> EngineResult<Option<String>> {
    let Some(filter) = filter else {
        return Ok(None);
    };
    if empty_filter(Some(filter)) {
        return Ok(Some(constant_filter(false)));
    }
    let mut clauses = Vec::new();
    if let Some(path) = &filter.path
        && !matches!(path, StoragePathFilter::All)
    {
        clauses.push(path_filter(resolve_directory, path, false)?);
    }
    if let Some(ids) = &filter.file_ids {
        clauses.push(file_id_filter(ids));
    }
    if let Some(ids) = &filter.entity_ids {
        clauses.push(in_filter("entity_id", ids.iter().map(EntityId::as_str)));
    }
    if let Some(names) = &filter.symbol_names {
        clauses.push(metadata_in_filter(
            CodeMetadata::SYMBOL_NAME,
            names.iter().map(String::as_str),
        ));
    }
    if let Some(types) = &filter.symbol_types {
        clauses.push(metadata_in_filter(
            CodeMetadata::SYMBOL_TYPE,
            types.iter().map(|kind| kind.as_str()),
        ));
    }
    Ok((!clauses.is_empty()).then(|| clauses.join(" AND ")))
}

fn encode_metadata_fields(metadata: &serde_json::Value) -> EngineResult<Vec<(IndexField, String)>> {
    let mut fields = Vec::new();
    for field in EntityMetadata::index_schema() {
        let Some(value) = metadata.get(field.name()).filter(|value| !value.is_null()) else {
            continue;
        };
        match field {
            IndexField::String(name) => {
                let value = value.as_str().ok_or_else(|| {
                    corrupt(format!("metadata index field {name:?} must be a string"))
                })?;
                fields.push((field, encode_metadata_value(field, value).into_owned()));
            }
        }
    }
    Ok(fields)
}

fn encode_metadata_value(field: IndexField, value: &str) -> Cow<'_, str> {
    if field == CodeMetadata::SYMBOL_NAME {
        Cow::Owned(primary_key("symbol", value))
    } else {
        Cow::Borrowed(value)
    }
}

fn metadata_in_filter<'a>(field: IndexField, values: impl Iterator<Item = &'a str>) -> String {
    in_filter(
        field.name(),
        values.map(|value| encode_metadata_value(field, value)),
    )
}

fn in_filter(field: &str, values: impl Iterator<Item = impl AsRef<str>>) -> String {
    format!(
        "{field} IN ({})",
        values
            .map(|value| quote(value.as_ref()))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Push negation to field operators; execution support and errors belong to zvec.
fn path_filter(
    resolve_directory: &impl Fn(&SourcePath) -> EngineResult<Option<DirectoryId>>,
    filter: &StoragePathFilter,
    negated: bool,
) -> EngineResult<String> {
    Ok(match filter {
        StoragePathFilter::All => constant_filter(!negated),
        StoragePathFilter::None => constant_filter(negated),
        StoragePathFilter::Directory(path) => {
            let Some(id) = resolve_directory(path)? else {
                return Ok(constant_filter(negated));
            };
            format!(
                "ancestor_directory_ids {}CONTAIN_ANY ({id})",
                if negated { "NOT " } else { "" }
            )
        }
        StoragePathFilter::FileNameExact(name) => {
            if name.contains(['\\', '\0']) {
                return Err(EngineError::invalid_argument(
                    "this file name requires path matching",
                ));
            }
            format!(
                "file_name {} {}",
                if negated { "!=" } else { "=" },
                quote(name)
            )
        }
        StoragePathFilter::FileNamePrefix(prefix) => format!(
            "file_name {}LIKE {}",
            if negated { "NOT " } else { "" },
            quote(&format!("{}%", like_literal(prefix)))
        ),
        StoragePathFilter::FileNameSuffix(suffix) => format!(
            "file_name {}LIKE {}",
            if negated { "NOT " } else { "" },
            quote(&format!("%{}", like_literal(suffix)))
        ),
        StoragePathFilter::And(filters) => {
            boolean_filter(resolve_directory, filters, !negated, negated)?
        }
        StoragePathFilter::Or(filters) => {
            boolean_filter(resolve_directory, filters, negated, negated)?
        }
        StoragePathFilter::Not(filter) => path_filter(resolve_directory, filter, !negated)?,
    })
}

fn file_id_filter(ids: &[FileId]) -> String {
    if ids.is_empty() {
        return constant_filter(false);
    }
    format!(
        "file_id IN ({})",
        ids.iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn constant_filter(value: bool) -> String {
    // Every stored document has a file ID, including ID zero.
    format!("file_id IS {}NULL", if value { "NOT " } else { "" })
}

fn boolean_filter(
    resolve_directory: &impl Fn(&SourcePath) -> EngineResult<Option<DirectoryId>>,
    filters: &[StoragePathFilter],
    conjunction: bool,
    negated: bool,
) -> EngineResult<String> {
    if filters.is_empty() {
        return Ok(constant_filter(conjunction));
    }
    let operator = if conjunction { " AND " } else { " OR " };
    Ok(format!(
        "({})",
        filters
            .iter()
            .map(|filter| path_filter(resolve_directory, filter, negated))
            .collect::<EngineResult<Vec<_>>>()?
            .join(operator)
    ))
}

fn like_literal(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

pub(super) fn empty_filter(filter: Option<&StorageSearchFilter>) -> bool {
    filter.is_some_and(|filter| {
        filter.file_ids.as_ref().is_some_and(Vec::is_empty)
            || filter.entity_ids.as_ref().is_some_and(Vec::is_empty)
            || filter.symbol_names.as_ref().is_some_and(Vec::is_empty)
            || filter.symbol_types.as_ref().is_some_and(Vec::is_empty)
    })
}

fn top_k(limit: usize) -> EngineResult<i32> {
    if limit > MAX_TOP_K {
        return Err(EngineError::invalid_argument(format!(
            "storage query limit must not exceed {MAX_TOP_K}"
        )));
    }
    i32::try_from(limit)
        .map_err(|_| EngineError::invalid_argument("storage query limit is too large"))
}

// C strings cannot contain NUL; indexed projections preserve token boundaries.
fn index_text(text: &str) -> Cow<'_, str> {
    if text.contains('\0') {
        Cow::Owned(text.replace('\0', " "))
    } else {
        Cow::Borrowed(text)
    }
}

fn reject_foreign_ids(
    collection: &Collection,
    keys: &[String],
    file_id: FileId,
) -> EngineResult<()> {
    for doc in fetch_map(collection, keys)?.into_values() {
        if u32_field(&doc, "file_id")? != file_id.get() {
            return Err(EngineError::invalid_argument(
                "entity or fragment ID is already owned by another file",
            ));
        }
    }
    Ok(())
}

/// Every canonical fragment has exactly one projection, and its entity has one model.
pub(super) fn validate_projections(
    entities: &[Entity],
    entries: &[IndexedFragment],
) -> EngineResult<()> {
    let expected = entities
        .iter()
        .flat_map(|entity| {
            entity
                .fragments()
                .iter()
                .map(move |fragment| (&fragment.id, entity.id()))
        })
        .collect::<HashMap<_, _>>();
    let mut seen = HashSet::new();
    let mut models = HashMap::new();
    for entry in entries {
        if expected.get(&entry.fragment_id).copied() != Some(&entry.entity_id) {
            return Err(EngineError::invalid_argument(
                "search projection does not reference its canonical entity fragment",
            ));
        }
        if !seen.insert(&entry.fragment_id) {
            return Err(EngineError::invalid_argument(
                "canonical fragment has more than one search projection",
            ));
        }
        if models
            .insert(&entry.entity_id, &entry.model)
            .is_some_and(|previous| previous != &entry.model)
        {
            return Err(EngineError::invalid_argument(
                "all fragments of an entity must use one embedding model",
            ));
        }
    }
    if expected.len() != seen.len() {
        return Err(EngineError::invalid_argument(
            "every canonical fragment requires a search projection",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;

#[cfg(feature = "experimental-sign-scale-v1")]
mod binary_diskann_generation;
