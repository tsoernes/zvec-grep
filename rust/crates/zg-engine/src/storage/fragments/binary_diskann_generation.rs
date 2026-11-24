//! Bounded extraction and staging of a packed-DiskANN generation from fragment rows.
//!
//! The caller must hold the IndexStore state lock throughout this operation.
//! The resulting immutable graph and identity sidecar must be published with
//! the matching workspace generation manifest.

use super::*;
use zg_zvec_fp16::{
    packed_diskann_generation::{self, CreateParams, Limits, Trust},
    packed_sign_index::{Document, Generation, Metric},
};

// Test-only miniature generation helper; production build limits are below.
#[cfg(test)]
const MAX_ROWS: usize = 128;
const MAX_ID_BYTES: usize = 1024;
#[cfg(test)]
const MAX_SOURCE_BYTES: usize = 1024 * 1024;
const MAX_BINARY_DISKANN_ROWS: usize = 100_000;
const MAX_BINARY_DISKANN_SOURCE_BYTES: usize = 256 * 1024 * 1024;
const MAX_BINARY_DISKANN_WORKING_BYTES: usize = 512 * 1024 * 1024;
const MAX_BINARY_DISKANN_IDENTITY_BYTES: usize = 256 * 1024 * 1024;
const MAX_BINARY_DISKANN_CODEBOOK_BYTES: usize = 256 * 1024 * 1024;
const MAX_BINARY_DISKANN_GENERATION_BYTES: usize = 512 * 1024 * 1024;
const MAX_BINARY_DISKANN_QUERY_BYTES: usize = 16 * 1024 * 1024;
const MAX_BINARY_DISKANN_BEAM: usize = 8192;

impl Fragments {
    #[cfg(test)]
    pub(in crate::storage) fn disposable_generation(
        &self,
        model: &str,
        generation: [u8; 16],
        generation_budget: usize,
    ) -> EngineResult<Generation> {
        self.generation_from_rows(
            model,
            generation,
            VectorIndexQuantization::BinarySign,
            MAX_ROWS,
            MAX_SOURCE_BYTES,
            generation_budget,
        )
    }

    fn generation_from_rows(
        &self,
        model: &str,
        generation: [u8; 16],
        expected_mode: VectorIndexQuantization,
        max_rows: usize,
        max_source_bytes: usize,
        generation_budget: usize,
    ) -> EngineResult<Generation> {
        let embedding = self
            .embeddings
            .get(model)
            .ok_or_else(|| EngineError::invalid_argument("unknown binary generation model"))?;
        if embedding.vector_index_quantization() != expected_mode {
            return Err(EngineError::unsupported(
                "binary generation mode does not match the stored fragment schema",
            ));
        }
        let index = self.index(model)?;
        let count = usize::try_from(native(index.stats(), "inspect binary collection")?.doc_count)
            .map_err(|_| corrupt("binary row count overflow"))?;
        if count == 0 || count > max_rows {
            return Err(EngineError::unsupported(format!(
                "binary row count {count} outside the configured generation bound {max_rows}"
            )));
        }
        let mut rows = Vec::new();
        rows.try_reserve_exact(count)
            .map_err(|_| EngineError::storage_failure("binary row allocation failed"))?;
        let mut bytes = 0usize;
        let iterator = native(
            index.iter_with_options(
                Some(&["document_id", "embedding", "embedding_scale"]),
                false,
            ),
            "iterate packed-vector rows",
        )?;
        for doc in iterator {
            let doc = native(doc, "read binary vector")?;
            let id = string_field(&doc, "document_id")?;
            if id.is_empty() || id.len() > MAX_ID_BYTES || doc_key(&doc)? != id {
                return Err(corrupt("invalid binary fragment identity"));
            }
            let signs = native(doc.get_binary("embedding"), "read packed signs")?
                .ok_or_else(|| corrupt("missing binary signs"))?;
            let scale = native(doc.get_f32("embedding_scale"), "read fixture scale")?
                .ok_or_else(|| corrupt("missing binary scale"))?;
            bytes = bytes
                .checked_add(id.len())
                .and_then(|n| n.checked_add(signs.len()))
                .ok_or_else(|| corrupt("binary source size overflow"))?;
            if bytes > max_source_bytes || rows.len() >= count {
                return Err(corrupt("binary source exceeds bounded snapshot"));
            }
            rows.push((id, signs, scale));
        }
        if rows.len() != count {
            return Err(corrupt("binary collection changed while extracting"));
        }
        let documents: Vec<_> = rows
            .iter()
            .map(|(id, signs, scale)| Document {
                id,
                signs,
                scale: *scale,
            })
            .collect();
        Generation::prepare(
            embedding.dimension(),
            generation,
            &documents,
            generation_budget,
        )
        .map_err(|error| corrupt(format!("invalid binary generation: {error:?}")))
    }

    pub(in crate::storage) fn build_binary_diskann(
        &mut self,
        index_root: &Path,
        storage_generation: &str,
    ) -> EngineResult<BinaryDiskAnnManifest> {
        let mut models = self.embeddings.iter().filter(|(_, embedding)| {
            embedding.vector_index_quantization() == VectorIndexQuantization::BinarySignDiskAnnV1
        });
        let (model, embedding) = models
            .next()
            .ok_or_else(|| EngineError::unsupported("workspace has no V1 embedding model"))?;
        if models.next().is_some() {
            return Err(EngineError::unsupported(
                "a binaryDiskAnn manifest currently supports one V1 embedding model",
            ));
        }
        let generation_id = uuid::Uuid::parse_str(storage_generation).map_err(|_| {
            EngineError::invalid_argument("binary DiskANN storage generation must be a UUID")
        })?;
        let generation = self.generation_from_rows(
            model,
            *generation_id.as_bytes(),
            VectorIndexQuantization::BinarySignDiskAnnV1,
            MAX_BINARY_DISKANN_ROWS,
            MAX_BINARY_DISKANN_SOURCE_BYTES,
            MAX_BINARY_DISKANN_WORKING_BYTES,
        )?;
        let identity_bytes = generation
            .identity_bytes(MAX_BINARY_DISKANN_IDENTITY_BYTES)
            .map_err(|error| corrupt(format!("encode binary DiskANN identity: {error:?}")))?;
        let identity_sha256 = packed_diskann_generation::identity_sha256(
            &identity_bytes,
            MAX_BINARY_DISKANN_IDENTITY_BYTES,
        )
        .map_err(|error| {
            EngineError::storage_failure(format!("hash binary DiskANN identity: {error:?}"))
        })?;
        let identity_record_sha256 = packed_diskann_generation::identity_record_sha256(
            &identity_bytes,
            generation.records(),
            MAX_BINARY_DISKANN_GENERATION_BYTES,
        )
        .map_err(|error| {
            EngineError::storage_failure(format!("bind binary DiskANN rows: {error:?}"))
        })?;
        let metric = match embedding.metric() {
            EmbeddingMetric::Cosine => Metric::Cosine,
            EmbeddingMetric::DotProduct => Metric::DotProduct,
            EmbeddingMetric::Euclidean => Metric::Euclidean,
        };
        let limits = Limits {
            max_documents: MAX_BINARY_DISKANN_ROWS,
            max_input_bytes: MAX_BINARY_DISKANN_SOURCE_BYTES,
            max_build_bytes: MAX_BINARY_DISKANN_WORKING_BYTES,
            max_codebook_bytes: MAX_BINARY_DISKANN_CODEBOOK_BYTES,
            max_generation_bytes: MAX_BINARY_DISKANN_GENERATION_BYTES,
            max_query_scratch_bytes: MAX_BINARY_DISKANN_QUERY_BYTES,
            max_beam: MAX_BINARY_DISKANN_BEAM,
            max_degree: 64,
            build_list_size: 100,
            build_threads: 2,
        };
        let count = u32::try_from(generation.ids().len())
            .map_err(|_| EngineError::storage_failure("binary DiskANN row count exceeds u32"))?;
        if count < 4 {
            return Err(EngineError::unsupported(
                "binary DiskANN V1 requires at least four vectors",
            ));
        }

        let index_root = std::fs::canonicalize(index_root)
            .map_err(|error| EngineError::from_io("locate staged binary DiskANN index", &error))?;
        let graph_parent = index_root.join("binary-diskann");
        let mut builder = std::fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder.create(&graph_parent).map_err(|error| {
            EngineError::from_io("create private binary DiskANN directory", &error)
        })?;
        let graph_path = graph_parent.join("graph");
        let keys = generation
            .keys(MAX_BINARY_DISKANN_IDENTITY_BYTES)
            .map_err(|error| corrupt(format!("encode binary DiskANN keys: {error:?}")))?;
        let create_params = CreateParams {
            staging_directory: &graph_path,
            dimension: embedding.dimension(),
            metric,
            keys: &keys,
            records: generation.records(),
            generation: *generation_id.as_bytes(),
            identity_digest: identity_record_sha256,
            limits,
        };
        let trust = packed_diskann_generation::create_v1(&create_params).map_err(|error| {
            EngineError::storage_failure(format!("build binary DiskANN graph: {error:?}"))
        })?;
        // The native create call has copied the packed rows into the graph;
        // release the build-side vector copy before reopening the generation.
        drop(generation);
        let identity_path = graph_parent.join("identity.bin");
        let mut identity_file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&identity_path)
            .map_err(|error| EngineError::from_io("create binary DiskANN identity", &error))?;
        use std::io::Write;
        identity_file
            .write_all(&identity_bytes)
            .map_err(|error| EngineError::from_io("write binary DiskANN identity", &error))?;
        identity_file
            .sync_all()
            .map_err(|error| EngineError::from_io("sync binary DiskANN identity", &error))?;
        crate::utils::sync_directory(&graph_parent)?;
        crate::utils::sync_directory(&index_root)?;

        let relative_graph = graph_path
            .strip_prefix(&index_root)
            .map_err(|_| EngineError::storage_failure("binary DiskANN graph escaped its index"))?
            .to_str()
            .ok_or_else(|| EngineError::storage_failure("binary DiskANN graph path is not UTF-8"))?
            .to_owned();
        let relative_identity = identity_path
            .strip_prefix(&index_root)
            .map_err(|_| EngineError::storage_failure("binary DiskANN identity escaped its index"))?
            .to_str()
            .ok_or_else(|| {
                EngineError::storage_failure("binary DiskANN identity path is not UTF-8")
            })?
            .to_owned();
        let metadata = BinaryDiskAnnManifest::new(
            storage_generation.to_owned(),
            relative_graph,
            relative_identity,
            u32::try_from(embedding.dimension()).map_err(|_| {
                EngineError::storage_failure("binary DiskANN dimension exceeds u32")
            })?,
            u64::from(count),
            identity_sha256,
            trust.to_bytes(),
        )?;
        let pinned_trust = Trust::from_bytes(metadata.trust()).map_err(|error| {
            EngineError::storage_failure(format!("invalid binary DiskANN trust: {error:?}"))
        })?;
        let worker = BinaryDiskAnnWorker::open(
            &graph_path,
            pinned_trust,
            &identity_bytes,
            identity_sha256,
            metric,
            limits,
        )
        .map_err(|error| {
            EngineError::storage_failure(format!("reopen binary DiskANN graph: {error:?}"))
        })?;
        let smoke_query = vec![1.0_f32; embedding.dimension()];
        if worker
            .search(&smoke_query, 1, 1)
            .map_err(|error| {
                EngineError::storage_failure(format!(
                    "smoke search binary DiskANN graph: {error:?}"
                ))
            })?
            .is_empty()
        {
            return Err(EngineError::storage_failure(
                "binary DiskANN smoke search returned no candidates",
            ));
        }
        self.binary_diskann = Some((model.clone(), Arc::new(worker)));
        Ok(metadata)
    }
}
