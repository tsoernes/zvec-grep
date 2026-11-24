//! Prepare, publish, recover, and remove workspace index storage.

use std::{fs, path::Path};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::manifest::{read_workspace_manifest, write_workspace_manifest};
use crate::{
    EngineError,
    domain::{BinaryDiskAnnManifest, Workspace, WorkspaceLayout},
    storage::IndexStore,
    utils::{
        atomic_write, directory_exists, ensure_directory, reject_symlink, sha256_file, sha256_hex,
        sync_directory,
    },
};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BuildRecord {
    index_version: u32,
    generation: String,
    previous_manifest: Option<String>,
    had_index: bool,
    publishing: bool,
}

impl BuildRecord {
    fn validate(&self) -> Result<(), EngineError> {
        if !super::is_supported_index_version(self.index_version) {
            return Err(EngineError::storage_failure(format!(
                "unsupported workspace build index version {}; supported versions are {} and {}",
                self.index_version,
                super::CURRENT_INDEX_VERSION,
                super::BINARY_DISKANN_INDEX_VERSION,
            )));
        }
        if Uuid::parse_str(&self.generation).is_err() {
            return Err(EngineError::storage_failure(
                "invalid workspace build ownership record",
            ));
        }
        Ok(())
    }
}

/// Call while holding the workspace read lock; a pending swap must recover before reads.
pub(crate) fn needs_recovery(layout: &WorkspaceLayout) -> Result<bool, EngineError> {
    let Some(record) = read_record(layout)? else {
        return directory_exists(layout.backup_index_dir());
    };
    Ok(record.publishing && !is_committed(layout, &record)?)
}

pub(crate) fn recover_build(layout: &WorkspaceLayout) -> Result<(), EngineError> {
    recover_build_with(layout, false, &IndexStore::delete)
}

pub(crate) fn recover_build_for_rebuild(layout: &WorkspaceLayout) -> Result<(), EngineError> {
    recover_build_with(layout, true, &IndexStore::delete)
}

fn recover_build_with(
    layout: &WorkspaceLayout,
    discard_invalid: bool,
    delete_storage: &impl Fn(&Path) -> Result<(), EngineError>,
) -> Result<(), EngineError> {
    let record = match read_record(layout) {
        Ok(Some(record)) => record,
        Ok(None) => {
            if directory_exists(layout.backup_index_dir())? {
                return Err(EngineError::storage_failure(
                    "workspace backup has no build record; cannot safely recover",
                ));
            }
            return cleanup_stage(layout, delete_storage);
        }
        // Without a backup there is no interrupted directory swap to roll back.
        Err(_) if discard_invalid && !directory_exists(layout.backup_index_dir())? => {
            return cleanup_stage(layout, delete_storage);
        }
        Err(error) => return Err(error),
    };
    if is_committed(layout, &record)? {
        let source_root = layout.storage_dir().parent().ok_or_else(|| {
            EngineError::storage_failure("workspace storage directory has no source-root parent")
        })?;
        let workspace = read_workspace_manifest(source_root, layout)?.ok_or_else(|| {
            EngineError::storage_failure("committed workspace manifest is missing")
        })?;
        crate::storage::validate_workspace_storage_format_at_path(
            layout.index_dir(),
            workspace.storage_format(),
        )?;
        delete_storage(layout.backup_index_dir())?;
    } else if record.publishing {
        let current = read_optional(layout.manifest_path())?;
        if current.as_deref().map(sha256_hex) != record.previous_manifest {
            return Err(EngineError::storage_failure(
                "workspace manifest changed during index publication; cannot safely recover",
            ));
        }
        let previous = layout.backup_index_dir();
        if directory_exists(previous)? {
            delete_storage(layout.index_dir())?;
            move_index(previous, layout.index_dir())?;
        } else if !record.had_index {
            delete_storage(layout.index_dir())?;
        }
    }
    cleanup_stage(layout, delete_storage)
}

/// Start a new attempt only after recovering the previous one.
pub(crate) fn prepare_build(target: &Workspace) -> Result<(), EngineError> {
    let generation = target.storage_generation().ok_or_else(|| {
        EngineError::invalid_argument(format!(
            "workspace {}: preparing an index build requires an enabled index",
            target.source_root().display(),
        ))
    })?;
    ensure_directory(target.layout().storage_dir())?;
    if target.layout().has_build_record()?
        || directory_exists(target.layout().backup_index_dir())?
        || directory_exists(target.layout().staged_index_dir())?
    {
        return Err(EngineError::resource_busy(
            "recover the unfinished workspace build before starting another",
        ));
    }
    let record = BuildRecord {
        index_version: target.index_version(),
        generation: generation.to_owned(),
        previous_manifest: read_optional(target.layout().manifest_path())?
            .as_deref()
            .map(sha256_hex),
        had_index: directory_exists(target.layout().index_dir())?,
        publishing: false,
    };
    ensure_directory(target.layout().temp_dir())?;
    write_record(target.layout(), &record)?;
    sync_directory(target.layout().storage_dir())?;
    if let Err(error) = ensure_directory(target.layout().staged_index_dir())
        .and_then(|()| sync_directory(target.layout().temp_dir()))
    {
        let _ = discard_build(target);
        return Err(error);
    }
    Ok(())
}

/// A stale completion must never discard a newer attempt or a published index.
pub(crate) fn discard_build(build: &Workspace) -> Result<(), EngineError> {
    let Some(record) = read_record(build.layout())? else {
        return Ok(());
    };
    if Some(record.generation.as_str()) != build.storage_generation()
        || is_committed(build.layout(), &record)?
    {
        return Ok(());
    }
    recover_build(build.layout())
}

pub(crate) fn publish_build(build: Workspace, updated_time: u64) -> Result<(), EngineError> {
    publish_build_with(build, updated_time, &IndexStore::delete)
}

fn publish_build_with(
    mut build: Workspace,
    updated_time: u64,
    delete_storage: &impl Fn(&Path) -> Result<(), EngineError>,
) -> Result<(), EngineError> {
    build.record_update(updated_time);
    let layout = build.layout();
    let result = build
        .storage_format()
        .map_or(Ok(()), |format| {
            crate::storage::validate_workspace_storage_format_at_path(
                layout.staged_index_dir(),
                Some(format),
            )
        })
        .and_then(|()| {
            if let Some(metadata) = build.binary_diskann_manifest() {
                validate_binary_diskann_bundle(layout.staged_index_dir(), metadata)
            } else {
                Ok(())
            }
        })
        .and_then(|()| begin_publication(&build))
        .and_then(|()| {
            if directory_exists(build.layout().index_dir())? {
                move_index(build.layout().index_dir(), layout.backup_index_dir())?;
            }
            move_index(
                build.layout().staged_index_dir(),
                build.layout().index_dir(),
            )?;
            // Only this atomic file replacement commits the new generation.
            let bytes = fs::read(layout.staged_manifest_path()).map_err(build_io)?;
            atomic_write(layout.manifest_path(), &bytes)
        });
    if let Err(error) = result {
        let _ = discard_build(&build);
        return Err(error);
    }
    // Keep the record if cleanup fails, so the next writer can finish it.
    let _ = recover_build_with(layout, false, delete_storage);
    Ok(())
}

fn validate_binary_diskann_bundle(
    index_root: &Path,
    metadata: &BinaryDiskAnnManifest,
) -> Result<(), EngineError> {
    const MAX_IDENTITY_BYTES: u64 = 256 * 1024 * 1024;

    reject_symlink(index_root)?;
    let canonical_root = fs::canonicalize(index_root).map_err(build_io)?;
    let graph_path = index_root.join(metadata.graph_path());
    let identity_path = index_root.join(metadata.identity_path());
    reject_symlink(&graph_path)?;
    reject_symlink(&identity_path)?;

    let graph_metadata = fs::symlink_metadata(&graph_path).map_err(build_io)?;
    if graph_metadata.file_type().is_symlink() || !graph_metadata.is_dir() {
        return Err(EngineError::storage_failure(
            "binary DiskANN graph path is not a regular directory",
        ));
    }
    let canonical_graph = fs::canonicalize(&graph_path).map_err(build_io)?;
    if !canonical_graph.starts_with(&canonical_root) {
        return Err(EngineError::storage_failure(
            "binary DiskANN graph path escapes the staged index",
        ));
    }

    let identity_metadata = fs::symlink_metadata(&identity_path).map_err(build_io)?;
    if identity_metadata.file_type().is_symlink()
        || !identity_metadata.is_file()
        || identity_metadata.len() > MAX_IDENTITY_BYTES
    {
        return Err(EngineError::storage_failure(
            "binary DiskANN identity path is not a bounded regular file",
        ));
    }
    let canonical_identity = fs::canonicalize(&identity_path).map_err(build_io)?;
    if !canonical_identity.starts_with(&canonical_root) {
        return Err(EngineError::storage_failure(
            "binary DiskANN identity path escapes the staged index",
        ));
    }
    let digest = sha256_file(&canonical_identity).map_err(build_io)?;
    if digest != metadata.identity_sha256() {
        return Err(EngineError::storage_failure(
            "binary DiskANN identity digest does not match the outer manifest",
        ));
    }
    Ok(())
}

fn begin_publication(build: &Workspace) -> Result<(), EngineError> {
    let layout = build.layout();
    let mut record = read_record(layout)?
        .ok_or_else(|| EngineError::storage_failure("workspace build record is missing"))?;
    if Some(record.generation.as_str()) != build.storage_generation() || record.publishing {
        return Err(EngineError::storage_failure(
            "workspace build no longer owns the pending publication",
        ));
    }
    if read_optional(layout.manifest_path())?
        .as_deref()
        .map(sha256_hex)
        != record.previous_manifest
    {
        return Err(EngineError::storage_failure(
            "workspace manifest changed while building the index",
        ));
    }
    write_workspace_manifest(layout.staged_manifest_path(), build)?;
    record.publishing = true;
    write_record(layout, &record)
}

/// Both parents must be durable before the manifest may commit a directory move.
fn move_index(from: &Path, to: &Path) -> Result<(), EngineError> {
    if !directory_exists(from)? {
        return Err(EngineError::storage_failure(format!(
            "index directory is missing: {}",
            from.display(),
        )));
    }
    reject_symlink(to)?;
    fs::rename(from, to).map_err(build_io)?;
    sync_directory(from.parent().expect("index parent"))?;
    sync_directory(to.parent().expect("index parent"))
}

pub(crate) fn drop_index_storage(layout: &WorkspaceLayout) -> Result<(), EngineError> {
    IndexStore::delete(layout.index_dir())?;
    reject_symlink(layout.temp_dir())?;
    IndexStore::delete(layout.backup_index_dir())?;
    cleanup_stage(layout, &IndexStore::delete)
}

fn cleanup_stage(
    layout: &WorkspaceLayout,
    delete_storage: &impl Fn(&Path) -> Result<(), EngineError>,
) -> Result<(), EngineError> {
    reject_symlink(layout.temp_dir())?;
    delete_storage(layout.staged_index_dir())?;
    for path in [layout.staged_manifest_path(), layout.build_record_path()] {
        reject_symlink(path)?;
        match fs::remove_file(path) {
            Ok(()) => sync_directory(layout.temp_dir())?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(build_io(error)),
        }
    }
    match fs::remove_dir(layout.temp_dir()) {
        Ok(()) => sync_directory(layout.storage_dir()),
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::NotFound | std::io::ErrorKind::DirectoryNotEmpty
            ) =>
        {
            Ok(())
        }
        Err(error) => Err(build_io(error)),
    }
}

fn read_record(layout: &WorkspaceLayout) -> Result<Option<BuildRecord>, EngineError> {
    reject_symlink(layout.temp_dir())?;
    read_optional(layout.build_record_path())?
        .as_deref()
        .map(decode_build)
        .transpose()
}

fn decode_build(bytes: &[u8]) -> Result<BuildRecord, EngineError> {
    let build: BuildRecord = serde_json::from_slice(bytes).map_err(|error| {
        EngineError::storage_failure(format!("invalid workspace build record: {error}"))
    })?;
    build.validate()?;
    Ok(build)
}

fn is_committed(layout: &WorkspaceLayout, build: &BuildRecord) -> Result<bool, EngineError> {
    let bytes = read_optional(layout.manifest_path())?;
    if bytes.as_deref().map(sha256_hex) == build.previous_manifest {
        // The manifest is still byte-for-byte the recorded pre-publication
        // state. It may be malformed; recovery can still safely restore its
        // matching backup without interpreting it as the committed generation.
        return Ok(false);
    }
    let source_root = layout.storage_dir().parent().ok_or_else(|| {
        EngineError::storage_failure("workspace storage directory has no source-root parent")
    })?;
    let Some(manifest) = read_workspace_manifest(source_root, layout)? else {
        return Ok(false);
    };
    if manifest.storage_generation() != Some(build.generation.as_str()) {
        return Ok(false);
    }
    let expected_format = match build.index_version {
        super::CURRENT_INDEX_VERSION => None,
        super::BINARY_DISKANN_INDEX_VERSION => {
            Some(crate::domain::WorkspaceStorageFormat::BinarySignDiskAnnV1)
        }
        _ => unreachable!("build records validate their index version before use"),
    };
    if manifest.index_version() != build.index_version
        || manifest.storage_format() != expected_format
    {
        return Err(EngineError::storage_failure(
            "workspace build record format does not match the committed manifest generation",
        ));
    }
    Ok(true)
}

fn write_record(layout: &WorkspaceLayout, record: &BuildRecord) -> Result<(), EngineError> {
    record.validate()?;
    let bytes = serde_json::to_vec_pretty(record)
        .map_err(|error| EngineError::internal(format!("encode workspace build: {error}")))?;
    atomic_write(layout.build_record_path(), &bytes)
}

fn read_optional(path: &Path) -> Result<Option<Vec<u8>>, EngineError> {
    reject_symlink(path)?;
    match fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(build_io(error)),
    }
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "used as a Result::map_err callback"
)]
fn build_io(error: std::io::Error) -> EngineError {
    EngineError::from_io("access workspace build state", &error)
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use crate::domain::{
        IndexDescriptor, IndexState, Workspace,
        model::{EmbeddingMetric, EmbeddingModelInfo},
    };
    use crate::workspace::manifest::{inspect_workspace_manifest, read_workspace_manifest};

    use super::*;

    #[cfg(unix)]
    #[test]
    fn build_operations_reject_a_symlinked_temporary_directory() {
        let directory = tempdir().expect("workspace");
        let external = tempdir().expect("external directory");
        let target = manifest(directory.path());
        fs::create_dir(target.layout().storage_dir()).expect("workspace home");
        std::os::unix::fs::symlink(external.path(), target.layout().temp_dir())
            .expect("temporary directory alias");
        let marker = external.path().join("manifest.json");
        fs::write(&marker, b"external data").expect("external marker");

        assert!(needs_recovery(target.layout()).is_err());
        assert!(prepare_build(&target).is_err());
        assert!(recover_build(target.layout()).is_err());
        assert!(drop_index_storage(target.layout()).is_err());
        assert_eq!(
            fs::read(marker).expect("preserved marker"),
            b"external data"
        );
        assert!(!external.path().join("new").exists());
        assert!(!external.path().join("build.json").exists());
    }

    fn fail_cleanup(_path: &Path) -> Result<(), EngineError> {
        Err(EngineError::storage_failure("injected cleanup failure"))
    }

    fn manifest(root: &Path) -> Workspace {
        let root = fs::canonicalize(root).expect("canonical fixture root");
        Workspace::new(
            &root,
            WorkspaceLayout::new(root.join(".zvec-grep")).expect("layout"),
            crate::domain::ScanRules::default(),
            crate::workspace::CURRENT_INDEX_VERSION,
            Some(Uuid::new_v4().to_string()),
            1,
            2,
            IndexState::Enabled(Box::new(
                IndexDescriptor::new(
                    EmbeddingModelInfo::new(
                        crate::domain::model::ModelInfo::new(
                            "local",
                            "example",
                            [
                                crate::domain::ContentKind::Text,
                                crate::domain::ContentKind::Code,
                            ],
                            32,
                        )
                        .expect("fixture model identity"),
                        8,
                        EmbeddingMetric::Cosine,
                        None,
                        None,
                    )
                    .expect("embedding model"),
                    std::collections::BTreeMap::new(),
                    std::collections::BTreeMap::from([(
                        "local/example".into(),
                        crate::domain::model::ModelConfig::default(),
                    )]),
                    false,
                    crate::domain::FTS_CONFIG,
                )
                .expect("index descriptor"),
            )),
        )
        .expect("manifest")
    }

    fn binary_diskann_workspace(base: &Workspace) -> Workspace {
        let generation = Uuid::new_v4().to_string();
        let target = Workspace::new(
            base.source_root(),
            base.layout().clone(),
            base.scan().clone(),
            super::super::BINARY_DISKANN_INDEX_VERSION,
            Some(generation.clone()),
            base.created_epoch_ms(),
            base.updated_epoch_ms(),
            binary_diskann_index(base.index()),
        )
        .expect("version three target")
        .with_storage_format(Some(
            crate::domain::WorkspaceStorageFormat::BinarySignDiskAnnV1,
        ));
        let dimension = u32::try_from(
            target
                .index()
                .descriptor()
                .expect("descriptor")
                .default_embedding_model()
                .dimension(),
        )
        .expect("dimension");
        let identity = binary_diskann_identity(&generation, dimension);
        let identity_digest = crate::utils::sha256_parts([identity.as_slice()]);
        let mut trust = Vec::with_capacity(300);
        trust.extend_from_slice(b"ZGV1TRST");
        trust.extend_from_slice(&1_u32.to_le_bytes());
        trust.extend_from_slice(&dimension.to_le_bytes());
        trust.extend_from_slice(&1_u32.to_le_bytes());
        trust.extend_from_slice(&1_u64.to_le_bytes());
        trust.extend_from_slice(Uuid::parse_str(&generation).expect("generation").as_bytes());
        for digest in 1_u8..=8 {
            trust.extend_from_slice(&[digest; 32]);
        }
        let metadata = BinaryDiskAnnManifest::new(
            generation,
            "binary-diskann/graph".to_owned(),
            "binary-diskann/identity.bin".to_owned(),
            dimension,
            1,
            identity_digest,
            trust,
        )
        .expect("test metadata");
        target
            .with_binary_diskann_manifest(Some(metadata))
            .expect("bind metadata to generation")
    }

    fn binary_diskann_identity(generation: &str, dimension: u32) -> Vec<u8> {
        let mut bytes = b"ZGIDV1\0\0".to_vec();
        bytes.extend_from_slice(Uuid::parse_str(generation).expect("generation").as_bytes());
        bytes.extend_from_slice(&dimension.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&5_u64.to_le_bytes());
        bytes.extend_from_slice(b"doc-1");
        bytes
    }

    fn write_binary_diskann_bundle(workspace: &Workspace, root: &Path) {
        if let Some(metadata) = workspace.binary_diskann_manifest() {
            let graph = root.join(metadata.graph_path());
            fs::create_dir_all(&graph).expect("binary graph directory");
            fs::write(graph.join("graph.zvec"), b"fixture graph").expect("graph bytes");
            let identity = root.join(metadata.identity_path());
            fs::create_dir_all(identity.parent().expect("identity parent"))
                .expect("identity directory");
            fs::write(
                identity,
                binary_diskann_identity(metadata.storage_generation(), metadata.dimension()),
            )
            .expect("identity bytes");
        }
    }

    fn write_active(manifest: &Workspace) {
        fs::create_dir_all(manifest.layout().index_dir()).expect("active storage");
        fs::write(manifest.layout().index_dir().join("old"), "old data").expect("old data");
        write_test_schema(manifest, manifest.layout().index_dir());
        write_binary_diskann_bundle(manifest, manifest.layout().index_dir());
        write_workspace_manifest(manifest.layout().manifest_path(), manifest)
            .expect("active manifest");
    }

    fn write_test_schema(workspace: &Workspace, storage_path: &Path) {
        let descriptor = workspace.index().descriptor().expect("index descriptor");
        let mut embeddings = vec![descriptor.default_embedding_model().clone()];
        embeddings.extend(descriptor.embedding_routes().values().cloned());
        let binary_storage = embeddings.iter().any(|embedding| {
            matches!(
                embedding.vector_index_quantization(),
                crate::domain::VectorIndexQuantization::BinarySign
                    | crate::domain::VectorIndexQuantization::BinarySignDiskAnnV1
            )
        });
        let diskann_v1 = embeddings.iter().any(|embedding| {
            embedding.vector_index_quantization()
                == crate::domain::VectorIndexQuantization::BinarySignDiskAnnV1
        });
        let schema = serde_json::json!({
            "path_encoding": "components",
            "binary_sign_storage": binary_storage,
            "binary_sign_diskann_storage_version": diskann_v1.then_some(1),
            "embeddings": embeddings,
        });
        fs::write(
            storage_path.join("schema.json"),
            serde_json::to_vec(&schema).expect("test schema JSON"),
        )
        .expect("test schema");
    }

    fn build_target(active: &Workspace) -> Workspace {
        let build = Workspace::new(
            active.source_root(),
            active.layout().clone(),
            active.scan().clone(),
            active.index_version(),
            Some(Uuid::new_v4().to_string()),
            active.created_epoch_ms(),
            active.updated_epoch_ms(),
            active.index().clone(),
        )
        .expect("build target")
        .with_storage_format(active.storage_format());
        if active.storage_format()
            == Some(crate::domain::WorkspaceStorageFormat::BinarySignDiskAnnV1)
        {
            binary_diskann_workspace(&build)
        } else {
            build
        }
    }

    fn binary_diskann_index(index: &IndexState) -> IndexState {
        let descriptor = index.descriptor().expect("enabled fixture index");
        let default = descriptor
            .default_embedding_model()
            .clone()
            .with_vector_index_quantization(
                crate::domain::VectorIndexQuantization::BinarySignDiskAnnV1,
            );
        let routes = descriptor
            .embedding_routes()
            .iter()
            .map(|(kind, embedding)| {
                (
                    *kind,
                    embedding.clone().with_vector_index_quantization(
                        crate::domain::VectorIndexQuantization::BinarySignDiskAnnV1,
                    ),
                )
            })
            .collect();
        IndexState::Enabled(Box::new(
            IndexDescriptor::new(
                default,
                routes,
                descriptor.embedding_configs().clone(),
                descriptor.graph_indexing(),
                descriptor.fts(),
            )
            .expect("binary DiskANN descriptor"),
        ))
    }

    fn stage(active: &Workspace) -> Workspace {
        let build = build_target(active);
        prepare_build(&build).expect("stage");
        write_test_schema(&build, build.layout().staged_index_dir());
        write_binary_diskann_bundle(&build, build.layout().staged_index_dir());
        fs::write(build.layout().staged_index_dir().join("new"), "new data").expect("new data");
        build
    }

    #[test]
    fn interrupted_builds_are_discarded_before_a_fresh_attempt() {
        let directory = tempdir().expect("workspace");
        let active = manifest(directory.path());
        write_active(&active);
        let build = stage(&active);
        recover_build_with(active.layout(), false, &fail_cleanup).expect_err("cleanup fails");
        assert!(active.layout().has_build_record().expect("build record"));
        assert!(build.layout().staged_index_dir().join("new").exists());
        recover_build(active.layout()).expect("retry cleanup");
        recover_build(active.layout()).expect("idempotent");
        assert!(!active.layout().temp_dir().exists());
        assert!(active.layout().index_dir().join("old").exists());
        let fresh = build_target(&active);
        prepare_build(&fresh).expect("fresh attempt");
        assert_ne!(fresh.storage_generation(), build.storage_generation());
        assert!(!fresh.layout().staged_index_dir().join("new").exists());
        discard_build(&build).expect("stale completion does not discard fresh attempt");
        assert!(fresh.layout().staged_index_dir().exists());
        discard_build(&fresh).expect("discard fresh attempt");
    }

    #[test]
    fn every_interrupted_publication_step_restores_the_previous_index() {
        for had_index in [false, true] {
            for step in 0..=2 {
                let directory = tempdir().expect("workspace");
                let active = manifest(directory.path());
                if had_index {
                    write_active(&active);
                }
                let build = stage(&active);
                begin_publication(&build).expect("prepare commit");
                if step >= 1 && had_index {
                    move_index(
                        active.layout().index_dir(),
                        active.layout().backup_index_dir(),
                    )
                    .expect("backup");
                }
                if step >= 2 {
                    move_index(
                        build.layout().staged_index_dir(),
                        active.layout().index_dir(),
                    )
                    .expect("install new index");
                }
                assert!(needs_recovery(active.layout()).expect("publication pending"));
                recover_build(active.layout()).expect("rollback interrupted publication");
                recover_build(active.layout()).expect("rollback is idempotent");
                assert!(!active.layout().temp_dir().exists());
                assert_eq!(active.layout().index_dir().join("old").exists(), had_index);
                assert!(!active.layout().index_dir().join("new").exists());
                assert_eq!(
                    read_workspace_manifest(active.source_root(), active.layout())
                        .expect("manifest"),
                    had_index.then_some(active)
                );
            }
        }
    }

    #[tokio::test]
    async fn the_first_read_recovers_an_interrupted_swap_under_the_write_lock() {
        use crate::workspace::lock::{LockMode, LockWait};
        for had_index in [false, true] {
            let directory = tempdir().expect("workspace");
            let active = manifest(directory.path());
            if had_index {
                write_active(&active);
            }
            let build = stage(&active);
            begin_publication(&build).expect("commit marker");
            if had_index {
                move_index(
                    active.layout().index_dir(),
                    active.layout().backup_index_dir(),
                )
                .expect("backup");
            }
            move_index(
                build.layout().staged_index_dir(),
                active.layout().index_dir(),
            )
            .expect("install");
            let _reader = LockWait::new(None, None)
                .expect("wait")
                .acquire(
                    active.source_root(),
                    active.layout(),
                    LockMode::Shared,
                    "test.read",
                )
                .await
                .expect("recovered read");
            assert_eq!(active.layout().index_dir().join("old").exists(), had_index);
            assert!(!active.layout().index_dir().join("new").exists());
            assert!(!active.layout().has_build_record().expect("build record"));
            assert_eq!(
                read_workspace_manifest(active.source_root(), active.layout()).expect("manifest"),
                had_index.then_some(active)
            );
        }
    }

    #[test]
    fn invalid_current_metadata_requires_explicit_recovery_before_rebuilding() {
        for field in ["scan", "descriptor", "generation"] {
            let directory = tempdir().expect("workspace");
            let active = manifest(directory.path());
            write_active(&active);
            let mut value = serde_json::to_value(&active).expect("manifest JSON");
            match field {
                "scan" => value["scan"] = serde_json::json!({ "include_hidden": "invalid" }),
                "descriptor" => value["index"]["descriptor"] = serde_json::Value::Null,
                "generation" => {
                    value["storageGeneration"] = serde_json::json!("invalid-generation");
                }
                _ => unreachable!(),
            }
            let bytes = serde_json::to_vec(&value).expect("invalid manifest JSON");
            assert_automatic_recovery_preserves_invalid_manifest(&active, &bytes);

            assert!(
                inspect_workspace_manifest(active.source_root(), active.layout())
                    .expect("inspect")
                    .into_workspace(true)
                    .expect("explicit rebuild ignores invalid metadata")
                    .is_none()
            );
            recover_build_for_rebuild(active.layout()).expect("explicit rollback");
            assert!(active.layout().index_dir().join("old").exists());
            assert!(!active.layout().index_dir().join("new").exists());
            assert!(!active.layout().temp_dir().exists());
            assert_eq!(
                fs::read(active.layout().manifest_path()).expect("preserved invalid manifest"),
                bytes
            );

            let fresh = stage(&active);
            let generation = fresh.storage_generation().map(str::to_owned);
            publish_build(fresh, 3).expect("publish replacement");
            let rebuilt = read_workspace_manifest(active.source_root(), active.layout())
                .expect("read replacement")
                .expect("replacement manifest");
            assert_eq!(rebuilt.storage_generation(), generation.as_deref());
            assert!(active.layout().index_dir().join("new").exists());
            assert!(!active.layout().index_dir().join("old").exists());
            assert!(!active.layout().temp_dir().exists());
        }
    }

    #[test]
    fn recovery_can_itself_be_interrupted_after_restoring_the_old_directory() {
        let directory = tempdir().expect("workspace");
        let active = manifest(directory.path());
        write_active(&active);
        let build = stage(&active);
        begin_publication(&build).expect("commit marker");
        move_index(
            active.layout().index_dir(),
            active.layout().backup_index_dir(),
        )
        .expect("backup");
        move_index(
            build.layout().staged_index_dir(),
            active.layout().index_dir(),
        )
        .expect("install");
        IndexStore::delete(active.layout().index_dir()).expect("remove uncommitted index");
        move_index(
            active.layout().backup_index_dir(),
            active.layout().index_dir(),
        )
        .expect("restore");
        recover_build(active.layout()).expect("resume interrupted rollback");
        assert!(active.layout().index_dir().join("old").exists());
        assert!(!active.layout().temp_dir().exists());
    }

    #[test]
    fn publication_cleanup_failure_preserves_new_index_and_retries_after_metadata_update() {
        let directory = tempdir().expect("workspace");
        let active = manifest(directory.path());
        write_active(&active);
        let build = stage(&active);
        publish_build_with(build.clone(), 3, &fail_cleanup)
            .expect("committed despite cleanup failure");
        let mut committed = read_workspace_manifest(active.source_root(), active.layout())
            .expect("read")
            .expect("manifest");
        assert_eq!(committed.storage_generation(), build.storage_generation());
        assert_eq!(committed.updated_epoch_ms(), 3);
        assert!(committed.layout().index_dir().join("new").exists());
        assert!(!needs_recovery(active.layout()).expect("safe for readers"));
        assert!(active.layout().has_build_record().expect("build record"));
        committed.record_update(4);
        write_workspace_manifest(active.layout().manifest_path(), &committed).expect("rename");
        discard_build(&build).expect("never discard published index");
        recover_build(active.layout()).expect("finish cleanup");
        assert!(!active.layout().temp_dir().exists());
        assert!(committed.layout().index_dir().join("new").exists());
        assert!(!committed.layout().index_dir().join("old").exists());
        assert_eq!(
            read_workspace_manifest(active.source_root(), active.layout()).expect("read"),
            Some(committed)
        );
    }

    #[test]
    fn moving_workspace_keeps_recovery_relative_to_its_new_home() {
        let directory = tempdir().expect("workspace");
        let original = directory.path().join("original");
        fs::create_dir(&original).expect("root");
        let active = manifest(&original);
        write_active(&active);
        let build = stage(&active);
        begin_publication(&build).expect("commit marker");
        move_index(
            active.layout().index_dir(),
            active.layout().backup_index_dir(),
        )
        .expect("backup");
        let moved = directory.path().join("moved");
        fs::rename(&original, &moved).expect("move workspace");
        let layout = WorkspaceLayout::new(moved.join(".zvec-grep")).expect("layout");
        recover_build(&layout).expect("recover relocated index");
        assert!(layout.index_dir().join("old").exists());
        assert!(!layout.temp_dir().exists());
    }

    fn assert_automatic_recovery_preserves_invalid_manifest(active: &Workspace, bytes: &[u8]) {
        let layout = active.layout();
        // Match the build's previous-manifest hash, so only the manifest preflight
        // prevents recovery from deleting the new index and restoring the backup.
        fs::write(layout.manifest_path(), bytes).expect("invalid manifest");
        let build = stage(active);
        begin_publication(&build).expect("pending publication");
        move_index(layout.index_dir(), layout.backup_index_dir()).expect("backup");
        move_index(layout.staged_index_dir(), layout.index_dir()).expect("install");
        let record = fs::read(layout.build_record_path()).expect("build record");
        let staged = fs::read(layout.staged_manifest_path()).expect("staged manifest");
        let error = crate::workspace::lock::try_acquire_workspace_read(
            active.source_root(),
            layout,
            "test.recovery",
        )
        .expect_err("invalid manifest must not recover");
        assert_eq!(error.code(), EngineError::INDEX_INCOMPATIBLE);
        assert_eq!(fs::read(layout.manifest_path()).expect("manifest"), bytes);
        assert_eq!(
            fs::read(layout.build_record_path()).expect("build record"),
            record
        );
        assert_eq!(
            fs::read(layout.staged_manifest_path()).expect("staged manifest"),
            staged
        );
        assert!(layout.index_dir().join("new").exists());
        assert!(layout.backup_index_dir().join("old").exists());
    }

    #[test]
    fn automatic_recovery_requires_current_version_in_every_manifest_state() {
        let current = super::super::CURRENT_INDEX_VERSION;
        let version_fields = [
            String::new(),
            "\"indexVersion\":null,".to_owned(),
            "\"indexVersion\":\"invalid\",".to_owned(),
            "\"indexVersion\":true,".to_owned(),
            "\"indexVersion\":-1,".to_owned(),
            "\"indexVersion\":1.5,".to_owned(),
            format!("\"indexVersion\":{},", u64::from(u32::MAX) + 1),
            format!("\"indexVersion\":{},", current - 1),
            format!(
                "\"indexVersion\":{},",
                super::super::BINARY_DISKANN_INDEX_VERSION + 1
            ),
            format!("\"indexVersion\":{current},\"indexVersion\":{current},"),
            format!(
                "\"indexVersion\":{},\"indexVersion\":{current},",
                current - 1
            ),
            format!("\"indexVersion\":null,\"indexVersion\":{current},"),
        ];
        for state in ["enabled", "disabled", "uninitialized"] {
            for fields in &version_fields {
                let directory = tempdir().expect("workspace");
                let active = manifest(directory.path());
                write_active(&active);
                let mut value = serde_json::to_value(&active).expect("manifest JSON");
                value
                    .as_object_mut()
                    .expect("object")
                    .remove("indexVersion");
                if state != "enabled" {
                    value["index"] = serde_json::json!({ "state": state });
                    value["storageGeneration"] = serde_json::Value::Null;
                }
                let json = serde_json::to_string(&value).expect("manifest JSON");
                let bytes = format!("{{{fields}{}", &json[1..]);
                assert_automatic_recovery_preserves_invalid_manifest(&active, bytes.as_bytes());
            }
        }
    }

    #[test]
    fn unsupported_build_versions_never_discard_a_possible_backup() {
        for version in [
            super::super::CURRENT_INDEX_VERSION - 1,
            super::super::BINARY_DISKANN_INDEX_VERSION + 1,
        ] {
            let directory = tempdir().expect("workspace");
            let active = manifest(directory.path());
            write_active(&active);
            let build = stage(&active);
            begin_publication(&build).expect("pending publication");
            let layout = active.layout();
            move_index(layout.index_dir(), layout.backup_index_dir()).expect("backup");
            let mut record: serde_json::Value = serde_json::from_slice(
                &fs::read(layout.build_record_path()).expect("build record"),
            )
            .expect("record JSON");
            record["index_version"] = serde_json::json!(version);
            fs::write(layout.build_record_path(), record.to_string()).expect("unsupported version");
            assert!(recover_build(layout).is_err());
            assert!(recover_build_for_rebuild(layout).is_err());
            assert!(build.layout().staged_index_dir().join("new").exists());
            assert!(layout.backup_index_dir().join("old").exists());
            assert!(layout.has_build_record().expect("build record"));
        }
    }

    #[test]
    fn binary_diskann_build_record_version_is_supported() {
        let record = BuildRecord {
            index_version: super::super::BINARY_DISKANN_INDEX_VERSION,
            generation: uuid::Uuid::nil().to_string(),
            previous_manifest: None,
            had_index: false,
            publishing: false,
        };
        record.validate().expect("V3 build record is supported");
    }

    #[test]
    fn a_matching_generation_with_a_different_manifest_version_is_not_committed() {
        let directory = tempdir().expect("workspace");
        let base = manifest(directory.path());
        let generation = Uuid::new_v4().to_string();
        let active = Workspace::new(
            base.source_root(),
            base.layout().clone(),
            base.scan().clone(),
            super::super::CURRENT_INDEX_VERSION,
            Some(generation.clone()),
            base.created_epoch_ms(),
            base.updated_epoch_ms(),
            base.index().clone(),
        )
        .expect("version two workspace");
        write_active(&active);
        fs::create_dir_all(active.layout().backup_index_dir()).expect("backup directory");
        fs::write(
            active.layout().backup_index_dir().join("sentinel"),
            b"prior generation",
        )
        .expect("backup sentinel");
        let record = BuildRecord {
            index_version: super::super::BINARY_DISKANN_INDEX_VERSION,
            generation,
            previous_manifest: Some(sha256_hex(b"different prior manifest")),
            had_index: true,
            publishing: true,
        };
        write_record(active.layout(), &record).expect("build record");

        assert!(needs_recovery(active.layout()).is_err());
        assert!(recover_build(active.layout()).is_err());
        assert_eq!(
            fs::read(active.layout().backup_index_dir().join("sentinel"))
                .expect("preserved backup"),
            b"prior generation",
        );
        assert!(active.layout().index_dir().join("old").is_file());
    }

    #[test]
    fn duplicate_manifest_version_keys_do_not_discard_the_backup() {
        let directory = tempdir().expect("workspace");
        let active = manifest(directory.path());
        write_active(&active);
        let build = stage(&active);
        begin_publication(&build).expect("begin publication");
        move_index(
            active.layout().index_dir(),
            active.layout().backup_index_dir(),
        )
        .expect("move old index to backup");
        let staged = fs::read(active.layout().staged_manifest_path()).expect("staged manifest");
        let staged = String::from_utf8(staged).expect("UTF-8 manifest");
        let duplicate = staged.replacen(
            "\"indexVersion\": 2,",
            "\"indexVersion\": 2,\"indexVersion\": 2,",
            1,
        );
        assert_ne!(duplicate, staged, "fixture must contain the version key");
        fs::write(active.layout().manifest_path(), duplicate).expect("duplicate-key manifest");

        assert!(needs_recovery(active.layout()).is_err());
        assert!(recover_build(active.layout()).is_err());
        assert!(active.layout().backup_index_dir().join("old").is_file());
        assert!(!active.layout().index_dir().exists());
    }

    #[test]
    fn duplicate_storage_format_keys_do_not_discard_the_backup() {
        let directory = tempdir().expect("workspace");
        let active = manifest(directory.path());
        write_active(&active);
        let target = binary_diskann_workspace(&active);
        prepare_build(&target).expect("stage version three publication");
        fs::write(target.layout().staged_index_dir().join("new"), "new format")
            .expect("staged marker");
        begin_publication(&target).expect("begin publication");
        move_index(
            active.layout().index_dir(),
            active.layout().backup_index_dir(),
        )
        .expect("move old index to backup");
        let staged = fs::read(active.layout().staged_manifest_path()).expect("staged manifest");
        let staged = String::from_utf8(staged).expect("UTF-8 manifest");
        let duplicate = staged.replacen(
            "\"storageFormat\": \"binary_sign_disk_ann_v1\",",
            "\"storageFormat\": \"binary_sign_disk_ann_v1\",\"storageFormat\": \"binary_sign_disk_ann_v1\",",
            1,
        );
        assert_ne!(
            duplicate, staged,
            "fixture must contain the storageFormat key"
        );
        fs::write(active.layout().manifest_path(), duplicate).expect("duplicate-key manifest");

        assert!(needs_recovery(active.layout()).is_err());
        assert!(recover_build(active.layout()).is_err());
        assert!(active.layout().backup_index_dir().join("old").is_file());
        assert!(!active.layout().index_dir().exists());
    }

    #[test]
    fn version_three_publication_recovers_the_previous_index() {
        let directory = tempdir().expect("workspace");
        let base = manifest(directory.path());
        let active = binary_diskann_workspace(&base);
        write_active(&active);

        let build = stage(&active);
        let record = read_record(active.layout())
            .expect("read build record")
            .expect("pending build record");
        assert_eq!(
            record.index_version,
            super::super::BINARY_DISKANN_INDEX_VERSION
        );
        begin_publication(&build).expect("begin publication");
        move_index(
            active.layout().index_dir(),
            active.layout().backup_index_dir(),
        )
        .expect("move old index to backup");

        assert!(needs_recovery(active.layout()).expect("recovery required"));
        recover_build(active.layout()).expect("restore prior version three index");
        assert!(active.layout().index_dir().join("old").is_file());
        assert!(!active.layout().backup_index_dir().exists());
        assert!(!active.layout().staged_index_dir().exists());
        assert_eq!(
            read_workspace_manifest(active.source_root(), active.layout()).expect("manifest"),
            Some(active),
        );
    }

    #[test]
    fn interrupted_version_two_to_three_publication_restores_version_two() {
        let directory = tempdir().expect("workspace");
        let active = manifest(directory.path());
        write_active(&active);
        let target = binary_diskann_workspace(&active);
        prepare_build(&target).expect("stage version three publication");
        fs::write(target.layout().staged_index_dir().join("new"), "new format")
            .expect("staged marker");
        begin_publication(&target).expect("begin publication");
        move_index(
            active.layout().index_dir(),
            active.layout().backup_index_dir(),
        )
        .expect("move version two index to backup");

        assert!(needs_recovery(active.layout()).expect("recovery required"));
        recover_build(active.layout()).expect("roll back interrupted format change");
        assert!(active.layout().index_dir().join("old").is_file());
        assert!(!active.layout().backup_index_dir().exists());
        assert!(!active.layout().staged_index_dir().exists());
        assert_eq!(
            read_workspace_manifest(active.source_root(), active.layout()).expect("manifest"),
            Some(active),
        );
    }

    #[test]
    fn committed_outer_v3_with_mismatched_inner_schema_keeps_rollback_backup() {
        let directory = tempdir().expect("workspace");
        let active = manifest(directory.path());
        write_active(&active);
        let target = binary_diskann_workspace(&active);
        prepare_build(&target).expect("stage version three publication");
        fs::write(target.layout().staged_index_dir().join("new"), "new format")
            .expect("staged marker");
        begin_publication(&target).expect("begin publication");
        move_index(
            active.layout().index_dir(),
            active.layout().backup_index_dir(),
        )
        .expect("move old index to backup");
        move_index(
            target.layout().staged_index_dir(),
            target.layout().index_dir(),
        )
        .expect("install new storage");
        write_workspace_manifest(target.layout().manifest_path(), &target)
            .expect("commit outer V3 manifest");

        let fp16 = active
            .index()
            .descriptor()
            .expect("descriptor")
            .default_embedding_model();
        let schema = serde_json::json!({
            "path_encoding": "components",
            "binary_sign_storage": false,
            "embeddings": [fp16],
        });
        fs::write(
            target.layout().index_dir().join("schema.json"),
            serde_json::to_vec(&schema).expect("inner schema JSON"),
        )
        .expect("mismatched inner schema");

        assert!(recover_build(target.layout()).is_err());
        assert!(target.layout().backup_index_dir().join("old").is_file());
        assert!(target.layout().index_dir().join("schema.json").is_file());
    }

    #[test]
    fn shared_read_rejects_v1_before_recovering_pending_publication() {
        let directory = tempdir().expect("workspace");
        let base = manifest(directory.path());
        let active = binary_diskann_workspace(&base);
        write_active(&active);

        let build = stage(&active);
        begin_publication(&build).expect("begin publication");
        move_index(
            active.layout().index_dir(),
            active.layout().backup_index_dir(),
        )
        .expect("move old index to backup");

        let error = crate::workspace::lock::try_acquire_workspace_read(
            active.source_root(),
            active.layout(),
            "binary-diskann-v1-test",
        )
        .expect_err("unsupported V1 must not enter recovery");
        assert_eq!(error.code(), EngineError::UNSUPPORTED);
        assert!(active.layout().backup_index_dir().join("old").is_file());
        assert!(build.layout().staged_index_dir().join("new").is_file());
        assert!(!active.layout().index_dir().exists());
    }

    #[test]
    fn version_three_publish_rejects_an_inner_schema_mismatch_before_commit() {
        let directory = tempdir().expect("workspace");
        let active = manifest(directory.path());
        write_active(&active);
        let original_manifest = fs::read(active.layout().manifest_path()).expect("old manifest");
        let target = binary_diskann_workspace(&active);
        prepare_build(&target).expect("stage version three publication");
        let fp16 = active
            .index()
            .descriptor()
            .expect("descriptor")
            .default_embedding_model();
        let schema = serde_json::json!({
            "path_encoding": "components",
            "binary_sign_storage": false,
            "embeddings": [fp16],
        });
        fs::write(
            target.layout().staged_index_dir().join("schema.json"),
            serde_json::to_vec(&schema).expect("inner schema JSON"),
        )
        .expect("mismatched staged schema");

        assert!(publish_build(target.clone(), 3).is_err());
        assert_eq!(
            fs::read(active.layout().manifest_path()).expect("unchanged manifest"),
            original_manifest,
        );
        assert!(active.layout().index_dir().join("old").is_file());
        assert!(!active.layout().backup_index_dir().exists());
        assert!(!active.layout().staged_index_dir().exists());
    }

    #[test]
    fn binary_diskann_bundle_validation_checks_identity_digest_and_bounds() {
        let directory = tempdir().expect("temporary bundle");
        let root = directory.path();
        let generation = Uuid::new_v4().to_string();
        let mut identity = b"ZGIDV1\0\0".to_vec();
        identity.extend_from_slice(Uuid::parse_str(&generation).expect("generation").as_bytes());
        identity.extend_from_slice(&2_u32.to_le_bytes());
        identity.extend_from_slice(&1_u64.to_le_bytes());
        identity.extend_from_slice(&1_u64.to_le_bytes());
        identity.extend_from_slice(&5_u64.to_le_bytes());
        identity.extend_from_slice(b"doc-1");
        let identity_digest = crate::utils::sha256_parts([identity.as_slice()]);
        let mut trust = Vec::with_capacity(300);
        trust.extend_from_slice(b"ZGV1TRST");
        trust.extend_from_slice(&1_u32.to_le_bytes());
        trust.extend_from_slice(&2_u32.to_le_bytes());
        trust.extend_from_slice(&1_u32.to_le_bytes());
        trust.extend_from_slice(&1_u64.to_le_bytes());
        trust.extend_from_slice(Uuid::parse_str(&generation).expect("generation").as_bytes());
        for digest in 1_u8..=8 {
            trust.extend_from_slice(&[digest; 32]);
        }
        let metadata = BinaryDiskAnnManifest::new(
            generation,
            "binary-diskann/graph".to_owned(),
            "binary-diskann/identity.bin".to_owned(),
            2,
            1,
            identity_digest,
            trust,
        )
        .expect("metadata");
        let graph = root.join(metadata.graph_path());
        fs::create_dir_all(&graph).expect("graph directory");
        fs::write(graph.join("graph.zvec"), b"candidate graph").expect("graph bytes");
        let identity_path = root.join(metadata.identity_path());
        fs::create_dir_all(identity_path.parent().expect("identity parent"))
            .expect("identity directory");
        fs::write(&identity_path, &identity).expect("identity bytes");

        validate_binary_diskann_bundle(root, &metadata).expect("valid bundle");
        fs::write(&identity_path, b"tampered").expect("tamper identity");
        assert!(validate_binary_diskann_bundle(root, &metadata).is_err());
    }

    #[test]
    fn binary_diskann_staged_generation_commits_pinned_sidecar_metadata() {
        let directory = tempdir().expect("workspace");
        let active = manifest(directory.path());
        write_active(&active);
        let build = binary_diskann_workspace(&active);
        prepare_build(&build).expect("stage V1 generation");
        write_test_schema(&build, build.layout().staged_index_dir());
        write_binary_diskann_bundle(&build, build.layout().staged_index_dir());
        fs::write(build.layout().staged_index_dir().join("new"), "new rows").expect("staged rows");
        let metadata = build
            .binary_diskann_manifest()
            .expect("outer trust metadata")
            .clone();

        publish_build(build.clone(), 3).expect("publish validated sidecar generation");
        let committed = read_workspace_manifest(directory.path(), build.layout())
            .expect("read committed manifest")
            .expect("manifest exists");
        assert_eq!(committed.storage_generation(), build.storage_generation());
        assert_eq!(committed.binary_diskann_manifest(), Some(&metadata));
        assert!(
            build
                .layout()
                .index_dir()
                .join(metadata.graph_path())
                .is_dir()
        );
        assert!(
            build
                .layout()
                .index_dir()
                .join(metadata.identity_path())
                .is_file()
        );
        assert!(!build.layout().backup_index_dir().exists());
    }

    #[test]
    fn malformed_original_manifest_is_preserved_until_successful_publication() {
        let directory = tempdir().expect("workspace");
        let active = manifest(directory.path());
        write_active(&active);
        fs::write(active.layout().manifest_path(), b"not JSON").expect("damaged manifest");
        let build = stage(&active);
        begin_publication(&build).expect("commit marker");
        move_index(
            active.layout().index_dir(),
            active.layout().backup_index_dir(),
        )
        .expect("backup");
        move_index(
            build.layout().staged_index_dir(),
            active.layout().index_dir(),
        )
        .expect("install");
        recover_build_for_rebuild(active.layout())
            .expect("explicit rebuild rolls back replacement");
        assert!(active.layout().index_dir().join("old").exists());
        assert_eq!(
            fs::read(active.layout().manifest_path()).expect("manifest"),
            b"not JSON"
        );
        let fresh = stage(&active);
        publish_build(fresh, 3).expect("replace corrupted index");
        assert!(active.layout().index_dir().join("new").exists());
        assert!(!active.layout().temp_dir().exists());
    }

    #[test]
    fn unexpected_manifest_change_prevents_a_destructive_rollback() {
        let directory = tempdir().expect("workspace");
        let active = manifest(directory.path());
        write_active(&active);
        let build = stage(&active);
        begin_publication(&build).expect("commit marker");
        move_index(
            active.layout().index_dir(),
            active.layout().backup_index_dir(),
        )
        .expect("backup");
        move_index(
            build.layout().staged_index_dir(),
            active.layout().index_dir(),
        )
        .expect("install");
        fs::write(active.layout().manifest_path(), b"unexpected modification").expect("modify");
        recover_build(active.layout()).expect_err("cannot determine commit safely");
        assert!(active.layout().index_dir().join("new").exists());
        assert!(active.layout().backup_index_dir().join("old").exists());
        assert!(active.layout().has_build_record().expect("build record"));
    }

    #[test]
    fn drop_removes_current_and_temporary_indexes_but_keeps_unrelated_files() {
        let directory = tempdir().expect("workspace");
        let active = manifest(directory.path());
        write_active(&active);
        let build = stage(&active);
        begin_publication(&build).expect("marker");
        move_index(
            active.layout().index_dir(),
            active.layout().backup_index_dir(),
        )
        .expect("backup");
        fs::write(active.layout().authorization_path(), b"keep").expect("unrelated data");
        assert!(directory_exists(active.layout().backup_index_dir()).expect("backup index"));
        drop_index_storage(active.layout()).expect("drop");
        drop_index_storage(active.layout()).expect("idempotent drop");
        assert!(!directory_exists(active.layout().index_dir()).expect("index"));
        assert!(!active.layout().temp_dir().exists());
        assert!(active.layout().authorization_path().exists());
    }

    fn with_api_key(workspace: &Workspace, api_key: &str) -> Workspace {
        let index = workspace.index().descriptor().expect("index");
        let mut configs = index.embedding_configs().clone();
        configs.insert(
            "local/example".into(),
            crate::domain::model::ModelConfig {
                api_key: Some(
                    crate::workspace::credentials::encrypt_api_key(
                        workspace.layout(),
                        "local/example",
                        api_key,
                    )
                    .expect("encrypted key"),
                ),
                endpoint: None,
                ..Default::default()
            },
        );
        let index = crate::domain::IndexDescriptor::new(
            index.default_embedding_model().clone(),
            index.embedding_routes().clone(),
            configs,
            index.graph_indexing(),
            index.fts(),
        )
        .expect("descriptor");
        crate::domain::Workspace::new(
            workspace.source_root(),
            workspace.layout().clone(),
            workspace.scan().clone(),
            workspace.index_version(),
            workspace.storage_generation().map(str::to_owned),
            workspace.created_epoch_ms(),
            workspace.updated_epoch_ms(),
            crate::domain::IndexState::Enabled(Box::new(index)),
        )
        .expect("workspace")
    }

    #[test]
    fn a_failed_publication_preserves_saved_credentials() {
        let directory = tempdir().expect("workspace");
        let mut active = manifest(directory.path());
        write_active(&active);
        active = with_api_key(&active, "original-key");
        write_workspace_manifest(active.layout().manifest_path(), &active)
            .expect("original manifest");
        let key = fs::read(active.layout().credentials_key_path()).expect("workspace key");
        let target = with_api_key(&active, "replacement-key");
        let build = stage(&target);
        begin_publication(&build).expect("publication");
        move_index(
            active.layout().index_dir(),
            active.layout().backup_index_dir(),
        )
        .expect("backup");
        move_index(
            build.layout().staged_index_dir(),
            active.layout().index_dir(),
        )
        .expect("install");
        discard_build(&build).expect("rollback");
        let restored = read_workspace_manifest(active.source_root(), active.layout())
            .expect("read")
            .expect("manifest");
        assert_eq!(restored, active);
        assert_eq!(
            restored
                .index()
                .descriptor()
                .expect("indexed workspace")
                .embedding_configs()["local/example"]
                .api_key
                .as_ref()
                .map(|key| crate::workspace::credentials::decrypt_api_key(
                    key,
                    active.layout(),
                    "local/example"
                ))
                .transpose()
                .expect("decrypt"),
            Some("original-key".into())
        );
        assert_eq!(
            fs::read(active.layout().credentials_key_path()).expect("preserved key"),
            key
        );
        let build = stage(&target);
        publish_build(build, 3).expect("publish replacement");
        let replaced = read_workspace_manifest(active.source_root(), active.layout())
            .expect("read")
            .expect("manifest");
        assert_eq!(
            replaced
                .index()
                .descriptor()
                .expect("indexed workspace")
                .embedding_configs()["local/example"]
                .api_key
                .as_ref()
                .map(|key| crate::workspace::credentials::decrypt_api_key(
                    key,
                    active.layout(),
                    "local/example"
                ))
                .transpose()
                .expect("decrypt"),
            Some("replacement-key".into())
        );
    }

    #[test]
    fn corrupt_build_metadata_never_discards_a_possible_backup() {
        let directory = tempdir().expect("workspace");
        let active = manifest(directory.path());
        write_active(&active);
        let build = stage(&active);
        begin_publication(&build).expect("publication");
        move_index(
            active.layout().index_dir(),
            active.layout().backup_index_dir(),
        )
        .expect("backup");
        fs::write(active.layout().build_record_path(), b"invalid").expect("corrupt record");
        assert!(recover_build_for_rebuild(active.layout()).is_err());
        assert!(active.layout().backup_index_dir().join("old").is_file());
        assert!(build.layout().staged_index_dir().join("new").is_file());
    }

    #[test]
    fn orphaned_stage_is_discarded_but_an_unowned_backup_is_preserved() {
        let directory = tempdir().expect("workspace");
        let active = manifest(directory.path());
        write_active(&active);
        let orphan = active.layout().staged_index_dir();
        fs::create_dir_all(orphan).expect("orphaned stage");
        fs::write(orphan.join("stale"), "stale data").expect("stale data");
        assert!(prepare_build(&active).is_err());
        recover_build(active.layout()).expect("discard orphaned stage");
        assert!(!orphan.exists());
        assert!(active.layout().index_dir().join("old").is_file());
        fs::create_dir_all(active.layout().backup_index_dir()).expect("unowned backup");
        assert!(recover_build_for_rebuild(active.layout()).is_err());
        assert!(active.layout().backup_index_dir().exists());
    }

    // Characterization only: this demonstrates why an inner storage schema
    // cannot protect a future format against pre-open recovery. It does not
    // provide a compatibility gate or test an older installed binary.
    #[test]
    fn orphan_recovery_attempts_cleanup_without_reading_inner_schema() {
        use std::cell::Cell;

        let directory = tempdir().expect("workspace");
        let active = manifest(directory.path());
        write_active(&active);
        let orphan = active.layout().staged_index_dir();
        fs::create_dir_all(orphan).expect("orphaned stage");
        let schema = orphan.join("schema.json");
        let future_schema = br#"{"binary_sign_diskann_storage_version":1}"#;
        fs::write(&schema, future_schema).expect("future inner schema");
        let attempted = Cell::new(false);
        let intercept = |path: &Path| {
            assert_eq!(path, orphan);
            attempted.set(true);
            Err(EngineError::storage_failure("intercepted fixture cleanup"))
        };

        assert!(recover_build_with(active.layout(), false, &intercept).is_err());
        assert!(attempted.get(), "recovery must reach the cleanup callback");
        assert_eq!(
            fs::read(schema).expect("fixture schema preserved"),
            future_schema
        );
        assert!(active.layout().index_dir().join("old").is_file());
    }
}
