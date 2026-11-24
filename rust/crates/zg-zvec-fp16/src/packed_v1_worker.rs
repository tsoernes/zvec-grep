//! Thread-owned bridge to the immutable packed-DiskANN generation API.
//!
//! Keep the native handle on one owner thread and communicate using owned
//! query/result messages. Never add unsafe auto-trait implementations to it.

use std::{
    path::{Path, PathBuf},
    sync::{
        Mutex,
        mpsc::{self, Receiver, SyncSender},
    },
    thread::{self, JoinHandle},
};

use crate::{
    packed_diskann_generation::{self, GenerationError, Limits, Trust},
    packed_sign_index::{GenerationIdentity, IndexError, Metric},
};

/// A result with owned document identity, safe to return across threads.
#[derive(Clone, Debug, PartialEq)]
pub struct Hit {
    pub id: String,
    pub key: u64,
    pub score: f64,
}

/// Errors while starting or communicating with the dedicated native owner.
#[derive(Debug)]
pub enum WorkerError {
    Native(GenerationError),
    Identity(IndexError),
    ThreadStart(std::io::Error),
    WorkerStopped,
}

impl From<GenerationError> for WorkerError {
    fn from(error: GenerationError) -> Self {
        Self::Native(error)
    }
}

impl From<IndexError> for WorkerError {
    fn from(error: IndexError) -> Self {
        Self::Identity(error)
    }
}

enum Request {
    Search {
        query: Vec<f32>,
        limit: usize,
        beam: usize,
        reply: SyncSender<Result<Vec<Hit>, GenerationError>>,
    },
    Shutdown,
}

struct OwnerContext {
    workspace: PathBuf,
    trust: Trust,
    keys: Vec<u64>,
    identity: GenerationIdentity,
    metric: Metric,
    limits: Limits,
}

/// A synchronous, thread-safe request handle for one immutable graph generation.
///
/// Queries are serialized on the native owner thread. This keeps the current
/// `!Send` native handle sound while providing a shareable handle for ZG's
/// async storage state. The C ABI forbids racing destroy with any call on the
/// same handle; the owner thread enforces that lifetime/concurrency contract.
pub struct BinaryDiskAnnWorker {
    requests: SyncSender<Request>,
    owner: Mutex<Option<JoinHandle<()>>>,
    document_count: usize,
}

impl BinaryDiskAnnWorker {
    /// Open an immutable persisted graph using caller-pinned outer-manifest
    /// trust and a separately pinned identity-sidecar digest.
    ///
    /// # Errors
    /// Returns an error if identity or trust validation fails, resources are
    /// unavailable, the owner thread cannot start, or the native graph cannot
    /// be opened.
    pub fn open(
        workspace: &Path,
        trust: Trust,
        identity_bytes: &[u8],
        expected_identity_sha256: [u8; 32],
        metric: Metric,
        limits: Limits,
    ) -> Result<Self, WorkerError> {
        let actual_identity_sha256 =
            packed_diskann_generation::identity_sha256(identity_bytes, limits.max_input_bytes)?;
        if actual_identity_sha256 != expected_identity_sha256 {
            return Err(GenerationError::Trust.into());
        }
        let identity = GenerationIdentity::from_bytes(
            identity_bytes,
            limits.max_input_bytes,
            limits.max_documents,
        )?;
        let document_count = identity.len();
        trust.validate_generation(
            identity.dimension(),
            metric,
            identity.len(),
            identity.generation(),
        )?;
        let keys = identity.keys(limits.max_input_bytes)?;
        let workspace = workspace.to_path_buf();
        let (requests, receiver) = mpsc::sync_channel(16);
        let (ready_sender, ready_receiver) = mpsc::sync_channel(1);
        let context = OwnerContext {
            workspace,
            trust,
            keys,
            identity,
            metric,
            limits,
        };
        let owner = thread::Builder::new()
            .name("zg-binary-diskann-owner".to_owned())
            .spawn(move || {
                Self::run_owner(receiver, ready_sender, context);
            })
            .map_err(WorkerError::ThreadStart)?;

        match ready_receiver.recv() {
            Ok(Ok(())) => Ok(Self {
                requests,
                owner: Mutex::new(Some(owner)),
                document_count,
            }),
            Ok(Err(error)) => {
                let _ = owner.join();
                Err(WorkerError::Native(error))
            }
            Err(_) => {
                let _ = owner.join();
                Err(WorkerError::WorkerStopped)
            }
        }
    }

    /// Search without a filter. Filtered retrieval must use an independently
    /// authorized fallback until native filtered ANN is integrated.
    ///
    /// # Errors
    /// Returns an error if the worker has stopped or the native search fails.
    pub fn search(
        &self,
        query: &[f32],
        limit: usize,
        beam: usize,
    ) -> Result<Vec<Hit>, WorkerError> {
        let limit = limit.min(self.document_count);
        if limit == 0 {
            return Ok(Vec::new());
        }
        let (reply, result) = mpsc::sync_channel(1);
        self.requests
            .send(Request::Search {
                query: query.to_vec(),
                limit,
                beam,
                reply,
            })
            .map_err(|_| WorkerError::WorkerStopped)?;
        result
            .recv()
            .map_err(|_| WorkerError::WorkerStopped)?
            .map_err(WorkerError::Native)
    }

    pub fn document_count(&self) -> usize {
        self.document_count
    }

    // Both endpoints and the full context intentionally move to the owner
    // thread, which must own their lifetimes for its entire request loop.
    #[allow(clippy::needless_pass_by_value)]
    fn run_owner(
        requests: Receiver<Request>,
        ready: SyncSender<Result<(), GenerationError>>,
        context: OwnerContext,
    ) {
        let OwnerContext {
            workspace,
            trust,
            keys,
            identity,
            metric,
            limits,
        } = context;
        let index = match packed_diskann_generation::open_v1(&workspace, trust, &keys, limits) {
            Ok(index) => index,
            Err(error) => {
                let _ = ready.send(Err(error));
                return;
            }
        };
        if ready.send(Ok(())).is_err() {
            return;
        }
        while let Ok(request) = requests.recv() {
            match request {
                Request::Search {
                    query,
                    limit,
                    beam,
                    reply,
                } => {
                    let result =
                        index
                            .search_v1(&query, metric, limit, beam)
                            .and_then(|candidates| {
                                let mut hits = candidates
                                    .into_iter()
                                    .map(|candidate| {
                                        identity
                                            .id_for_key(candidate.key)
                                            .map(|id| Hit {
                                                id: id.to_owned(),
                                                key: candidate.key,
                                                score: candidate.score,
                                            })
                                            .map_err(GenerationError::from)
                                    })
                                    .collect::<Result<Vec<_>, _>>()?;
                                hits.sort_unstable_by(|left, right| {
                                    right
                                        .score
                                        .total_cmp(&left.score)
                                        .then_with(|| left.id.cmp(&right.id))
                                });
                                Ok(hits)
                            });
                    let _ = reply.send(result);
                }
                Request::Shutdown => break,
            }
        }
        // Make the drop order explicit: the native handle is destroyed on the
        // same thread on which it was opened and queried.
        drop(index);
    }
}

impl Drop for BinaryDiskAnnWorker {
    fn drop(&mut self) {
        let _ = self.requests.send(Request::Shutdown);
        let owner = self
            .owner
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(owner) = owner {
            let _ = owner.join();
        }
    }
}

#[cfg(all(
    test,
    feature = "experimental-sign-scale-v1",
    target_os = "linux",
    target_arch = "x86_64"
))]
mod tests {
    use super::*;
    use crate::{
        packed_diskann_generation,
        packed_sign_index::{Document, Generation},
    };
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn owner_thread_reopens_pinned_graph_and_serializes_queries() {
        crate::tests::initialize();
        let limits = Limits {
            max_documents: 10,
            max_input_bytes: 1024 * 1024,
            max_build_bytes: 16 * 1024 * 1024,
            max_codebook_bytes: 1024 * 1024,
            max_generation_bytes: 8 * 1024 * 1024,
            max_query_scratch_bytes: 1024 * 1024,
            max_beam: 16,
            max_degree: 4,
            build_list_size: 16,
            build_threads: 2,
        };
        let documents = [
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
        ];
        let generation = Generation::prepare(2, [0x42; 16], &documents, limits.max_build_bytes)
            .expect("generation");
        let identity_bytes = generation
            .identity_bytes(limits.max_input_bytes)
            .expect("identity");
        let identity_sha256 =
            packed_diskann_generation::identity_sha256(&identity_bytes, limits.max_input_bytes)
                .expect("identity digest");
        let identity_record_sha256 = packed_diskann_generation::identity_record_sha256(
            &identity_bytes,
            generation.records(),
            limits.max_generation_bytes,
        )
        .expect("row-bound identity digest");
        let parent = tempfile::Builder::new()
            .permissions(std::fs::Permissions::from_mode(0o700))
            .tempdir()
            .expect("private graph parent");
        let workspace = parent.path().join("graph");
        let keys = generation.keys(limits.max_input_bytes).expect("keys");
        let trust =
            packed_diskann_generation::create_v1(&packed_diskann_generation::CreateParams {
                staging_directory: &workspace,
                dimension: generation.dimension(),
                metric: Metric::Cosine,
                keys: &keys,
                records: generation.records(),
                generation: generation.generation(),
                identity_digest: identity_record_sha256,
                limits,
            })
            .expect("build graph");
        let trust = Trust::from_bytes(&trust.to_bytes()).expect("pin trust bundle");
        drop(generation);

        let worker = BinaryDiskAnnWorker::open(
            &workspace,
            trust,
            &identity_bytes,
            identity_sha256,
            Metric::Cosine,
            limits,
        )
        .expect("open graph on owner thread");
        let first = worker.search(&[10.0, 1.0], 1, 2).expect("first search");
        let second = worker.search(&[10.0, 1.0], 2, 2).expect("second search");
        let oversized_top_k = worker
            .search(&[10.0, 1.0], 10, 10)
            .expect("top-k clamps to document count");
        assert_eq!(first[0].id, "heavy");
        assert_eq!(second[0].id, "heavy");
        assert_eq!(second[1].id, "light");
        assert_eq!(oversized_top_k.len(), documents.len());
    }
}
