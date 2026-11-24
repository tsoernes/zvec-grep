//! Semantic baseline for a future native binary DiskANN implementation.
//!
//! These tests currently exercise the existing exhaustive binary-sign backend.
//! Passing them does not establish that a native DiskANN path exists or ran.

use super::*;

fn binary_model(metric: EmbeddingMetric) -> EmbeddingModelInfo {
    EmbeddingModelInfo::new(
        crate::domain::model::ModelInfo::new(
            "fixture",
            "fixture",
            [
                crate::domain::ContentKind::Text,
                crate::domain::ContentKind::Code,
            ],
            32,
        )
        .expect("model"),
        64,
        metric,
        None,
        None,
    )
    .expect("embedding model")
    .with_vector_index_quantization(VectorIndexQuantization::BinarySign)
}

// All fixture document components have equal magnitude, so the exact original
// vector equals the binary-sign reconstruction. This reference does not call
// the implementation's encoder or scorer.
fn exact_score(query: &[f32], document: &[f32], metric: EmbeddingMetric) -> f64 {
    let dot = query
        .iter()
        .zip(document)
        .map(|(&q, &d)| f64::from(q) * f64::from(d))
        .sum::<f64>();
    let query_norm_sq = query.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>();
    let doc_norm_sq = document.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>();
    match metric {
        EmbeddingMetric::Cosine if doc_norm_sq == 0.0 => 0.0,
        EmbeddingMetric::Cosine => dot / (query_norm_sq.sqrt() * doc_norm_sq.sqrt()),
        EmbeddingMetric::DotProduct => dot,
        EmbeddingMetric::Euclidean => -(query_norm_sq + doc_norm_sq - 2.0 * dot).max(0.0).sqrt(),
    }
}

#[test]
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn float_query_scoring_respects_weights_scales_filters_and_reopen() {
    super::super::super::zvec::initialize().expect("initialize zvec");
    let mut query = vec![0.1_f32; 64];
    query[0] = 10.0;
    query[1] = 1.0;

    // Both signs are Hamming distance one from the query signs, but negating
    // dimension zero loses much more weighted dot than negating dimension one.
    let mut matches_heavy_dimension = vec![1.0_f32; 64];
    matches_heavy_dimension[1] = -1.0;
    let mut misses_heavy_dimension = vec![1.0_f32; 64];
    misses_heavy_dimension[0] = -1.0;
    let hamming = |doc: &[f32]| {
        query
            .iter()
            .zip(doc)
            .filter(|&(&q, &d)| (q >= 0.0) != (d >= 0.0))
            .count()
    };
    assert_eq!(
        hamming(&matches_heavy_dimension),
        hamming(&misses_heavy_dimension)
    );
    assert!(
        exact_score(&query, &matches_heavy_dimension, EmbeddingMetric::Cosine)
            > exact_score(&query, &misses_heavy_dimension, EmbeddingMetric::Cosine)
    );

    for metric in [
        EmbeddingMetric::Cosine,
        EmbeddingMetric::DotProduct,
        EmbeddingMetric::Euclidean,
    ] {
        let home = tempfile::tempdir().expect("disposable local index");
        let model = binary_model(metric);
        let store = Fragments::open(home.path(), std::slice::from_ref(&model), false)
            .expect("create binary-sign fixture");
        let mut expected = Vec::new();
        for (file_id, vectors) in [
            (
                1,
                vec![
                    matches_heavy_dimension.clone(),
                    misses_heavy_dimension.clone(),
                ],
            ),
            (
                2,
                vec![
                    matches_heavy_dimension.iter().map(|&x| 2.0 * x).collect(),
                    vec![0.0; 64],
                ],
            ),
        ] {
            let (source, entities, mut entries) =
                metadata_fragments(file_id, "binary-diskann-fixture");
            for (entry, document) in entries.iter_mut().zip(vectors) {
                entry.vector = document;
                expected.push((
                    source.id(),
                    entry.fragment_id.as_str().to_owned(),
                    exact_score(&query, &entry.vector, metric),
                ));
            }
            write_fixture(&store, &source, &entities, &entries);
        }
        expected.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.1.cmp(&b.1)));
        store.flush().expect("flush fixture");
        let verify = |store: &Fragments| {
            for selected_file in [None, Some(FileId::new(1)), Some(FileId::new(2))] {
                let filter = StorageSearchFilter {
                    file_ids: selected_file.map(|id| vec![id]),
                    ..Default::default()
                };
                let sql = build_filter(Some(&filter), &DIRECTORY_LOOKUP).expect("filter SQL");
                let results = store
                    .search_vector("fixture/fixture", &query, 4, sql.as_deref())
                    .expect("float query against stored signs");
                let reference: Vec<_> = expected
                    .iter()
                    .filter(|(file, _, _)| selected_file.is_none_or(|selected| *file == selected))
                    .collect();
                assert_eq!(
                    results.len(),
                    reference.len(),
                    "metric {metric:?}, filter {selected_file:?}"
                );
                for (actual, (_, id, score)) in results.iter().zip(reference) {
                    assert_eq!(
                        &actual.document_id, id,
                        "metric {metric:?}, filter {selected_file:?}"
                    );
                    assert!(
                        (actual.score - score).abs() < 1e-10,
                        "metric {metric:?}: {} != {score}",
                        actual.score
                    );
                    assert_eq!(actual.path, StorageSearchPath::Vector);
                }
            }
        };
        verify(&store);
        drop(store);
        let reopened = Fragments::open(home.path(), &[model], true).expect("reopen fixture");
        verify(&reopened);
    }
}

/// Directly build/open the stable generation ABI and compare candidates against
/// ZG's exhaustive binary-sign scan. This is not a workspace-service test.
#[cfg(all(
    feature = "experimental-sign-scale-v1",
    target_os = "linux",
    target_arch = "x86_64"
))]
#[test]
fn native_generation_abi_matches_disposable_zg_scan() {
    use std::os::unix::fs::PermissionsExt;

    use zg_zvec_fp16::{
        packed_diskann_generation::{self, Limits, Trust},
        packed_sign_index::{Document, Generation, Metric},
        packed_v1_worker::BinaryDiskAnnWorker,
    };

    super::super::super::zvec::initialize().expect("initialize zvec");
    let mut query = vec![0.1_f32; 64];
    query[0] = 10.0;
    query[1] = 1.0;
    let mut heavy = vec![1.0_f32; 64];
    heavy[1] = -1.0;
    let mut light = vec![1.0_f32; 64];
    light[0] = -1.0;

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
    for (metric, native_metric) in [
        (EmbeddingMetric::Cosine, Metric::Cosine),
        (EmbeddingMetric::DotProduct, Metric::DotProduct),
        (EmbeddingMetric::Euclidean, Metric::Euclidean),
    ] {
        let scan_home = tempfile::tempdir().expect("disposable scalar scan index");
        let model = binary_model(metric);
        let scan = Fragments::open(scan_home.path(), &[model], false)
            .expect("open synthetic exhaustive scan");
        let mut encoded = Vec::new();
        for (file_id, vectors) in [
            (1, vec![heavy.clone(), light.clone()]),
            (
                2,
                vec![
                    heavy.iter().map(|value| 2.0 * value).collect(),
                    vec![0.0; 64],
                ],
            ),
        ] {
            let (source, entities, mut entries) =
                metadata_fragments(file_id, "binary-diskann-native-fixture");
            for (entry, vector) in entries.iter_mut().zip(vectors) {
                entry.vector = vector;
                let (signs, scale) = encode_sign_vector(&entry.vector).expect("encode signs");
                encoded.push((entry.fragment_id.as_str().to_owned(), signs, scale));
            }
            write_fixture(&scan, &source, &entities, &entries);
        }
        scan.flush().expect("flush exhaustive scan");
        let documents: Vec<_> = encoded
            .iter()
            .map(|(id, signs, scale)| Document {
                id,
                signs,
                scale: *scale,
            })
            .collect();
        let generation = Generation::prepare(64, [0x42; 16], &documents, limits.max_build_bytes)
            .expect("stable native generation");
        let identity_bytes = generation
            .identity_bytes(limits.max_input_bytes)
            .expect("identity sidecar");
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
            .expect("private native fixture parent");
        let workspace = parent.path().join("graph");
        let keys = generation.keys(limits.max_input_bytes).expect("keys");
        let trust = packed_diskann_generation::create_v1(
            &workspace,
            generation.dimension(),
            native_metric,
            &keys,
            generation.records(),
            generation.generation(),
            identity_record_sha256,
            limits,
        )
        .expect("build native generation");
        let pinned = Trust::from_bytes(&trust.to_bytes()).expect("pin generation trust");
        drop(generation);
        let native = BinaryDiskAnnWorker::open(
            &workspace,
            pinned,
            identity_bytes.clone(),
            identity_sha256,
            native_metric,
            limits,
        )
        .expect("reopen trusted native generation");

        let expected = scan
            .search_vector("fixture/fixture", &query, 4, None)
            .expect("exhaustive scan oracle");
        let actual = native.search(&query, 4, 4).expect("native graph search");
        assert_eq!(actual.len(), expected.len(), "metric {metric:?}");
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_eq!(actual.id, expected.document_id, "metric {metric:?}");
            assert!(
                (actual.score - expected.score).abs() < 1e-10,
                "metric {metric:?}"
            );
        }

        drop(native);
        let reopened = BinaryDiskAnnWorker::open(
            &workspace,
            pinned,
            identity_bytes,
            identity_sha256,
            native_metric,
            limits,
        )
        .expect("reopen generation through stable ABI");
        let reopened_hits = reopened
            .search(&query, 4, 4)
            .expect("search reopened generation");
        assert_eq!(reopened_hits.len(), expected.len());
        for (actual, expected) in reopened_hits.iter().zip(&expected) {
            assert_eq!(actual.id, expected.document_id, "metric {metric:?}");
            assert!(
                (actual.score - expected.score).abs() < 1e-10,
                "metric {metric:?}"
            );
        }
    }
}

#[test]
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn binary_diskann_fragment_rows_round_trip_before_graph_search_is_enabled() {
    super::super::super::zvec::initialize().expect("initialize zvec");
    let home = tempfile::tempdir().expect("disposable fragment index");
    let model = binary_model(EmbeddingMetric::Cosine)
        .with_vector_index_quantization(VectorIndexQuantization::BinarySignDiskAnnV1);
    let model_reference = model.model().reference();
    let vector = (0..64)
        .map(|index| if index % 3 == 0 { -0.03125 } else { 0.03125 })
        .collect::<Vec<_>>();
    let (source, entities, mut entries) = metadata_fragments(1, "binary-diskann-row-format");
    for entry in &mut entries {
        entry.vector.clone_from(&vector);
    }
    let (expected_signs, expected_scale) = encode_sign_vector(&vector).expect("encode signs");
    let store = Fragments::open(home.path(), std::slice::from_ref(&model), false)
        .expect("open binary DiskANN fragment schema");
    write_fixture(&store, &source, &entities, &entries);
    store.flush().expect("flush binary DiskANN rows");

    let check_rows = |store: &Fragments| {
        let index = store.index(&model_reference).expect("model collection");
        let iterator = native(
            index.iter_with_options(
                Some(&["document_id", "embedding", "embedding_scale"]),
                false,
            ),
            "iterate binary DiskANN rows",
        )
        .expect("iterator");
        let mut count = 0_usize;
        for document in iterator {
            let document = native(document, "read binary DiskANN row").expect("document");
            let signs = native(document.get_binary("embedding"), "read sign bits")
                .expect("sign field")
                .expect("packed embedding");
            let scale = native(document.get_f32("embedding_scale"), "read sign scale")
                .expect("scale field")
                .expect("embedding scale");
            assert_eq!(signs, expected_signs);
            assert_eq!(scale, expected_scale);
            count += 1;
        }
        assert_eq!(count, entries.len());
        let error = store
            .search_vector(&model_reference, &vector, 1, None)
            .expect_err("ANN stays gated until the graph is integrated");
        assert_eq!(error.code(), crate::EngineError::UNSUPPORTED);
    };
    check_rows(&store);
    drop(store);

    let reopened = Fragments::open(home.path(), &[model], true).expect("reopen V1 rows");
    check_rows(&reopened);
}

#[test]
#[cfg(all(
    feature = "experimental-sign-scale-v1",
    target_os = "linux",
    target_arch = "x86_64"
))]
fn fragment_search_routes_unfiltered_v1_queries_through_native_diskann() {
    use std::os::unix::fs::PermissionsExt;

    super::super::super::zvec::initialize().expect("initialize zvec");
    let mut query = vec![0.1_f32; 64];
    query[0] = 10.0;
    query[1] = 1.0;
    let mut heavy = vec![1.0_f32; 64];
    heavy[1] = -1.0;
    let mut light = vec![1.0_f32; 64];
    light[0] = -1.0;
    let heavy_scaled: Vec<_> = heavy.iter().map(|value| 2.0 * value).collect();
    let light_scaled: Vec<_> = light.iter().map(|value| 2.0 * value).collect();

    let binary_sign_home = tempfile::tempdir().expect("binary-sign oracle");
    let binary_sign_model = binary_model(EmbeddingMetric::Cosine);
    let binary_sign_store = Fragments::open(
        binary_sign_home.path(),
        std::slice::from_ref(&binary_sign_model),
        false,
    )
    .expect("open binary-sign oracle");
    for (file_id, vectors) in [
        (1, vec![heavy.clone(), light.clone()]),
        (2, vec![heavy_scaled.clone(), light_scaled.clone()]),
    ] {
        let (source, entities, mut entries) =
            metadata_fragments(file_id, "binary-diskann-fragment-search");
        for (entry, document) in entries.iter_mut().zip(vectors) {
            entry.vector = document;
        }
        write_fixture(&binary_sign_store, &source, &entities, &entries);
    }
    binary_sign_store.flush().expect("flush binary-sign oracle");
    let expected = binary_sign_store
        .search_vector("fixture/fixture", &query, 4, None)
        .expect("binary-sign exact scan");

    let v1_home = tempfile::Builder::new()
        .permissions(std::fs::Permissions::from_mode(0o700))
        .tempdir()
        .expect("private V1 fragment generation");
    let v1_model = binary_model(EmbeddingMetric::Cosine)
        .with_vector_index_quantization(VectorIndexQuantization::BinarySignDiskAnnV1);
    let mut v1_store = Fragments::open(v1_home.path(), std::slice::from_ref(&v1_model), false)
        .expect("open V1 fragment rows");
    for (file_id, vectors) in [
        (1, vec![heavy, light]),
        (2, vec![heavy_scaled, light_scaled]),
    ] {
        let (source, entities, mut entries) =
            metadata_fragments(file_id, "binary-diskann-fragment-search");
        for (entry, document) in entries.iter_mut().zip(vectors) {
            entry.vector = document;
        }
        write_fixture(&v1_store, &source, &entities, &entries);
    }
    v1_store.flush().expect("flush V1 fragment rows");
    let generation = uuid::Uuid::new_v4().to_string();
    let metadata = v1_store
        .build_binary_diskann(v1_home.path(), &generation)
        .expect("build and reopen native DiskANN generation");
    let actual = v1_store
        .search_vector("fixture/fixture", &query, 100, None)
        .expect("native DiskANN search and fragment hydration");
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(&expected) {
        assert_eq!(actual.document_id, expected.document_id);
        assert!((actual.score - expected.score).abs() < 1e-10);
    }
    let filtered = StorageSearchFilter {
        file_ids: Some(vec![FileId::new(2)]),
        ..Default::default()
    };
    let filtered_sql = build_filter(Some(&filtered), &DIRECTORY_LOOKUP).expect("filter SQL");
    let expected_filtered = binary_sign_store
        .search_vector("fixture/fixture", &query, 4, filtered_sql.as_deref())
        .expect("binary-sign filtered scan oracle");
    let actual_filtered = v1_store
        .search_vector("fixture/fixture", &query, 4, filtered_sql.as_deref())
        .expect("binary DiskANN filtered exact fallback");
    assert_eq!(actual_filtered.len(), expected_filtered.len());
    for (actual, expected) in actual_filtered.iter().zip(&expected_filtered) {
        assert_eq!(actual.document_id, expected.document_id);
        assert!((actual.score - expected.score).abs() < 1e-10);
    }
    drop(v1_store);

    let reopened = Fragments::open_with_binary_diskann_manifest(
        v1_home.path(),
        std::slice::from_ref(&v1_model),
        true,
        Some(&metadata),
    )
    .expect("reopen persisted native DiskANN generation");
    let reopened_hits = reopened
        .search_vector("fixture/fixture", &query, 100, None)
        .expect("search reopened native DiskANN generation");
    assert_eq!(reopened_hits.len(), expected.len());
    for (actual, expected) in reopened_hits.iter().zip(&expected) {
        assert_eq!(actual.document_id, expected.document_id);
        assert!((actual.score - expected.score).abs() < 1e-10);
    }
}
