//! Legacy schema-reader baseline for a future binary DiskANN format.
//!
//! The marker below is a synthetic unknown field, not a proposed production
//! schema. It proves only that this reader fails closed on unknown fields.

use super::*;

#[test]
fn unsupported_storage_marker_rejected_without_changing_snapshot() {
    let home = tempfile::tempdir().expect("disposable snapshot");
    let model = schema().with_vector_index_quantization(VectorIndexQuantization::BinarySign);
    let mut value = serde_json::to_value(SchemaRecord::new(std::slice::from_ref(&model)))
        .expect("serialize existing binary-sign schema");
    value["synthetic_future_storage_marker"] = serde_json::json!(true);
    let original = serde_json::to_vec(&value).expect("serialize synthetic future schema");
    let schema_path = home.path().join("schema.json");
    fs::write(&schema_path, &original).expect("write disposable snapshot schema");

    for options in [
        WorkspaceIndexStorageOptions::ReadOnly {
            graph_enabled: false,
            storage_path: home.path().to_owned(),
        },
        WorkspaceIndexStorageOptions::ReadWrite {
            graph_enabled: false,
            storage_path: home.path().to_owned(),
            embeddings: vec![model.clone()],
        },
    ] {
        let error =
            load_schema(home.path(), options).expect_err("unknown storage marker must fail closed");
        assert_eq!(error.code(), EngineError::STORAGE_FAILURE);
        assert!(
            error
                .message()
                .contains("unsupported or invalid storage schema")
        );
        assert_eq!(
            fs::read(&schema_path).expect("read preserved schema"),
            original
        );
        assert_eq!(fs::read_dir(home.path()).expect("list snapshot").count(), 1);
    }
}

#[test]
fn future_binary_diskann_marker_fails_closed_before_native_open() {
    let home = tempfile::tempdir().expect("disposable snapshot");
    let legacy_model = schema().with_vector_index_quantization(VectorIndexQuantization::BinarySign);
    let future_model =
        schema().with_vector_index_quantization(VectorIndexQuantization::BinarySignDiskAnnV1);
    let original = serde_json::to_value(SchemaRecord::new(std::slice::from_ref(&future_model)))
        .expect("serialize future schema");
    assert_eq!(original["binary_sign_storage"], true);
    assert_eq!(original["binary_sign_diskann_storage_version"], 1);
    assert_eq!(
        original["embeddings"][0]["vectorIndexQuantization"],
        "binary_sign_diskann_v1"
    );

    // The strict inner marker now accepts the V1 row format. Workspace opens
    // must still supply an outer generation-pinned trust manifest.
    for version in [None, Some(0), Some(2)] {
        let mut record = SchemaRecord::new(std::slice::from_ref(&future_model));
        record.binary_sign_diskann_storage_version = version;
        let error = record.embeddings().expect_err("mismatched V1 marker");
        assert!(error.message().contains("marker does not match"));
    }
    let mut matched = SchemaRecord::new(std::slice::from_ref(&future_model));
    matched.binary_sign_diskann_storage_version = Some(1);
    assert_eq!(
        matched.embeddings().expect("V1 row schema"),
        [future_model.clone()]
    );
    let mut legacy_with_marker = SchemaRecord::new(std::slice::from_ref(&legacy_model));
    legacy_with_marker.binary_sign_diskann_storage_version = Some(1);
    assert!(
        legacy_with_marker
            .embeddings()
            .expect_err("legacy model and V1 marker must disagree")
            .message()
            .contains("marker does not match")
    );

    // Both the mandatory marker and the distinct model enum differ from the
    // currently installed legacy format. An old strict reader must not infer
    // the existing scalar-sign mode when either part is missing or unknown.
    #[allow(dead_code)]
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct LegacySchemaRecord {
        path_encoding: String,
        binary_sign_storage: bool,
        embeddings: Vec<LegacyEmbedding>,
    }
    #[allow(dead_code)]
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase", deny_unknown_fields)]
    struct LegacyEmbedding {
        model: serde_json::Value,
        dimension: usize,
        metric: String,
        vector_index_quantization: LegacyQuantization,
        max_input_tokens: Option<usize>,
        max_image_bytes: Option<usize>,
    }
    #[allow(dead_code)]
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum LegacyQuantization {
        Fp16,
        BinarySign,
    }
    assert!(serde_json::from_value::<LegacySchemaRecord>(original.clone()).is_err());

    let mut missing_marker = original.clone();
    missing_marker
        .as_object_mut()
        .expect("schema object")
        .remove("binary_sign_diskann_storage_version");
    assert!(serde_json::from_value::<LegacySchemaRecord>(missing_marker.clone()).is_err());
    let mut unknown_version = original.clone();
    unknown_version["binary_sign_diskann_storage_version"] = serde_json::json!(2);
    let mut mismatched_mode = original.clone();
    mismatched_mode["embeddings"][0]["vectorIndexQuantization"] = serde_json::json!("binary_sign");

    let schema_path = home.path().join("schema.json");
    fs::write(home.path().join("storage.lock"), b"").expect("precreate storage lock");
    let collection_path = home.path().join("files");
    fs::create_dir(&collection_path).expect("synthetic collection directory");
    for collection in ["directories", "entities"] {
        fs::create_dir(home.path().join(collection)).expect("synthetic base collection");
    }
    fs::create_dir(
        home.path()
            .join(fragments::fragment_collection_name(&future_model)),
    )
    .expect("synthetic V1 fragment collection");
    let orphan = collection_path.join("999999999.tmp");
    fs::create_dir(&orphan).expect("synthetic orphan directory");
    let sentinel = orphan.join("sentinel");
    fs::write(&sentinel, b"retained").expect("write sentinel");

    for (name, value, expected_code) in [
        (
            "matched-but-no-outer-manifest",
            original,
            EngineError::UNSUPPORTED,
        ),
        (
            "missing-version",
            missing_marker,
            EngineError::STORAGE_FAILURE,
        ),
        (
            "unknown-version",
            unknown_version,
            EngineError::STORAGE_FAILURE,
        ),
        (
            "mismatched-mode",
            mismatched_mode,
            EngineError::STORAGE_FAILURE,
        ),
    ] {
        let bytes = serde_json::to_vec(&value).expect("serialize test schema");
        fs::write(&schema_path, &bytes).expect("write test schema");
        for (options, expected_code) in [
            (
                WorkspaceIndexStorageOptions::ReadOnly {
                    graph_enabled: false,
                    storage_path: home.path().to_owned(),
                },
                expected_code,
            ),
            (
                WorkspaceIndexStorageOptions::ReadWrite {
                    graph_enabled: false,
                    storage_path: home.path().to_owned(),
                    embeddings: vec![future_model.clone()],
                },
                EngineError::UNSUPPORTED,
            ),
        ] {
            let error = IndexStore::open(options)
                .err()
                .unwrap_or_else(|| panic!("{name} must fail before native collection open"));
            assert_eq!(error.code(), expected_code, "{name}");
            assert_eq!(fs::read(&schema_path).expect("schema preserved"), bytes);
            assert_eq!(fs::read(&sentinel).expect("orphan preserved"), b"retained");
            assert!(
                !collection_path.join("LOCK").exists(),
                "{name}: native open"
            );
        }
    }
}

#[test]
fn requesting_unavailable_binary_diskann_does_not_create_storage() {
    let home = tempfile::tempdir().expect("disposable parent");
    let target = home.path().join("must-not-be-created");
    let model =
        schema().with_vector_index_quantization(VectorIndexQuantization::BinarySignDiskAnnV1);
    let error = IndexStore::open(WorkspaceIndexStorageOptions::ReadWrite {
        graph_enabled: false,
        storage_path: target.clone(),
        embeddings: vec![model],
    })
    .err()
    .expect("not-yet-implemented native mode must fail closed");
    assert_eq!(error.code(), EngineError::UNSUPPORTED);
    assert!(!target.exists(), "invalid request must not create storage");
}
