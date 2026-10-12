//! `HelixStore` behaviour beyond the shared contract: modes, sharing, locking, persistence.
#![cfg(feature = "helix")]

use std::sync::Arc;

use ce_store::{EntityNode, GraphStore, HelixMode, HelixOptions, HelixStore, StoreError};
use pretty_assertions::assert_eq;

const DIM: usize = 4;

#[test]
fn handles_are_shared_per_database() {
    let options = HelixOptions::memory(format!("shared-{}", std::process::id()), DIM);
    let first = HelixStore::open(&options).expect("opens");
    let second = HelixStore::open(&options).expect("reuses");
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(first.mode(), HelixMode::Memory);
    assert_eq!(first.location(), "memory");

    let mut other_dim = options.clone();
    other_dim.embedding_dim = DIM + 1;
    let error = HelixStore::open(&other_dim).expect_err("dimension mismatch");
    assert!(matches!(error, StoreError::Connection(message) if message.contains("embedding_dim")));
}

#[test]
fn closed_handles_report_database_lost() {
    let options = HelixOptions::memory(format!("closed-{}", std::process::id()), DIM);
    let store = HelixStore::open(&options).expect("opens");
    store.create_entity(&EntityNode::new("Acme", "ORG")).expect("write");
    store.close().expect("close");
    store.close().expect("closing twice is a no-op");
    let error = store.create_entity(&EntityNode::new("Beta", "ORG")).expect_err("closed");
    assert!(error.is_fatal(), "{error:?}");
    let reopened = HelixStore::open(&options).expect("a closed handle is replaced");
    assert!(!Arc::ptr_eq(&store, &reopened));
}

#[test]
fn embedded_database_persists_and_holds_the_writer_lock() {
    let dir = tempfile::tempdir().expect("temp dir");
    let options = HelixOptions::embedded(dir.path(), "case-a", DIM);
    let (store_id, entity_id) = {
        let store = HelixStore::open(&options).expect("opens on disk");
        assert!(dir.path().join("case-a.lock").exists());
        let acme = store.create_entity(&EntityNode::new("Acme Corp", "ORG")).expect("write");
        store.create_entity_embedding(&acme.id, &[1.0, 0.0, 0.0, 0.0]).expect("vector");

        // A second handle on the same lock file (another writer) is refused.
        let lock_path = dir.path().join("case-a.lock");
        let other =
            std::fs::OpenOptions::new().read(true).write(true).open(&lock_path).expect("lock file");
        assert!(fs4::FileExt::try_lock(&other).is_err(), "the store holds the lock");

        let id = store.store_id();
        store.close().expect("close");
        assert!(fs4::FileExt::try_lock(&other).is_ok(), "close releases the lock");
        fs4::FileExt::unlock(&other).expect("unlock");
        (id, acme.id)
    };

    let reopened = HelixStore::open(&options).expect("reopens");
    assert_eq!(reopened.store_id(), store_id, "the store id persists");
    let hits = reopened.find_entity_by_name("acme").expect("text index persists");
    assert_eq!(hits[0].node.id, entity_id);
    let hits = reopened.find_similar_entities(&[1.0, 0.0, 0.0, 0.0], 3).expect("vectors persist");
    assert_eq!(hits[0].node.id, entity_id);
    reopened.close().expect("close");
}

#[test]
fn a_held_lock_fails_open() {
    let dir = tempfile::tempdir().expect("temp dir");
    let lock_path = dir.path().join("busy.lock");
    let holder = std::fs::File::create(&lock_path).expect("lock file");
    fs4::FileExt::try_lock(&holder).expect("lock taken by another writer");
    let error =
        HelixStore::open(&HelixOptions::embedded(dir.path(), "busy", DIM)).expect_err("busy");
    assert!(
        matches!(&error, StoreError::Connection(message) if message.contains("held by another process")),
        "{error:?}"
    );
}

#[test]
fn server_mode_requires_a_url() {
    let mut options = HelixOptions::server("http://127.0.0.1:9", None, "srv", DIM);
    options.url = None;
    assert!(matches!(HelixStore::open(&options), Err(StoreError::Connection(_))));
}

#[test]
#[ignore = "needs a HelixDB v3 server at HELIX_TEST_URL"]
fn server_mode_round_trip() {
    let url = std::env::var("HELIX_TEST_URL").expect("HELIX_TEST_URL is set");
    let store =
        HelixStore::open(&HelixOptions::server(url, None, "contract", DIM)).expect("connects");
    let acme = store.create_entity(&EntityNode::new("Acme", "ORG")).expect("write");
    assert_eq!(
        store.get_all_entities(1_000).expect("read").iter().filter(|e| e.id == acme.id).count(),
        1
    );
}

#[test]
fn from_config_reads_the_helix_block() {
    let yaml = "mode: memory\nnamespace: from-config\nbatch_size: 50\nembedding_dim: 4\n";
    let config: ce_config::pipeline::HelixConfig = yaml_serde::from_str(yaml).expect("helix block");
    let options = HelixOptions::from_config(&config);
    assert_eq!(options.mode, HelixMode::Memory);
    assert_eq!(options.namespace, "from-config");
    assert_eq!(options.api_key, None);
    let store = HelixStore::from_config(&config).expect("opens");
    assert_eq!(store.embedding_dim(), 4);
}
