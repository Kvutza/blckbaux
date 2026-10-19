use blckbaux_neighbor_index::mmap_store::MmapColumnStore;
use ndarray::array;
use std::sync::Mutex;
use tempfile::TempDir;

#[test]
fn check_dims() {
    blckbaux_neighbor_index::observation::check_dims(4).unwrap();
    assert!(
        blckbaux_neighbor_index::observation::check_dims(
            blckbaux_neighbor_index::observation::MAX_DIM + 1
        )
        .is_err()
    );
}

#[test]
fn check_rows() {
    blckbaux_neighbor_index::observation::check_rows(10).unwrap();
}

#[test]
fn check_backend() {
    let dir = TempDir::new().unwrap();
    blckbaux_neighbor_index::observation::write_metadata(dir.path(), 0, 4, 1, false, 0).unwrap();
    blckbaux_neighbor_index::observation::check_backend(
        dir.path(),
        blckbaux_neighbor_index::observation::INDEX_BACKEND,
    )
    .unwrap();
}

#[test]
fn load_rows() {
    let dir = TempDir::new().unwrap();
    blckbaux_neighbor_index::observation::write_metadata(dir.path(), 0, 4, 1, false, 2).unwrap();
    assert_eq!(
        blckbaux_neighbor_index::observation::load_rows(dir.path()),
        Some(2)
    );
}

#[test]
fn load_backend() {
    let dir = TempDir::new().unwrap();
    blckbaux_neighbor_index::observation::write_metadata(dir.path(), 0, 4, 1, false, 0).unwrap();
    assert_eq!(
        blckbaux_neighbor_index::observation::load_backend(dir.path()).as_deref(),
        Some(blckbaux_neighbor_index::observation::INDEX_BACKEND)
    );
}

#[test]
fn write_metadata() {
    let dir = TempDir::new().unwrap();
    blckbaux_neighbor_index::observation::write_metadata(dir.path(), 1, 2, 1, false, 0).unwrap();
}

#[test]
fn open_yvar() {
    let dir = TempDir::new().unwrap();
    assert!(
        blckbaux_neighbor_index::observation::open_yvar(dir.path(), 1, Some(&array![[0.1]]))
            .unwrap()
            .is_some()
    );
}

#[test]
fn append_yvar() {
    let dir = TempDir::new().unwrap();
    let mut slot = None;
    blckbaux_neighbor_index::observation::append_yvar(
        dir.path(),
        1,
        &mut slot,
        Some(&array![[0.2]].view()),
    )
    .unwrap();
    assert!(slot.is_some());
}

#[test]
fn train_rows() {
    let dir = TempDir::new().unwrap();
    let mut x = MmapColumnStore::open_mmap(dir.path().join("x.bin"), 2, None).unwrap();
    let mut y = MmapColumnStore::open_mmap(dir.path().join("y.bin"), 1, None).unwrap();
    x.mmap_append(&array![[0.0, 0.0]].view()).unwrap();
    y.mmap_append(&array![[0.0]].view()).unwrap();
    blckbaux_neighbor_index::observation::train_rows(1, &x, &y, None, &[0]).unwrap();
}

#[test]
fn mark_dirty() {
    let dirty = Mutex::new(false);
    blckbaux_neighbor_index::observation::mark_dirty(&dirty);
    assert!(*dirty.lock().unwrap());
}

#[test]
fn num_sidecars() {
    let dir = TempDir::new().unwrap();
    let mut counter =
        blckbaux_neighbor_index::observation::NumObsCounter::open(dir.path()).unwrap();
    counter.set(42);
    assert_eq!(
        blckbaux_neighbor_index::observation::bpann_obs(dir.path()),
        Some(42)
    );
    blckbaux_neighbor_index::observation::write_obs(dir.path(), 7).unwrap();
    assert_eq!(
        blckbaux_neighbor_index::observation::bpann_obs(dir.path()),
        Some(7)
    );
    blckbaux_neighbor_index::observation::write_rows(dir.path(), 5).unwrap();
    assert_eq!(
        blckbaux_neighbor_index::observation::load_rows(dir.path()),
        Some(5)
    );
}

#[test]
fn parse_string() {
    assert_eq!(
        blckbaux_neighbor_index::observation::parse_string(
            "{\"index_backend\":\"bpann_disk\"}",
            "index_backend"
        )
        .as_deref(),
        Some("bpann_disk")
    );
}
