"""Generated from Cargo metadata by tools/build-parity --sync. Do not edit."""

BLCKBAUX_CRATES = ["anstream", "anstyle", "deser", "memmap2", "ndarray", "rand", "rand_distr", "rayon", "sha2", "sobol", "thiserror", "tokenizers"]

BLCKBAUX_TEST_CRATES = ["anstream", "anstyle", "deser", "memmap2", "ndarray", "numpy", "rand", "rand_chacha", "rand_distr", "rayon", "sha2", "sobol", "tempfile", "thiserror", "tokenizers"]

BLCKBAUX_PROGRAMS = [
    {
        "version": "0.2.3",
        "name": "turbo-enn",
        "crate": "turbo_enn",
        "root": "src/bin/turbo-enn/main.rs",
        "crate_deps": ["anstream", "anstyle", "deser", "memmap2", "ndarray", "rand", "rand_distr", "rayon", "sha2", "sobol", "thiserror", "tokenizers"],
        "local_deps": ["artifact", "blckbaux", "neighbor-index", "semantics", "tooling"],
        "required_features": [],
    },
    {
        "version": "0.2.3",
        "name": "turbo-enn-worker",
        "crate": "turbo_enn_worker",
        "root": "src/bin/turbo-enn-worker/main.rs",
        "crate_deps": ["anstream", "anstyle", "deser", "memmap2", "ndarray", "rand", "rand_distr", "rayon", "sha2", "sobol", "thiserror", "tokenizers"],
        "local_deps": ["artifact", "blckbaux", "neighbor-index", "semantics", "tooling"],
        "required_features": [],
    },
    {
        "version": "0.2.3",
        "name": "cuda_trial",
        "crate": "cuda_trial",
        "root": "examples/cuda_trial.rs",
        "crate_deps": ["anstream", "anstyle", "deser", "memmap2", "ndarray", "numpy", "rand", "rand_chacha", "rand_distr", "rayon", "sha2", "sobol", "tempfile", "thiserror", "tokenizers"],
        "local_deps": ["artifact", "blckbaux", "neighbor-index", "semantics", "tooling"],
        "required_features": ["cuda"],
    },
    {
        "version": "0.2.3",
        "name": "knn_frontier",
        "crate": "knn_frontier",
        "root": "examples/knn_frontier.rs",
        "crate_deps": ["anstream", "anstyle", "deser", "memmap2", "ndarray", "numpy", "rand", "rand_chacha", "rand_distr", "rayon", "sha2", "sobol", "tempfile", "thiserror", "tokenizers"],
        "local_deps": ["artifact", "blckbaux", "neighbor-index", "semantics", "tooling"],
        "required_features": ["metal"],
    },
    {
        "version": "0.2.3",
        "name": "posterior_frontier",
        "crate": "posterior_frontier",
        "root": "examples/posterior_frontier.rs",
        "crate_deps": ["anstream", "anstyle", "deser", "memmap2", "ndarray", "numpy", "rand", "rand_chacha", "rand_distr", "rayon", "sha2", "sobol", "tempfile", "thiserror", "tokenizers"],
        "local_deps": ["artifact", "blckbaux", "neighbor-index", "semantics", "tooling"],
        "required_features": [],
    },
    {
        "version": "0.2.3",
        "name": "stable_api",
        "crate": "stable_api",
        "root": "examples/stable_api.rs",
        "crate_deps": ["anstream", "anstyle", "deser", "memmap2", "ndarray", "numpy", "rand", "rand_chacha", "rand_distr", "rayon", "sha2", "sobol", "tempfile", "thiserror", "tokenizers"],
        "local_deps": ["artifact", "blckbaux", "neighbor-index", "semantics", "tooling"],
        "required_features": [],
    },
    {
        "version": "0.2.3",
        "name": "trial_bench",
        "crate": "trial_bench",
        "root": "examples/trial_bench.rs",
        "crate_deps": ["anstream", "anstyle", "deser", "memmap2", "ndarray", "numpy", "rand", "rand_chacha", "rand_distr", "rayon", "sha2", "sobol", "tempfile", "thiserror", "tokenizers"],
        "local_deps": ["artifact", "blckbaux", "neighbor-index", "semantics", "tooling"],
        "required_features": [],
    },
]

ARTIFACT_CRATES = ["deser", "deser-json", "deser-toml", "deser-value", "sha2"]

ARTIFACT_TEST_CRATES = ["deser", "deser-json", "deser-toml", "deser-value", "sha2", "tempfile"]

ARTIFACT_PROGRAMS = [
]

CLI_CRATES = ["clap"]

CLI_TEST_CRATES = ["clap"]

CLI_PROGRAMS = [
    {
        "version": "0.2.3",
        "name": "blckbaux",
        "crate": "blckbaux",
        "root": "src/bin/blckbaux/main.rs",
        "crate_deps": ["clap"],
        "local_deps": ["artifact", "tooling"],
        "required_features": [],
    },
]

CORPUS_CRATES = ["arrow-array", "deser", "futures", "opendal", "parquet", "parquet_opendal", "sha2", "tokenizers", "tokio"]

CORPUS_TEST_CRATES = ["arrow-array", "deser", "futures", "opendal", "parquet", "parquet_opendal", "sha2", "tokenizers", "tokio"]

CORPUS_PROGRAMS = [
    {
        "version": "0.2.3",
        "name": "blckbaux-corpus",
        "crate": "blckbaux_corpus",
        "root": "src/main.rs",
        "crate_deps": ["arrow-array", "deser", "futures", "opendal", "parquet", "parquet_opendal", "sha2", "tokenizers", "tokio"],
        "local_deps": ["artifact"],
        "required_features": [],
    },
]

MODAL_RUNNER_CRATES = ["modal-rs", "tempfile", "tokio"]

MODAL_RUNNER_TEST_CRATES = ["modal-rs", "tempfile", "tokio"]

MODAL_RUNNER_PROGRAMS = [
    {
        "version": "0.2.3",
        "name": "blckbaux-modal",
        "crate": "blckbaux_modal",
        "root": "src/main.rs",
        "crate_deps": ["modal-rs", "tempfile", "tokio"],
        "local_deps": ["artifact", "blckbaux", "semantics"],
        "required_features": [],
    },
]

NEIGHBOR_INDEX_CRATES = ["deser", "memmap2", "ndarray", "rand", "rayon", "thiserror"]

NEIGHBOR_INDEX_TEST_CRATES = ["deser", "memmap2", "ndarray", "rand", "rand_chacha", "rayon", "tempfile", "thiserror"]

NEIGHBOR_INDEX_PROGRAMS = [
]

PROPERTIES_CRATES = ["hegeltest", "ndarray"]

PROPERTIES_TEST_CRATES = ["hegeltest", "ndarray"]

PROPERTIES_PROGRAMS = [
]

BLCKBAUX_PY_CRATES = ["ndarray", "numpy", "pyo3", "rand"]

BLCKBAUX_PY_TEST_CRATES = ["ndarray", "numpy", "pyo3", "rand"]

BLCKBAUX_PY_PROGRAMS = [
]

RUNNER_CRATES = ["clap", "sha2"]

RUNNER_TEST_CRATES = ["clap", "sha2"]

RUNNER_PROGRAMS = [
    {
        "version": "0.2.3",
        "name": "blckbaux-runner",
        "crate": "blckbaux_runner",
        "root": "src/main.rs",
        "crate_deps": ["clap", "sha2"],
        "local_deps": ["artifact", "blckbaux", "semantics", "tooling"],
        "required_features": [],
    },
]

SEMANTICS_CRATES = ["deser", "rand"]

SEMANTICS_TEST_CRATES = ["deser", "rand"]

SEMANTICS_PROGRAMS = [
]

TOOLING_CRATES = ["clap", "deser"]

TOOLING_TEST_CRATES = ["clap", "deser"]

TOOLING_PROGRAMS = [
    {
        "version": "0.2.3",
        "name": "build-graph",
        "crate": "build_graph",
        "root": "src/bin/build-graph.rs",
        "crate_deps": ["clap", "deser"],
        "local_deps": ["artifact", "tooling"],
        "required_features": [],
    },
]

PTX_SYNTH_CRATES = []

PTX_SYNTH_TEST_CRATES = []

PTX_SYNTH_PROGRAMS = [
]

RUST_EDITION = "2024"
