use super::*;

const PRETRAIN_MUON: &str = "version=4\nexperiment='pretrain'\nmodel='fbt-pisa1-hnet-mhc4-v1'\n[data]\nrecipe='stack-v3-rust-go-v1'\n[optimizer]\nalgorithm='muon'\n[objective]\nfunction='cross-entropy'\nsampling='natural'\n";

#[cfg(feature = "buck2-test-data")]
const GAUSSIAN: &str = include_str!("../../../turbo-enn.toml/hnet-4k-gaussian.toml");
#[cfg(not(feature = "buck2-test-data"))]
const GAUSSIAN: &str = include_str!("../../../../../../examples/tuning/hnet-4k-gaussian.toml");

#[cfg(feature = "buck2-test-data")]
const OPERATOR_BO: &str = include_str!("../../../turbo-enn.toml/hnet-1m-operator-bo.toml");
#[cfg(not(feature = "buck2-test-data"))]
const OPERATOR_BO: &str =
    include_str!("../../../../../../examples/tuning/hnet-1m-operator-bo.toml");

fn resident(acquisition: &str, candidates: u32, arms: u32, fit: &str) -> String {
    format!(
        "version=4\nexperiment='end-to-end'\n[optimizer]\nalgorithm='turbo-enn'\n[optimizer.candidate-generation]\ngenerator='independent'\ncandidates={candidates}\narms={arms}\n[optimizer.surrogate]\nestimator='enn'\n{fit}\n[optimizer.acquisition]\n{acquisition}\n[optimizer.trust-region]\npolicy='turbo'\n"
    )
}

#[test]
fn roundtrip() {
    macro_rules! examples {
        ($($name:literal),+ $(,)?) => {
            [$({
                #[cfg(feature = "buck2-test-data")]
                let text = include_str!(concat!("../../../turbo-enn.toml/", $name));
                #[cfg(not(feature = "buck2-test-data"))]
                let text = include_str!(concat!("../../../../../../examples/tuning/", $name));
                ($name, text)
            }),+]
        };
    }
    for (name, text) in examples!(
        "hnet-1m-active-rotate.toml",
        "hnet-1m-gaussian.toml",
        "hnet-1m-lie.toml",
        "hnet-natural-adamw.toml",
        "hnet-1m-metal-1000.toml",
        "hnet-1m-metal-adaptive-1000.toml",
        "hnet-1m-metal-adaptive-200.toml",
        "hnet-natural-muon.toml",
        "hnet-1m-operator-bo.toml",
        "hnet-4k-gaussian.toml",
        "hnet-4k-mean-audit.toml",
        "hnet-4k-pool-audit.toml",
        "hnet-4k-operator-bo.toml",
        "hnet-4k-rademacher.toml",
        "hnet-4k-self-tuning-audit.toml",
        "hnet-64k-gaussian.toml",
        "hnet-fineweb-active-rotate.toml",
        "hnet-fineweb-muon.toml",
    ) {
        let spec = ExperimentSpec::parse(text).unwrap_or_else(|error| panic!("{name}: {error}"));
        let resolved = spec.to_toml().unwrap();
        let restored = ExperimentSpec::parse(&resolved).unwrap();
        assert_eq!(resolved, restored.to_toml().unwrap(), "{name}");
    }
}

#[test]
fn fixed_data() {
    let spec = ExperimentSpec::parse(GAUSSIAN).unwrap();
    assert_eq!(spec.objective.window_tokens, Some(4_096));
    assert!(spec.generation.is_none());
    let restored = ExperimentSpec::parse(&spec.to_toml().unwrap()).unwrap();
    assert_eq!(restored.objective.window_tokens, Some(4_096));
}

#[test]
fn closed() {
    for text in [
        format!("{PRETRAIN_MUON}neighbor=10"),
        format!("{PRETRAIN_MUON}[run]\nround=10"),
        PRETRAIN_MUON.replace("algorithm='muon'", "algorithm='unknown'"),
        PRETRAIN_MUON.replace("algorithm='muon'", "algorithm='muon-v1'"),
        format!("{PRETRAIN_MUON}[run]\nselection='enn'"),
        PRETRAIN_MUON.replace("stack-v3-rust-go-v1", "../fineweb"),
        PRETRAIN_MUON.replace(
            "algorithm='muon'",
            "algorithm='muon'\n[optimizer.acquisition]\nfunction='thompson'",
        ),
        GAUSSIAN.replace(
            "generator = \"independent\"",
            "generator = \"independent\"\ncandidate = 4",
        ),
        GAUSSIAN.replace("estimator = \"knn\"", "estimator = \"knn\"\nneigbors = 10"),
        GAUSSIAN.replace("fit = \"leave-one-out\"", "procedure = \"leave-one-out\""),
        GAUSSIAN.replace("fit = \"leave-one-out\"\n", ""),
        OPERATOR_BO.replace("estimator = \"enn\"", "estimator = \"sub-nn\""),
        GAUSSIAN.replace(
            "function = \"upper-confidence-bound\"",
            "function = \"thompson\"\nbeta = 1.0",
        ),
        GAUSSIAN.replace(
            "policy = \"reliability\"",
            "policy = \"reliability\"\nevidence-decy = 0.9",
        ),
    ] {
        assert!(parse_tune(&text).is_err(), "accepted:\n{text}");
    }
}

#[test]
fn recipe_names() {
    let text = PRETRAIN_MUON.replace("stack-v3-rust-go-v1", "code-mixture-v1");
    assert!(ExperimentSpec::parse(&text).is_ok());
    for name in ["123", "Code-Mixture", "code--mixture", "code-mixture-"] {
        assert!(
            ExperimentSpec::parse(&PRETRAIN_MUON.replace("stack-v3-rust-go-v1", name)).is_err()
        );
    }
}

#[test]
fn guards() {
    for text in [
        PRETRAIN_MUON.replace(
            "algorithm='muon'",
            "algorithm='muon'\n[execution]\nrounds=0",
        ),
        GAUSSIAN.replace("candidates = 4", "candidates = 0"),
        GAUSSIAN.replace("arms = 4", "arms = 3"),
        GAUSSIAN.replace("neighbors = 32", "neighbors = 0"),
        GAUSSIAN.replace("epistemic-scale = 0.05", "epistemic-scale = nan"),
        GAUSSIAN.replace("min = 0.0001", "min = 0.2"),
        GAUSSIAN.replace(
            "policy = \"reliability\"",
            "policy = \"reliability\"\nevidence-decay = 2.0",
        ),
        GAUSSIAN.replace("min = 0.0001", "min = nan"),
        GAUSSIAN.replace("max = 0.016", "max = 0.00001"),
        GAUSSIAN.replace("initial = 0.001", "initial = -1.0"),
    ] {
        assert!(
            parse_tune(&text).is_err(),
            "accepted invalid document:\n{text}"
        );
    }
}

#[test]
fn distinct_roles() {
    assert!(parse_tune(&GAUSSIAN.replace("knn", "enn")).is_err());
    assert!(parse_tune(&GAUSSIAN.replace("operator-bo", "turbo-enn")).is_err());
}

#[test]
fn random_acquisition() {
    let text = GAUSSIAN.replace("upper-confidence-bound", "random");
    let config = parse_tune(&text).unwrap();
    assert_eq!(config.selection, Some(PretrainSelection::Random));
    let spec = ExperimentSpec::from_runtime(&config).unwrap();
    let OptimizerSpec::OperatorBo { acquisition, .. } = spec.optimizer else {
        panic!("expected operator-bo");
    };
    assert!(matches!(acquisition, AcquisitionSpec::Random));
}

#[test]
fn reliability() {
    let config = parse_tune(GAUSSIAN).unwrap();
    assert_eq!(
        config.reliability_controller(),
        Some(crate::ReliabilityPolicy::default())
    );
    let spec = ExperimentSpec::from_runtime(&config).unwrap();
    let roundtrip = parse_tune(&spec.to_toml().unwrap()).unwrap();
    assert_eq!(
        roundtrip.reliability_controller(),
        config.reliability_controller()
    );
}

#[test]
fn turbo_enn() {
    let config = parse_tune(OPERATOR_BO).unwrap();
    assert_eq!(config.k_neighbors, Some(10));
    assert_eq!(config.local_scale_neighbors, None);
    assert_eq!(
        config
            .resident_enn(config.acquisition_seed())
            .unwrap()
            .distance_scaling,
        DistanceScaling::Global
    );
    let spec = ExperimentSpec::from_runtime(&config).unwrap();
    let OptimizerSpec::TurboEnn { surrogate, .. } = &spec.optimizer else {
        panic!("expected turbo-enn");
    };
    assert_eq!(surrogate.estimator, SurrogateEstimator::Enn);
    assert_eq!(surrogate.neighbors, 10);
    assert_eq!(surrogate.scaling, DistanceScaling::Global);
    assert_eq!(surrogate.local_neighbors, None);
    let SurrogateFit::LeaveOneOut {
        fit_candidates,
        fit_samples,
    } = &surrogate.fit
    else {
        panic!("expected leave-one-out fitting");
    };
    assert_eq!(*fit_candidates, 30);
    assert_eq!(*fit_samples, 10);
}

#[test]
fn failure_budget() {
    let text = OPERATOR_BO.replace("failure-tolerance = 4", "failure-tolerance = 7");
    let config = parse_tune(&text).unwrap();
    let policy = config.reliability_controller().unwrap();
    assert_eq!(policy.failure_tolerance, 7);
    let spec = ExperimentSpec::from_runtime(&config).unwrap();
    let roundtrip = parse_tune(&spec.to_toml().unwrap()).unwrap();
    assert_eq!(
        roundtrip
            .reliability_controller()
            .unwrap()
            .failure_tolerance,
        7
    );
}

#[test]
fn paths() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("experiment.toml");
    let text = resident("function='thompson'", 4, 1, "fit='fixed'").replace(
        "experiment='end-to-end'",
        "experiment='end-to-end'\noutput='results'",
    );
    std::fs::write(&path, &text).unwrap();
    let authored = std::fs::read_to_string(&path).unwrap();
    let (config, text) = load_tune(&path).unwrap();
    let restored = parse_tune(&text).unwrap();
    assert_eq!(
        restored.output(),
        directory.path().canonicalize().unwrap().join("results")
    );
    assert_eq!(restored.length(), config.length());
    assert_eq!(restored.acquisition_seed(), config.acquisition_seed());
    assert!(matches!(
        restored.acquisition,
        Some(AcquisitionConfig::Thompson)
    ));
    assert!(text.contains("version = 4"));
    assert!(text.contains("[optimizer.candidate-generation]"));
    assert_eq!(std::fs::read_to_string(path).unwrap(), authored);
}

#[test]
fn models() {
    assert_eq!(
        ModelArchitecture::ALL.map(ModelArchitecture::id),
        [
            "fbt-pisa1-legacy-v1",
            "fbt-pisa1-residual1-v1",
            "fbt-pisa1-projected-boundary-v1",
            "fbt-pisa1-hc4-v1",
            "fbt-pisa1-mhc4-v1",
            "fbt-pisa1-looped-mhc4-v1",
            "fbt-pisa1-diffusion-mhc4-v1",
            "fbt-pisa1-hnet-mhc4-v1",
        ]
    );
}

#[test]
fn budgets() {
    let config = parse_tune(&resident(
        "function='upper-confidence-bound'",
        4,
        2,
        "fit='leave-one-out'\nfit-candidates=64\nfit-samples=32",
    ))
    .unwrap();
    let surrogate = config.resident_enn(17).unwrap();
    assert_eq!(surrogate.pool.count(), 4);
    assert_eq!(surrogate.pool.arms(), 2);
    assert_eq!(surrogate.num_candidates, 64);
    assert_eq!(surrogate.num_samples, 32);
    assert!(surrogate.fit_params);
    assert!(!surrogate.adapt_neighbors);
}

#[test]
fn search_conflict() {
    let ambiguous = resident(
        "function='upper-confidence-bound'",
        4,
        1,
        "fit='leave-one-out-neighbor-search-v1'\nfit-candidates=4\nfit-samples=4",
    );
    assert!(parse_tune(&ambiguous).is_err());
}

#[test]
fn neighbor_controls() {
    let fixed = resident("function='upper-confidence-bound'", 4, 1, "fit='fixed'");
    let fixed = parse_tune(&fixed).unwrap().resident_enn(17).unwrap();
    assert!(!fixed.fit_params);
    assert!(!fixed.adapt_neighbors);

    let fitted = resident(
        "function='upper-confidence-bound'",
        4,
        1,
        "fit='leave-one-out'\nfit-candidates=4\nfit-samples=4",
    );
    let fitted = parse_tune(&fitted).unwrap().resident_enn(17).unwrap();
    assert!(fitted.fit_params);
    assert!(!fitted.adapt_neighbors);

    let adaptive = resident("function='upper-confidence-bound'", 4, 1, "fit='fixed'").replace(
        "estimator='enn'",
        "estimator='enn'\nneighbor-rule='risk-prefix'",
    );
    let adaptive = parse_tune(&adaptive).unwrap().resident_enn(17).unwrap();
    assert!(!adaptive.fit_params);
    assert!(adaptive.adapt_neighbors);

    let fitted_adaptive = resident(
        "function='upper-confidence-bound'",
        4,
        1,
        "fit='leave-one-out'\nfit-candidates=4\nfit-samples=4",
    )
    .replace(
        "estimator='enn'",
        "estimator='enn'\nneighbor-rule='risk-prefix'",
    );
    let fitted_adaptive = parse_tune(&fitted_adaptive)
        .unwrap()
        .resident_enn(17)
        .unwrap();
    assert!(fitted_adaptive.fit_params);
    assert!(fitted_adaptive.adapt_neighbors);
}

#[test]
fn vectors() {
    let generation = "\n[generation]\npurpose='systems-probe'\nmax-tokens=16\ntemperature=0.0\n[[generation.tasks]]\nprompt=[1]\n[generation.reward]\nkind='command-objectives'\nprogram='score'\nargs=[]\ntimeout-ms=10\n";
    let base = resident("function='pareto'\nscales=[1.0,2.0]", 4, 1, "fit='fixed'").replace(
        "experiment='end-to-end'",
        "experiment='generation'\nmodel='fbt-pisa1-legacy-v1'",
    );
    let config = parse_tune(&(base.clone() + generation)).unwrap();
    assert_eq!(
        config.objective_acquisition.unwrap().mode,
        ResidentObjectiveMode::Pareto
    );
    assert!(
        parse_tune(
            &(base.replace(
                "scales=[1.0,2.0]",
                "scales=[1.0,2.0]\npreferences=[1.0,1.0]"
            ) + generation)
        )
        .is_err()
    );
}

#[test]
fn causal_recipe() {
    let base = resident("function='thompson'", 1, 1, "fit='fixed'").replace(
        "experiment='end-to-end'",
        "experiment='pretrain'\nmodel='fbt-pisa1-hnet-mhc4-v1'\n[data]\nrecipe='stack-v3-rust-go-v1'\n[objective]\nfunction='cross-entropy'\nsampling='natural'\nwindow-tokens=1048576",
    );
    let config = parse_tune(&base).unwrap();
    assert_eq!(config.window_tokens, Some(1_048_576));
    assert!(config.generation.is_none());
    assert!(config.objective_acquisition.is_none());
    assert!(parse_tune(&base.replace("recipe=", "mixture=")).is_err());
    assert!(parse_tune(&(base.clone() + "\n[objective]\nreference='moving-incumbent'\n")).is_err());
    assert!(parse_tune(&(base + "\n[generation]\npurpose='pretrain'\nmax-tokens=16\ntemperature=1.0\n[generation.reward]\nkind='causal-objectives'\n")).is_err());
}

#[test]
fn natural_objective() {
    let spec = ExperimentSpec::parse(PRETRAIN_MUON).unwrap();
    let authored = spec.to_toml().unwrap();
    assert!(authored.contains("function = \"cross-entropy\""));
    assert!(authored.contains("sampling = \"natural\""));
    assert!(!authored.contains("[generation]"));
    assert!(!authored.contains("max-tokens"));
    assert!(!authored.contains("corpus-prompt-tokens"));
}

#[test]
fn window_budget() {
    assert!(ExperimentSpec::parse(&format!("{PRETRAIN_MUON}window-tokens=4096\n")).is_err());
    assert!(ExperimentSpec::parse(&GAUSSIAN.replace("window-tokens = 4096\n", "")).is_err());
    assert!(
        ExperimentSpec::parse(&GAUSSIAN.replace("sampling = \"natural\"", "sampling = \"window\""))
            .is_err()
    );
}
