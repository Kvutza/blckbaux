use super::BayesOptimizer;
use crate::config::{RunConfig, turbo_zero};
use crate::error::ENNError;
use crate::optimizer_factory::enn_configured;
use ndarray::{Array2, array};
use rand::SeedableRng;
use rand::rngs::StdRng;

#[derive(Debug, PartialEq)]
struct Trajectory {
    candidates: Vec<u64>,
    incumbents: Vec<u64>,
    observations: usize,
    restarts: usize,
}

fn objective(x: &[f64], round: usize) -> f64 {
    let phase = round as f64 * 0.013;
    -x.iter()
        .enumerate()
        .map(|(axis, value)| {
            let target = 0.2 + 0.15 * axis as f64 + phase;
            (value - target).powi(2)
        })
        .sum::<f64>()
}

fn trajectory(seed: u64) -> Trajectory {
    let bounds = array![[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]];
    let overrides = RunConfig {
        noise_aware: Some(true),
        num_candidates_factor: Some(1.0),
        min_candidates: Some(32),
        max_candidates: Some(32),
        ..Default::default()
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let mut optimizer = enn_configured(bounds, 3, 4, &mut rng, Some(&overrides)).unwrap();
    let mut candidates = Vec::new();
    let mut incumbents = Vec::new();
    let mut expected_observations = 0;

    for round in 0..16 {
        let arms = 1 + (round + seed as usize) % 4;
        let x = optimizer.ask(arms, &mut rng).unwrap();
        assert_eq!(x.shape(), &[arms, 3], "seed={seed} round={round}");
        assert!(
            x.iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value)),
            "candidate escaped its declared domain: seed={seed} round={round}"
        );
        candidates.extend(x.iter().map(|value| value.to_bits()));

        let y = Array2::from_shape_fn((arms, 1), |(row, _)| {
            objective(x.row(row).as_slice().unwrap(), round)
        });
        optimizer.tell(&x.view(), &y.view(), &mut rng).unwrap();
        expected_observations += arms;

        assert_eq!(optimizer.obs_count(), expected_observations);
        assert_eq!(optimizer.x_obs().unwrap().nrows(), expected_observations);
        assert_eq!(optimizer.y_obs().unwrap().nrows(), expected_observations);
        assert_eq!(
            optimizer.incumbent_tracker.observation_count(),
            expected_observations,
            "posterior tracker diverged: seed={seed} round={round}"
        );
        assert!(optimizer.tr_length().is_finite() && optimizer.tr_length() > 0.0);
        let incumbent = optimizer
            .y_scalar()
            .expect("tell must establish an incumbent");
        assert!(incumbent.iter().all(|value| value.is_finite()));
        incumbents.extend(incumbent.iter().map(|value| value.to_bits()));
    }

    Trajectory {
        candidates,
        incumbents,
        observations: optimizer.obs_count(),
        restarts: optimizer.restart_generation(),
    }
}

#[test]
fn trajectory_replay() {
    for seed in [0, 1, 17, 0x5eed] {
        assert_eq!(trajectory(seed), trajectory(seed), "replay seed={seed}");
    }
}

#[test]
fn invalid_observation() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(42);
    let mut optimizer = BayesOptimizer::new(bounds, turbo_zero(), &mut rng).unwrap();
    let x = array![[0.5, 0.5], [0.3, 0.7], [0.1, 0.9]];
    let y = array![[1.0], [2.0]];

    let error = optimizer
        .add_observations(&x.view(), &y.view())
        .expect_err("mismatched rows must fail");
    assert!(matches!(error, ENNError::InvalidShape { .. }));
    assert_eq!(optimizer.obs_count(), 0);
}

#[test]
fn tracker_restarts() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let overrides = RunConfig {
        noise_aware: Some(true),
        ..Default::default()
    };
    let mut rng = StdRng::seed_from_u64(57);
    let mut optimizer = enn_configured(bounds, 3, 0, &mut rng, Some(&overrides)).unwrap();

    for round in 0..64 {
        let x = array![[0.1 + 0.01 * round as f64, 0.2]];
        let y = array![[-(round as f64)]];
        optimizer.tell(&x.view(), &y.view(), &mut rng).unwrap();
        assert_eq!(
            optimizer.incumbent_tracker.observation_count(),
            optimizer.obs_count(),
            "tracker diverged after round {round}"
        );
    }
    assert!(optimizer.restart_generation() >= 1);
}
