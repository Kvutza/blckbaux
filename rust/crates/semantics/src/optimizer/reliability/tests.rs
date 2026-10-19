use super::*;

#[test]
fn paired_streams() {
    assert_eq!(candidate_seed(73, 0), candidate_seed(73, 1));
    assert_eq!(candidate_seed(73, 2), candidate_seed(73, 3));
    assert_ne!(candidate_seed(73, 0), candidate_seed(73, 2));
}
#[test]
fn pool_parity() {
    let legacy = PoolShape::new(1, 4).unwrap();
    assert_eq!(legacy, PoolShape::legacy());
    assert_eq!(legacy.seed(73, 0), legacy.seed(73, 1));
    assert_eq!(legacy.seed(73, 2), legacy.seed(73, 3));

    let paired = PoolShape::new(2, 4).unwrap();
    assert_eq!(paired.seed(73, 0), paired.seed(73, 1));
    assert_eq!(paired.seed(73, 2), paired.seed(73, 3));
    assert_ne!(paired.seed(73, 0), paired.seed(73, 2));
}

#[test]
fn arm_distinct() {
    let shape = PoolShape::new(4, 4).unwrap();
    let seeds = std::array::from_fn::<_, 4, _>(|candidate| shape.seed(73, candidate));
    for left in 0..4 {
        for right in left + 1..4 {
            assert_ne!(seeds[left], seeds[right]);
        }
    }
}

#[test]
fn neighbor_query() {
    let estimator = NeighborValueEstimator::new(1.25, 0.04, 73).unwrap();
    let prediction = estimator.predict(&[0.0]).unwrap();
    assert_eq!(prediction.mean.to_bits(), 1.25_f32.to_bits());
    assert!(prediction.standard_error.is_finite());
    assert!(estimator.predict(&[]).is_err());
}

#[test]
fn radius_bounds() {
    let mut controller = RadiusController::new(0.1, 0.1, 2.0).unwrap();
    for _ in 0..64 {
        let decision = controller
            .observe(-1.0, 0.01, 1.0, 0.01, 0.1, true)
            .unwrap();
        assert!((0.1..=2.0).contains(&decision.next_radius));
    }
}

#[test]
fn calibration_trace() {
    let mut reference = crate::calibration::Calibration::default();
    let mut controller = RadiusController::new(0.125, 0.001, 2.0).unwrap();
    for step in 0..192 {
        let observed = ((step * 13) % 47) as f32 * 0.017 - 0.2;
        let predicted = ((step * 7) % 41) as f32 * 0.011 - 0.1;
        let variance = 0.001 + (step % 5) as f32 * 0.0003;
        let prediction = 0.002 + (step % 7) as f32 * 0.0002;
        let threshold = f64::from(reference.multiplier()) * f64::from(variance).sqrt();
        let expected = f64::from(observed) > threshold;
        let result = controller
            .observe(observed, variance, predicted, prediction, 0.1, true)
            .unwrap();
        assert_eq!(result.accepted, expected);
        assert_eq!(result.threshold.to_bits(), threshold.to_bits());
        reference
            .observe(
                observed,
                predicted,
                (f64::from(variance) + f64::from(prediction)).sqrt() as f32,
            )
            .unwrap();
        assert_eq!(
            controller.calibration.multiplier().to_bits(),
            reference.multiplier().to_bits()
        );
    }
}

#[test]
fn calibration_atomic() {
    let mut state = RadiusController::new(0.125, 0.001, 2.0).unwrap();
    let radius = state.radius();
    assert!(
        state
            .observe(f32::MAX, 0.0, -f32::MAX, 0.0, 0.1, true)
            .is_err()
    );
    assert_eq!(state.radius().to_bits(), radius.to_bits());
    assert_eq!(state.calibration.count(), 0);
}

#[test]
fn action_priority() {
    assert_eq!(
        decide_action(true, true, true, true, true, true, false),
        ReliabilityAction::Shrink
    );
    assert_eq!(
        decide_action(true, false, true, true, true, true, true),
        ReliabilityAction::Expand
    );
    assert_eq!(
        decide_action(false, false, false, false, false, false, true),
        ReliabilityAction::Shrink
    );
    assert_eq!(
        decide_action(false, false, false, true, false, false, true),
        ReliabilityAction::Hold
    );
}

#[test]
fn acceptance_margin() {
    assert!(!accept_candidate(0.0, 0.0));
    assert!(!accept_candidate(0.2, 0.2));
    assert!(accept_candidate(0.200_001, 0.2));
    assert!(!accept_candidate(-0.1, 0.2));
}

#[test]
fn bits_match() {
    let edge = [
        f64::NEG_INFINITY,
        -f64::MAX,
        -1.0,
        -f64::MIN_POSITIVE,
        -f64::from_bits(1),
        -0.0,
        0.0,
        f64::from_bits(1),
        f64::MIN_POSITIVE,
        1.0,
        f64::MAX,
        f64::INFINITY,
        f64::NAN,
        f64::from_bits(0xfff8_0000_0000_0001),
    ];
    for improvement in edge {
        for margin in edge {
            assert_eq!(
                accept_bits(improvement.to_bits(), margin.to_bits()),
                improvement > margin,
                "improvement={improvement:?} margin={margin:?}",
            );
        }
    }

    let mut left = 0x243f_6a88_85a3_08d3;
    let mut right = 0x1319_8a2e_0370_7344;
    for _ in 0..100_000 {
        left = splitmix64(left);
        right = splitmix64(right);
        let improvement = f64::from_bits(left);
        let margin = f64::from_bits(right);
        assert_eq!(accept_bits(left, right), improvement > margin);
    }
}
