//! Property tests for optimizer decisions, retention, distances, and accelerator ABI.
//! This package has no production interface; only test builds compile these properties.
#![cfg(test)]

use blckbaux::posterior::neighbor_dist::squared_l2;
use hegel::TestCase;
use hegel::generators as gs;
use ndarray::Array1;

#[hegel::test]
fn proposal_flags(tc: TestCase) {
    let count = tc.draw(gs::integers::<usize>().min_value(1).max_value(512));
    let finite = tc.draw(gs::integers::<u8>().min_value(0).max_value(1)) != 0;
    let allow = tc.draw(gs::integers::<u8>().min_value(0).max_value(1)) != 0;
    let mut flags = 0;
    let mut invalid = false;
    let mut changed = false;
    for _ in 0..count {
        let base = tc.draw(gs::integers::<u16>());
        let raw = tc.draw(gs::integers::<u16>());
        let candidate = if finite { raw & 0x7bff } else { raw };
        flags |= blckbaux_semantics::proposal::fp16_flags(base, candidate);
        invalid |= (candidate >> 10) & 31 == 31;
        changed |= base != candidate;
    }
    assert_eq!(
        blckbaux_semantics::proposal::validate(flags, allow).is_ok(),
        !invalid && (changed || allow)
    );
}

#[hegel::test]
fn reuse_trace(tc: TestCase) {
    let mut actual = blckbaux_semantics::optimizer::SeedReuse::default();
    let mut previous: Option<(u64, u32, bool)> = None;
    for step in 0..128 {
        let (seed, reused) = actual.select(73, step).unwrap();
        assert_eq!(
            (seed, reused),
            previous.map_or((73 + step, false), |(seed, _, _)| (seed, true))
        );
        let accepted = tc.draw(gs::integers::<u8>().min_value(0).max_value(1)) != 0;
        let initializing = tc.draw(gs::integers::<u8>().min_value(0).max_value(9)) == 0;
        previous = if accepted {
            if initializing {
                None
            } else {
                Some((seed, previous.map_or(1, |(_, n, _)| n + 1), false))
            }
        } else {
            match previous {
                Some((seed, n, false)) if n >= 1 && !initializing => Some((seed, n, true)),
                _ => None,
            }
        };
        actual.observe(seed, accepted, initializing);
    }
}

/// Retention protects the incumbent and evicts the oldest other identity.
/// Independently scan the full window rather than calling the rule twice.
#[hegel::test]
fn history_slot(tc: TestCase) {
    let count = tc.draw(gs::integers::<usize>().min_value(0).max_value(128));
    let rotation = tc.draw(gs::integers::<usize>().min_value(0).max_value(127));
    let identities = (0..count)
        .map(|i| ((i + rotation) % 128 + 1) as i64)
        .collect::<Vec<_>>();
    let base = if count == 0 {
        1
    } else {
        identities[tc.draw(gs::integers::<usize>().min_value(0).max_value(count - 1))]
    };
    let actual = blckbaux_semantics::neighbor::retention(&identities, base).unwrap();
    if count < 128 {
        assert_eq!(actual, count);
    } else {
        assert_ne!(identities[actual], base);
        for &identity in &identities {
            assert!(identity == base || identities[actual] <= identity);
        }
    }
    assert!(blckbaux_semantics::neighbor::retention(&[1; 129], 1).is_err());
    assert!(blckbaux_semantics::neighbor::retention(&[1; 128], 1).is_err());
}

#[hegel::test]
fn surrogate_spec(tc: TestCase) {
    let spec = blckbaux_semantics::optimizer::SurrogateSpec {
        neighbors: tc.draw(gs::integers::<usize>().min_value(1).max_value(128)),
        epistemic: tc.draw(gs::floats::<f32>().min_value(0.0).max_value(10.0)),
        aleatoric: tc.draw(gs::floats::<f32>().min_value(0.0).max_value(10.0)),
        scale: tc.draw(gs::floats::<f32>().min_value(0.0).max_value(10.0)),
        fit_candidates: tc.draw(gs::integers::<usize>().min_value(1).max_value(64)),
        fit_samples: tc.draw(gs::integers::<usize>().min_value(1).max_value(128)),
        fit: tc.draw(gs::integers::<u8>().min_value(0).max_value(1)) != 0,
        adaptive: tc.draw(gs::integers::<u8>().min_value(0).max_value(1)) != 0,
    }
    .validate()
    .unwrap();
    let json = blckbaux_artifact::json::to_string(&spec).unwrap();
    let actual: blckbaux_semantics::optimizer::SurrogateSpec =
        blckbaux_artifact::json::from_str(&json).unwrap();
    assert_eq!(actual.validate().unwrap(), spec);
    let mut invalid = spec;
    invalid.neighbors = 0;
    assert!(invalid.validate().is_err());
    invalid.neighbors = 129;
    assert!(invalid.validate().is_err());
    invalid = spec;
    invalid.scale = f32::INFINITY;
    assert!(invalid.validate().is_err());
}

/// The shared prefix scan must choose the same first risk minimizer as a
/// separately recomputed batch calculation over sorted neighbour distances.
#[hegel::test]
fn prefix_choice(tc: TestCase) {
    let count = tc.draw(gs::integers::<usize>().min_value(1).max_value(32));
    let epistemic = tc.draw(gs::floats::<f32>().min_value(0.01).max_value(2.0));
    let mut points = (0..count)
        .map(|_| {
            (
                tc.draw(gs::floats::<f32>().min_value(0.0).max_value(4.0)),
                tc.draw(gs::floats::<f32>().min_value(0.01).max_value(2.0)),
            )
        })
        .collect::<Vec<_>>();
    points.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut scan = blckbaux::neighbor_rule::Prefix::default();
    let mut risks = Vec::new();
    for (index, &(distance, noise)) in points.iter().enumerate() {
        scan.push(index as u32 + 1, distance, epistemic, noise);
        let weights = points[..=index]
            .iter()
            .map(|&(d, n)| 1.0 / (1.0e-9 + epistemic * d + n).max(1.0e-12))
            .collect::<Vec<f32>>();
        let total = weights.iter().copied().sum::<f32>();
        let noise = weights
            .iter()
            .zip(&points)
            .map(|(&weight, &(_, noise))| weight * weight * noise)
            .sum::<f32>();
        risks.push(epistemic * distance + (noise / (total * total).max(1.0e-24)).sqrt());
    }
    let expected = risks
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;
    assert_eq!(scan.count as usize, expected + 1);
}

/// Recurrence has one shared ordering: prefix, repeated core, suffix. This
/// checks arbitrary valid layer spans, not only the current model constants.
#[hegel::test]
fn recurrence_trace(tc: TestCase) {
    use blckbaux_semantics::recurrence::RecurrentCore;
    let layers = tc.draw(gs::integers::<usize>().min_value(1).max_value(32));
    let first = tc.draw(gs::integers::<usize>().min_value(0).max_value(layers - 1));
    let count = tc.draw(
        gs::integers::<usize>()
            .min_value(1)
            .max_value(layers - first),
    );
    let visits = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let core = RecurrentCore {
        first_layer: first,
        layer_count: count,
        training_visits: 1,
        min_inference_visits: 1,
        max_inference_visits: 8,
    };
    let actual = core.layer_visits(layers, visits).unwrap();
    let expected = (0..first)
        .chain((0..visits).flat_map(|_| first..first + count))
        .chain(first + count..layers)
        .collect::<Vec<_>>();
    assert_eq!(actual.iter().map(|v| v.layer).collect::<Vec<_>>(), expected);
    for (index, visit) in actual.iter().enumerate() {
        assert_eq!(visit.execution, index);
        assert_eq!(
            visit.recurrent,
            (first..first + count).contains(&visit.layer)
        );
    }
    assert!(core.layer_visits(layers, 0).is_err());
    assert!(core.layer_visits(layers, 9).is_err());
}

/// Known integer fixtures protect stream identity independently of aliases.
#[hegel::test]
fn stream_identity(tc: TestCase) {
    let seed = tc.draw(gs::integers::<u64>());
    assert_eq!(
        blckbaux::hash::splitmix64(seed),
        blckbaux_semantics::optimizer::splitmix64(seed)
    );
    assert_eq!(blckbaux::hash::splitmix64(0), 0xe220_a839_7b1d_cdaf);
    assert_eq!(blckbaux::hash::splitmix64(1), 0x910a_2dec_8902_5cc1);
    assert_eq!(blckbaux::hash::splitmix64(u64::MAX), 0xe4d9_7177_1b65_2c20);
}

/// Compiling generated declarations checks every Metal field offset. Reading
/// sentinel fields on the GPU also catches order/type mistakes in the ABI.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[hegel::test]
fn wave_abi(tc: TestCase) {
    use blckbaux_semantics::hnet::{EncodeShape, ShardStats, WaveShape};
    use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
    let wave = WaveShape {
        blocks: tc.draw(gs::integers::<u32>()),
        tokens: tc.draw(gs::integers::<u32>()),
        proposal_start: tc.draw(gs::integers::<u32>()),
        intra: tc.draw(gs::integers::<u32>()),
        vocabulary: tc.draw(gs::integers::<u32>()),
        shards: tc.draw(gs::integers::<u32>()),
        temperature: tc.draw(gs::floats::<f32>().min_value(0.0).max_value(4.0)),
        score: tc.draw(gs::integers::<u32>()),
    };
    let shard = ShardStats {
        maximum: 1.0,
        sum: 2.0,
        target: 3.0,
        best: 4.0,
        token: tc.draw(gs::integers::<u32>()),
        reserved: tc.draw(gs::integers::<u32>()),
    };
    let encode = EncodeShape {
        rows: tc.draw(gs::integers::<u32>()),
        width: tc.draw(gs::integers::<u32>()),
        vocabulary: tc.draw(gs::integers::<u32>()),
        patch: tc.draw(gs::integers::<u32>()),
    };
    let source = blckbaux_semantics::hnet::metal_source(
        r#"
kernel void abi_read(constant WaveShape& w [[buffer(0)]],
                     constant ShardStats& s [[buffer(1)]],
                     device uint* output [[buffer(2)]],
                     constant EncodeShape& e [[buffer(3)]]) {
    output[0]=w.blocks; output[1]=w.tokens; output[2]=w.proposal_start;
    output[3]=w.intra; output[4]=w.vocabulary; output[5]=w.shards;
    output[6]=as_type<uint>(w.temperature); output[7]=w.score;
    output[8]=as_type<uint>(s.maximum); output[9]=as_type<uint>(s.sum);
    output[10]=as_type<uint>(s.target); output[11]=as_type<uint>(s.best);
    output[12]=s.token; output[13]=s.reserved;
    output[14]=e.rows; output[15]=e.width; output[16]=e.vocabulary; output[17]=e.patch;
}
"#,
    );
    static GPU: std::sync::OnceLock<(Device, metal::ComputePipelineState, metal::CommandQueue)> =
        std::sync::OnceLock::new();
    let (device, pipeline, queue) = GPU.get_or_init(|| {
        let device = Device::system_default().expect("Metal device required");
        let library = device
            .new_library_with_source(&source, &CompileOptions::new())
            .unwrap();
        let function = library.get_function("abi_read", None).unwrap();
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .unwrap();
        let queue = device.new_command_queue();
        (device, pipeline, queue)
    });
    let output = device.new_buffer(18 * 4, MTLResourceOptions::StorageModeShared);
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_bytes(
        0,
        size_of::<WaveShape>() as u64,
        std::ptr::from_ref(&wave).cast(),
    );
    encoder.set_bytes(
        1,
        size_of::<ShardStats>() as u64,
        std::ptr::from_ref(&shard).cast(),
    );
    encoder.set_buffer(2, Some(&output), 0);
    encoder.set_bytes(
        3,
        size_of::<EncodeShape>() as u64,
        std::ptr::from_ref(&encode).cast(),
    );
    let one = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_threads(one, one);
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), metal::MTLCommandBufferStatus::Completed);
    // SAFETY: completed command initialized all 18 u32 values in shared storage.
    let actual = unsafe { std::slice::from_raw_parts(output.contents().cast::<u32>(), 18) };
    assert_eq!(
        actual,
        &[
            wave.blocks,
            wave.tokens,
            wave.proposal_start,
            wave.intra,
            wave.vocabulary,
            wave.shards,
            wave.temperature.to_bits(),
            wave.score,
            shard.maximum.to_bits(),
            shard.sum.to_bits(),
            shard.target.to_bits(),
            shard.best.to_bits(),
            shard.token,
            shard.reserved,
            encode.rows,
            encode.width,
            encode.vocabulary,
            encode.patch,
        ]
    );
}

/// Both public adapters must execute the same policy over an adaptive trace,
/// including configured failure budgets and non-default evidence parameters.
#[hegel::test]
fn controller_parity(tc: TestCase) {
    let policy = blckbaux::ReliabilityPolicy {
        evidence_decay: tc.draw(gs::floats::<f64>().min_value(0.0).max_value(1.0)),
        confidence_z: tc.draw(gs::floats::<f64>().min_value(0.25).max_value(4.0)),
        reliability_threshold: tc.draw(gs::floats::<f64>().min_value(0.01).max_value(0.99)),
        minimum_evidence: tc.draw(gs::floats::<f64>().min_value(0.1).max_value(8.0)),
        acceptance_ratio: tc.draw(gs::floats::<f64>().min_value(0.01).max_value(2.0)),
        boundary_fraction: tc.draw(gs::floats::<f64>().min_value(0.0).max_value(1.0)),
        expand_factor: tc.draw(gs::floats::<f64>().min_value(1.25).max_value(4.0)),
        shrink_factor: tc.draw(gs::floats::<f64>().min_value(0.125).max_value(0.9)),
        failure_tolerance: tc.draw(gs::integers::<usize>().min_value(0).max_value(8)),
    };
    let encoded = blckbaux_artifact::json::to_string(&policy).unwrap();
    let decoded: blckbaux::ReliabilityPolicy = blckbaux_artifact::json::from_str(&encoded).unwrap();
    assert_eq!(policy, decoded);
    let mut host = blckbaux_semantics::optimizer::RadiusController::configured(
        0.125,
        1.0 / 65536.0,
        2.0,
        policy,
    )
    .unwrap();
    let mut metal = blckbaux::ReliabilityController::new(
        policy,
        blckbaux::trust_region::TRLengthConfig::new(0.125, 1.0 / 65536.0, 2.0),
    )
    .unwrap();
    let steps = tc.draw(gs::integers::<usize>().min_value(1).max_value(48));
    for _ in 0..steps {
        let observed = tc.draw(gs::floats::<f32>().min_value(-2.0).max_value(2.0));
        let predicted = tc.draw(gs::floats::<f32>().min_value(-2.0).max_value(2.0));
        let variance = tc.draw(gs::floats::<f32>().min_value(0.0).max_value(1.0));
        let prediction = tc.draw(gs::floats::<f32>().min_value(0.0).max_value(1.0));
        let response = tc.draw(gs::floats::<f32>().min_value(0.0).max_value(1.0));
        let nominal = f64::from(host.radius());
        let actual = host
            .observe(observed, variance, predicted, prediction, response, false)
            .unwrap();
        let expected = metal
            .update(blckbaux::ReliabilityEvidence {
                observed_improvement: f64::from(observed),
                observation_variance: f64::from(variance),
                predicted_improvement: f64::from(predicted),
                prediction_variance: f64::from(prediction),
                accepted: actual.accepted,
                nominal_radius: nominal,
                response_radius: f64::from(response),
            })
            .unwrap();
        assert_eq!(actual.action, expected.action);
        assert_eq!(
            actual.next_radius.to_bits(),
            (expected.length as f32).to_bits()
        );
        assert_eq!(
            actual.standardized_error.to_bits(),
            expected.standardized_error.unwrap().to_bits()
        );
        assert_eq!(
            actual.sharpness_ratio.to_bits(),
            expected.sharpness_ratio.unwrap().to_bits()
        );
        assert_eq!(
            actual.agreement_ratio.map(f64::to_bits),
            expected.agreement_ratio.map(f64::to_bits)
        );
        assert!((1.0 / 65536.0..=2.0).contains(&actual.next_radius));
    }
}

/// An accepted observation resets consecutive rejection evidence. With the
/// adequacy gate inactive, contraction occurs at exactly the declared budget.
#[hegel::test]
fn failure_reset(tc: TestCase) {
    let budget = tc.draw(gs::integers::<usize>().min_value(1).max_value(32));
    let policy = blckbaux::ReliabilityPolicy {
        failure_tolerance: budget,
        minimum_evidence: f64::MAX,
        ..Default::default()
    };
    let mut state =
        blckbaux_semantics::optimizer::RadiusController::configured(0.125, 0.001, 2.0, policy)
            .unwrap();
    for _ in 0..budget - 1 {
        assert_eq!(
            state
                .observe(0.0, 0.0, 0.0, 0.0, 0.0, false)
                .unwrap()
                .action,
            blckbaux::ReliabilityAction::Hold
        );
    }
    assert!(
        state
            .observe(1.0, 0.0, 1.0, 0.0, 1.0, false)
            .unwrap()
            .accepted
    );
    for step in 1..=budget {
        let result = state.observe(0.0, 0.0, 0.0, 0.0, 0.0, false).unwrap();
        let expected = if step == budget {
            blckbaux::ReliabilityAction::Shrink
        } else {
            blckbaux::ReliabilityAction::Hold
        };
        assert_eq!(result.action, expected);
    }
    assert_eq!(state.radius(), 0.0625);
}

/// Invalid evidence must not consume a failure or change the next transition.
#[hegel::test]
fn invalid_atomic(tc: TestCase) {
    let mut state =
        blckbaux_semantics::optimizer::RadiusControl::new(Default::default(), 0.125, 0.001, 2.0)
            .unwrap();
    let valid = blckbaux::ReliabilityEvidence {
        observed_improvement: tc.draw(gs::floats::<f64>().min_value(-1.0).max_value(1.0)),
        observation_variance: 0.01,
        predicted_improvement: 0.0,
        prediction_variance: 0.01,
        accepted: false,
        nominal_radius: 0.125,
        response_radius: 0.1,
    };
    state.update(valid).unwrap();
    let mut reference = state.clone();
    let mut invalid = valid;
    match tc.draw(gs::integers::<u8>().min_value(0).max_value(5)) {
        0 => invalid.observed_improvement = f64::NAN,
        1 => invalid.observation_variance = -1.0,
        2 => invalid.predicted_improvement = f64::INFINITY,
        3 => invalid.prediction_variance = -1.0,
        4 => invalid.nominal_radius = 0.0,
        _ => invalid.response_radius = -1.0,
    }
    assert!(state.update(invalid).is_err());
    assert_eq!(
        state.update(valid).unwrap(),
        reference.update(valid).unwrap()
    );
}

#[hegel::test]
fn native_order(tc: TestCase) {
    let left = tc.draw(gs::integers::<u64>());
    let right = tc.draw(gs::integers::<u64>());
    assert_eq!(
        blckbaux_semantics::optimizer::accept_bits(left, right),
        f64::from_bits(left) > f64::from_bits(right)
    );
}

#[hegel::test]
fn test_identity(tc: TestCase) {
    let dim = tc.draw(gs::integers::<usize>().min_value(1).max_value(16));
    let vals: Vec<f64> = (0..dim)
        .map(|_| tc.draw(gs::floats::<f64>().min_value(-100.0).max_value(100.0)))
        .collect();
    let scales: Vec<f64> = (0..dim)
        .map(|_| tc.draw(gs::floats::<f64>().min_value(0.1).max_value(10.0)))
        .collect();
    let x = Array1::from_vec(vals);
    let s = Array1::from_vec(scales);
    let d = squared_l2(x.view(), x.view(), true, s.view());
    assert!(d.abs() < 1e-10, "self distance must be zero: {d}");
}

#[hegel::test]
fn test_symmetry(tc: TestCase) {
    let dim = tc.draw(gs::integers::<usize>().min_value(1).max_value(16));
    let x_vals: Vec<f64> = (0..dim)
        .map(|_| tc.draw(gs::floats::<f64>().min_value(-100.0).max_value(100.0)))
        .collect();
    let y_vals: Vec<f64> = (0..dim)
        .map(|_| tc.draw(gs::floats::<f64>().min_value(-100.0).max_value(100.0)))
        .collect();
    let scales: Vec<f64> = (0..dim)
        .map(|_| tc.draw(gs::floats::<f64>().min_value(0.1).max_value(10.0)))
        .collect();
    let x = Array1::from_vec(x_vals);
    let y = Array1::from_vec(y_vals);
    let s = Array1::from_vec(scales);
    let d_xy = squared_l2(x.view(), y.view(), true, s.view());
    let d_yx = squared_l2(y.view(), x.view(), true, s.view());
    assert!(
        (d_xy - d_yx).abs() < 1e-10,
        "distance must be symmetric: {d_xy} != {d_yx}"
    );
}

#[hegel::test]
fn test_positivity(tc: TestCase) {
    let dim = tc.draw(gs::integers::<usize>().min_value(1).max_value(16));
    let x_vals: Vec<f64> = (0..dim)
        .map(|_| tc.draw(gs::floats::<f64>().min_value(-100.0).max_value(100.0)))
        .collect();
    let y_vals: Vec<f64> = (0..dim)
        .map(|_| tc.draw(gs::floats::<f64>().min_value(-100.0).max_value(100.0)))
        .collect();
    let scales: Vec<f64> = (0..dim)
        .map(|_| tc.draw(gs::floats::<f64>().min_value(0.1).max_value(10.0)))
        .collect();
    let x = Array1::from_vec(x_vals);
    let y = Array1::from_vec(y_vals);
    let s = Array1::from_vec(scales);
    let d = squared_l2(x.view(), y.view(), true, s.view());
    assert!(d >= 0.0, "squared distance must be non-negative: {d}");
}
