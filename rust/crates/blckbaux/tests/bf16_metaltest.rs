//! Compile and exercise the standalone backend without changing library registrations.
#![cfg(all(target_os = "macos", feature = "metal"))]
#![allow(dead_code)]

#[path = "../src/apple_gpu.rs"]
mod apple_gpu;
#[path = "../src/bf16_metal.rs"]
mod bf16_metal;
#[path = "../src/objective_observation.rs"]
mod objective_observation;
use blckbaux::procedural_pool;

pub use blckbaux::{
    ProposalKernel, Rescalarize, calibration, config, fit, fitter, hash, hypervolume, mbtrregn,
    params, reliability_region, response_observation, tensor_store, threshold, trust_region,
};
mod trials {
    pub use blckbaux::experimental::RoundConfig;
}
mod weights {
    pub use blckbaux::experimental::AcquisitionKind;
}
