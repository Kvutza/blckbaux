//! Factory functions for creating optimizers with preset configs.

use ndarray::Array2;
use rand::RngCore;

use crate::config::{InitStrategy, RunConfig, SurrogateConfig, lhd_only, turbo_enn, turbo_zero};
use crate::error::ENNError;
use crate::optimizer::BayesOptimizer;
use crate::strategy::Strategy;

/// Create an optimizer for TuRBO-ENN.
pub fn enn_optimizer(
    bounds: Array2<f64>,
    k: i32,
    num_init: usize,
    rng: &mut dyn RngCore,
) -> Result<BayesOptimizer, ENNError> {
    enn_configured(bounds, k, num_init, rng, None)
}

/// Create an experimental multi-trust-region optimizer for TuRBO-ENN.
pub fn enn_tr(
    bounds: Array2<f64>,
    k: i32,
    num_init: usize,
    num_regions: usize,
    rng: &mut dyn RngCore,
) -> Result<BayesOptimizer, ENNError> {
    region_configured(bounds, k, num_init, num_regions, rng, None)
}

/// Create TuRBO-ENN from an optional resolved runtime configuration.
pub fn enn_configured(
    bounds: Array2<f64>,
    k: i32,
    num_init: usize,
    rng: &mut dyn RngCore,
    settings: Option<&RunConfig>,
) -> Result<BayesOptimizer, ENNError> {
    let mut config = turbo_enn();
    if let SurrogateConfig::ENN(enn_cfg) = &mut config.surrogate {
        enn_cfg.k = k;
    }
    if let Some(settings) = settings {
        config = settings.apply_to(config);
    }
    let strategy = Strategy::hybrid(InitStrategy::LHD, num_init);
    BayesOptimizer::new_strategy(bounds, config, strategy, rng)
}

/// Create experimental multi-trust-region TuRBO-ENN from resolved settings.
pub fn region_configured(
    bounds: Array2<f64>,
    k: i32,
    num_init: usize,
    num_regions: usize,
    rng: &mut dyn RngCore,
    settings: Option<&RunConfig>,
) -> Result<BayesOptimizer, ENNError> {
    let mut config = turbo_enn();
    if let SurrogateConfig::ENN(enn_cfg) = &mut config.surrogate {
        enn_cfg.k = k;
    }
    if let Some(settings) = settings {
        config = settings.apply_to(config);
    }
    let strategy = Strategy::experimental(bounds.nrows(), num_regions, num_init, rng)?;
    BayesOptimizer::new_strategy(bounds, config, strategy, rng)
}

/// Create an optimizer for TuRBO-ZERO.
pub fn create_optimizer(
    bounds: Array2<f64>,
    num_init: usize,
    rng: &mut dyn RngCore,
) -> Result<BayesOptimizer, ENNError> {
    create_configured(bounds, num_init, rng, None)
}

/// Create TuRBO-ZERO from an optional resolved runtime configuration.
pub fn create_configured(
    bounds: Array2<f64>,
    num_init: usize,
    rng: &mut dyn RngCore,
    settings: Option<&RunConfig>,
) -> Result<BayesOptimizer, ENNError> {
    let mut config = turbo_zero();
    if let Some(settings) = settings {
        config = settings.apply_to(config);
    }
    let strategy = Strategy::hybrid(InitStrategy::LHD, num_init);
    BayesOptimizer::new_strategy(bounds, config, strategy, rng)
}

/// Create an optimizer for LHD-only.
pub fn create_lhd(
    bounds: Array2<f64>,
    num_init: usize,
    rng: &mut dyn RngCore,
) -> Result<BayesOptimizer, ENNError> {
    lhd_configured(bounds, num_init, rng, None)
}

/// Create LHD-only from an optional resolved runtime configuration.
pub fn lhd_configured(
    bounds: Array2<f64>,
    num_init: usize,
    rng: &mut dyn RngCore,
    settings: Option<&RunConfig>,
) -> Result<BayesOptimizer, ENNError> {
    let mut config = lhd_only();
    if let Some(settings) = settings {
        config = settings.apply_to(config);
    }
    let strategy = Strategy::init(InitStrategy::LHD, num_init);
    BayesOptimizer::new_strategy(bounds, config, strategy, rng)
}
