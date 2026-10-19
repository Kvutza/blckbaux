//! A scalar value prediction with its standard error.

#[derive(Clone, Copy, Debug, Default)]
pub struct ValuePrediction {
    pub mean: f32,
    pub standard_error: f32,
}
