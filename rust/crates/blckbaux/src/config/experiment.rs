use deser::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum TrainingOptimizer {
    #[default]
    ResidentEnnV1,
    OperatorBo,
    MuonV1,
    AdamwV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum PretrainSelection {
    #[default]
    Enn,
    Random,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum DistanceScaling {
    #[default]
    Global,
    #[deser(rename = "self-tuning")]
    SelfTuning,
}

impl DistanceScaling {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::SelfTuning => "self-tuning",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum HistoryGeometry {
    Realized,
    #[default]
    Latent,
}

impl HistoryGeometry {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Realized => "realized",
            Self::Latent => "latent",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum TrustRegionShape {
    Scalar,
    TensorFamilyStatic,
    TensorFamilyLearned,
    ActiveRotate,
    ActiveRandom,
    ActiveExperts,
    ActiveProjections,
    ActiveRouter,
    ActiveNorms,
}

impl TrustRegionShape {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::TensorFamilyStatic => "tensor-family-static",
            Self::TensorFamilyLearned => "tensor-family-learned",
            Self::ActiveRotate => "active-rotate",
            Self::ActiveRandom => "active-random",
            Self::ActiveExperts => "active-experts",
            Self::ActiveProjections => "active-projections",
            Self::ActiveRouter => "active-router",
            Self::ActiveNorms => "active-norms",
        }
    }

    pub const fn is_active(self) -> bool {
        matches!(
            self,
            Self::ActiveRotate
                | Self::ActiveRandom
                | Self::ActiveExperts
                | Self::ActiveProjections
                | Self::ActiveRouter
                | Self::ActiveNorms
        )
    }

    pub fn active_group(self, step: u32, seed: u64) -> Option<usize> {
        match self {
            Self::Scalar | Self::TensorFamilyStatic | Self::TensorFamilyLearned => None,
            Self::ActiveRotate => Some((step as usize) % 4),
            Self::ActiveRandom => {
                Some((crate::hash::splitmix64(seed ^ u64::from(step)) as usize) % 4)
            }
            Self::ActiveExperts => Some(0),
            Self::ActiveProjections => Some(1),
            Self::ActiveRouter => Some(2),
            Self::ActiveNorms => Some(3),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum ObjectiveReference {
    #[default]
    MovingIncumbent,
    FrozenInitial,
}

impl ObjectiveReference {
    pub const fn name(self) -> &'static str {
        match self {
            Self::MovingIncumbent => "moving-incumbent",
            Self::FrozenInitial => "frozen-initial",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_names() {
        assert_eq!(TrustRegionShape::ActiveRotate.name(), "active-rotate");
        assert_eq!(TrustRegionShape::ActiveRandom.name(), "active-random");
        assert!(TrustRegionShape::ActiveRotate.is_active());
        assert!(!TrustRegionShape::Scalar.is_active());
        assert_eq!(TrustRegionShape::ActiveRotate.active_group(0, 42), Some(0));
        assert_eq!(TrustRegionShape::ActiveRotate.active_group(1, 42), Some(1));
        assert_eq!(TrustRegionShape::ActiveRotate.active_group(2, 42), Some(2));
        assert_eq!(TrustRegionShape::ActiveRotate.active_group(3, 42), Some(3));
        assert_eq!(TrustRegionShape::ActiveRotate.active_group(4, 42), Some(0));
    }
}
