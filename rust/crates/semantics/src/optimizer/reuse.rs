//! Successful proposal-root reuse in the causal pretraining loop.
//! Retain a successful root, permit one retry after rejection, then regenerate.
//! This policy does not claim that repeated directions improve the objective.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SeedReuse {
    retained: Option<u64>,
    retried: bool,
}

impl SeedReuse {
    pub fn select(&self, root: u64, round: u64) -> Result<(u64, bool), &'static str> {
        match self.retained {
            Some(seed) => Ok((seed, true)),
            None => root
                .checked_add(round)
                .map(|seed| (seed, false))
                .ok_or("proposal seed overflow"),
        }
    }

    pub fn observe(&mut self, seed: u64, accepted: bool, initializing: bool) {
        if initializing {
            *self = Self::default();
        } else if accepted {
            self.retained = Some(seed);
            self.retried = false;
        } else if self.retained.is_some() && !self.retried {
            self.retried = true;
        } else {
            *self = Self::default();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transitions() {
        let mut state = SeedReuse::default();
        assert_eq!(state.select(73, 0).unwrap(), (73, false));
        state.observe(73, true, true);
        assert_eq!(state.select(73, 1).unwrap(), (74, false));
        state.observe(74, true, false);
        assert_eq!(state.select(73, 2).unwrap(), (74, true));
        state.observe(74, false, false);
        assert_eq!(state.select(73, 3).unwrap(), (74, true));
        state.observe(74, false, false);
        assert_eq!(state.select(73, 4).unwrap(), (77, false));
        assert!(state.select(u64::MAX, 1).is_err());
    }
}
