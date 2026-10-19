//! Mathematical and execution semantics shared by every BLCKBAUX adapter.
//!
//! No backend owns these types. Schedules describe implementations, not proofs
//! of numerical equivalence. CUDA enables only scalar ABI trait implementations;
//! the default build has no dependencies and does not require a GPU toolchain.

mod abi;
pub mod context;
pub mod gaussian;
pub mod graph;
pub mod hnet;
pub mod model;
pub mod muon;
pub mod neighbor;
pub mod optimizer;
pub use optimizer::*;
pub mod proposal;
pub mod random;
pub mod recurrence;
pub mod schedule;
pub mod shape;
pub mod tensor;
