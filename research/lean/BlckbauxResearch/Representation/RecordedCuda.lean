import BlckbauxResearch.Representation.Blocks
import BlckbauxResearch.Optimization.ResearchProblem

/-!
# Recorded CUDA coefficient representation

The completed T4 run recorded one coefficient for each of 70 resident tensor
buffers while holding each buffer's dense pseudorandom field fixed.  This file
states the resulting real-valued rank consequence conditionally on the audited
Rust/CUDA map being represented by `BlockBasis`.

It is not a Rust refinement proof and it does not make a claim about the extra
states introduced by FP16 rounding.  Those remain separate obligations.
-/

namespace BlckbauxResearch.Representation.RecordedCuda

open BlckbauxResearch.Optimization.ResearchProblem

/-- Buffer-coefficient count recorded by the completed T4 experiment. -/
def blocks : ℕ := 70

/-- Source-audited proposal counts for the completed 10,000-round T4 run. -/
def searchCount : SearchCount where
  scored := 40_000
  materialized := 10_000
  evaluated := 10_000
  rounds := 10_000
  materializedLe := by norm_num
  evaluatedLe := by norm_num
  roundsLe := by norm_num

/-- Baseline plus paired incumbent and candidate evaluations in every round. -/
def objectiveQueries : ℕ := 20_001

/-- The recorded candidate search was fully sequential, not a 10,000-item batch. -/
theorem wasSequential : searchCount.sequential := by
  norm_num [SearchCount.sequential, searchCount]

/-- Candidate evaluations are only part of the total objective-query count. -/
theorem queriesCover : searchCount.evaluated ≤ objectiveQueries := by
  norm_num [searchCount, objectiveQueries]

/-- Measured mean complete round wall time in milliseconds. -/
def roundMs : ℝ := 1202.43794

/-- The recorded mean exceeded the 1000 ms ceiling. -/
theorem missesCeiling : (currentTarget.ceilingMs : ℝ) < roundMs := by
  norm_num [currentTarget, roundMs]

/-- The recorded mean exceeded six times the 200 ms operating target. -/
theorem overSix : 6 * (currentTarget.targetMs : ℝ) < roundMs := by
  norm_num [currentTarget, roundMs]

/-- The latency value is recorded evidence, not a universal theorem about CUDA. -/
def latencyClaim : ResearchClaim where
  statement :=
    "The completed 10,000-round T4 run measured 1202.43794 ms mean complete round wall time."
  proposition := roundMs = 1202.43794
  status := .measurement

/-- The audited fixed-field proposal span has rank at most 70. -/
theorem spanRank
    (basis : BlockBasis currentTarget.parameterCount blocks) :
    Module.finrank ℝ (Submodule.span ℝ (Set.range basis.field)) ≤ 70 := by
  simpa [blocks] using basis.spanRank

/-- The fixed 70-field span has strictly smaller rank than the full proposal space. -/
theorem rankGap
    (basis : BlockBasis currentTarget.parameterCount blocks) :
    Module.finrank ℝ (Submodule.span ℝ (Set.range basis.field)) <
      Module.finrank ℝ (Parameter.Proposal currentTarget.parameterCount) := by
  calc
    Module.finrank ℝ (Submodule.span ℝ (Set.range basis.field)) ≤ 70 :=
      spanRank basis
    _ < 1_177_756_440 := by norm_num
    _ = Module.finrank ℝ (Parameter.Proposal currentTarget.parameterCount) := by
      exact Proposal.currentRank.symm

end BlckbauxResearch.Representation.RecordedCuda
