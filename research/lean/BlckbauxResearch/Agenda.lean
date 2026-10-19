import BlckbauxResearch.Protocol

/-!
The initial dialogue is a finite research record.  Its numbered moves and
evidence locators refer to claims and replies, not theorem premises silently
available to an optimizer proof.
-/

namespace BlckbauxResearch.Agenda

open BlckbauxResearch

def initialDialogue : Except ProtocolError Dialogue := do
  let d1 ← Dialogue.empty.append {
    author := .first
    kind := .question
    target := none
    statement := "How should Lean participate in the BLCKBAUX optimizer research loop?"
  }
  let d2 ← d1.append {
    author := .first
    kind := .proposal
    target := some 0
    statement := "Encode the two-author scientific dialogue before selecting an optimizer theorem."
    evidence := some {
      kind := .conjecture
      locator := "BLCKBAUX research dialogue, 2026-10-12"
    }
  }
  let d3 ← d2.append {
    author := .second
    kind := .paraphrase
    target := some 1
    statement := "The communication protocol itself must become a checked research object."
  }
  let d4 ← d3.append {
    author := .first
    kind := .confirm
    target := some 2
    statement := "Confirmed; implement the dialogue protocol and then use it for the research agenda."
  }
  let d5 ← d4.append {
    author := .first
    kind := .question
    target := some 1
    statement := "Which unresolved property of the current optimizer should Lean investigate first?"
  }
  let d6 ← d5.append {
    author := .second
    kind := .proposal
    target := some 4
    statement := "Formalize Epistemic Nearest Neighbors surrogate error and trajectory total-variation bounds."
    evidence := some {
      kind := .leanProof
      locator := "BlckbauxResearch.Optimization.Surrogate and Trajectory"
    }
  }
  let d7 ← d6.append {
    author := .first
    kind := .paraphrase
    target := some 5
    statement := "The BO algorithm foundations require Epistemic Nearest Neighbors precision weighting and trajectory-level trust regions."
  }
  let d8 ← d7.append {
    author := .second
    kind := .confirm
    target := some 6
    statement := "Confirmed; investigate Epistemic Nearest Neighbors and trajectory trust regions."
  }
  let d9 ← d8.append {
    author := .first
    kind := .question
    target := some 5
    statement := "How should trust-region length adaptation be derived when black-box evaluations are noisy?"
  }
  let d10 ← d9.append {
    author := .second
    kind := .proposal
    target := some 8
    statement := "Decouple step acceptance from radius adaptation through Beta-discounted model-agreement evidence."
    evidence := some {
      kind := .leanProof
      locator := "BlckbauxResearch.Optimization.Reliability"
    }
  }
  let d11 ← d10.append {
    author := .first
    kind := .paraphrase
    target := some 9
    statement := "Length adaptation must track surrogate calibration rather than noisy candidate progress, preventing spurious collapse."
  }
  let d12 ← d11.append {
    author := .second
    kind := .confirm
    target := some 10
    statement := "Confirmed; decouple candidate acceptance from trust-region adaptation."
  }
  let d13 ← d12.append {
    author := .first
    kind := .question
    target := some 9
    statement := "How does operator perturbation theory connect to Epistemic Nearest Neighbors surrogate uncertainty?"
  }
  let d14 ← d13.append {
    author := .second
    kind := .proposal
    target := some 12
    statement := "Null perturbations on reached states have zero operator response, while neighbor precision growth strictly contracts epistemic variance."
    evidence := some {
      kind := .leanProof
      locator := "BlckbauxResearch.Optimization.Operator and Surrogate"
    }
  }
  let d15 ← d14.append {
    author := .first
    kind := .paraphrase
    target := some 13
    statement := "Perturbations outside reached states produce zero objective change, requiring ENN distance to measure operator response rather than Euclidean weight coordinates."
  }
  d15.append {
    author := .second
    kind := .confirm
    target := some 14
    statement := "Confirmed; retain the checked operator and variance lemmas without treating them as an optimizer guarantee."
  }

def initialAccepted : Bool :=
  match initialDialogue with
  | .ok dialogue =>
      dialogue.moves.length == 16 && dialogue.targetsPrecede &&
        dialogue.meaningConfirmed 1 && dialogue.meaningConfirmed 5 &&
        dialogue.meaningConfirmed 9 && dialogue.meaningConfirmed 13
  | .error _ => false

example : initialAccepted = true := by native_decide

def orphanInvalid : Except ProtocolError Dialogue :=
  Dialogue.empty.append {
    author := .second
    kind := .challenge
    target := some 9
    statement := "This challenge has no earlier claim."
    evidence := some {
      kind := .derivation
      locator := "none"
    }
  }

def orphanRejected : Bool :=
  match orphanInvalid with
  | .error (.targetDoesNotPrecede 9 0) => true
  | _ => false

example : orphanRejected = true := by native_decide

end BlckbauxResearch.Agenda
