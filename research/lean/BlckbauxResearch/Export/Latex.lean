import Lean
import Mathlib.Data.Real.Basic
import BlckbauxResearch.Optimization.Operator
import BlckbauxResearch.Optimization.Perturbation

/-!
# Formal Metaprogramming and LaTeX Pretty-Printer (`LeanTeX` Engine)

This module provides a direct, machine-checked bridge from formal Lean 4
kernel expressions (`Lean.Expr`) to publication-grade LaTeX mathematical notation.
Inspired by Kyle Miller's `LeanTeX` architecture (`kmill/LeanTeX`), it defines:
1. `BindingPower` and `LatexData` for operator precedence-aware parenthesization.
2. An expression translator `latexPP` mapping kernel ASTs to standard mathematical symbols.
3. User commands `#latex` and `#latex_type` for theorem reflection.
-/

namespace BlckbauxResearch.Export

open Lean Meta Elab

/-- Binding power for precedence and parenthesization. -/
inductive BindingPower where
  | min : BindingPower
  | assoc (bp : Nat) : BindingPower
  | nonAssoc (bp : Nat) : BindingPower
  | max : BindingPower
deriving Repr, Inhabited, DecidableEq

/-- Safely checks whether an inner operator with `inner` BP needs parentheses inside an `outer` BP. -/
def BindingPower.needsParens (inner : BindingPower) (outer : Nat) : Bool :=
  match inner with
  | .max => false
  | .min => true
  | .nonAssoc bp => bp ≤ outer
  | .assoc bp => bp < outer

/-- Structural representation of pretty-printed LaTeX code with left and right binding powers. -/
structure LatexData where
  latex : String
  lbp : BindingPower := .max
  rbp : BindingPower := .max
deriving Repr, Inhabited

namespace LatexData

/-- Wraps an expression in `\left( ... \right)` if its binding power is weaker than `outerBp`. -/
def protect (d : LatexData) (outerBp : Nat) : LatexData :=
  if d.lbp.needsParens outerBp || d.rbp.needsParens outerBp then
    { latex := "\\left(" ++ d.latex ++ "\\right)", lbp := .max, rbp := .max }
  else
    d

/-- Construct an atomic LaTeX atom (never needs external parentheses on its own). -/
def atom (s : String) : LatexData :=
  { latex := s, lbp := .max, rbp := .max }

/-- Infix binary operator formatting with left and right protection. -/
def infixOp (op : String) (bp : Nat) (left right : LatexData) : LatexData :=
  let l := left.protect bp
  let r := right.protect (bp + 1)
  { latex := l.latex ++ " " ++ op ++ " " ++ r.latex, lbp := .assoc bp, rbp := .assoc bp }

/-- Prefix unary operator formatting. -/
def prefixOp (op : String) (bp : Nat) (arg : LatexData) : LatexData :=
  let a := arg.protect bp
  { latex := op ++ a.latex, lbp := .assoc bp, rbp := .assoc bp }

end LatexData

/-- Core recursive pretty-printer converting a Lean `Expr` into `LatexData`. -/
partial def latexPP (e : Expr) : MetaM LatexData := do
  let e ← instantiateMVars e
  match e with
  | .bvar idx =>
      return LatexData.atom ("x_{" ++ toString idx ++ "}")
  | .fvar fvarId => do
      let decl ← fvarId.getDecl
      let name := decl.userName.toString
      return LatexData.atom (cleanName name)
  | .const name _ =>
      return LatexData.atom (constToLatex name)
  | .lit (.natVal n) =>
      return LatexData.atom (toString n)
  | .lit (.strVal s) =>
      return LatexData.atom ("\\text{\"" ++ s ++ "\"}")
  | .app .. =>
      ppApp e
  | .forallE name type body bi =>
      ppForall name type body bi
  | .lam name type body _ =>
      ppLambda name type body
  | .sort _ =>
      return LatexData.atom "\\mathrm{Type}"
  | .mdata _ expr =>
      latexPP expr
  | .proj _ idx expr => do
      let inner ← latexPP expr
      return LatexData.atom (inner.latex ++ ".\\pi_{" ++ toString idx ++ "}")
  | _ =>
      return LatexData.atom ("\\mathrm{" ++ e.ctorName ++ "}")

where
  cleanName (s : String) : String :=
    match s with
    | "alpha" => "\\alpha"
    | "beta" => "\\beta"
    | "theta" => "\\theta"
    | "delta" => "\\delta"
    | "eps" | "epsilon" => "\\epsilon"
    | "lambda" => "\\lambda"
    | "sigma" => "\\sigma"
    | "mu" => "\\mu"
    | "gamma" => "\\gamma"
    | "xi" => "\\xi"
    | _ => s

  constToLatex (n : Name) : String :=
    match n.toString with
    | "Real" => "\\mathbb{R}"
    | "Nat" => "\\mathbb{N}"
    | "Prop" => "\\mathrm{Prop}"
    | "True" => "\\top"
    | "False" => "\\bot"
    | "Zero.zero" | "OfNat.ofNat" => "0"
    | s =>
        if s.startsWith "BlckbauxResearch.Optimization.Operator." then
          "\\mathcal{O}_{\\mathrm{" ++ s.drop 35 ++ "}}"
        else
          "\\mathrm{" ++ n.getString! ++ "}"

  ppApp (e : Expr) : MetaM LatexData := do
    let fn := e.getAppFn
    let args := e.getAppArgs
    let fnName := if fn.isConst then fn.constName!.toString else ""
    match fnName, args with
    | "OfNat.ofNat", #[_, n, _] => latexPP n
    | "Nat.cast", #[_, _, n] => latexPP n
    | "HAdd.hAdd", #[_, _, _, _, a, b] => do
        let la ← latexPP a
        let lb ← latexPP b
        return LatexData.infixOp "+" 50 la lb
    | "HSub.hSub", #[_, _, _, _, a, b] => do
        let la ← latexPP a
        let lb ← latexPP b
        return LatexData.infixOp "-" 50 la lb
    | "HMul.hMul", #[_, _, _, _, a, b] => do
        let la ← latexPP a
        let lb ← latexPP b
        return LatexData.infixOp "\\cdot" 60 la lb
    | "HDiv.hDiv", #[_, _, _, _, a, b] => do
        let la ← latexPP a
        let lb ← latexPP b
        return LatexData.atom ("\\frac{" ++ la.latex ++ "}{" ++ lb.latex ++ "}")
    | "HPow.hPow", #[_, _, _, _, a, b] => do
        let la ← latexPP a
        let lb ← latexPP b
        return LatexData.atom ("{" ++ (la.protect 80).latex ++ "}^{" ++ lb.latex ++ "}")
    | "LE.le", #[_, _, a, b] => do
        let la ← latexPP a
        let lb ← latexPP b
        return LatexData.infixOp "\\le" 40 la lb
    | "GE.ge", #[_, _, a, b] => do
        let la ← latexPP a
        let lb ← latexPP b
        return LatexData.infixOp "\\ge" 40 la lb
    | "LT.lt", #[_, _, a, b] => do
        let la ← latexPP a
        let lb ← latexPP b
        return LatexData.infixOp "<" 40 la lb
    | "GT.gt", #[_, _, a, b] => do
        let la ← latexPP a
        let lb ← latexPP b
        return LatexData.infixOp ">" 40 la lb
    | "Eq", #[_, a, b] => do
        let la ← latexPP a
        let lb ← latexPP b
        return LatexData.infixOp "=" 40 la lb
    | "Ne", #[_, a, b] => do
        let la ← latexPP a
        let lb ← latexPP b
        return LatexData.infixOp "\\ne" 40 la lb
    | "Min.min", #[_, _, a, b] => do
        let la ← latexPP a
        let lb ← latexPP b
        return LatexData.atom ("\\min\\left(" ++ la.latex ++ ", " ++ lb.latex ++ "\\right)")
    | "Max.max", #[_, _, a, b] => do
        let la ← latexPP a
        let lb ← latexPP b
        return LatexData.atom ("\\max\\left(" ++ la.latex ++ ", " ++ lb.latex ++ "\\right)")
    | "Neg.neg", #[_, _, a] => do
        let la ← latexPP a
        return LatexData.prefixOp "-" 70 la
    | "Abs.abs", #[_, _, a] => do
        let la ← latexPP a
        return LatexData.atom ("\\left|" ++ la.latex ++ "\\right|")
    | "dist", #[_, _, a, b] => do
        let la ← latexPP a
        let lb ← latexPP b
        return LatexData.atom ("\\operatorname{dist}\\left(" ++ la.latex ++ ", " ++ lb.latex ++ "\\right)")
    | _, _ =>
        let lfn ← latexPP fn
        let mut visibleArgs := #[]
        for arg in args do
          let t ← inferType arg
          if !t.isSort && (← isClass? t).isNone then
            visibleArgs := visibleArgs.push arg
        let largs ← visibleArgs.mapM latexPP
        if largs.isEmpty then
          return lfn
        else
          let joined := ", ".intercalate (largs.map (·.latex)).toList
          return LatexData.atom (lfn.latex ++ "\\left(" ++ joined ++ "\\right)")

  ppForall (name : Name) (type body : Expr) (_bi : BinderInfo) : MetaM LatexData := do
    if body.hasLooseBVar 0 then
      withLocalDeclD name type fun fvar => do
        let bodyInst := body.instantiate1 fvar
        let ltype ← latexPP type
        let lbody ← latexPP bodyInst
        let varName := cleanName name.toString
        return {
          latex := "\\forall " ++ varName ++ " \\in " ++ ltype.latex ++ ",\\, " ++ lbody.latex,
          lbp := .min,
          rbp := .min
        }
    else
      -- Simple arrow / implication
      let ldom ← latexPP type
      let lcod ← latexPP body
      let domProt := ldom.protect 25
      let codProt := lcod.protect 24
      return {
        latex := domProt.latex ++ " \\implies " ++ codProt.latex,
        lbp := .assoc 25,
        rbp := .assoc 25
      }

  ppLambda (name : Name) (type body : Expr) : MetaM LatexData := do
    withLocalDeclD name type fun fvar => do
      let bodyInst := body.instantiate1 fvar
      let ltype ← latexPP type
      let lbody ← latexPP bodyInst
      let varName := cleanName name.toString
      return LatexData.atom ("\\left(" ++ varName ++ " \\in " ++ ltype.latex ++ "\\right) \\mapsto " ++ lbody.latex)

/-- Pretty-print an elaborated term expression to LaTeX string. -/
def toLatex (e : Expr) : MetaM String := do
  let data ← latexPP e
  return data.latex

/-- Pretty-print the type of a declared constant in the environment. -/
def declLatex (name : Name) : MetaM String := do
  let env ← getEnv
  match env.find? name with
  | some decl =>
      toLatex decl.type
  | none =>
      throwError s!"Unknown declaration {name}"

/-- Command `#latex <term>`: prints the LaTeX representation of `<term>`. -/
elab tk:"#latex " t:term : command => do
  Command.runTermElabM fun _ => do
    let e ← Term.elabTerm t none
    Term.synthesizeSyntheticMVarsNoPostponing
    let s ← toLatex e
    logInfoAt tk s!"[LeanTeX Output]\n{s}"

/-- Command `#latex_type <id>`: prints the LaTeX type signature of `<id>`. -/
elab tk:"#latex_type " id:ident : command => do
  Command.runTermElabM fun _ => do
    let name ← realizeGlobalConstNoOverload id
    let s ← declLatex name
    logInfoAt tk s!"[LeanTeX Type of {name}]\n{s}"

#latex (2 + 3 : Nat)
#latex (fun (x : Real) => x + 1)
#latex_type Optimization.Operator.nullInvariant
#latex_type Optimization.Operator.zeroProgress
#latex_type BlckbauxResearch.Optimization.Perturbation.Finding.mirrorDescent

end BlckbauxResearch.Export
