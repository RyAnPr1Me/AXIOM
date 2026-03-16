//! E-graph equality saturation optimizer for the Axiom compiler.
//!
//! This module uses the [`egg`](https://docs.rs/egg) crate to represent Axiom
//! HIR expressions in an e-graph, apply algebraic rewrite rules, and extract
//! the lowest-cost equivalent program.

use egg::{define_language, rewrite, CostFunction, Id, Language, RecExpr, Rewrite, Runner};

use crate::hir::{HirId, HirNode, HirOp, HirProgram};

// ---------------------------------------------------------------------------
// FloatBits – Ord/Hash-safe wrapper for f64 bit patterns
// ---------------------------------------------------------------------------

/// Stores f64 values as their bit representation so the type is `Ord + Hash`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FloatBits(pub i64);

impl FloatBits {
    /// Store f64 as i64 bit pattern. The cast is lossless—all 64 bits are
    /// preserved—but we use i64 because egg's `define_language!` leaf types
    /// must implement `Ord`, and `i64` does while `u64` would also work but
    /// `i64` integrates more naturally with the `Num(i64)` variant.
    pub fn from_f64(v: f64) -> Self {
        Self(v.to_bits() as i64)
    }

    pub fn to_f64(self) -> f64 {
        f64::from_bits(self.0 as u64)
    }
}

impl std::fmt::Display for FloatBits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}f64", self.0)
    }
}

impl std::str::FromStr for FloatBits {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.strip_suffix("f64")
            .ok_or_else(|| "expected float bits suffix 'f64'".to_string())
            .and_then(|inner| {
                inner
                    .parse::<i64>()
                    .map(FloatBits)
                    .map_err(|e| e.to_string())
            })
    }
}

// ---------------------------------------------------------------------------
// AxiomLang – e-graph node language
// ---------------------------------------------------------------------------

define_language! {
    /// The e-graph node types mirroring Axiom HIR operations.
    pub enum AxiomLang {
        // Arithmetic
        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = Mul([Id; 2]),
        "/" = Div([Id; 2]),
        "%" = Mod([Id; 2]),
        "neg" = Neg([Id; 1]),

        // Comparison
        "==" = Eq([Id; 2]),
        "!=" = Ne([Id; 2]),
        "<"  = Lt([Id; 2]),
        ">"  = Gt([Id; 2]),
        "<=" = Le([Id; 2]),
        ">=" = Ge([Id; 2]),

        // Logical
        "&&" = And([Id; 2]),
        "||" = Or([Id; 2]),
        "!" = Not([Id; 1]),

        // Bitwise
        "&" = BitAnd([Id; 2]),
        "|" = BitOr([Id; 2]),
        "^" = BitXor([Id; 2]),
        "<<" = Shl([Id; 2]),
        ">>" = Shr([Id; 2]),

        // Memory
        "load" = Load([Id; 1]),
        "store" = Store([Id; 2]),

        // Control flow
        "if" = If([Id; 3]),
        "loop" = Loop([Id; 2]),

        // Functions (variable arity: first child is callee, rest are args)
        "call" = Call(Box<[Id]>),

        // SIMD
        "vec_add" = VecAdd([Id; 2]),
        "vec_mul" = VecMul([Id; 2]),
        "vec_load" = VecLoad([Id; 1]),
        "vec_store" = VecStore([Id; 2]),

        // Leaf variants – order matters: most specific first, Symbol last.
        Num(i64),
        Float(FloatBits),
        Bool(bool),
        Var(egg::Symbol),
    }
}

// ---------------------------------------------------------------------------
// Rewrite rules
// ---------------------------------------------------------------------------

/// Build the full set of algebraic rewrite rules for Axiom (85+ rules).
pub fn rules() -> Vec<Rewrite<AxiomLang, ()>> {
    let mut rules: Vec<Rewrite<AxiomLang, ()>> = Vec::new();

    macro_rules! add {
        ($($rule:expr),+ $(,)?) => { $( rules.push($rule); )+ };
    }

    // ── Arithmetic identities ────────────────────────────────────────────

    add!(
        rewrite!("add-zero-r";  "(+ ?x 0)" => "?x"),
        rewrite!("add-zero-l";  "(+ 0 ?x)" => "?x"),
        rewrite!("sub-zero";    "(- ?x 0)" => "?x"),
        rewrite!("sub-self";    "(- ?x ?x)" => "0")
    );

    add!(
        rewrite!("mul-one-r";  "(* ?x 1)" => "?x"),
        rewrite!("mul-one-l";  "(* 1 ?x)" => "?x"),
        rewrite!("mul-zero-r"; "(* ?x 0)" => "0"),
        rewrite!("mul-zero-l"; "(* 0 ?x)" => "0")
    );

    add!(
        rewrite!("div-one";   "(/ ?x 1)" => "?x"),
        rewrite!("div-self";  "(/ ?x ?x)" => "1")
    );

    add!(
        rewrite!("neg-sub";    "(- 0 ?x)" => "(neg ?x)"),
        rewrite!("neg-neg";    "(neg (neg ?x))" => "?x"),
        rewrite!("mul-neg1";   "(* ?x -1)" => "(neg ?x)"),
        rewrite!("mul-neg1-l"; "(* -1 ?x)" => "(neg ?x)")
    );

    add!(
        rewrite!("mul2-to-add";   "(* ?x 2)" => "(+ ?x ?x)"),
        rewrite!("mul2-to-add-l"; "(* 2 ?x)" => "(+ ?x ?x)")
    );

    // Commutativity
    add!(
        rewrite!("add-comm"; "(+ ?a ?b)" => "(+ ?b ?a)"),
        rewrite!("mul-comm"; "(* ?a ?b)" => "(* ?b ?a)")
    );

    // Associativity
    add!(
        rewrite!("add-assoc-l"; "(+ (+ ?a ?b) ?c)" => "(+ ?a (+ ?b ?c))"),
        rewrite!("add-assoc-r"; "(+ ?a (+ ?b ?c))" => "(+ (+ ?a ?b) ?c)"),
        rewrite!("mul-assoc-l"; "(* (* ?a ?b) ?c)" => "(* ?a (* ?b ?c))"),
        rewrite!("mul-assoc-r"; "(* ?a (* ?b ?c))" => "(* (* ?a ?b) ?c)")
    );

    // Distribution
    add!(
        rewrite!("dist-mul-add";   "(* ?a (+ ?b ?c))" => "(+ (* ?a ?b) (* ?a ?c))"),
        rewrite!("factor-mul-add"; "(+ (* ?a ?b) (* ?a ?c))" => "(* ?a (+ ?b ?c))"),
        rewrite!("dist-mul-add-r";   "(* (+ ?b ?c) ?a)" => "(+ (* ?b ?a) (* ?c ?a))"),
        rewrite!("factor-mul-add-r"; "(+ (* ?b ?a) (* ?c ?a))" => "(* (+ ?b ?c) ?a)"),
        rewrite!("dist-mul-sub";   "(* ?a (- ?b ?c))" => "(- (* ?a ?b) (* ?a ?c))"),
        rewrite!("factor-mul-sub"; "(- (* ?a ?b) (* ?a ?c))" => "(* ?a (- ?b ?c))")
    );

    // Strength reduction: multiply/divide by powers of two → shift.
    // NOTE: division-to-shift is only correct for unsigned or non-negative
    // signed integers (arithmetic right shift rounds toward −∞, not zero).
    // A type-aware analysis pass should gate these rules in production.
    add!(
        rewrite!("mul-4-to-shl";  "(* ?x 4)"  => "(<< ?x 2)"),
        rewrite!("mul-8-to-shl";  "(* ?x 8)"  => "(<< ?x 3)"),
        rewrite!("mul-16-to-shl"; "(* ?x 16)" => "(<< ?x 4)"),
        rewrite!("mul-32-to-shl"; "(* ?x 32)" => "(<< ?x 5)"),
        rewrite!("mul-64-to-shl"; "(* ?x 64)" => "(<< ?x 6)"),
        rewrite!("div-2-to-shr";  "(/ ?x 2)"  => "(>> ?x 1)"),
        rewrite!("div-4-to-shr";  "(/ ?x 4)"  => "(>> ?x 2)"),
        rewrite!("div-8-to-shr";  "(/ ?x 8)"  => "(>> ?x 3)"),
        rewrite!("div-16-to-shr"; "(/ ?x 16)" => "(>> ?x 4)"),
        rewrite!("div-32-to-shr"; "(/ ?x 32)" => "(>> ?x 5)")
    );

    // Modular arithmetic
    add!(
        rewrite!("mod-1"; "(% ?x 1)" => "0")
    );

    // Additional arithmetic simplifications
    add!(
        rewrite!("add-neg";       "(+ ?x (neg ?x))" => "0"),
        rewrite!("sub-neg";       "(- ?x (neg ?y))" => "(+ ?x ?y)"),
        rewrite!("neg-sub-flip";  "(neg (- ?a ?b))" => "(- ?b ?a)"),
        rewrite!("add-sub-cancel"; "(- (+ ?a ?b) ?b)" => "?a"),
        rewrite!("sub-add-cancel"; "(+ (- ?a ?b) ?b)" => "?a"),
        rewrite!("mul-neg-neg";   "(* (neg ?a) (neg ?b))" => "(* ?a ?b)")
    );

    // ── Bitwise rules ────────────────────────────────────────────────────

    add!(
        rewrite!("bitand-zero-r"; "(& ?x 0)" => "0"),
        rewrite!("bitand-zero-l"; "(& 0 ?x)" => "0"),
        rewrite!("bitor-zero-r";  "(| ?x 0)" => "?x"),
        rewrite!("bitor-zero-l";  "(| 0 ?x)" => "?x"),
        rewrite!("bitxor-zero-r"; "(^ ?x 0)" => "?x"),
        rewrite!("bitxor-zero-l"; "(^ 0 ?x)" => "?x"),
        rewrite!("bitxor-self";   "(^ ?x ?x)" => "0"),
        rewrite!("bitand-self";   "(& ?x ?x)" => "?x"),
        rewrite!("bitor-self";    "(| ?x ?x)" => "?x")
    );

    add!(
        rewrite!("bitand-comm"; "(& ?a ?b)" => "(& ?b ?a)"),
        rewrite!("bitor-comm";  "(| ?a ?b)" => "(| ?b ?a)"),
        rewrite!("bitxor-comm"; "(^ ?a ?b)" => "(^ ?b ?a)")
    );

    add!(
        rewrite!("bitand-assoc-l"; "(& (& ?a ?b) ?c)" => "(& ?a (& ?b ?c))"),
        rewrite!("bitor-assoc-l";  "(| (| ?a ?b) ?c)" => "(| ?a (| ?b ?c))")
    );

    // ── Logical rules ────────────────────────────────────────────────────

    add!(
        rewrite!("not-not";     "(! (! ?x))" => "?x"),
        rewrite!("and-true-r";  "(&& ?x true)"  => "?x"),
        rewrite!("and-true-l";  "(&& true ?x)"  => "?x"),
        rewrite!("or-false-r";  "(|| ?x false)" => "?x"),
        rewrite!("or-false-l";  "(|| false ?x)" => "?x"),
        rewrite!("and-false-r"; "(&& ?x false)" => "false"),
        rewrite!("and-false-l"; "(&& false ?x)" => "false"),
        rewrite!("or-true-r";   "(|| ?x true)"  => "true"),
        rewrite!("or-true-l";   "(|| true ?x)"  => "true"),
        rewrite!("and-self";    "(&& ?x ?x)" => "?x"),
        rewrite!("or-self";     "(|| ?x ?x)" => "?x")
    );

    add!(
        rewrite!("and-comm"; "(&& ?a ?b)" => "(&& ?b ?a)"),
        rewrite!("or-comm";  "(|| ?a ?b)" => "(|| ?b ?a)")
    );

    // ── Memory rules ─────────────────────────────────────────────────────

    add!(
        rewrite!("load-store"; "(load (store ?addr ?val))" => "?val")
    );

    // ── Comparison simplifications ───────────────────────────────────────

    add!(
        rewrite!("eq-self";  "(== ?x ?x)" => "true"),
        rewrite!("ne-self";  "(!= ?x ?x)" => "false"),
        rewrite!("lt-self";  "(<  ?x ?x)" => "false"),
        rewrite!("gt-self";  "(>  ?x ?x)" => "false"),
        rewrite!("le-self";  "(<= ?x ?x)" => "true"),
        rewrite!("ge-self";  "(>= ?x ?x)" => "true")
    );

    add!(
        rewrite!("lt-flip"; "(< ?a ?b)"  => "(> ?b ?a)"),
        rewrite!("le-flip"; "(<= ?a ?b)" => "(>= ?b ?a)")
    );

    // ── Control-flow simplifications ─────────────────────────────────────

    add!(
        rewrite!("if-true";  "(if true ?a ?b)"  => "?a"),
        rewrite!("if-false"; "(if false ?a ?b)" => "?b")
    );

    // ── SIMD commutativity ───────────────────────────────────────────────

    add!(
        rewrite!("vec-add-comm"; "(vec_add ?a ?b)" => "(vec_add ?b ?a)"),
        rewrite!("vec-mul-comm"; "(vec_mul ?a ?b)" => "(vec_mul ?b ?a)")
    );

    rules
}

// ---------------------------------------------------------------------------
// Cost function
// ---------------------------------------------------------------------------

/// A simple AST-size cost model that prefers cheaper machine operations.
pub struct AxiomCost;

impl CostFunction<AxiomLang> for AxiomCost {
    type Cost = usize;

    fn cost<C>(&mut self, enode: &AxiomLang, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        let op_cost: usize = match enode {
            // Constants and variables are essentially free.
            AxiomLang::Num(_) | AxiomLang::Float(_) | AxiomLang::Bool(_) | AxiomLang::Var(_) => 1,

            // Shifts are the cheapest arithmetic.
            AxiomLang::Shl(_) | AxiomLang::Shr(_) => 2,

            // Bitwise ops.
            AxiomLang::BitAnd(_) | AxiomLang::BitOr(_) | AxiomLang::BitXor(_) => 2,

            // Addition / subtraction / negation.
            AxiomLang::Add(_) | AxiomLang::Sub(_) | AxiomLang::Neg(_) => 3,

            // Multiply is more expensive than add.
            AxiomLang::Mul(_) => 5,

            // Division / mod are the most expensive arithmetic.
            AxiomLang::Div(_) | AxiomLang::Mod(_) => 8,

            // Comparisons.
            AxiomLang::Eq(_) | AxiomLang::Ne(_) | AxiomLang::Lt(_) | AxiomLang::Gt(_)
            | AxiomLang::Le(_) | AxiomLang::Ge(_) => 2,

            // Logical.
            AxiomLang::And(_) | AxiomLang::Or(_) | AxiomLang::Not(_) => 2,

            // Memory.
            AxiomLang::Load(_) => 6,
            AxiomLang::Store(_) => 6,

            // Control.
            AxiomLang::If(_) => 4,
            AxiomLang::Loop(_) => 8,

            // Functions.
            AxiomLang::Call(_) => 15,

            // SIMD.
            AxiomLang::VecAdd(_) | AxiomLang::VecMul(_) => 3,
            AxiomLang::VecLoad(_) | AxiomLang::VecStore(_) => 5,
        };

        enode.fold(op_cost, |sum, id| sum + costs(id))
    }
}

// ---------------------------------------------------------------------------
// AxiomOptimizer
// ---------------------------------------------------------------------------

/// Equality-saturation optimizer for Axiom expressions.
#[derive(Debug, Clone)]
pub struct AxiomOptimizer {
    node_limit: usize,
    iter_limit: usize,
}

impl Default for AxiomOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl AxiomOptimizer {
    /// Create a new optimizer with default parameters.
    pub fn new() -> Self {
        Self {
            node_limit: 50_000,
            iter_limit: 20,
        }
    }

    /// Return the full set of rewrite rules.
    pub fn rules(&self) -> Vec<Rewrite<AxiomLang, ()>> {
        rules()
    }

    /// Run equality saturation on `expr` and extract the cheapest equivalent.
    pub fn optimize(&self, expr: RecExpr<AxiomLang>) -> RecExpr<AxiomLang> {
        let runner = Runner::<AxiomLang, ()>::default()
            .with_expr(&expr)
            .with_node_limit(self.node_limit)
            .with_iter_limit(self.iter_limit)
            .run(&rules());

        let root = runner.roots[0];
        let extractor = egg::Extractor::new(&runner.egraph, AxiomCost);
        let (_best_cost, best_expr) = extractor.find_best(root);
        best_expr
    }
}

// ---------------------------------------------------------------------------
// HIR → egg conversion
// ---------------------------------------------------------------------------

/// Convert a single HIR node tree (rooted at `root`) into an egg `RecExpr`.
///
/// The function walks the HIR operand graph depth-first, mapping each [`HirOp`]
/// to the corresponding [`AxiomLang`] node.
pub fn hir_to_expr(root: &HirNode, program: &HirProgram) -> RecExpr<AxiomLang> {
    let mut expr = RecExpr::default();
    let node_map: std::collections::HashMap<HirId, &HirNode> = program
        .functions
        .iter()
        .flat_map(|f| f.body.iter())
        .map(|n| (n.id, n))
        .collect();

    fn build(
        node: &HirNode,
        node_map: &std::collections::HashMap<HirId, &HirNode>,
        expr: &mut RecExpr<AxiomLang>,
        cache: &mut std::collections::HashMap<HirId, Id>,
    ) -> Id {
        if let Some(&id) = cache.get(&node.id) {
            return id;
        }

        // Recursively convert operands first.
        let child_ids: Vec<Id> = node
            .operands
            .iter()
            .filter_map(|hid| node_map.get(hid).map(|n| build(n, node_map, expr, cache)))
            .collect();

        let lang_node = match &node.op {
            // Arithmetic
            HirOp::Add => AxiomLang::Add([child_ids[0], child_ids[1]]),
            HirOp::Sub => AxiomLang::Sub([child_ids[0], child_ids[1]]),
            HirOp::Mul => AxiomLang::Mul([child_ids[0], child_ids[1]]),
            HirOp::Div => AxiomLang::Div([child_ids[0], child_ids[1]]),
            HirOp::Mod => AxiomLang::Mod([child_ids[0], child_ids[1]]),
            HirOp::Neg => AxiomLang::Neg([child_ids[0]]),

            // Comparison
            HirOp::Eq => AxiomLang::Eq([child_ids[0], child_ids[1]]),
            HirOp::Ne => AxiomLang::Ne([child_ids[0], child_ids[1]]),
            HirOp::Lt => AxiomLang::Lt([child_ids[0], child_ids[1]]),
            HirOp::Gt => AxiomLang::Gt([child_ids[0], child_ids[1]]),
            HirOp::Le => AxiomLang::Le([child_ids[0], child_ids[1]]),
            HirOp::Ge => AxiomLang::Ge([child_ids[0], child_ids[1]]),

            // Logical
            HirOp::And => AxiomLang::And([child_ids[0], child_ids[1]]),
            HirOp::Or  => AxiomLang::Or([child_ids[0], child_ids[1]]),
            HirOp::Not => AxiomLang::Not([child_ids[0]]),

            // Bitwise
            HirOp::BitAnd => AxiomLang::BitAnd([child_ids[0], child_ids[1]]),
            HirOp::BitOr  => AxiomLang::BitOr([child_ids[0], child_ids[1]]),
            HirOp::BitXor => AxiomLang::BitXor([child_ids[0], child_ids[1]]),
            HirOp::Shl    => AxiomLang::Shl([child_ids[0], child_ids[1]]),
            HirOp::Shr    => AxiomLang::Shr([child_ids[0], child_ids[1]]),

            // Memory
            HirOp::Load  => AxiomLang::Load([child_ids[0]]),
            HirOp::Store => AxiomLang::Store([child_ids[0], child_ids[1]]),

            // Constants
            HirOp::IntConst(v) => AxiomLang::Num(*v),
            HirOp::FloatConst(v) => AxiomLang::Float(FloatBits::from_f64(*v)),
            HirOp::BoolConst(v) => AxiomLang::Bool(*v),

            // Variables
            HirOp::VarRef { name } => AxiomLang::Var(egg::Symbol::from(name.as_str())),
            HirOp::Param { index } => {
                AxiomLang::Var(egg::Symbol::from(format!("__param_{index}").as_str()))
            }

            // Function calls
            HirOp::Call { name } => {
                let name_id = expr.add(AxiomLang::Var(egg::Symbol::from(name.as_str())));
                let mut ids = vec![name_id];
                ids.extend_from_slice(&child_ids);
                AxiomLang::Call(ids.into_boxed_slice())
            }

            // Control flow – validate arity before indexing.
            HirOp::Branch if child_ids.len() >= 3 => {
                AxiomLang::If([child_ids[0], child_ids[1], child_ids[2]])
            }
            HirOp::Branch => {
                let tag = format!("__malformed_branch_{}", child_ids.len());
                AxiomLang::Var(egg::Symbol::from(tag.as_str()))
            }
            HirOp::Loop if child_ids.len() >= 2 => {
                AxiomLang::Loop([child_ids[0], child_ids[1]])
            }
            HirOp::Loop => {
                let tag = format!("__malformed_loop_{}", child_ids.len());
                AxiomLang::Var(egg::Symbol::from(tag.as_str()))
            }

            // BitNot has no direct e-graph node; lower to XOR with -1.
            HirOp::BitNot if !child_ids.is_empty() => {
                let neg1 = expr.add(AxiomLang::Num(-1));
                AxiomLang::BitXor([child_ids[0], neg1])
            }

            // Fallback: represent unsupported ops as named variables so we
            // don't lose information.
            other => {
                let tag = format!("__unsupported_{other:?}");
                AxiomLang::Var(egg::Symbol::from(tag.as_str()))
            }
        };

        let id = expr.add(lang_node);
        cache.insert(node.id, id);
        id
    }

    let mut cache = std::collections::HashMap::new();
    build(root, &node_map, &mut expr, &mut cache);
    expr
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse an s-expression, optimize it, and return the optimized string.
    fn opt(input: &str) -> String {
        let expr: RecExpr<AxiomLang> = input.parse().unwrap();
        let optimizer = AxiomOptimizer::new();
        optimizer.optimize(expr).to_string()
    }

    // -- additive identity --------------------------------------------------

    #[test]
    fn add_zero_right() {
        assert_eq!(opt("(+ x 0)"), "x");
    }

    #[test]
    fn add_zero_left() {
        assert_eq!(opt("(+ 0 x)"), "x");
    }

    #[test]
    fn sub_zero() {
        assert_eq!(opt("(- x 0)"), "x");
    }

    // -- multiplicative identity / annihilator ------------------------------

    #[test]
    fn mul_one_right() {
        assert_eq!(opt("(* x 1)"), "x");
    }

    #[test]
    fn mul_one_left() {
        assert_eq!(opt("(* 1 x)"), "x");
    }

    #[test]
    fn mul_zero() {
        assert_eq!(opt("(* x 0)"), "0");
    }

    // -- division -----------------------------------------------------------

    #[test]
    fn div_one() {
        assert_eq!(opt("(/ x 1)"), "x");
    }

    #[test]
    fn div_self() {
        assert_eq!(opt("(/ x x)"), "1");
    }

    // -- sub self / negation ------------------------------------------------

    #[test]
    fn sub_self() {
        assert_eq!(opt("(- x x)"), "0");
    }

    #[test]
    fn double_negation() {
        assert_eq!(opt("(neg (neg x))"), "x");
    }

    #[test]
    fn mul_neg_one() {
        assert_eq!(opt("(* x -1)"), "(neg x)");
    }

    // -- strength reduction: mul → shift ------------------------------------

    #[test]
    fn mul_2_to_add() {
        assert_eq!(opt("(* x 2)"), "(+ x x)");
    }

    #[test]
    fn mul_4_to_shl() {
        assert_eq!(opt("(* x 4)"), "(<< x 2)");
    }

    #[test]
    fn mul_8_to_shl() {
        assert_eq!(opt("(* x 8)"), "(<< x 3)");
    }

    #[test]
    fn div_2_to_shr() {
        assert_eq!(opt("(/ x 2)"), "(>> x 1)");
    }

    // -- bitwise rules ------------------------------------------------------

    #[test]
    fn bitand_zero() {
        assert_eq!(opt("(& x 0)"), "0");
    }

    #[test]
    fn bitor_zero() {
        assert_eq!(opt("(| x 0)"), "x");
    }

    #[test]
    fn bitxor_zero() {
        assert_eq!(opt("(^ x 0)"), "x");
    }

    #[test]
    fn bitxor_self() {
        assert_eq!(opt("(^ x x)"), "0");
    }

    #[test]
    fn bitand_self() {
        assert_eq!(opt("(& x x)"), "x");
    }

    #[test]
    fn bitor_self() {
        assert_eq!(opt("(| x x)"), "x");
    }

    // -- logical rules ------------------------------------------------------

    #[test]
    fn not_not() {
        assert_eq!(opt("(! (! x))"), "x");
    }

    #[test]
    fn and_true() {
        assert_eq!(opt("(&& x true)"), "x");
    }

    #[test]
    fn or_false() {
        assert_eq!(opt("(|| x false)"), "x");
    }

    #[test]
    fn and_false() {
        assert_eq!(opt("(&& x false)"), "false");
    }

    #[test]
    fn or_true() {
        assert_eq!(opt("(|| x true)"), "true");
    }

    // -- memory rules -------------------------------------------------------

    #[test]
    fn load_store_elim() {
        assert_eq!(opt("(load (store addr val))"), "val");
    }

    // -- comparison simplifications -----------------------------------------

    #[test]
    fn eq_self() {
        assert_eq!(opt("(== x x)"), "true");
    }

    #[test]
    fn ne_self() {
        assert_eq!(opt("(!= x x)"), "false");
    }

    #[test]
    fn lt_self() {
        assert_eq!(opt("(< x x)"), "false");
    }

    #[test]
    fn le_self() {
        assert_eq!(opt("(<= x x)"), "true");
    }

    // -- control flow -------------------------------------------------------

    #[test]
    fn if_true() {
        assert_eq!(opt("(if true a b)"), "a");
    }

    #[test]
    fn if_false() {
        assert_eq!(opt("(if false a b)"), "b");
    }

    // -- modular arithmetic -------------------------------------------------

    #[test]
    fn mod_one() {
        assert_eq!(opt("(% x 1)"), "0");
    }

    // -- combined: sub cancel -----------------------------------------------

    #[test]
    fn add_sub_cancel() {
        assert_eq!(opt("(- (+ a b) b)"), "a");
    }

    // -- optimizer struct ---------------------------------------------------

    #[test]
    fn optimizer_returns_rules() {
        let o = AxiomOptimizer::new();
        assert!(o.rules().len() >= 50, "expected ≥50 rules, got {}", o.rules().len());
    }

    // -- cost function prefers shifts over mul/div --------------------------

    #[test]
    fn cost_prefers_shift_over_div() {
        assert_eq!(opt("(/ x 4)"), "(>> x 2)");
    }

    // -- parse round-trip ---------------------------------------------------

    #[test]
    fn parse_roundtrip_num() {
        let expr: RecExpr<AxiomLang> = "42".parse().unwrap();
        assert_eq!(expr.to_string(), "42");
    }

    #[test]
    fn parse_roundtrip_complex() {
        let input = "(+ (* a b) 1)";
        let expr: RecExpr<AxiomLang> = input.parse().unwrap();
        assert_eq!(expr.to_string(), input);
    }

    #[test]
    fn float_bits_roundtrip() {
        let fb = FloatBits::from_f64(3.14);
        let s = fb.to_string();
        let fb2: FloatBits = s.parse().unwrap();
        assert_eq!(fb, fb2);
        assert!((fb2.to_f64() - 3.14).abs() < 1e-10);
    }
}