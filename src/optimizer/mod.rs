//! Higher-level optimization passes for the Axiom compiler.
//!
//! This module orchestrates optimizations beyond the e-graph rewrite rules,
//! including loop optimization, auto-vectorization analysis, superoptimization
//! for small blocks, and a cost model for comparing code sequences.

use std::cmp::Ordering;

use crate::hir::{HirFunction, HirId, HirNode, HirOp, HirProgram};

// ---------------------------------------------------------------------------
// Loop Pattern Detection
// ---------------------------------------------------------------------------

/// Recognised loop idioms that enable specialised transformations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoopPattern {
    /// Accumulation into a single variable (e.g. `sum += arr[i]`).
    Reduction {
        accumulator: String,
        operator: ReductionOp,
        source_array: String,
    },
    /// Element-wise transformation (e.g. `result[i] = f(arr[i])`).
    Map {
        source_array: String,
        dest_array: String,
    },
    /// Conditional selection of elements.
    Filter {
        source_array: String,
        dest_array: String,
    },
    /// Prefix-sum / inclusive scan.
    Scan {
        accumulator: String,
        operator: ReductionOp,
        source_array: String,
        dest_array: String,
    },
    /// Pattern could not be classified.
    Unknown,
}

/// Binary operators that appear in reduction / scan patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionOp {
    Add,
    Mul,
    Min,
    Max,
    BitAnd,
    BitOr,
    BitXor,
}

/// Detect the dominant pattern in a loop body represented as a slice of HIR
/// nodes.  This is a heuristic analysis.
pub fn detect_loop_pattern(body: &[HirNode]) -> LoopPattern {
    let has_index = body.iter().any(|n| matches!(n.op, HirOp::Index));
    let has_store = body.iter().any(|n| matches!(n.op, HirOp::Store));

    // Look for accumulator-style variable mutation (VarDef with an arithmetic
    // operand that references the same variable).
    if let Some(pattern) = try_detect_reduction(body) {
        return pattern;
    }
    if let Some(pattern) = try_detect_scan(body) {
        return pattern;
    }
    if let Some(pattern) = try_detect_filter(body) {
        return pattern;
    }
    if has_index && has_store {
        if let Some(pattern) = try_detect_map(body) {
            return pattern;
        }
    }
    LoopPattern::Unknown
}

fn try_detect_reduction(body: &[HirNode]) -> Option<LoopPattern> {
    // Pattern: VarDef that accumulates via Add/Mul/… from an indexed array.
    for node in body {
        if let HirOp::VarDef { name, mutable: true } = &node.op {
            for op_id in &node.operands {
                if let Some(arith) = body.iter().find(|n| n.id == *op_id) {
                    let reduction_op = match &arith.op {
                        HirOp::Add => Some(ReductionOp::Add),
                        HirOp::Mul => Some(ReductionOp::Mul),
                        HirOp::BitAnd => Some(ReductionOp::BitAnd),
                        HirOp::BitOr => Some(ReductionOp::BitOr),
                        HirOp::BitXor => Some(ReductionOp::BitXor),
                        _ => None,
                    };
                    if let Some(op) = reduction_op {
                        let source = find_indexed_array(body, &arith.operands);
                        return Some(LoopPattern::Reduction {
                            accumulator: name.clone(),
                            operator: op,
                            source_array: source.unwrap_or_default(),
                        });
                    }
                }
            }
        }
    }
    None
}

fn try_detect_map(body: &[HirNode]) -> Option<LoopPattern> {
    // Pattern: Store to dest[i] of some f(src[i]).
    let stores: Vec<_> = body.iter().filter(|n| matches!(n.op, HirOp::Store)).collect();
    let indices: Vec<_> = body.iter().filter(|n| matches!(n.op, HirOp::Index)).collect();
    if stores.len() == 1 && indices.len() >= 2 {
        let dest = find_indexed_array(body, &stores[0].operands).unwrap_or_default();
        let src = indices
            .iter()
            .filter_map(|idx| find_base_var(body, idx.operands.first().copied()))
            .find(|name| name != &dest)
            .unwrap_or_default();
        return Some(LoopPattern::Map {
            source_array: src,
            dest_array: dest,
        });
    }
    None
}

fn try_detect_filter(body: &[HirNode]) -> Option<LoopPattern> {
    let has_branch = body.iter().any(|n| matches!(n.op, HirOp::Branch));
    let has_store = body.iter().any(|n| matches!(n.op, HirOp::Store));
    let indices: Vec<_> = body.iter().filter(|n| matches!(n.op, HirOp::Index)).collect();
    if has_branch && has_store && !indices.is_empty() {
        let src = indices
            .iter()
            .filter_map(|idx| find_base_var(body, idx.operands.first().copied()))
            .next()
            .unwrap_or_default();
        return Some(LoopPattern::Filter {
            source_array: src,
            dest_array: String::new(),
        });
    }
    None
}

fn try_detect_scan(body: &[HirNode]) -> Option<LoopPattern> {
    // Scan: accumulator update + store to dest array in same iteration.
    let has_store = body.iter().any(|n| matches!(n.op, HirOp::Store));
    if !has_store {
        return None;
    }
    for node in body {
        if let HirOp::VarDef { name, mutable: true } = &node.op {
            for op_id in &node.operands {
                if let Some(arith) = body.iter().find(|n| n.id == *op_id) {
                    let reduction_op = match &arith.op {
                        HirOp::Add => Some(ReductionOp::Add),
                        HirOp::Mul => Some(ReductionOp::Mul),
                        _ => None,
                    };
                    if let Some(op) = reduction_op {
                        let source = find_indexed_array(body, &arith.operands);
                        return Some(LoopPattern::Scan {
                            accumulator: name.clone(),
                            operator: op,
                            source_array: source.unwrap_or_default(),
                            dest_array: String::new(),
                        });
                    }
                }
            }
        }
    }
    None
}

/// Walk operands looking for an `Index` whose base is a `VarRef`.
fn find_indexed_array(body: &[HirNode], operand_ids: &[HirId]) -> Option<String> {
    for id in operand_ids {
        if let Some(node) = body.iter().find(|n| n.id == *id) {
            if matches!(node.op, HirOp::Index) {
                return find_base_var(body, node.operands.first().copied());
            }
            // Recurse one level.
            if let Some(name) = find_indexed_array(body, &node.operands) {
                return Some(name);
            }
        }
    }
    None
}

fn find_base_var(body: &[HirNode], id: Option<HirId>) -> Option<String> {
    let id = id?;
    body.iter().find(|n| n.id == id).and_then(|n| match &n.op {
        HirOp::VarRef { name } => Some(name.clone()),
        _ => None,
    })
}

// ---------------------------------------------------------------------------
// Loop Optimisation Transforms
// ---------------------------------------------------------------------------

/// Configuration for loop transformations.
#[derive(Debug, Clone)]
pub struct LoopOptConfig {
    /// Unroll factor (default 4).
    pub unroll_factor: usize,
    /// Tile size in elements (default 64).
    pub tile_size: usize,
    /// Whether to attempt loop fusion.
    pub enable_fusion: bool,
    /// Whether to emit vectorization hints.
    pub enable_vectorization_hints: bool,
}

impl Default for LoopOptConfig {
    fn default() -> Self {
        Self {
            unroll_factor: 4,
            tile_size: 64,
            enable_fusion: true,
            enable_vectorization_hints: true,
        }
    }
}

/// Annotation attached to a loop after analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoopTransform {
    Unroll { factor: usize },
    Tile { tile_size: usize },
    Fuse { partner_loop: HirId },
    VectorizationHint,
}

/// Apply loop-level optimizations to every function in a program.
pub fn optimize_loops(program: &mut HirProgram, config: &LoopOptConfig) {
    for func in &mut program.functions {
        optimize_function_loops(func, config);
    }
}

fn optimize_function_loops(func: &mut HirFunction, config: &LoopOptConfig) {
    let loop_indices: Vec<usize> = func
        .body
        .iter()
        .enumerate()
        .filter(|(_, n)| matches!(n.op, HirOp::Loop))
        .map(|(i, _)| i)
        .collect();

    for &idx in loop_indices.iter().rev() {
        let body_slice = extract_loop_body(&func.body, idx);
        let pattern = detect_loop_pattern(&body_slice);

        match pattern {
            LoopPattern::Reduction { .. } | LoopPattern::Map { .. } => {
                if config.enable_vectorization_hints {
                    // Mark the loop header with a vectorization-hint nop.
                    let hint = HirNode {
                        id: HirId(func.body.len() + 1000),
                        op: HirOp::Nop,
                        operands: vec![func.body[idx].id],
                        ty: None,
                        span: func.body[idx].span,
                    };
                    func.body.insert(idx, hint);
                }
                if config.unroll_factor > 1 {
                    unroll_loop(&mut func.body, idx, config.unroll_factor);
                }
            }
            _ => {}
        }
    }

    if config.enable_fusion {
        fuse_adjacent_loops(&mut func.body);
    }
}

fn extract_loop_body(nodes: &[HirNode], loop_idx: usize) -> Vec<HirNode> {
    let loop_node = &nodes[loop_idx];
    if loop_node.operands.len() < 2 {
        return vec![];
    }
    let body_start_id = loop_node.operands[1];
    let start = nodes.iter().position(|n| n.id == body_start_id);
    match start {
        Some(s) => {
            let end = nodes[s..]
                .iter()
                .position(|n| matches!(n.op, HirOp::Jump))
                .map(|off| s + off + 1)
                .unwrap_or(nodes.len().min(s + 20));
            nodes[s..end].to_vec()
        }
        None => vec![],
    }
}

fn unroll_loop(_body: &mut Vec<HirNode>, _loop_idx: usize, _factor: usize) {
    // TODO: Implement loop unrolling — duplicate the loop body `factor` times
    // and adjust induction variable references accordingly. This is intentionally
    // stubbed; the optimizer records the intent for future implementation.
}

fn fuse_adjacent_loops(_body: &mut Vec<HirNode>) {
    // TODO: Implement loop fusion — detect adjacent Loop nodes whose iteration
    // ranges are identical and merge their bodies. This is intentionally stubbed
    // for future implementation.
}

// ---------------------------------------------------------------------------
// Auto-Vectorization Analysis
// ---------------------------------------------------------------------------

/// Describes how a loop can be vectorized.
#[derive(Debug, Clone)]
pub struct VectorizationPlan {
    /// The loop header node id.
    pub loop_id: HirId,
    /// SIMD vector width in elements (e.g. 4 for f32 on SSE).
    pub vector_width: usize,
    /// Operations that map directly to SIMD intrinsics.
    pub simd_ops: Vec<SimdMapping>,
    /// Whether a scalar tail loop is needed.
    pub needs_tail_loop: bool,
    /// Estimated speedup factor.
    pub estimated_speedup: f64,
}

/// Maps an HIR operation to a target SIMD intrinsic.
#[derive(Debug, Clone)]
pub struct SimdMapping {
    pub hir_op: SimdCompatibleOp,
    pub intrinsic_name: String,
}

/// Operations that have direct SIMD equivalents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdCompatibleOp {
    Add,
    Sub,
    Mul,
    Div,
    Load,
    Store,
    Min,
    Max,
    Fma,
}

/// Analyse a loop body and produce a vectorization plan, if possible.
pub fn analyze_vectorization(
    loop_id: HirId,
    body: &[HirNode],
    vector_width: usize,
) -> Option<VectorizationPlan> {
    if !is_vectorizable(body) {
        return None;
    }

    let mut simd_ops = Vec::new();
    for node in body {
        if let Some(mapping) = map_to_simd(&node.op) {
            simd_ops.push(mapping);
        }
    }

    if simd_ops.is_empty() {
        return None;
    }

    let estimated_speedup = simd_ops.len() as f64 / body.len().max(1) as f64
        * vector_width as f64;

    Some(VectorizationPlan {
        loop_id,
        vector_width,
        simd_ops,
        needs_tail_loop: true,
        estimated_speedup,
    })
}

fn is_vectorizable(body: &[HirNode]) -> bool {
    // A loop is vectorizable when it contains sequential array accesses and no
    // control-flow divergence.
    let has_branch = body.iter().any(|n| matches!(n.op, HirOp::Branch));
    let has_call = body.iter().any(|n| matches!(n.op, HirOp::Call { .. }));
    let has_index = body.iter().any(|n| matches!(n.op, HirOp::Index));
    has_index && !has_branch && !has_call
}

fn map_to_simd(op: &HirOp) -> Option<SimdMapping> {
    let (compat, name) = match op {
        HirOp::Add => (SimdCompatibleOp::Add, "vadd"),
        HirOp::Sub => (SimdCompatibleOp::Sub, "vsub"),
        HirOp::Mul => (SimdCompatibleOp::Mul, "vmul"),
        HirOp::Div => (SimdCompatibleOp::Div, "vdiv"),
        HirOp::Load => (SimdCompatibleOp::Load, "vload"),
        HirOp::Store => (SimdCompatibleOp::Store, "vstore"),
        _ => return None,
    };
    Some(SimdMapping {
        hir_op: compat,
        intrinsic_name: name.to_string(),
    })
}

// ---------------------------------------------------------------------------
// Superoptimization & Peephole Rules
// ---------------------------------------------------------------------------

/// A peephole optimisation rule matching a small pattern and replacing it.
#[derive(Debug, Clone)]
pub struct PeepholeRule {
    pub name: &'static str,
    /// Predicate: returns `true` if the rule applies to the given window.
    pub matches: fn(&[HirNode]) -> bool,
    /// Rewrite the matched window, returning replacement nodes.
    pub apply: fn(&[HirNode]) -> Vec<HirNode>,
}

/// Return the built-in set of peephole rules.
pub fn default_peephole_rules() -> Vec<PeepholeRule> {
    vec![
        PeepholeRule {
            name: "add_zero_elim",
            matches: |nodes| {
                nodes.len() == 2
                    && matches!(nodes[0].op, HirOp::IntConst(0))
                    && matches!(nodes[1].op, HirOp::Add)
            },
            apply: |nodes| {
                // x + 0 → x : keep only the non-zero operand reference.
                let add = &nodes[1];
                let kept = add
                    .operands
                    .iter()
                    .copied()
                    .find(|id| *id != nodes[0].id)
                    .unwrap_or(add.operands[0]);
                vec![HirNode {
                    id: add.id,
                    op: HirOp::VarRef {
                        name: format!("__peep_{}", kept.0),
                    },
                    operands: vec![kept],
                    ty: add.ty,
                    span: add.span,
                }]
            },
        },
        PeepholeRule {
            name: "mul_one_elim",
            matches: |nodes| {
                nodes.len() == 2
                    && matches!(nodes[0].op, HirOp::IntConst(1))
                    && matches!(nodes[1].op, HirOp::Mul)
            },
            apply: |nodes| {
                let mul = &nodes[1];
                let kept = mul
                    .operands
                    .iter()
                    .copied()
                    .find(|id| *id != nodes[0].id)
                    .unwrap_or(mul.operands[0]);
                vec![HirNode {
                    id: mul.id,
                    op: HirOp::VarRef {
                        name: format!("__peep_{}", kept.0),
                    },
                    operands: vec![kept],
                    ty: mul.ty,
                    span: mul.span,
                }]
            },
        },
        PeepholeRule {
            name: "mul_zero_elim",
            matches: |nodes| {
                nodes.len() == 2
                    && matches!(nodes[0].op, HirOp::IntConst(0))
                    && matches!(nodes[1].op, HirOp::Mul)
            },
            apply: |nodes| {
                vec![HirNode {
                    id: nodes[1].id,
                    op: HirOp::IntConst(0),
                    operands: vec![],
                    ty: nodes[1].ty,
                    span: nodes[1].span,
                }]
            },
        },
        PeepholeRule {
            name: "double_neg_elim",
            matches: |nodes| {
                nodes.len() == 2
                    && matches!(nodes[0].op, HirOp::Neg)
                    && matches!(nodes[1].op, HirOp::Neg)
            },
            apply: |nodes| {
                let inner_operand = nodes[0]
                    .operands
                    .first()
                    .copied()
                    .unwrap_or(HirId(0));
                vec![HirNode {
                    id: nodes[1].id,
                    op: HirOp::VarRef {
                        name: format!("__peep_{}", inner_operand.0),
                    },
                    operands: vec![inner_operand],
                    ty: nodes[1].ty,
                    span: nodes[1].span,
                }]
            },
        },
        PeepholeRule {
            name: "strength_reduce_mul_pow2",
            matches: |nodes| {
                nodes.len() == 2
                    && matches!(nodes[1].op, HirOp::Mul)
                    && matches!(nodes[0].op, HirOp::IntConst(v) if v > 0 && (v as u64).is_power_of_two())
            },
            apply: |nodes| {
                if let HirOp::IntConst(v) = nodes[0].op {
                    let shift = v.trailing_zeros() as i64;
                    let mul = &nodes[1];
                    let other = mul
                        .operands
                        .iter()
                        .copied()
                        .find(|id| *id != nodes[0].id)
                        .unwrap_or(mul.operands[0]);
                    let shift_node = HirNode {
                        id: nodes[0].id,
                        op: HirOp::IntConst(shift),
                        operands: vec![],
                        ty: nodes[0].ty,
                        span: nodes[0].span,
                    };
                    let shl_node = HirNode {
                        id: mul.id,
                        op: HirOp::Shl,
                        operands: vec![other, nodes[0].id],
                        ty: mul.ty,
                        span: mul.span,
                    };
                    vec![shift_node, shl_node]
                } else {
                    nodes.to_vec()
                }
            },
        },
    ]
}

/// Run superoptimization on small blocks within a function.
pub fn superoptimize(func: &mut HirFunction) {
    let rules = default_peephole_rules();
    apply_peephole_rules(&mut func.body, &rules);
}

/// Slide a window over `body` and apply the first matching peephole rule.
pub fn apply_peephole_rules(body: &mut Vec<HirNode>, rules: &[PeepholeRule]) {
    let max_window = 4;
    let mut changed = true;
    let mut iterations = 0;
    while changed && iterations < 50 {
        changed = false;
        iterations += 1;
        let mut i = 0;
        while i < body.len() {
            let mut applied = false;
            for window_size in (2..=max_window.min(body.len() - i)).rev() {
                let window = &body[i..i + window_size];
                for rule in rules {
                    if (rule.matches)(window) {
                        let replacement = (rule.apply)(window);
                        let old_len = window_size;
                        body.splice(i..i + old_len, replacement.into_iter());
                        changed = true;
                        applied = true;
                        break;
                    }
                }
                if applied {
                    break;
                }
            }
            if !applied {
                i += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Cost Model
// ---------------------------------------------------------------------------

/// Quantitative metrics for a code sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct CostMetrics {
    pub instruction_count: usize,
    pub estimated_latency: f64,
    pub estimated_throughput: f64,
    pub memory_accesses: usize,
    pub branch_count: usize,
    pub register_pressure: usize,
}

impl Default for CostMetrics {
    fn default() -> Self {
        Self {
            instruction_count: 0,
            estimated_latency: 0.0,
            estimated_throughput: 0.0,
            memory_accesses: 0,
            branch_count: 0,
            register_pressure: 0,
        }
    }
}

/// Static cost model used to compare alternative code sequences.
#[derive(Debug, Clone)]
pub struct CostModel {
    /// Weight for instruction count in the composite score.
    pub w_instructions: f64,
    /// Weight for latency.
    pub w_latency: f64,
    /// Weight for memory accesses.
    pub w_memory: f64,
    /// Weight for branches.
    pub w_branches: f64,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            w_instructions: 1.0,
            w_latency: 2.0,
            w_memory: 1.5,
            w_branches: 1.0,
        }
    }
}

impl CostModel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Estimate cost metrics for a sequence of HIR nodes.
    pub fn estimate(&self, nodes: &[HirNode]) -> CostMetrics {
        let mut m = CostMetrics::default();
        m.instruction_count = nodes.len();

        let mut live: usize = 0;
        let mut max_live: usize = 0;

        for node in nodes {
            match &node.op {
                HirOp::Load | HirOp::Store | HirOp::Index => {
                    m.memory_accesses += 1;
                    m.estimated_latency += 4.0; // cache-hit assumption
                }
                HirOp::Branch | HirOp::Jump => {
                    m.branch_count += 1;
                    m.estimated_latency += 1.0;
                }
                HirOp::Mul | HirOp::Div | HirOp::Mod => {
                    m.estimated_latency += 3.0;
                }
                HirOp::Call { .. } => {
                    m.estimated_latency += 5.0;
                }
                _ => {
                    m.estimated_latency += 1.0;
                }
            }

            // Simple liveness heuristic: each node produces one value,
            // consumed by operands of subsequent nodes.
            live += 1;
            live = live.saturating_sub(node.operands.len());
            max_live = max_live.max(live);
        }

        m.register_pressure = max_live;
        m.estimated_throughput = if m.estimated_latency > 0.0 {
            m.instruction_count as f64 / m.estimated_latency
        } else {
            0.0
        };

        m
    }

    /// Compare two cost metrics: returns `Less` when `a` is cheaper (better).
    pub fn compare(&self, a: &CostMetrics, b: &CostMetrics) -> Ordering {
        let score_a = self.composite_score(a);
        let score_b = self.composite_score(b);
        score_a
            .partial_cmp(&score_b)
            .unwrap_or(Ordering::Equal)
    }

    fn composite_score(&self, m: &CostMetrics) -> f64 {
        self.w_instructions * m.instruction_count as f64
            + self.w_latency * m.estimated_latency
            + self.w_memory * m.memory_accesses as f64
            + self.w_branches * m.branch_count as f64
    }
}

// ---------------------------------------------------------------------------
// Optimisation Pipeline
// ---------------------------------------------------------------------------

/// Optimisation level, mirroring classic compiler `-O` flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    /// No optimizations.
    O0,
    /// Basic peephole and simple loop opts.
    O1,
    /// Full loop optimization + auto-vectorization analysis.
    O2,
    /// Aggressive: superoptimization on small blocks.
    O3,
}

/// Orchestrates all optimization passes over an HIR program.
#[derive(Debug, Clone)]
pub struct OptimizationPipeline {
    pub level: OptLevel,
    pub loop_config: LoopOptConfig,
    pub cost_model: CostModel,
}

impl OptimizationPipeline {
    pub fn new(level: OptLevel) -> Self {
        let loop_config = match level {
            OptLevel::O0 => LoopOptConfig {
                unroll_factor: 1,
                enable_fusion: false,
                enable_vectorization_hints: false,
                ..Default::default()
            },
            OptLevel::O1 => LoopOptConfig {
                unroll_factor: 2,
                enable_fusion: false,
                enable_vectorization_hints: false,
                ..Default::default()
            },
            OptLevel::O2 => LoopOptConfig::default(),
            OptLevel::O3 => LoopOptConfig {
                unroll_factor: 8,
                tile_size: 128,
                ..Default::default()
            },
        };
        Self {
            level,
            loop_config,
            cost_model: CostModel::default(),
        }
    }

    /// Run all applicable optimization passes on the program.
    pub fn optimize(&self, program: &mut HirProgram) {
        if self.level == OptLevel::O0 {
            return;
        }

        // Phase 1: peephole on every function.
        let rules = default_peephole_rules();
        for func in &mut program.functions {
            apply_peephole_rules(&mut func.body, &rules);
        }

        if self.level == OptLevel::O1 {
            return;
        }

        // Phase 2: loop optimizations.
        optimize_loops(program, &self.loop_config);

        if self.level == OptLevel::O2 {
            return;
        }

        // Phase 3 (O3): superoptimize small blocks.
        for func in &mut program.functions {
            superoptimize(func);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{HirId, HirNode, HirOp};

    fn make_node(id: usize, op: HirOp, operands: Vec<usize>) -> HirNode {
        HirNode {
            id: HirId(id),
            op,
            operands: operands.into_iter().map(HirId).collect(),
            ty: None,
            span: None,
        }
    }

    // -- Loop pattern detection -------------------------------------------

    #[test]
    fn detect_reduction_pattern() {
        let body = vec![
            make_node(0, HirOp::VarRef { name: "arr".into() }, vec![]),
            make_node(1, HirOp::VarRef { name: "i".into() }, vec![]),
            make_node(2, HirOp::Index, vec![0, 1]),
            make_node(3, HirOp::VarRef { name: "sum".into() }, vec![]),
            make_node(4, HirOp::Add, vec![3, 2]),
            make_node(
                5,
                HirOp::VarDef { name: "sum".into(), mutable: true },
                vec![4],
            ),
        ];
        let pattern = detect_loop_pattern(&body);
        match pattern {
            LoopPattern::Reduction { accumulator, operator, source_array } => {
                assert_eq!(accumulator, "sum");
                assert_eq!(operator, ReductionOp::Add);
                assert_eq!(source_array, "arr");
            }
            other => panic!("expected Reduction, got {:?}", other),
        }
    }

    #[test]
    fn detect_map_pattern() {
        let body = vec![
            make_node(0, HirOp::VarRef { name: "src".into() }, vec![]),
            make_node(1, HirOp::VarRef { name: "i".into() }, vec![]),
            make_node(2, HirOp::Index, vec![0, 1]),
            make_node(3, HirOp::Mul, vec![2, 2]),
            make_node(4, HirOp::VarRef { name: "dst".into() }, vec![]),
            make_node(5, HirOp::Index, vec![4, 1]),
            make_node(6, HirOp::Store, vec![5, 3]),
        ];
        let pattern = detect_loop_pattern(&body);
        match pattern {
            LoopPattern::Map { source_array, dest_array } => {
                assert_eq!(source_array, "src");
                assert_eq!(dest_array, "dst");
            }
            other => panic!("expected Map, got {:?}", other),
        }
    }

    #[test]
    fn detect_unknown_for_empty() {
        assert_eq!(detect_loop_pattern(&[]), LoopPattern::Unknown);
    }

    // -- Cost model -------------------------------------------------------

    #[test]
    fn cost_model_basic_estimate() {
        let cm = CostModel::new();
        let nodes = vec![
            make_node(0, HirOp::IntConst(1), vec![]),
            make_node(1, HirOp::IntConst(2), vec![]),
            make_node(2, HirOp::Add, vec![0, 1]),
        ];
        let metrics = cm.estimate(&nodes);
        assert_eq!(metrics.instruction_count, 3);
        assert!(metrics.estimated_latency > 0.0);
        assert!(metrics.estimated_throughput > 0.0);
        assert_eq!(metrics.memory_accesses, 0);
        assert_eq!(metrics.branch_count, 0);
    }

    #[test]
    fn cost_model_memory_ops() {
        let cm = CostModel::new();
        let nodes = vec![
            make_node(0, HirOp::Load, vec![]),
            make_node(1, HirOp::Store, vec![0]),
        ];
        let metrics = cm.estimate(&nodes);
        assert_eq!(metrics.memory_accesses, 2);
    }

    #[test]
    fn cost_model_compare_prefers_fewer_instructions() {
        let cm = CostModel::new();
        let cheap = CostMetrics {
            instruction_count: 2,
            estimated_latency: 2.0,
            estimated_throughput: 1.0,
            memory_accesses: 0,
            branch_count: 0,
            register_pressure: 1,
        };
        let expensive = CostMetrics {
            instruction_count: 10,
            estimated_latency: 20.0,
            estimated_throughput: 0.5,
            memory_accesses: 5,
            branch_count: 2,
            register_pressure: 4,
        };
        assert_eq!(cm.compare(&cheap, &expensive), Ordering::Less);
        assert_eq!(cm.compare(&expensive, &cheap), Ordering::Greater);
        assert_eq!(cm.compare(&cheap, &cheap), Ordering::Equal);
    }

    // -- Peephole rules ---------------------------------------------------

    #[test]
    fn peephole_add_zero_elimination() {
        let rules = default_peephole_rules();
        let mut body = vec![
            make_node(0, HirOp::IntConst(0), vec![]),
            make_node(1, HirOp::Add, vec![0, 99]),
        ];
        apply_peephole_rules(&mut body, &rules);
        assert_eq!(body.len(), 1);
        assert!(matches!(body[0].op, HirOp::VarRef { .. }));
    }

    #[test]
    fn peephole_mul_one_elimination() {
        let rules = default_peephole_rules();
        let mut body = vec![
            make_node(0, HirOp::IntConst(1), vec![]),
            make_node(1, HirOp::Mul, vec![0, 99]),
        ];
        apply_peephole_rules(&mut body, &rules);
        assert_eq!(body.len(), 1);
        assert!(matches!(body[0].op, HirOp::VarRef { .. }));
    }

    #[test]
    fn peephole_mul_zero_elimination() {
        let rules = default_peephole_rules();
        let mut body = vec![
            make_node(0, HirOp::IntConst(0), vec![]),
            make_node(1, HirOp::Mul, vec![0, 99]),
        ];
        apply_peephole_rules(&mut body, &rules);
        assert_eq!(body.len(), 1);
        assert!(matches!(body[0].op, HirOp::IntConst(0)));
    }

    #[test]
    fn peephole_strength_reduce_pow2() {
        let rules = default_peephole_rules();
        let mut body = vec![
            make_node(0, HirOp::IntConst(8), vec![]),
            make_node(1, HirOp::Mul, vec![0, 99]),
        ];
        apply_peephole_rules(&mut body, &rules);
        assert_eq!(body.len(), 2);
        assert!(matches!(body[0].op, HirOp::IntConst(3))); // log2(8)=3
        assert!(matches!(body[1].op, HirOp::Shl));
    }

    #[test]
    fn pipeline_o0_no_changes() {
        let pipeline = OptimizationPipeline::new(OptLevel::O0);
        let mut program = HirProgram {
            functions: vec![],
            structs: vec![],
            enums: vec![],
            type_context: crate::types::TypeContext::new(),
        };
        pipeline.optimize(&mut program);
        // O0 returns immediately, no crash.
    }
}