//! Higher-level optimization passes for the Axiom compiler.
//!
//! This module orchestrates optimizations beyond the e-graph rewrite rules,
//! including loop optimization, auto-vectorization analysis, superoptimization
//! for small blocks, constant folding, dead code elimination, common
//! subexpression elimination, strength reduction, branch optimization,
//! memory optimization, function inlining, tail-call detection, loop-invariant
//! code motion, and a cost model for comparing code sequences.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use crate::hir::{HirFunction, HirId, HirNode, HirOp, HirProgram};
use crate::types::TypeId;

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

/// Duplicate loop body nodes `factor` times with fresh HirIds.
fn unroll_loop(body: &mut Vec<HirNode>, loop_idx: usize, factor: usize) {
    if loop_idx >= body.len() {
        return;
    }
    let loop_node = &body[loop_idx];
    if loop_node.operands.len() < 2 {
        return;
    }

    let loop_body = extract_loop_body(body, loop_idx);
    if loop_body.is_empty() {
        return;
    }

    let max_id = body.iter().map(|n| n.id.0).max().unwrap_or(0);
    let body_len = loop_body.len();

    let body_start_id = body[loop_idx].operands[1];
    let start_pos = body.iter().position(|n| n.id == body_start_id).unwrap_or(loop_idx + 1);
    let end_pos = body[start_pos..]
        .iter()
        .position(|n| matches!(n.op, HirOp::Jump))
        .map(|off| start_pos + off + 1)
        .unwrap_or(body.len().min(start_pos + 20));

    let mut new_nodes = Vec::new();
    for copy_idx in 1..factor {
        let base_offset = max_id + 1;
        let id_offset = copy_idx * body_len + base_offset;

        let mut id_map: HashMap<HirId, HirId> = HashMap::new();
        for (i, node) in loop_body.iter().enumerate() {
            id_map.insert(node.id, HirId(id_offset + i));
        }

        for (i, node) in loop_body.iter().enumerate() {
            let new_operands: Vec<HirId> = node
                .operands
                .iter()
                .map(|op| *id_map.get(op).unwrap_or(op))
                .collect();
            new_nodes.push(HirNode {
                id: HirId(id_offset + i),
                op: node.op.clone(),
                operands: new_operands,
                ty: node.ty,
                span: node.span,
            });
        }
    }

    let insert_at = end_pos.min(body.len());
    for (i, node) in new_nodes.into_iter().enumerate() {
        body.insert(insert_at + i, node);
    }
}

/// Find consecutive Loop nodes with same condition operand and merge bodies.
fn fuse_adjacent_loops(body: &mut Vec<HirNode>) {
    let mut i = 0;
    while i + 1 < body.len() {
        let is_loop_pair = matches!(body[i].op, HirOp::Loop)
            && matches!(body[i + 1].op, HirOp::Loop);

        if is_loop_pair {
            let same_cond = !body[i].operands.is_empty()
                && !body[i + 1].operands.is_empty()
                && body[i].operands[0] == body[i + 1].operands[0];

            if same_cond {
                let second_body_ops = body[i + 1].operands.clone();
                for op in second_body_ops.iter().skip(1) {
                    if !body[i].operands.contains(op) {
                        body[i].operands.push(*op);
                    }
                }
                body.remove(i + 1);
                continue;
            }
        }
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// Auto-Vectorization Analysis
// ---------------------------------------------------------------------------

/// Describes how a loop can be vectorized.
#[derive(Debug, Clone)]
pub struct VectorizationPlan {
    pub loop_id: HirId,
    pub vector_width: usize,
    pub simd_ops: Vec<SimdMapping>,
    pub needs_tail_loop: bool,
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
        // --- New peephole rules ---
        PeepholeRule {
            name: "sub_zero_elim",
            matches: |nodes| {
                nodes.len() == 2
                    && matches!(nodes[0].op, HirOp::IntConst(0))
                    && matches!(nodes[1].op, HirOp::Sub)
                    && nodes[1].operands.len() >= 2
                    && nodes[1].operands[1] == nodes[0].id
            },
            apply: |nodes| {
                let sub = &nodes[1];
                let kept = sub.operands[0];
                vec![HirNode {
                    id: sub.id,
                    op: HirOp::VarRef {
                        name: format!("__peep_{}", kept.0),
                    },
                    operands: vec![kept],
                    ty: sub.ty,
                    span: sub.span,
                }]
            },
        },
        PeepholeRule {
            name: "div_one_elim",
            matches: |nodes| {
                nodes.len() == 2
                    && matches!(nodes[0].op, HirOp::IntConst(1))
                    && matches!(nodes[1].op, HirOp::Div)
                    && nodes[1].operands.len() >= 2
                    && nodes[1].operands[1] == nodes[0].id
            },
            apply: |nodes| {
                let div = &nodes[1];
                let kept = div.operands[0];
                vec![HirNode {
                    id: div.id,
                    op: HirOp::VarRef {
                        name: format!("__peep_{}", kept.0),
                    },
                    operands: vec![kept],
                    ty: div.ty,
                    span: div.span,
                }]
            },
        },
        PeepholeRule {
            name: "and_self",
            matches: |nodes| {
                nodes.len() == 1
                    && matches!(nodes[0].op, HirOp::BitAnd)
                    && nodes[0].operands.len() == 2
                    && nodes[0].operands[0] == nodes[0].operands[1]
            },
            apply: |nodes| {
                let kept = nodes[0].operands[0];
                vec![HirNode {
                    id: nodes[0].id,
                    op: HirOp::VarRef {
                        name: format!("__peep_{}", kept.0),
                    },
                    operands: vec![kept],
                    ty: nodes[0].ty,
                    span: nodes[0].span,
                }]
            },
        },
        PeepholeRule {
            name: "or_self",
            matches: |nodes| {
                nodes.len() == 1
                    && matches!(nodes[0].op, HirOp::BitOr)
                    && nodes[0].operands.len() == 2
                    && nodes[0].operands[0] == nodes[0].operands[1]
            },
            apply: |nodes| {
                let kept = nodes[0].operands[0];
                vec![HirNode {
                    id: nodes[0].id,
                    op: HirOp::VarRef {
                        name: format!("__peep_{}", kept.0),
                    },
                    operands: vec![kept],
                    ty: nodes[0].ty,
                    span: nodes[0].span,
                }]
            },
        },
        PeepholeRule {
            name: "xor_self",
            matches: |nodes| {
                nodes.len() == 1
                    && matches!(nodes[0].op, HirOp::BitXor)
                    && nodes[0].operands.len() == 2
                    && nodes[0].operands[0] == nodes[0].operands[1]
            },
            apply: |nodes| {
                vec![HirNode {
                    id: nodes[0].id,
                    op: HirOp::IntConst(0),
                    operands: vec![],
                    ty: nodes[0].ty,
                    span: nodes[0].span,
                }]
            },
        },
        PeepholeRule {
            name: "and_zero",
            matches: |nodes| {
                nodes.len() == 2
                    && matches!(nodes[0].op, HirOp::IntConst(0))
                    && matches!(nodes[1].op, HirOp::BitAnd)
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
            name: "or_neg_one",
            matches: |nodes| {
                nodes.len() == 2
                    && matches!(nodes[0].op, HirOp::IntConst(-1))
                    && matches!(nodes[1].op, HirOp::BitOr)
            },
            apply: |nodes| {
                vec![HirNode {
                    id: nodes[1].id,
                    op: HirOp::IntConst(-1),
                    operands: vec![],
                    ty: nodes[1].ty,
                    span: nodes[1].span,
                }]
            },
        },
        PeepholeRule {
            name: "shl_zero",
            matches: |nodes| {
                nodes.len() == 2
                    && matches!(nodes[0].op, HirOp::IntConst(0))
                    && matches!(nodes[1].op, HirOp::Shl)
                    && nodes[1].operands.len() >= 2
                    && nodes[1].operands[1] == nodes[0].id
            },
            apply: |nodes| {
                let shl = &nodes[1];
                let kept = shl.operands[0];
                vec![HirNode {
                    id: shl.id,
                    op: HirOp::VarRef {
                        name: format!("__peep_{}", kept.0),
                    },
                    operands: vec![kept],
                    ty: shl.ty,
                    span: shl.span,
                }]
            },
        },
        PeepholeRule {
            name: "shr_zero",
            matches: |nodes| {
                nodes.len() == 2
                    && matches!(nodes[0].op, HirOp::IntConst(0))
                    && matches!(nodes[1].op, HirOp::Shr)
                    && nodes[1].operands.len() >= 2
                    && nodes[1].operands[1] == nodes[0].id
            },
            apply: |nodes| {
                let shr = &nodes[1];
                let kept = shr.operands[0];
                vec![HirNode {
                    id: shr.id,
                    op: HirOp::VarRef {
                        name: format!("__peep_{}", kept.0),
                    },
                    operands: vec![kept],
                    ty: shr.ty,
                    span: shr.span,
                }]
            },
        },
        PeepholeRule {
            name: "not_not",
            matches: |nodes| {
                nodes.len() == 2
                    && matches!(nodes[0].op, HirOp::BitNot)
                    && matches!(nodes[1].op, HirOp::BitNot)
                    && nodes[1].operands.contains(&nodes[0].id)
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
            name: "branch_const_true",
            matches: |nodes| {
                nodes.len() == 2
                    && matches!(nodes[0].op, HirOp::BoolConst(true))
                    && matches!(nodes[1].op, HirOp::Branch)
                    && nodes[1].operands.len() >= 2
            },
            apply: |nodes| {
                let branch = &nodes[1];
                let then_target = branch.operands[1];
                vec![HirNode {
                    id: branch.id,
                    op: HirOp::Jump,
                    operands: vec![then_target],
                    ty: branch.ty,
                    span: branch.span,
                }]
            },
        },
        PeepholeRule {
            name: "branch_const_false",
            matches: |nodes| {
                nodes.len() == 2
                    && matches!(nodes[0].op, HirOp::BoolConst(false))
                    && matches!(nodes[1].op, HirOp::Branch)
                    && nodes[1].operands.len() >= 3
            },
            apply: |nodes| {
                let branch = &nodes[1];
                let else_target = branch.operands[2];
                vec![HirNode {
                    id: branch.id,
                    op: HirOp::Jump,
                    operands: vec![else_target],
                    ty: branch.ty,
                    span: branch.span,
                }]
            },
        },
        PeepholeRule {
            name: "consecutive_store",
            matches: |nodes| {
                nodes.len() == 2
                    && matches!(nodes[0].op, HirOp::Store)
                    && matches!(nodes[1].op, HirOp::Store)
                    && !nodes[0].operands.is_empty()
                    && !nodes[1].operands.is_empty()
                    && nodes[0].operands[0] == nodes[1].operands[0]
            },
            apply: |nodes| {
                vec![nodes[1].clone()]
            },
        },
        PeepholeRule {
            name: "load_after_store",
            matches: |nodes| {
                nodes.len() == 2
                    && matches!(nodes[0].op, HirOp::Store)
                    && matches!(nodes[1].op, HirOp::Load)
                    && !nodes[0].operands.is_empty()
                    && !nodes[1].operands.is_empty()
                    && nodes[0].operands[0] == nodes[1].operands[0]
            },
            apply: |nodes| {
                let store = &nodes[0];
                let load = &nodes[1];
                let val = if store.operands.len() >= 2 {
                    store.operands[1]
                } else {
                    store.operands[0]
                };
                vec![
                    store.clone(),
                    HirNode {
                        id: load.id,
                        op: HirOp::VarRef {
                            name: format!("__peep_{}", val.0),
                        },
                        operands: vec![val],
                        ty: load.ty,
                        span: load.span,
                    },
                ]
            },
        },
        PeepholeRule {
            name: "identity_bitand_neg",
            matches: |nodes| {
                nodes.len() == 2
                    && matches!(nodes[0].op, HirOp::IntConst(-1))
                    && matches!(nodes[1].op, HirOp::BitAnd)
            },
            apply: |nodes| {
                let band = &nodes[1];
                let kept = band
                    .operands
                    .iter()
                    .copied()
                    .find(|id| *id != nodes[0].id)
                    .unwrap_or(band.operands[0]);
                vec![HirNode {
                    id: band.id,
                    op: HirOp::VarRef {
                        name: format!("__peep_{}", kept.0),
                    },
                    operands: vec![kept],
                    ty: band.ty,
                    span: band.span,
                }]
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
            let max_ws = max_window.min(body.len() - i);
            for window_size in (1..=max_ws).rev() {
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
    pub w_instructions: f64,
    pub w_latency: f64,
    pub w_memory: f64,
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
                    m.estimated_latency += 4.0;
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
// Constant Folding / Propagation
// ---------------------------------------------------------------------------

/// Fold constant expressions and propagate known constant values.
pub fn constant_folding(body: &mut Vec<HirNode>) {
    let mut changed = true;
    while changed {
        changed = false;

        let const_vals: HashMap<HirId, i64> = body
            .iter()
            .filter_map(|n| {
                if let HirOp::IntConst(v) = n.op {
                    Some((n.id, v))
                } else {
                    None
                }
            })
            .collect();

        for node in body.iter_mut() {
            let folded = match &node.op {
                HirOp::Add | HirOp::Sub | HirOp::Mul | HirOp::Div | HirOp::Mod
                | HirOp::BitAnd | HirOp::BitOr | HirOp::BitXor | HirOp::Shl | HirOp::Shr => {
                    if node.operands.len() == 2 {
                        let a = const_vals.get(&node.operands[0]);
                        let b = const_vals.get(&node.operands[1]);
                        if let (Some(&av), Some(&bv)) = (a, b) {
                            match &node.op {
                                HirOp::Add => Some(av.wrapping_add(bv)),
                                HirOp::Sub => Some(av.wrapping_sub(bv)),
                                HirOp::Mul => Some(av.wrapping_mul(bv)),
                                HirOp::Div => {
                                    if bv != 0 { Some(av.wrapping_div(bv)) } else { None }
                                }
                                HirOp::Mod => {
                                    if bv != 0 { Some(av.wrapping_rem(bv)) } else { None }
                                }
                                HirOp::BitAnd => Some(av & bv),
                                HirOp::BitOr => Some(av | bv),
                                HirOp::BitXor => Some(av ^ bv),
                                HirOp::Shl => {
                                    if bv >= 0 && bv < 64 { Some(av.wrapping_shl(bv as u32)) } else { None }
                                }
                                HirOp::Shr => {
                                    if bv >= 0 && bv < 64 { Some(av.wrapping_shr(bv as u32)) } else { None }
                                }
                                _ => None,
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                HirOp::Neg => {
                    if node.operands.len() == 1 {
                        const_vals.get(&node.operands[0]).map(|v| v.wrapping_neg())
                    } else {
                        None
                    }
                }
                _ => None,
            };

            if let Some(result) = folded {
                node.op = HirOp::IntConst(result);
                node.operands.clear();
                changed = true;
            }
        }

        // Constant propagation: if VarDef binds to IntConst, replace VarRef
        let mut var_consts: HashMap<String, i64> = HashMap::new();
        for node in body.iter() {
            if let HirOp::VarDef { name, .. } = &node.op {
                if node.operands.len() == 1 {
                    if let Some(&val) = const_vals.get(&node.operands[0]) {
                        var_consts.insert(name.clone(), val);
                    }
                }
            }
        }

        for node in body.iter_mut() {
            if let HirOp::VarRef { name } = &node.op {
                if let Some(&val) = var_consts.get(name) {
                    node.op = HirOp::IntConst(val);
                    node.operands.clear();
                    changed = true;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dead Code Elimination
// ---------------------------------------------------------------------------

fn is_side_effectful(op: &HirOp) -> bool {
    matches!(
        op,
        HirOp::Store
            | HirOp::Call { .. }
            | HirOp::Return
            | HirOp::Branch
            | HirOp::Jump
            | HirOp::Loop
            | HirOp::Break
            | HirOp::Continue
            | HirOp::VarDef { .. }
            | HirOp::Block
    )
}

/// Remove nodes whose results are never used, keeping side-effectful ops.
pub fn dead_code_elimination(body: &mut Vec<HirNode>) {
    let mut changed = true;
    while changed {
        changed = false;
        let used_ids: HashSet<HirId> = body
            .iter()
            .flat_map(|n| n.operands.iter().copied())
            .collect();

        let old_len = body.len();
        body.retain(|node| {
            used_ids.contains(&node.id) || is_side_effectful(&node.op)
        });
        if body.len() != old_len {
            changed = true;
        }
    }
}

// ---------------------------------------------------------------------------
// Common Subexpression Elimination
// ---------------------------------------------------------------------------

fn op_key(op: &HirOp) -> Option<String> {
    match op {
        HirOp::Add => Some("Add".into()),
        HirOp::Sub => Some("Sub".into()),
        HirOp::Mul => Some("Mul".into()),
        HirOp::Div => Some("Div".into()),
        HirOp::Mod => Some("Mod".into()),
        HirOp::Neg => Some("Neg".into()),
        HirOp::Eq => Some("Eq".into()),
        HirOp::Ne => Some("Ne".into()),
        HirOp::Lt => Some("Lt".into()),
        HirOp::Gt => Some("Gt".into()),
        HirOp::Le => Some("Le".into()),
        HirOp::Ge => Some("Ge".into()),
        HirOp::And => Some("And".into()),
        HirOp::Or => Some("Or".into()),
        HirOp::Not => Some("Not".into()),
        HirOp::BitAnd => Some("BitAnd".into()),
        HirOp::BitOr => Some("BitOr".into()),
        HirOp::BitXor => Some("BitXor".into()),
        HirOp::BitNot => Some("BitNot".into()),
        HirOp::Shl => Some("Shl".into()),
        HirOp::Shr => Some("Shr".into()),
        HirOp::Index => Some("Index".into()),
        HirOp::IntConst(v) => Some(format!("IntConst({})", v)),
        HirOp::BoolConst(v) => Some(format!("BoolConst({})", v)),
        _ => None,
    }
}

/// Eliminate common subexpressions by rewriting downstream operands.
pub fn common_subexpression_elimination(body: &mut Vec<HirNode>) {
    let mut seen: HashMap<(String, Vec<HirId>), HirId> = HashMap::new();
    let mut replacements: HashMap<HirId, HirId> = HashMap::new();

    for node in body.iter() {
        if let Some(key) = op_key(&node.op) {
            let expr_key = (key, node.operands.clone());
            if let Some(&first_id) = seen.get(&expr_key) {
                replacements.insert(node.id, first_id);
            } else {
                seen.insert(expr_key, node.id);
            }
        }
    }

    if replacements.is_empty() {
        return;
    }

    for node in body.iter_mut() {
        for op in node.operands.iter_mut() {
            if let Some(&replacement) = replacements.get(op) {
                *op = replacement;
            }
        }
    }

    body.retain(|node| !replacements.contains_key(&node.id));
}

// ---------------------------------------------------------------------------
// Loop-Invariant Code Motion (LICM)
// ---------------------------------------------------------------------------

/// Hoist loop-invariant computations before the loop.
pub fn loop_invariant_code_motion(body: &mut Vec<HirNode>) {
    let mut i = 0;
    while i < body.len() {
        if !matches!(body[i].op, HirOp::Loop) {
            i += 1;
            continue;
        }

        let loop_idx = i;
        let loop_node = &body[loop_idx];
        if loop_node.operands.len() < 2 {
            i += 1;
            continue;
        }

        let body_start_id = loop_node.operands[1];
        let start = body.iter().position(|n| n.id == body_start_id);
        let (body_start, body_end) = match start {
            Some(s) => {
                let end = body[s..]
                    .iter()
                    .position(|n| matches!(n.op, HirOp::Jump))
                    .map(|off| s + off + 1)
                    .unwrap_or(body.len().min(s + 20));
                (s, end)
            }
            None => {
                i += 1;
                continue;
            }
        };

        let loop_body_ids: HashSet<HirId> = body[body_start..body_end]
            .iter()
            .map(|n| n.id)
            .collect();

        let mut to_hoist: Vec<usize> = Vec::new();
        for j in body_start..body_end {
            let node = &body[j];
            if is_side_effectful(&node.op) {
                continue;
            }
            let all_external = node
                .operands
                .iter()
                .all(|op| !loop_body_ids.contains(op));
            if all_external && !node.operands.is_empty() {
                to_hoist.push(j);
            }
        }

        let mut offset = 0;
        for &idx in &to_hoist {
            let actual_idx = idx - offset;
            let node = body.remove(actual_idx);
            body.insert(loop_idx, node);
            offset += 1;
        }

        i += 1 + to_hoist.len();
    }
}

// ---------------------------------------------------------------------------
// Strength Reduction
// ---------------------------------------------------------------------------

fn is_power_of_two(v: i64) -> bool {
    v > 0 && (v as u64).is_power_of_two()
}

fn log2_of(v: i64) -> i64 {
    v.trailing_zeros() as i64
}

/// Apply strength reduction transformations.
pub fn strength_reduction(body: &mut Vec<HirNode>) {
    let const_vals: HashMap<HirId, i64> = body
        .iter()
        .filter_map(|n| {
            if let HirOp::IntConst(v) = n.op {
                Some((n.id, v))
            } else {
                None
            }
        })
        .collect();

    let max_id = body.iter().map(|n| n.id.0).max().unwrap_or(0);
    let mut next_id = max_id + 1;
    let mut insertions: Vec<(usize, Vec<HirNode>)> = Vec::new();

    for (i, node) in body.iter_mut().enumerate() {
        match &node.op {
            HirOp::Div => {
                if node.operands.len() == 2 {
                    if let Some(&divisor) = const_vals.get(&node.operands[1]) {
                        if is_power_of_two(divisor) {
                            let shift = log2_of(divisor);
                            let x = node.operands[0];
                            let shift_id = HirId(next_id);
                            next_id += 1;
                            let shift_node = HirNode {
                                id: shift_id,
                                op: HirOp::IntConst(shift),
                                operands: vec![],
                                ty: node.ty,
                                span: node.span,
                            };
                            node.op = HirOp::Shr;
                            node.operands = vec![x, shift_id];
                            insertions.push((i, vec![shift_node]));
                        }
                    }
                }
            }
            HirOp::Mod => {
                if node.operands.len() == 2 {
                    if let Some(&divisor) = const_vals.get(&node.operands[1]) {
                        if is_power_of_two(divisor) {
                            let mask = divisor - 1;
                            let x = node.operands[0];
                            let mask_id = HirId(next_id);
                            next_id += 1;
                            let mask_node = HirNode {
                                id: mask_id,
                                op: HirOp::IntConst(mask),
                                operands: vec![],
                                ty: node.ty,
                                span: node.span,
                            };
                            node.op = HirOp::BitAnd;
                            node.operands = vec![x, mask_id];
                            insertions.push((i, vec![mask_node]));
                        }
                    }
                }
            }
            HirOp::Mul => {
                if node.operands.len() == 2 {
                    let (const_op_idx, val) =
                        if let Some(&v) = const_vals.get(&node.operands[1]) {
                            (1, Some(v))
                        } else if let Some(&v) = const_vals.get(&node.operands[0]) {
                            (0, Some(v))
                        } else {
                            (0, None)
                        };

                    if let Some(v) = val {
                        let x = node.operands[1 - const_op_idx];
                        match v {
                            3 => {
                                let one_id = HirId(next_id);
                                next_id += 1;
                                let shl_id = HirId(next_id);
                                next_id += 1;
                                let one_node = HirNode {
                                    id: one_id,
                                    op: HirOp::IntConst(1),
                                    operands: vec![],
                                    ty: node.ty,
                                    span: node.span,
                                };
                                let shl_node = HirNode {
                                    id: shl_id,
                                    op: HirOp::Shl,
                                    operands: vec![x, one_id],
                                    ty: node.ty,
                                    span: node.span,
                                };
                                node.op = HirOp::Add;
                                node.operands = vec![x, shl_id];
                                insertions.push((i, vec![one_node, shl_node]));
                            }
                            5 => {
                                let two_id = HirId(next_id);
                                next_id += 1;
                                let shl_id = HirId(next_id);
                                next_id += 1;
                                let two_node = HirNode {
                                    id: two_id,
                                    op: HirOp::IntConst(2),
                                    operands: vec![],
                                    ty: node.ty,
                                    span: node.span,
                                };
                                let shl_node = HirNode {
                                    id: shl_id,
                                    op: HirOp::Shl,
                                    operands: vec![x, two_id],
                                    ty: node.ty,
                                    span: node.span,
                                };
                                node.op = HirOp::Add;
                                node.operands = vec![x, shl_id];
                                insertions.push((i, vec![two_node, shl_node]));
                            }
                            7 => {
                                let three_id = HirId(next_id);
                                next_id += 1;
                                let shl_id = HirId(next_id);
                                next_id += 1;
                                let three_node = HirNode {
                                    id: three_id,
                                    op: HirOp::IntConst(3),
                                    operands: vec![],
                                    ty: node.ty,
                                    span: node.span,
                                };
                                let shl_node = HirNode {
                                    id: shl_id,
                                    op: HirOp::Shl,
                                    operands: vec![x, three_id],
                                    ty: node.ty,
                                    span: node.span,
                                };
                                node.op = HirOp::Sub;
                                node.operands = vec![shl_id, x];
                                insertions.push((i, vec![three_node, shl_node]));
                            }
                            _ => {}
                        }
                    }
                }
            }
            _ => {}
        }
    }

    let mut offset = 0;
    insertions.sort_by_key(|(i, _)| *i);
    for (idx, nodes) in insertions {
        let insert_at = idx + offset;
        for (j, node) in nodes.into_iter().enumerate() {
            body.insert(insert_at + j, node);
            offset += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Branch Optimization
// ---------------------------------------------------------------------------

/// Simplify branches with constant conditions.
pub fn branch_optimization(body: &mut Vec<HirNode>) {
    let const_vals: HashMap<HirId, bool> = body
        .iter()
        .filter_map(|n| {
            if let HirOp::BoolConst(v) = n.op {
                Some((n.id, v))
            } else {
                None
            }
        })
        .collect();

    for node in body.iter_mut() {
        if matches!(node.op, HirOp::Branch) && !node.operands.is_empty() {
            let cond_id = node.operands[0];
            if let Some(&cond_val) = const_vals.get(&cond_id) {
                if cond_val && node.operands.len() >= 2 {
                    let then_target = node.operands[1];
                    node.op = HirOp::Jump;
                    node.operands = vec![then_target];
                } else if !cond_val && node.operands.len() >= 3 {
                    let else_target = node.operands[2];
                    node.op = HirOp::Jump;
                    node.operands = vec![else_target];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Memory Optimization
// ---------------------------------------------------------------------------

/// Optimize memory access patterns.
pub fn memory_optimization(body: &mut Vec<HirNode>) {
    // Load-after-store
    let mut i = 0;
    while i + 1 < body.len() {
        if matches!(body[i].op, HirOp::Store)
            && matches!(body[i + 1].op, HirOp::Load)
            && !body[i].operands.is_empty()
            && !body[i + 1].operands.is_empty()
            && body[i].operands[0] == body[i + 1].operands[0]
        {
            let val = if body[i].operands.len() >= 2 {
                body[i].operands[1]
            } else {
                body[i].operands[0]
            };
            body[i + 1].op = HirOp::VarRef {
                name: format!("__mem_{}", val.0),
            };
            body[i + 1].operands = vec![val];
        }
        i += 1;
    }

    // Dead store
    let mut i = 0;
    while i + 1 < body.len() {
        if matches!(body[i].op, HirOp::Store)
            && matches!(body[i + 1].op, HirOp::Store)
            && !body[i].operands.is_empty()
            && !body[i + 1].operands.is_empty()
            && body[i].operands[0] == body[i + 1].operands[0]
        {
            body.remove(i);
            continue;
        }
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// Function Inlining
// ---------------------------------------------------------------------------

/// Inline functions whose body is smaller than `max_size` nodes.
pub fn inline_small_functions(program: &mut HirProgram, max_size: usize) {
    let small_funcs: HashMap<String, HirFunction> = program
        .functions
        .iter()
        .filter(|f| f.body.len() < max_size && !f.body.is_empty())
        .map(|f| (f.name.clone(), f.clone()))
        .collect();

    if small_funcs.is_empty() {
        return;
    }

    let global_max_id = program
        .functions
        .iter()
        .flat_map(|f| f.body.iter().map(|n| n.id.0))
        .max()
        .unwrap_or(0);
    let mut next_id = global_max_id + 1000;

    for func in &mut program.functions {
        let mut i = 0;
        while i < func.body.len() {
            let should_inline = if let HirOp::Call { name } = &func.body[i].op {
                if let Some(target) = small_funcs.get(name) {
                    if target.name != func.name {
                        Some((name.clone(), func.body[i].id, func.body[i].operands.clone()))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            if let Some((callee_name, call_id, call_operands)) = should_inline {
                let target = &small_funcs[&callee_name];
                let mut inlined_nodes = Vec::new();
                let mut id_map: HashMap<HirId, HirId> = HashMap::new();

                for node in &target.body {
                    if let HirOp::Param { index } = &node.op {
                        if *index < call_operands.len() {
                            id_map.insert(node.id, call_operands[*index]);
                        }
                    }
                }

                for node in &target.body {
                    if !id_map.contains_key(&node.id) {
                        id_map.insert(node.id, HirId(next_id));
                        next_id += 1;
                    }
                }

                for node in &target.body {
                    if matches!(node.op, HirOp::Param { .. }) {
                        continue;
                    }
                    if matches!(node.op, HirOp::Return) {
                        if let Some(&ret_val) = node.operands.first() {
                            let mapped_val = *id_map.get(&ret_val).unwrap_or(&ret_val);
                            id_map.insert(node.id, mapped_val);
                            id_map.entry(call_id).or_insert(mapped_val);
                        }
                        continue;
                    }

                    let new_id = *id_map.get(&node.id).unwrap_or(&node.id);
                    let new_operands: Vec<HirId> = node
                        .operands
                        .iter()
                        .map(|op| *id_map.get(op).unwrap_or(op))
                        .collect();
                    inlined_nodes.push(HirNode {
                        id: new_id,
                        op: node.op.clone(),
                        operands: new_operands,
                        ty: node.ty,
                        span: node.span,
                    });
                }

                if !inlined_nodes.is_empty() {
                    let last_inlined_id = inlined_nodes.last().unwrap().id;
                    body_replace_call(&mut func.body, i, &inlined_nodes, call_id, last_inlined_id);
                    i += inlined_nodes.len();
                } else {
                    i += 1;
                }
            } else {
                i += 1;
            }
        }
    }
}

fn body_replace_call(
    body: &mut Vec<HirNode>,
    call_idx: usize,
    inlined: &[HirNode],
    call_id: HirId,
    replacement_id: HirId,
) {
    body.remove(call_idx);
    for (j, node) in inlined.iter().enumerate() {
        body.insert(call_idx + j, node.clone());
    }
    if call_id != replacement_id {
        for node in body.iter_mut() {
            for op in node.operands.iter_mut() {
                if *op == call_id {
                    *op = replacement_id;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tail Call Optimization
// ---------------------------------------------------------------------------

/// Information about a detected tail call.
#[derive(Debug, Clone)]
pub struct TailCallInfo {
    pub function_name: String,
    pub call_node: HirId,
}

/// Detect tail calls: Return whose operand is a Call node.
pub fn detect_tail_calls(func: &HirFunction) -> Vec<TailCallInfo> {
    let mut result = Vec::new();

    let call_ids: HashMap<HirId, String> = func
        .body
        .iter()
        .filter_map(|n| {
            if let HirOp::Call { name } = &n.op {
                Some((n.id, name.clone()))
            } else {
                None
            }
        })
        .collect();

    for node in &func.body {
        if matches!(node.op, HirOp::Return) {
            for op_id in &node.operands {
                if let Some(callee) = call_ids.get(op_id) {
                    result.push(TailCallInfo {
                        function_name: callee.clone(),
                        call_node: *op_id,
                    });
                }
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Optimization Statistics
// ---------------------------------------------------------------------------

/// Statistics about optimizations applied.
#[derive(Debug, Clone, Default)]
pub struct OptimizationStats {
    pub dce_removed: usize,
    pub constants_folded: usize,
    pub cse_eliminated: usize,
    pub licm_hoisted: usize,
    pub loops_unrolled: usize,
    pub loops_fused: usize,
    pub branches_optimized: usize,
    pub memory_ops_optimized: usize,
    pub functions_inlined: usize,
    pub tail_calls_detected: usize,
    pub strength_reductions: usize,
    pub peephole_applied: usize,
}

// ---------------------------------------------------------------------------
// Optimisation Pipeline
// ---------------------------------------------------------------------------

/// Optimisation level, mirroring classic compiler `-O` flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    O0,
    O1,
    O2,
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

        // O1: DCE + constant folding + basic peephole
        let rules = default_peephole_rules();
        for func in &mut program.functions {
            dead_code_elimination(&mut func.body);
            constant_folding(&mut func.body);
            apply_peephole_rules(&mut func.body, &rules);
        }

        if self.level == OptLevel::O1 {
            return;
        }

        // O2: + CSE + LICM + loop opts + strength reduction
        for func in &mut program.functions {
            common_subexpression_elimination(&mut func.body);
            loop_invariant_code_motion(&mut func.body);
            strength_reduction(&mut func.body);
        }
        optimize_loops(program, &self.loop_config);

        if self.level == OptLevel::O2 {
            return;
        }

        // O3: + inlining + superoptimization + branch opt + memory opt
        inline_small_functions(program, 20);
        for func in &mut program.functions {
            superoptimize(func);
            branch_optimization(&mut func.body);
            memory_optimization(&mut func.body);
        }
        for func in &program.functions {
            let _tail_calls = detect_tail_calls(func);
        }
    }

    /// Run all optimization passes and collect statistics.
    pub fn optimize_with_stats(&self, program: &mut HirProgram) -> OptimizationStats {
        let mut stats = OptimizationStats::default();

        if self.level == OptLevel::O0 {
            return stats;
        }

        let rules = default_peephole_rules();
        for func in &mut program.functions {
            let before = func.body.len();
            dead_code_elimination(&mut func.body);
            let after = func.body.len();
            stats.dce_removed += before.saturating_sub(after);

            let before_fold: usize = func.body.iter()
                .filter(|n| matches!(n.op, HirOp::IntConst(_)))
                .count();
            constant_folding(&mut func.body);
            let after_fold: usize = func.body.iter()
                .filter(|n| matches!(n.op, HirOp::IntConst(_)))
                .count();
            stats.constants_folded += after_fold.saturating_sub(before_fold);

            let before = func.body.len();
            apply_peephole_rules(&mut func.body, &rules);
            let after = func.body.len();
            stats.peephole_applied += before.saturating_sub(after);
        }

        if self.level == OptLevel::O1 {
            return stats;
        }

        for func in &mut program.functions {
            let before = func.body.len();
            common_subexpression_elimination(&mut func.body);
            let after = func.body.len();
            stats.cse_eliminated += before.saturating_sub(after);

            let before_ids: Vec<HirId> = func.body.iter().map(|n| n.id).collect();
            loop_invariant_code_motion(&mut func.body);
            let after_ids: Vec<HirId> = func.body.iter().map(|n| n.id).collect();
            if before_ids != after_ids {
                stats.licm_hoisted += 1;
            }

            let sr_before = func.body.len();
            strength_reduction(&mut func.body);
            let sr_after = func.body.len();
            if sr_after != sr_before {
                stats.strength_reductions += sr_after.saturating_sub(sr_before);
            }
        }

        let loop_count_before: usize = program.functions.iter()
            .flat_map(|f| f.body.iter())
            .filter(|n| matches!(n.op, HirOp::Loop))
            .count();
        optimize_loops(program, &self.loop_config);
        let loop_count_after: usize = program.functions.iter()
            .flat_map(|f| f.body.iter())
            .filter(|n| matches!(n.op, HirOp::Loop))
            .count();
        if loop_count_after < loop_count_before {
            stats.loops_fused += loop_count_before - loop_count_after;
        }
        stats.loops_unrolled += program.functions.iter()
            .flat_map(|f| f.body.iter())
            .filter(|n| matches!(n.op, HirOp::Nop))
            .count();

        if self.level == OptLevel::O2 {
            return stats;
        }

        let total_calls_before: usize = program.functions.iter()
            .flat_map(|f| f.body.iter())
            .filter(|n| matches!(n.op, HirOp::Call { .. }))
            .count();
        inline_small_functions(program, 20);
        let total_calls_after: usize = program.functions.iter()
            .flat_map(|f| f.body.iter())
            .filter(|n| matches!(n.op, HirOp::Call { .. }))
            .count();
        stats.functions_inlined += total_calls_before.saturating_sub(total_calls_after);

        for func in &mut program.functions {
            let before = func.body.len();
            superoptimize(func);
            let after = func.body.len();
            stats.peephole_applied += before.saturating_sub(after);

            let branch_before = func.body.iter()
                .filter(|n| matches!(n.op, HirOp::Branch))
                .count();
            branch_optimization(&mut func.body);
            let branch_after = func.body.iter()
                .filter(|n| matches!(n.op, HirOp::Branch))
                .count();
            stats.branches_optimized += branch_before.saturating_sub(branch_after);

            let mem_before = func.body.len();
            memory_optimization(&mut func.body);
            let mem_after = func.body.len();
            stats.memory_ops_optimized += mem_before.saturating_sub(mem_after);
        }

        for func in &program.functions {
            let tail_calls = detect_tail_calls(func);
            stats.tail_calls_detected += tail_calls.len();
        }

        stats
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
    }

    // -- New optimization pass tests --------------------------------------

    #[test]
    fn test_constant_folding_add() {
        let mut body = vec![
            make_node(0, HirOp::IntConst(3), vec![]),
            make_node(1, HirOp::IntConst(5), vec![]),
            make_node(2, HirOp::Add, vec![0, 1]),
        ];
        constant_folding(&mut body);
        let result = body.iter().find(|n| n.id == HirId(2)).unwrap();
        assert!(matches!(result.op, HirOp::IntConst(8)));
    }

    #[test]
    fn test_constant_folding_mul() {
        let mut body = vec![
            make_node(0, HirOp::IntConst(4), vec![]),
            make_node(1, HirOp::IntConst(6), vec![]),
            make_node(2, HirOp::Mul, vec![0, 1]),
        ];
        constant_folding(&mut body);
        let result = body.iter().find(|n| n.id == HirId(2)).unwrap();
        assert!(matches!(result.op, HirOp::IntConst(24)));
    }

    #[test]
    fn test_constant_folding_sub() {
        let mut body = vec![
            make_node(0, HirOp::IntConst(10), vec![]),
            make_node(1, HirOp::IntConst(3), vec![]),
            make_node(2, HirOp::Sub, vec![0, 1]),
        ];
        constant_folding(&mut body);
        let result = body.iter().find(|n| n.id == HirId(2)).unwrap();
        assert!(matches!(result.op, HirOp::IntConst(7)));
    }

    #[test]
    fn test_dce_removes_unused() {
        let mut body = vec![
            make_node(0, HirOp::IntConst(1), vec![]),
            make_node(1, HirOp::IntConst(2), vec![]),
            make_node(2, HirOp::Add, vec![0, 0]),
            make_node(3, HirOp::Mul, vec![1, 1]),
            make_node(4, HirOp::Return, vec![2]),
        ];
        dead_code_elimination(&mut body);
        assert!(!body.iter().any(|n| n.id == HirId(3)));
    }

    #[test]
    fn test_dce_keeps_side_effects() {
        let mut body = vec![
            make_node(0, HirOp::IntConst(42), vec![]),
            make_node(1, HirOp::Store, vec![0]),
        ];
        dead_code_elimination(&mut body);
        assert!(body.iter().any(|n| matches!(n.op, HirOp::Store)));
    }

    #[test]
    fn test_cse_basic() {
        let mut body = vec![
            make_node(0, HirOp::IntConst(5), vec![]),
            make_node(1, HirOp::IntConst(10), vec![]),
            make_node(2, HirOp::Add, vec![0, 1]),
            make_node(3, HirOp::Add, vec![0, 1]),
            make_node(4, HirOp::Return, vec![3]),
        ];
        common_subexpression_elimination(&mut body);
        assert!(!body.iter().any(|n| n.id == HirId(3)));
        let ret = body.iter().find(|n| n.id == HirId(4)).unwrap();
        assert_eq!(ret.operands[0], HirId(2));
    }

    #[test]
    fn test_strength_reduction_div_pow2() {
        let mut body = vec![
            make_node(0, HirOp::VarRef { name: "x".into() }, vec![]),
            make_node(1, HirOp::IntConst(8), vec![]),
            make_node(2, HirOp::Div, vec![0, 1]),
        ];
        strength_reduction(&mut body);
        let div_node = body.iter().find(|n| n.id == HirId(2)).unwrap();
        assert!(matches!(div_node.op, HirOp::Shr));
    }

    #[test]
    fn test_strength_reduction_mod_pow2() {
        let mut body = vec![
            make_node(0, HirOp::VarRef { name: "x".into() }, vec![]),
            make_node(1, HirOp::IntConst(16), vec![]),
            make_node(2, HirOp::Mod, vec![0, 1]),
        ];
        strength_reduction(&mut body);
        let mod_node = body.iter().find(|n| n.id == HirId(2)).unwrap();
        assert!(matches!(mod_node.op, HirOp::BitAnd));
        let mask_node = body.iter().find(|n| matches!(n.op, HirOp::IntConst(15)));
        assert!(mask_node.is_some());
    }

    #[test]
    fn test_branch_const_true() {
        let mut body = vec![
            make_node(0, HirOp::BoolConst(true), vec![]),
            make_node(1, HirOp::Branch, vec![0, 10, 20]),
        ];
        branch_optimization(&mut body);
        let node = body.iter().find(|n| n.id == HirId(1)).unwrap();
        assert!(matches!(node.op, HirOp::Jump));
        assert_eq!(node.operands[0], HirId(10));
    }

    #[test]
    fn test_branch_const_false() {
        let mut body = vec![
            make_node(0, HirOp::BoolConst(false), vec![]),
            make_node(1, HirOp::Branch, vec![0, 10, 20]),
        ];
        branch_optimization(&mut body);
        let node = body.iter().find(|n| n.id == HirId(1)).unwrap();
        assert!(matches!(node.op, HirOp::Jump));
        assert_eq!(node.operands[0], HirId(20));
    }

    #[test]
    fn test_memory_load_after_store() {
        let mut body = vec![
            make_node(0, HirOp::IntConst(100), vec![]),
            make_node(1, HirOp::IntConst(42), vec![]),
            make_node(2, HirOp::Store, vec![0, 1]),
            make_node(3, HirOp::Load, vec![0]),
        ];
        memory_optimization(&mut body);
        let load_node = body.iter().find(|n| n.id == HirId(3)).unwrap();
        assert!(matches!(load_node.op, HirOp::VarRef { .. }));
    }

    #[test]
    fn test_memory_dead_store() {
        let mut body = vec![
            make_node(0, HirOp::IntConst(100), vec![]),
            make_node(1, HirOp::IntConst(42), vec![]),
            make_node(2, HirOp::IntConst(99), vec![]),
            make_node(3, HirOp::Store, vec![0, 1]),
            make_node(4, HirOp::Store, vec![0, 2]),
        ];
        memory_optimization(&mut body);
        let stores: Vec<_> = body.iter().filter(|n| matches!(n.op, HirOp::Store)).collect();
        assert_eq!(stores.len(), 1);
        assert_eq!(stores[0].id, HirId(4));
    }

    #[test]
    fn test_peephole_sub_zero() {
        let rules = default_peephole_rules();
        let mut body = vec![
            make_node(0, HirOp::IntConst(0), vec![]),
            make_node(1, HirOp::Sub, vec![99, 0]),
        ];
        apply_peephole_rules(&mut body, &rules);
        assert_eq!(body.len(), 1);
        assert!(matches!(body[0].op, HirOp::VarRef { .. }));
    }

    #[test]
    fn test_peephole_div_one() {
        let rules = default_peephole_rules();
        let mut body = vec![
            make_node(0, HirOp::IntConst(1), vec![]),
            make_node(1, HirOp::Div, vec![99, 0]),
        ];
        apply_peephole_rules(&mut body, &rules);
        assert_eq!(body.len(), 1);
        assert!(matches!(body[0].op, HirOp::VarRef { .. }));
    }

    #[test]
    fn test_peephole_xor_self() {
        let rules = default_peephole_rules();
        let mut body = vec![
            make_node(0, HirOp::BitXor, vec![5, 5]),
        ];
        apply_peephole_rules(&mut body, &rules);
        assert_eq!(body.len(), 1);
        assert!(matches!(body[0].op, HirOp::IntConst(0)));
    }

    #[test]
    fn test_peephole_and_zero() {
        let rules = default_peephole_rules();
        let mut body = vec![
            make_node(0, HirOp::IntConst(0), vec![]),
            make_node(1, HirOp::BitAnd, vec![0, 99]),
        ];
        apply_peephole_rules(&mut body, &rules);
        assert_eq!(body.len(), 1);
        assert!(matches!(body[0].op, HirOp::IntConst(0)));
    }

    #[test]
    fn test_tail_call_detection() {
        use crate::hir::HirParam;
        let func = HirFunction {
            name: "factorial".into(),
            params: vec![HirParam {
                name: "n".into(),
                ty: TypeId(0),
            }],
            body: vec![
                make_node(0, HirOp::VarRef { name: "n".into() }, vec![]),
                make_node(1, HirOp::Call { name: "factorial".into() }, vec![0]),
                make_node(2, HirOp::Return, vec![1]),
            ],
            return_type: Some(TypeId(0)),
            is_pub: false,
            span: None,
        };
        let tail_calls = detect_tail_calls(&func);
        assert_eq!(tail_calls.len(), 1);
        assert_eq!(tail_calls[0].function_name, "factorial");
        assert_eq!(tail_calls[0].call_node, HirId(1));
    }

    #[test]
    fn test_loop_unrolling_creates_nodes() {
        let mut body = vec![
            make_node(0, HirOp::IntConst(0), vec![]),
            make_node(1, HirOp::IntConst(10), vec![]),
            make_node(10, HirOp::Loop, vec![0, 20]),
            make_node(20, HirOp::VarRef { name: "i".into() }, vec![]),
            make_node(21, HirOp::Add, vec![20, 1]),
            make_node(22, HirOp::Jump, vec![10]),
        ];
        let before_count = body.len();
        unroll_loop(&mut body, 2, 4);
        assert!(body.len() > before_count, "Unrolling should create more nodes");
    }

    #[test]
    fn test_loop_fusion_merges() {
        let mut body = vec![
            make_node(0, HirOp::IntConst(10), vec![]),
            make_node(1, HirOp::Loop, vec![0, 10]),
            make_node(2, HirOp::Loop, vec![0, 20]),
        ];
        fuse_adjacent_loops(&mut body);
        let loops: Vec<_> = body.iter().filter(|n| matches!(n.op, HirOp::Loop)).collect();
        assert_eq!(loops.len(), 1, "Two loops should be fused into one");
    }

    #[test]
    fn test_licm_hoists_invariant() {
        let mut body = vec![
            make_node(0, HirOp::IntConst(5), vec![]),
            make_node(1, HirOp::IntConst(10), vec![]),
            make_node(10, HirOp::Loop, vec![0, 20]),
            make_node(20, HirOp::Add, vec![0, 1]),
            make_node(21, HirOp::VarRef { name: "i".into() }, vec![]),
            make_node(22, HirOp::Jump, vec![10]),
        ];
        loop_invariant_code_motion(&mut body);
        let add_pos = body.iter().position(|n| n.id == HirId(20)).unwrap();
        let loop_pos = body.iter().position(|n| n.id == HirId(10)).unwrap();
        assert!(add_pos < loop_pos, "Invariant node should be hoisted before the loop");
    }

    #[test]
    fn test_inline_small_function() {
        use crate::hir::HirParam;
        let mut program = HirProgram {
            functions: vec![
                HirFunction {
                    name: "double".into(),
                    params: vec![HirParam { name: "x".into(), ty: TypeId(0) }],
                    body: vec![
                        make_node(100, HirOp::Param { index: 0 }, vec![]),
                        make_node(101, HirOp::IntConst(2), vec![]),
                        make_node(102, HirOp::Mul, vec![100, 101]),
                        make_node(103, HirOp::Return, vec![102]),
                    ],
                    return_type: Some(TypeId(0)),
                    is_pub: false,
                    span: None,
                },
                HirFunction {
                    name: "main".into(),
                    params: vec![],
                    body: vec![
                        make_node(200, HirOp::IntConst(5), vec![]),
                        make_node(201, HirOp::Call { name: "double".into() }, vec![200]),
                        make_node(202, HirOp::Return, vec![201]),
                    ],
                    return_type: Some(TypeId(0)),
                    is_pub: true,
                    span: None,
                },
            ],
            structs: vec![],
            enums: vec![],
            type_context: crate::types::TypeContext::new(),
        };
        inline_small_functions(&mut program, 10);
        let main_func = program.functions.iter().find(|f| f.name == "main").unwrap();
        let has_call = main_func.body.iter().any(|n| matches!(n.op, HirOp::Call { .. }));
        assert!(!has_call, "Call should have been inlined");
        let has_mul = main_func.body.iter().any(|n| matches!(n.op, HirOp::Mul));
        assert!(has_mul, "Inlined body should contain Mul");
    }

    #[test]
    fn test_optimization_stats() {
        let pipeline = OptimizationPipeline::new(OptLevel::O3);
        let mut program = HirProgram {
            functions: vec![
                HirFunction {
                    name: "test_fn".into(),
                    params: vec![],
                    body: vec![
                        make_node(0, HirOp::IntConst(3), vec![]),
                        make_node(1, HirOp::IntConst(5), vec![]),
                        make_node(2, HirOp::Add, vec![0, 1]),
                        make_node(3, HirOp::BoolConst(true), vec![]),
                        make_node(4, HirOp::Branch, vec![3, 10, 20]),
                        make_node(5, HirOp::Return, vec![2]),
                    ],
                    return_type: Some(TypeId(0)),
                    is_pub: true,
                    span: None,
                },
            ],
            structs: vec![],
            enums: vec![],
            type_context: crate::types::TypeContext::new(),
        };
        let stats = pipeline.optimize_with_stats(&mut program);
        let total = stats.dce_removed
            + stats.constants_folded
            + stats.peephole_applied
            + stats.branches_optimized;
        assert!(total > 0, "O3 should perform some optimizations, stats: {:?}", stats);
    }

    #[test]
    fn test_pipeline_o1() {
        let pipeline = OptimizationPipeline::new(OptLevel::O1);
        let mut program = HirProgram {
            functions: vec![HirFunction {
                name: "test".into(),
                params: vec![],
                body: vec![
                    make_node(0, HirOp::IntConst(3), vec![]),
                    make_node(1, HirOp::IntConst(5), vec![]),
                    make_node(2, HirOp::Add, vec![0, 1]),
                    make_node(3, HirOp::Return, vec![2]),
                ],
                return_type: Some(TypeId(0)),
                is_pub: true,
                span: None,
            }],
            structs: vec![],
            enums: vec![],
            type_context: crate::types::TypeContext::new(),
        };
        pipeline.optimize(&mut program);
        let func = &program.functions[0];
        let has_const_8 = func.body.iter().any(|n| matches!(n.op, HirOp::IntConst(8)));
        assert!(has_const_8, "O1 should fold constants: {:?}", func.body);
    }

    #[test]
    fn test_pipeline_o2() {
        let pipeline = OptimizationPipeline::new(OptLevel::O2);
        let mut program = HirProgram {
            functions: vec![HirFunction {
                name: "test".into(),
                params: vec![],
                body: vec![
                    make_node(0, HirOp::IntConst(5), vec![]),
                    make_node(1, HirOp::IntConst(10), vec![]),
                    make_node(2, HirOp::Add, vec![0, 1]),
                    make_node(3, HirOp::Add, vec![0, 1]),
                    make_node(4, HirOp::Return, vec![3]),
                ],
                return_type: Some(TypeId(0)),
                is_pub: true,
                span: None,
            }],
            structs: vec![],
            enums: vec![],
            type_context: crate::types::TypeContext::new(),
        };
        pipeline.optimize(&mut program);
        assert!(!program.functions.is_empty());
    }

    #[test]
    fn test_pipeline_o3() {
        use crate::hir::HirParam;
        let pipeline = OptimizationPipeline::new(OptLevel::O3);
        let mut program = HirProgram {
            functions: vec![
                HirFunction {
                    name: "helper".into(),
                    params: vec![HirParam { name: "x".into(), ty: TypeId(0) }],
                    body: vec![
                        make_node(50, HirOp::Param { index: 0 }, vec![]),
                        make_node(51, HirOp::Return, vec![50]),
                    ],
                    return_type: Some(TypeId(0)),
                    is_pub: false,
                    span: None,
                },
                HirFunction {
                    name: "main".into(),
                    params: vec![],
                    body: vec![
                        make_node(0, HirOp::IntConst(42), vec![]),
                        make_node(1, HirOp::Call { name: "helper".into() }, vec![0]),
                        make_node(2, HirOp::Return, vec![1]),
                    ],
                    return_type: Some(TypeId(0)),
                    is_pub: true,
                    span: None,
                },
            ],
            structs: vec![],
            enums: vec![],
            type_context: crate::types::TypeContext::new(),
        };
        pipeline.optimize(&mut program);
        assert!(!program.functions.is_empty());
    }

    #[test]
    fn test_peephole_or_neg_one() {
        let rules = default_peephole_rules();
        let mut body = vec![
            make_node(0, HirOp::IntConst(-1), vec![]),
            make_node(1, HirOp::BitOr, vec![0, 99]),
        ];
        apply_peephole_rules(&mut body, &rules);
        assert_eq!(body.len(), 1);
        assert!(matches!(body[0].op, HirOp::IntConst(-1)));
    }

    #[test]
    fn test_peephole_shl_zero() {
        let rules = default_peephole_rules();
        let mut body = vec![
            make_node(0, HirOp::IntConst(0), vec![]),
            make_node(1, HirOp::Shl, vec![99, 0]),
        ];
        apply_peephole_rules(&mut body, &rules);
        assert_eq!(body.len(), 1);
        assert!(matches!(body[0].op, HirOp::VarRef { .. }));
    }

    #[test]
    fn test_peephole_not_not() {
        let rules = default_peephole_rules();
        let mut body = vec![
            make_node(0, HirOp::BitNot, vec![99]),
            make_node(1, HirOp::BitNot, vec![0]),
        ];
        apply_peephole_rules(&mut body, &rules);
        assert_eq!(body.len(), 1);
        assert!(matches!(body[0].op, HirOp::VarRef { .. }));
    }
}
