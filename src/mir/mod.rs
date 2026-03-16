//! Mid-Level Intermediate Representation (MIR).
//!
//! The MIR sits between the HIR and the code-generation backend.  It uses
//! SSA (Static Single Assignment) form where every value is defined exactly
//! once.  Instructions operate on machine-level types and are organised
//! into basic blocks with explicit control-flow edges.

use std::collections::HashMap;
use std::fmt;

use crate::hir::{HirFunction, HirId, HirNode, HirOp, HirProgram};
use crate::types::{AxiomType, FloatKind, IntegerKind, SimdKind, TypeContext, TypeId};

// ---------------------------------------------------------------------------
// MirValue – SSA value handle
// ---------------------------------------------------------------------------

/// Lightweight handle for an SSA value in the MIR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MirValue(pub usize);

impl fmt::Display for MirValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// BlockId – basic-block identifier
// ---------------------------------------------------------------------------

/// Lightweight handle identifying a basic block inside a [`MirFunction`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub usize);

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// MirType – machine-level types
// ---------------------------------------------------------------------------

/// Machine-level type used by the MIR.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MirType {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    Bool,
    /// Generic pointer (target-width).
    Ptr,
    /// 4-lane f32 SIMD vector register.
    Vec4F,
    /// 8-lane f32 SIMD vector register.
    Vec8F,
    /// 16-lane f32 SIMD vector register.
    Vec16F,
    Void,
    /// Aggregate (struct / array) with a known byte size.
    Aggregate(usize),
}

impl MirType {
    /// Size in bytes of this MIR type.
    pub fn size_bytes(&self) -> usize {
        match self {
            MirType::I8 | MirType::U8 | MirType::Bool => 1,
            MirType::I16 | MirType::U16 => 2,
            MirType::I32 | MirType::U32 | MirType::F32 => 4,
            MirType::I64 | MirType::U64 | MirType::F64 | MirType::Ptr => 8,
            MirType::Vec4F => 16,
            MirType::Vec8F => 32,
            MirType::Vec16F => 64,
            MirType::Void => 0,
            MirType::Aggregate(sz) => *sz,
        }
    }
}

impl fmt::Display for MirType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MirType::I8 => write!(f, "i8"),
            MirType::I16 => write!(f, "i16"),
            MirType::I32 => write!(f, "i32"),
            MirType::I64 => write!(f, "i64"),
            MirType::U8 => write!(f, "u8"),
            MirType::U16 => write!(f, "u16"),
            MirType::U32 => write!(f, "u32"),
            MirType::U64 => write!(f, "u64"),
            MirType::F32 => write!(f, "f32"),
            MirType::F64 => write!(f, "f64"),
            MirType::Bool => write!(f, "bool"),
            MirType::Ptr => write!(f, "ptr"),
            MirType::Vec4F => write!(f, "vec4f"),
            MirType::Vec8F => write!(f, "vec8f"),
            MirType::Vec16F => write!(f, "vec16f"),
            MirType::Void => write!(f, "void"),
            MirType::Aggregate(sz) => write!(f, "agg({})", sz),
        }
    }
}

// ---------------------------------------------------------------------------
// CmpOp – comparison kind
// ---------------------------------------------------------------------------

/// Comparison operation used by [`MirInst::Cmp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl fmt::Display for CmpOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            CmpOp::Eq => "eq",
            CmpOp::Ne => "ne",
            CmpOp::Lt => "lt",
            CmpOp::Le => "le",
            CmpOp::Gt => "gt",
            CmpOp::Ge => "ge",
        };
        write!(f, "{}", s)
    }
}

// ---------------------------------------------------------------------------
// VecOpKind – SIMD vector operations
// ---------------------------------------------------------------------------

/// Kind of SIMD vector operation used by [`MirInst::VecOp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VecOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Load,
    Store,
    Broadcast,
    Shuffle,
    Extract,
    Insert,
}

impl fmt::Display for VecOpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            VecOpKind::Add => "vadd",
            VecOpKind::Sub => "vsub",
            VecOpKind::Mul => "vmul",
            VecOpKind::Div => "vdiv",
            VecOpKind::Load => "vload",
            VecOpKind::Store => "vstore",
            VecOpKind::Broadcast => "vbroadcast",
            VecOpKind::Shuffle => "vshuffle",
            VecOpKind::Extract => "vextract",
            VecOpKind::Insert => "vinsert",
        };
        write!(f, "{}", s)
    }
}

// ---------------------------------------------------------------------------
// MirConst – constant payloads
// ---------------------------------------------------------------------------

/// Constant value embedded in a [`MirInst::Const`] instruction.
#[derive(Debug, Clone, PartialEq)]
pub enum MirConst {
    Int(i64),
    Float(f64),
    Bool(bool),
    /// Zero-initialiser for any type.
    Zero,
}

impl fmt::Display for MirConst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MirConst::Int(v) => write!(f, "{}", v),
            MirConst::Float(v) => write!(f, "{:.6}", v),
            MirConst::Bool(v) => write!(f, "{}", v),
            MirConst::Zero => write!(f, "zeroinit"),
        }
    }
}

// ---------------------------------------------------------------------------
// MirInst – SSA instructions
// ---------------------------------------------------------------------------

/// A single SSA instruction in the MIR.
#[derive(Debug, Clone, PartialEq)]
pub enum MirInst {
    // -- binary arithmetic --------------------------------------------------
    Add { dst: MirValue, ty: MirType, lhs: MirValue, rhs: MirValue },
    Sub { dst: MirValue, ty: MirType, lhs: MirValue, rhs: MirValue },
    Mul { dst: MirValue, ty: MirType, lhs: MirValue, rhs: MirValue },
    Div { dst: MirValue, ty: MirType, lhs: MirValue, rhs: MirValue },
    Mod { dst: MirValue, ty: MirType, lhs: MirValue, rhs: MirValue },

    // -- binary bitwise / logical -------------------------------------------
    And { dst: MirValue, ty: MirType, lhs: MirValue, rhs: MirValue },
    Or  { dst: MirValue, ty: MirType, lhs: MirValue, rhs: MirValue },
    Xor { dst: MirValue, ty: MirType, lhs: MirValue, rhs: MirValue },
    Shl { dst: MirValue, ty: MirType, lhs: MirValue, rhs: MirValue },
    Shr { dst: MirValue, ty: MirType, lhs: MirValue, rhs: MirValue },

    // -- comparison ---------------------------------------------------------
    Cmp { dst: MirValue, op: CmpOp, ty: MirType, lhs: MirValue, rhs: MirValue },

    // -- unary --------------------------------------------------------------
    Neg { dst: MirValue, ty: MirType, operand: MirValue },
    Not { dst: MirValue, ty: MirType, operand: MirValue },

    // -- memory -------------------------------------------------------------
    Load  { dst: MirValue, ty: MirType, addr: MirValue },
    Store { ty: MirType, addr: MirValue, value: MirValue },

    // -- stack allocation ---------------------------------------------------
    StackAlloc { dst: MirValue, size: usize, align: usize },

    // -- constants ----------------------------------------------------------
    Const { dst: MirValue, ty: MirType, value: MirConst },

    // -- control flow -------------------------------------------------------
    Br   { cond: MirValue, then_bb: BlockId, else_bb: BlockId },
    Jump { target: BlockId },
    Ret  { value: Option<MirValue> },

    // -- calls --------------------------------------------------------------
    Call { dst: Option<MirValue>, func: String, args: Vec<MirValue> },

    // -- SIMD ---------------------------------------------------------------
    VecOp { dst: MirValue, op: VecOpKind, ty: MirType, operands: Vec<MirValue> },

    // -- casts --------------------------------------------------------------
    Cast { dst: MirValue, from_ty: MirType, to_ty: MirType, value: MirValue },

    // -- pointer arithmetic -------------------------------------------------
    Gep { dst: MirValue, base: MirValue, indices: Vec<MirValue> },

    // -- SSA phi node -------------------------------------------------------
    Phi { dst: MirValue, ty: MirType, incoming: Vec<(BlockId, MirValue)> },
}

impl fmt::Display for MirInst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // binary
            MirInst::Add { dst, ty, lhs, rhs } => write!(f, "{} = add {} {}, {}", dst, ty, lhs, rhs),
            MirInst::Sub { dst, ty, lhs, rhs } => write!(f, "{} = sub {} {}, {}", dst, ty, lhs, rhs),
            MirInst::Mul { dst, ty, lhs, rhs } => write!(f, "{} = mul {} {}, {}", dst, ty, lhs, rhs),
            MirInst::Div { dst, ty, lhs, rhs } => write!(f, "{} = div {} {}, {}", dst, ty, lhs, rhs),
            MirInst::Mod { dst, ty, lhs, rhs } => write!(f, "{} = mod {} {}, {}", dst, ty, lhs, rhs),
            MirInst::And { dst, ty, lhs, rhs } => write!(f, "{} = and {} {}, {}", dst, ty, lhs, rhs),
            MirInst::Or  { dst, ty, lhs, rhs } => write!(f, "{} = or {} {}, {}", dst, ty, lhs, rhs),
            MirInst::Xor { dst, ty, lhs, rhs } => write!(f, "{} = xor {} {}, {}", dst, ty, lhs, rhs),
            MirInst::Shl { dst, ty, lhs, rhs } => write!(f, "{} = shl {} {}, {}", dst, ty, lhs, rhs),
            MirInst::Shr { dst, ty, lhs, rhs } => write!(f, "{} = shr {} {}, {}", dst, ty, lhs, rhs),

            // comparison
            MirInst::Cmp { dst, op, ty, lhs, rhs } => {
                write!(f, "{} = cmp {} {} {}, {}", dst, op, ty, lhs, rhs)
            }

            // unary
            MirInst::Neg { dst, ty, operand } => write!(f, "{} = neg {} {}", dst, ty, operand),
            MirInst::Not { dst, ty, operand } => write!(f, "{} = not {} {}", dst, ty, operand),

            // memory
            MirInst::Load { dst, ty, addr } => write!(f, "{} = load {} {}", dst, ty, addr),
            MirInst::Store { ty, addr, value } => write!(f, "store {} {}, {}", ty, value, addr),

            // stack
            MirInst::StackAlloc { dst, size, align } => {
                write!(f, "{} = stackalloc size={} align={}", dst, size, align)
            }

            // constants
            MirInst::Const { dst, ty, value } => {
                write!(f, "{} = const {} {}", dst, ty, value)
            }

            // control flow
            MirInst::Br { cond, then_bb, else_bb } => {
                write!(f, "br {}, {}, {}", cond, then_bb, else_bb)
            }
            MirInst::Jump { target } => write!(f, "jump {}", target),
            MirInst::Ret { value: Some(v) } => write!(f, "ret {}", v),
            MirInst::Ret { value: None } => write!(f, "ret void"),

            // call
            MirInst::Call { dst: Some(d), func, args } => {
                let arg_str: Vec<String> = args.iter().map(|a| a.to_string()).collect();
                write!(f, "{} = call {}({})", d, func, arg_str.join(", "))
            }
            MirInst::Call { dst: None, func, args } => {
                let arg_str: Vec<String> = args.iter().map(|a| a.to_string()).collect();
                write!(f, "call {}({})", func, arg_str.join(", "))
            }

            // SIMD
            MirInst::VecOp { dst, op, ty, operands } => {
                let ops: Vec<String> = operands.iter().map(|o| o.to_string()).collect();
                write!(f, "{} = {} {} {}", dst, op, ty, ops.join(", "))
            }

            // cast
            MirInst::Cast { dst, from_ty, to_ty, value } => {
                write!(f, "{} = cast {} {} to {}", dst, from_ty, value, to_ty)
            }

            // GEP
            MirInst::Gep { dst, base, indices } => {
                let idx_str: Vec<String> = indices.iter().map(|i| i.to_string()).collect();
                write!(f, "{} = gep {}, {}", dst, base, idx_str.join(", "))
            }

            // phi
            MirInst::Phi { dst, ty, incoming } => {
                let entries: Vec<String> = incoming
                    .iter()
                    .map(|(bb, val)| format!("[{}: {}]", bb, val))
                    .collect();
                write!(f, "{} = phi {} {}", dst, ty, entries.join(", "))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MirBlock – basic block
// ---------------------------------------------------------------------------

/// A basic block in the MIR control-flow graph.
#[derive(Debug, Clone)]
pub struct MirBlock {
    pub id: BlockId,
    pub label: String,
    pub instructions: Vec<MirInst>,
    pub predecessors: Vec<BlockId>,
    pub successors: Vec<BlockId>,
}

impl MirBlock {
    /// Create a new, empty basic block.
    pub fn new(id: BlockId, label: impl Into<String>) -> Self {
        Self {
            id,
            label: label.into(),
            instructions: Vec::new(),
            predecessors: Vec::new(),
            successors: Vec::new(),
        }
    }
}

impl fmt::Display for MirBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}:", self.label)?;
        for inst in &self.instructions {
            writeln!(f, "    {}", inst)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MirFunction
// ---------------------------------------------------------------------------

/// A function in MIR form.
#[derive(Debug, Clone)]
pub struct MirFunction {
    pub name: String,
    pub params: Vec<(MirValue, MirType)>,
    pub return_type: MirType,
    pub blocks: Vec<MirBlock>,
    pub entry_block: BlockId,
    pub next_value: usize,
}

impl MirFunction {
    /// Allocate a fresh [`MirValue`] id.
    pub fn fresh_value(&mut self) -> MirValue {
        let v = MirValue(self.next_value);
        self.next_value += 1;
        v
    }
}

impl fmt::Display for MirFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let params: Vec<String> = self
            .params
            .iter()
            .map(|(v, ty)| format!("{} {}", ty, v))
            .collect();
        writeln!(
            f,
            "fn {}({}) -> {} {{",
            self.name,
            params.join(", "),
            self.return_type
        )?;
        for block in &self.blocks {
            write!(f, "{}", block)?;
        }
        writeln!(f, "}}")
    }
}

// ---------------------------------------------------------------------------
// MirProgram
// ---------------------------------------------------------------------------

/// The complete MIR program (collection of functions).
#[derive(Debug, Clone)]
pub struct MirProgram {
    pub functions: Vec<MirFunction>,
}

impl fmt::Display for MirProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, func) in self.functions.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{}", func)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MirError
// ---------------------------------------------------------------------------

/// Error produced during HIR → MIR lowering.
#[derive(Debug, Clone)]
pub struct MirError {
    pub message: String,
}

impl MirError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self { message: msg.into() }
    }
}

impl fmt::Display for MirError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MIR error: {}", self.message)
    }
}

impl std::error::Error for MirError {}

// ---------------------------------------------------------------------------
// Type conversion helpers
// ---------------------------------------------------------------------------

/// Convert an [`AxiomType`] into the corresponding [`MirType`].
fn axiom_type_to_mir(ty: &AxiomType) -> MirType {
    match ty {
        AxiomType::Integer(kind) => match kind {
            IntegerKind::I8 => MirType::I8,
            IntegerKind::I16 => MirType::I16,
            IntegerKind::I32 => MirType::I32,
            IntegerKind::I64 => MirType::I64,
            IntegerKind::U8 => MirType::U8,
            IntegerKind::U16 => MirType::U16,
            IntegerKind::U32 => MirType::U32,
            IntegerKind::U64 => MirType::U64,
        },
        AxiomType::Float(kind) => match kind {
            FloatKind::F32 => MirType::F32,
            FloatKind::F64 => MirType::F64,
        },
        AxiomType::Bool => MirType::Bool,
        AxiomType::Char => MirType::U32,
        AxiomType::Simd(kind) => match kind {
            SimdKind::Vec4F => MirType::Vec4F,
            SimdKind::Vec8F => MirType::Vec8F,
            SimdKind::Vec16F => MirType::Vec16F,
        },
        AxiomType::Reference { .. } | AxiomType::Pointer { .. } => MirType::Ptr,
        AxiomType::Unit => MirType::Void,
        AxiomType::Array { .. } | AxiomType::Slice { .. } => MirType::Ptr,
        AxiomType::Struct { .. } => {
            let sz = ty.size_bytes();
            if sz > 0 { MirType::Aggregate(sz) } else { MirType::Ptr }
        }
        AxiomType::Enum { .. } => MirType::I64,
        AxiomType::Function { .. } => MirType::Ptr,
        AxiomType::GenericParam { .. } => MirType::Ptr,
    }
}

/// Resolve a [`TypeId`] through the [`TypeContext`] and produce a [`MirType`].
fn resolve_mir_type(type_ctx: &TypeContext, tid: TypeId) -> MirType {
    axiom_type_to_mir(type_ctx.resolve(tid))
}

// ---------------------------------------------------------------------------
// MirBuilder – HIR → MIR lowering
// ---------------------------------------------------------------------------

/// Lowers an [`HirProgram`] into a [`MirProgram`].
pub struct MirBuilder<'a> {
    type_ctx: &'a TypeContext,
    /// Current function being built.
    blocks: Vec<MirBlock>,
    current_block: BlockId,
    next_value: usize,
    next_block: usize,
    /// Map from HIR id → MIR SSA value.
    value_map: HashMap<HirId, MirValue>,
    /// Map from variable name → current MIR SSA value.
    var_map: HashMap<String, MirValue>,
}

impl<'a> MirBuilder<'a> {
    // -- public entry point ------------------------------------------------

    /// Lower an entire [`HirProgram`] into a [`MirProgram`].
    pub fn lower_program(hir: &HirProgram) -> Result<MirProgram, MirError> {
        let mut functions = Vec::with_capacity(hir.functions.len());
        for func in &hir.functions {
            let mir_func = Self::lower_function(func, &hir.type_context)?;
            functions.push(mir_func);
        }
        Ok(MirProgram { functions })
    }

    // -- per-function lowering ---------------------------------------------

    fn lower_function(
        func: &HirFunction,
        type_ctx: &TypeContext,
    ) -> Result<MirFunction, MirError> {
        let mut builder = MirBuilder {
            type_ctx,
            blocks: Vec::new(),
            current_block: BlockId(0),
            next_value: 0,
            next_block: 0,
            value_map: HashMap::new(),
            var_map: HashMap::new(),
        };

        let entry = builder.new_block("entry");
        builder.current_block = entry;

        // Lower parameters.
        let mut params = Vec::with_capacity(func.params.len());
        for param in &func.params {
            let mir_ty = resolve_mir_type(type_ctx, param.ty);
            let val = builder.fresh_value();
            builder.var_map.insert(param.name.clone(), val);
            params.push((val, mir_ty));
        }

        // Lower body nodes.
        for node in &func.body {
            builder.lower_node(node)?;
        }

        // Ensure every block ends with a terminator.
        builder.ensure_terminators();

        // Build CFG edges.
        builder.build_cfg_edges();

        let return_type = func
            .return_type
            .map(|tid| resolve_mir_type(type_ctx, tid))
            .unwrap_or(MirType::Void);

        Ok(MirFunction {
            name: func.name.clone(),
            params,
            return_type,
            blocks: builder.blocks,
            entry_block: entry,
            next_value: builder.next_value,
        })
    }

    // -- helpers -----------------------------------------------------------

    fn fresh_value(&mut self) -> MirValue {
        let v = MirValue(self.next_value);
        self.next_value += 1;
        v
    }

    fn new_block(&mut self, label: &str) -> BlockId {
        let id = BlockId(self.next_block);
        self.next_block += 1;
        let blk = MirBlock::new(id, format!("{}_{}", label, id.0));
        self.blocks.push(blk);
        id
    }

    fn emit(&mut self, inst: MirInst) {
        let idx = self.current_block.0;
        self.blocks[idx].instructions.push(inst);
    }

    fn is_terminator(inst: &MirInst) -> bool {
        matches!(inst, MirInst::Br { .. } | MirInst::Jump { .. } | MirInst::Ret { .. })
    }

    fn block_has_terminator(&self, id: BlockId) -> bool {
        self.blocks[id.0]
            .instructions
            .last()
            .map_or(false, Self::is_terminator)
    }

    /// Ensure every basic block ends with a terminator.
    fn ensure_terminators(&mut self) {
        for i in 0..self.blocks.len() {
            let needs = self.blocks[i]
                .instructions
                .last()
                .map_or(true, |inst| !Self::is_terminator(inst));
            if needs {
                self.blocks[i].instructions.push(MirInst::Ret { value: None });
            }
        }
    }

    /// Populate predecessor / successor lists by scanning terminators.
    fn build_cfg_edges(&mut self) {
        let edges: Vec<(BlockId, Vec<BlockId>)> = self
            .blocks
            .iter()
            .map(|blk| {
                let mut succs = Vec::new();
                if let Some(term) = blk.instructions.last() {
                    match term {
                        MirInst::Jump { target } => succs.push(*target),
                        MirInst::Br { then_bb, else_bb, .. } => {
                            succs.push(*then_bb);
                            succs.push(*else_bb);
                        }
                        _ => {}
                    }
                }
                (blk.id, succs)
            })
            .collect();

        // Clear existing edges.
        for blk in &mut self.blocks {
            blk.successors.clear();
            blk.predecessors.clear();
        }

        for (src, succs) in &edges {
            self.blocks[src.0].successors = succs.clone();
            for &tgt in succs {
                self.blocks[tgt.0].predecessors.push(*src);
            }
        }
    }

    /// Determine the MIR type for an HIR node, falling back to I64.
    fn node_mir_type(&self, node: &HirNode) -> MirType {
        node.ty
            .map(|tid| resolve_mir_type(self.type_ctx, tid))
            .unwrap_or(MirType::I64)
    }

    /// Look up the MIR value previously produced for an HIR id.
    fn resolve_operand(&self, hir_id: HirId) -> Result<MirValue, MirError> {
        self.value_map
            .get(&hir_id)
            .copied()
            .ok_or_else(|| MirError::new(format!("unresolved HIR operand {}", hir_id)))
    }

    // -- node lowering -----------------------------------------------------

    fn lower_node(&mut self, node: &HirNode) -> Result<Option<MirValue>, MirError> {
        match &node.op {
            // -- constants --------------------------------------------------
            HirOp::IntConst(v) => {
                let dst = self.fresh_value();
                let ty = self.node_mir_type(node);
                self.emit(MirInst::Const { dst, ty, value: MirConst::Int(*v) });
                self.value_map.insert(node.id, dst);
                Ok(Some(dst))
            }
            HirOp::FloatConst(v) => {
                let dst = self.fresh_value();
                let ty = self.node_mir_type(node);
                self.emit(MirInst::Const { dst, ty, value: MirConst::Float(*v) });
                self.value_map.insert(node.id, dst);
                Ok(Some(dst))
            }
            HirOp::BoolConst(v) => {
                let dst = self.fresh_value();
                self.emit(MirInst::Const { dst, ty: MirType::Bool, value: MirConst::Bool(*v) });
                self.value_map.insert(node.id, dst);
                Ok(Some(dst))
            }
            HirOp::StringConst(_) => {
                // Strings are lowered as pointers for now.
                let dst = self.fresh_value();
                self.emit(MirInst::Const { dst, ty: MirType::Ptr, value: MirConst::Zero });
                self.value_map.insert(node.id, dst);
                Ok(Some(dst))
            }

            // -- binary arithmetic ------------------------------------------
            HirOp::Add => self.lower_binary_op(node, |dst, ty, lhs, rhs| {
                MirInst::Add { dst, ty, lhs, rhs }
            }),
            HirOp::Sub => self.lower_binary_op(node, |dst, ty, lhs, rhs| {
                MirInst::Sub { dst, ty, lhs, rhs }
            }),
            HirOp::Mul => self.lower_binary_op(node, |dst, ty, lhs, rhs| {
                MirInst::Mul { dst, ty, lhs, rhs }
            }),
            HirOp::Div => self.lower_binary_op(node, |dst, ty, lhs, rhs| {
                MirInst::Div { dst, ty, lhs, rhs }
            }),
            HirOp::Mod => self.lower_binary_op(node, |dst, ty, lhs, rhs| {
                MirInst::Mod { dst, ty, lhs, rhs }
            }),

            // -- bitwise / logical ------------------------------------------
            HirOp::BitAnd | HirOp::And => self.lower_binary_op(node, |dst, ty, lhs, rhs| {
                MirInst::And { dst, ty, lhs, rhs }
            }),
            HirOp::BitOr | HirOp::Or => self.lower_binary_op(node, |dst, ty, lhs, rhs| {
                MirInst::Or { dst, ty, lhs, rhs }
            }),
            HirOp::BitXor => self.lower_binary_op(node, |dst, ty, lhs, rhs| {
                MirInst::Xor { dst, ty, lhs, rhs }
            }),
            HirOp::Shl => self.lower_binary_op(node, |dst, ty, lhs, rhs| {
                MirInst::Shl { dst, ty, lhs, rhs }
            }),
            HirOp::Shr => self.lower_binary_op(node, |dst, ty, lhs, rhs| {
                MirInst::Shr { dst, ty, lhs, rhs }
            }),

            // -- comparisons ------------------------------------------------
            HirOp::Eq => self.lower_cmp(node, CmpOp::Eq),
            HirOp::Ne => self.lower_cmp(node, CmpOp::Ne),
            HirOp::Lt => self.lower_cmp(node, CmpOp::Lt),
            HirOp::Le => self.lower_cmp(node, CmpOp::Le),
            HirOp::Gt => self.lower_cmp(node, CmpOp::Gt),
            HirOp::Ge => self.lower_cmp(node, CmpOp::Ge),

            // -- unary ------------------------------------------------------
            HirOp::Neg => {
                let operand = self.resolve_operand(node.operands[0])?;
                let dst = self.fresh_value();
                let ty = self.node_mir_type(node);
                self.emit(MirInst::Neg { dst, ty, operand });
                self.value_map.insert(node.id, dst);
                Ok(Some(dst))
            }
            HirOp::Not | HirOp::BitNot => {
                let operand = self.resolve_operand(node.operands[0])?;
                let dst = self.fresh_value();
                let ty = self.node_mir_type(node);
                self.emit(MirInst::Not { dst, ty, operand });
                self.value_map.insert(node.id, dst);
                Ok(Some(dst))
            }

            // -- memory / allocation ----------------------------------------
            HirOp::Load => {
                let addr = self.resolve_operand(node.operands[0])?;
                let dst = self.fresh_value();
                let ty = self.node_mir_type(node);
                self.emit(MirInst::Load { dst, ty, addr });
                self.value_map.insert(node.id, dst);
                Ok(Some(dst))
            }
            HirOp::Store => {
                let addr = self.resolve_operand(node.operands[0])?;
                let value = self.resolve_operand(node.operands[1])?;
                let ty = self.node_mir_type(node);
                self.emit(MirInst::Store { ty, addr, value });
                Ok(None)
            }
            HirOp::StackAlloc => {
                let dst = self.fresh_value();
                let ty = self.node_mir_type(node);
                let size = ty.size_bytes().max(1);
                let align = size.min(8);
                self.emit(MirInst::StackAlloc { dst, size, align });
                self.value_map.insert(node.id, dst);
                Ok(Some(dst))
            }
            HirOp::Alloc => {
                // Heap allocation → call to runtime allocator.
                let dst = self.fresh_value();
                let ty = self.node_mir_type(node);
                let size_val = self.fresh_value();
                let size = ty.size_bytes().max(1);
                self.emit(MirInst::Const {
                    dst: size_val,
                    ty: MirType::I64,
                    value: MirConst::Int(size as i64),
                });
                self.emit(MirInst::Call {
                    dst: Some(dst),
                    func: "__axiom_alloc".to_string(),
                    args: vec![size_val],
                });
                self.value_map.insert(node.id, dst);
                Ok(Some(dst))
            }

            // -- variables --------------------------------------------------
            HirOp::VarDef { name, .. } => {
                let val = self.resolve_operand(node.operands[0])?;
                self.var_map.insert(name.clone(), val);
                self.value_map.insert(node.id, val);
                Ok(Some(val))
            }
            HirOp::VarRef { name } => {
                let val = self.var_map.get(name).copied().ok_or_else(|| {
                    MirError::new(format!("undefined variable `{}`", name))
                })?;
                self.value_map.insert(node.id, val);
                Ok(Some(val))
            }

            // -- parameters -------------------------------------------------
            HirOp::Param { .. } => {
                // During function lowering, params are pre-populated in var_map.
                // Create a placeholder value if needed.
                let dst = self.fresh_value();
                self.value_map.insert(node.id, dst);
                Ok(Some(dst))
            }

            // -- control flow -----------------------------------------------
            HirOp::Return => {
                let val = if node.operands.is_empty() {
                    None
                } else {
                    Some(self.resolve_operand(node.operands[0])?)
                };
                self.emit(MirInst::Ret { value: val });
                Ok(None)
            }
            HirOp::Branch => {
                let cond = self.resolve_operand(node.operands[0])?;
                let then_bb = self.new_block("then");
                let else_bb = self.new_block("else");
                self.emit(MirInst::Br { cond, then_bb, else_bb });
                self.current_block = then_bb;
                Ok(None)
            }
            HirOp::Jump => {
                let target = self.new_block("target");
                self.emit(MirInst::Jump { target });
                self.current_block = target;
                Ok(None)
            }
            HirOp::Loop => {
                let header = self.new_block("loop_header");
                let body = self.new_block("loop_body");
                let exit = self.new_block("loop_exit");

                if !self.block_has_terminator(self.current_block) {
                    self.emit(MirInst::Jump { target: header });
                }
                self.current_block = header;

                if !node.operands.is_empty() {
                    let cond = self.resolve_operand(node.operands[0])?;
                    self.emit(MirInst::Br { cond, then_bb: body, else_bb: exit });
                } else {
                    self.emit(MirInst::Jump { target: body });
                }
                self.current_block = body;
                Ok(None)
            }

            // -- calls ------------------------------------------------------
            HirOp::Call { name } => {
                let args: Result<Vec<MirValue>, MirError> = node
                    .operands
                    .iter()
                    .map(|id| self.resolve_operand(*id))
                    .collect();
                let args = args?;
                let dst = self.fresh_value();
                self.emit(MirInst::Call {
                    dst: Some(dst),
                    func: name.clone(),
                    args,
                });
                self.value_map.insert(node.id, dst);
                Ok(Some(dst))
            }

            // -- aggregates / misc ------------------------------------------
            HirOp::Index => {
                let base = self.resolve_operand(node.operands[0])?;
                let index = self.resolve_operand(node.operands[1])?;
                let dst = self.fresh_value();
                self.emit(MirInst::Gep { dst, base, indices: vec![index] });
                self.value_map.insert(node.id, dst);
                Ok(Some(dst))
            }
            HirOp::FieldAccess { .. } | HirOp::Slice { .. } => {
                let base = self.resolve_operand(node.operands[0])?;
                let dst = self.fresh_value();
                self.emit(MirInst::Gep { dst, base, indices: vec![] });
                self.value_map.insert(node.id, dst);
                Ok(Some(dst))
            }
            HirOp::StructInit { .. } | HirOp::ArrayLiteral => {
                let dst = self.fresh_value();
                let ty = self.node_mir_type(node);
                let size = ty.size_bytes().max(1);
                let align = size.min(8);
                self.emit(MirInst::StackAlloc { dst, size, align });
                // Store each field / element.
                for &operand_id in &node.operands {
                    let _val = self.resolve_operand(operand_id)?;
                }
                self.value_map.insert(node.id, dst);
                Ok(Some(dst))
            }

            // -- block / nop / break / continue -----------------------------
            HirOp::Block | HirOp::Nop | HirOp::Break | HirOp::Continue => {
                // Record the last operand's value (if any) as the block result.
                if let Some(&last) = node.operands.last() {
                    if let Some(val) = self.value_map.get(&last).copied() {
                        self.value_map.insert(node.id, val);
                        return Ok(Some(val));
                    }
                }
                Ok(None)
            }
        }
    }

    // -- lowering helpers --------------------------------------------------

    fn lower_binary_op<F>(
        &mut self,
        node: &HirNode,
        make: F,
    ) -> Result<Option<MirValue>, MirError>
    where
        F: FnOnce(MirValue, MirType, MirValue, MirValue) -> MirInst,
    {
        let lhs = self.resolve_operand(node.operands[0])?;
        let rhs = self.resolve_operand(node.operands[1])?;
        let dst = self.fresh_value();
        let ty = self.node_mir_type(node);
        self.emit(make(dst, ty, lhs, rhs));
        self.value_map.insert(node.id, dst);
        Ok(Some(dst))
    }

    fn lower_cmp(
        &mut self,
        node: &HirNode,
        op: CmpOp,
    ) -> Result<Option<MirValue>, MirError> {
        let lhs = self.resolve_operand(node.operands[0])?;
        let rhs = self.resolve_operand(node.operands[1])?;
        let dst = self.fresh_value();
        let ty = self.node_mir_type(node);
        self.emit(MirInst::Cmp { dst, op, ty, lhs, rhs });
        self.value_map.insert(node.id, dst);
        Ok(Some(dst))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- MirType sizes -----------------------------------------------------

    #[test]
    fn mir_type_sizes() {
        assert_eq!(MirType::I8.size_bytes(), 1);
        assert_eq!(MirType::I16.size_bytes(), 2);
        assert_eq!(MirType::I32.size_bytes(), 4);
        assert_eq!(MirType::I64.size_bytes(), 8);
        assert_eq!(MirType::U8.size_bytes(), 1);
        assert_eq!(MirType::U16.size_bytes(), 2);
        assert_eq!(MirType::U32.size_bytes(), 4);
        assert_eq!(MirType::U64.size_bytes(), 8);
        assert_eq!(MirType::F32.size_bytes(), 4);
        assert_eq!(MirType::F64.size_bytes(), 8);
        assert_eq!(MirType::Bool.size_bytes(), 1);
        assert_eq!(MirType::Ptr.size_bytes(), 8);
        assert_eq!(MirType::Vec4F.size_bytes(), 16);
        assert_eq!(MirType::Vec8F.size_bytes(), 32);
        assert_eq!(MirType::Vec16F.size_bytes(), 64);
        assert_eq!(MirType::Void.size_bytes(), 0);
        assert_eq!(MirType::Aggregate(24).size_bytes(), 24);
    }

    // -- Display formatting ------------------------------------------------

    #[test]
    fn display_add() {
        let inst = MirInst::Add {
            dst: MirValue(0),
            ty: MirType::I64,
            lhs: MirValue(1),
            rhs: MirValue(2),
        };
        assert_eq!(inst.to_string(), "%0 = add i64 %1, %2");
    }

    #[test]
    fn display_cmp() {
        let inst = MirInst::Cmp {
            dst: MirValue(3),
            op: CmpOp::Lt,
            ty: MirType::I32,
            lhs: MirValue(1),
            rhs: MirValue(2),
        };
        assert_eq!(inst.to_string(), "%3 = cmp lt i32 %1, %2");
    }

    #[test]
    fn display_const() {
        let inst = MirInst::Const {
            dst: MirValue(0),
            ty: MirType::I64,
            value: MirConst::Int(42),
        };
        assert_eq!(inst.to_string(), "%0 = const i64 42");
    }

    #[test]
    fn display_br() {
        let inst = MirInst::Br {
            cond: MirValue(5),
            then_bb: BlockId(1),
            else_bb: BlockId(2),
        };
        assert_eq!(inst.to_string(), "br %5, bb1, bb2");
    }

    #[test]
    fn display_ret() {
        let ret_val = MirInst::Ret { value: Some(MirValue(7)) };
        assert_eq!(ret_val.to_string(), "ret %7");

        let ret_void = MirInst::Ret { value: None };
        assert_eq!(ret_void.to_string(), "ret void");
    }

    #[test]
    fn display_call() {
        let inst = MirInst::Call {
            dst: Some(MirValue(10)),
            func: "foo".to_string(),
            args: vec![MirValue(1), MirValue(2)],
        };
        assert_eq!(inst.to_string(), "%10 = call foo(%1, %2)");

        let void_call = MirInst::Call {
            dst: None,
            func: "bar".to_string(),
            args: vec![],
        };
        assert_eq!(void_call.to_string(), "call bar()");
    }

    #[test]
    fn display_phi() {
        let inst = MirInst::Phi {
            dst: MirValue(4),
            ty: MirType::I32,
            incoming: vec![(BlockId(0), MirValue(1)), (BlockId(1), MirValue(2))],
        };
        assert_eq!(inst.to_string(), "%4 = phi i32 [bb0: %1], [bb1: %2]");
    }

    #[test]
    fn display_cast() {
        let inst = MirInst::Cast {
            dst: MirValue(5),
            from_ty: MirType::I32,
            to_ty: MirType::I64,
            value: MirValue(3),
        };
        assert_eq!(inst.to_string(), "%5 = cast i32 %3 to i64");
    }

    #[test]
    fn display_store() {
        let inst = MirInst::Store {
            ty: MirType::I64,
            addr: MirValue(0),
            value: MirValue(1),
        };
        assert_eq!(inst.to_string(), "store i64 %1, %0");
    }

    #[test]
    fn display_gep() {
        let inst = MirInst::Gep {
            dst: MirValue(6),
            base: MirValue(0),
            indices: vec![MirValue(1), MirValue(2)],
        };
        assert_eq!(inst.to_string(), "%6 = gep %0, %1, %2");
    }

    #[test]
    fn display_vecop() {
        let inst = MirInst::VecOp {
            dst: MirValue(7),
            op: VecOpKind::Add,
            ty: MirType::Vec4F,
            operands: vec![MirValue(1), MirValue(2)],
        };
        assert_eq!(inst.to_string(), "%7 = vadd vec4f %1, %2");
    }

    // -- Block construction ------------------------------------------------

    #[test]
    fn block_construction() {
        let mut blk = MirBlock::new(BlockId(0), "entry");
        blk.instructions.push(MirInst::Const {
            dst: MirValue(0),
            ty: MirType::I64,
            value: MirConst::Int(1),
        });
        blk.instructions.push(MirInst::Ret { value: Some(MirValue(0)) });

        assert_eq!(blk.id, BlockId(0));
        assert_eq!(blk.label, "entry");
        assert_eq!(blk.instructions.len(), 2);

        let displayed = blk.to_string();
        assert!(displayed.contains("entry:"));
        assert!(displayed.contains("%0 = const i64 1"));
        assert!(displayed.contains("ret %0"));
    }

    // -- Function display --------------------------------------------------

    #[test]
    fn function_display() {
        let blk = MirBlock {
            id: BlockId(0),
            label: "entry_0".to_string(),
            instructions: vec![
                MirInst::Const { dst: MirValue(2), ty: MirType::I64, value: MirConst::Int(10) },
                MirInst::Add {
                    dst: MirValue(3),
                    ty: MirType::I64,
                    lhs: MirValue(0),
                    rhs: MirValue(2),
                },
                MirInst::Ret { value: Some(MirValue(3)) },
            ],
            predecessors: vec![],
            successors: vec![],
        };

        let func = MirFunction {
            name: "add_ten".to_string(),
            params: vec![(MirValue(0), MirType::I64)],
            return_type: MirType::I64,
            blocks: vec![blk],
            entry_block: BlockId(0),
            next_value: 4,
        };

        let s = func.to_string();
        assert!(s.contains("fn add_ten(i64 %0) -> i64 {"));
        assert!(s.contains("%3 = add i64 %0, %2"));
    }

    // -- HIR → MIR lowering ------------------------------------------------

    #[test]
    fn lower_simple_return() {
        use crate::hir::{HirFunction, HirParam, HirProgram};
        use crate::types::{AxiomType, IntegerKind, TypeContext};

        let mut type_ctx = TypeContext::new();
        let i64_tid = type_ctx.register(AxiomType::Integer(IntegerKind::I64));

        // fn identity(x: i64) -> i64 { return x; }
        let var_ref = HirNode {
            id: HirId(1),
            op: HirOp::VarRef { name: "x".to_string() },
            operands: vec![],
            ty: Some(i64_tid),
            span: None,
        };
        let ret_node = HirNode {
            id: HirId(2),
            op: HirOp::Return,
            operands: vec![HirId(1)],
            ty: None,
            span: None,
        };

        let func = HirFunction {
            name: "identity".to_string(),
            params: vec![HirParam { name: "x".to_string(), ty: i64_tid }],
            body: vec![var_ref, ret_node],
            return_type: Some(i64_tid),
            is_pub: true,
            span: None,
        };

        let program = HirProgram {
            functions: vec![func],
            structs: vec![],
            enums: vec![],
            type_context: type_ctx,
        };

        let mir = MirBuilder::lower_program(&program).expect("lowering should succeed");
        assert_eq!(mir.functions.len(), 1);

        let mf = &mir.functions[0];
        assert_eq!(mf.name, "identity");
        assert_eq!(mf.params.len(), 1);
        assert_eq!(mf.params[0].1, MirType::I64);
        assert_eq!(mf.return_type, MirType::I64);

        // Should have at least one block with instructions.
        assert!(!mf.blocks.is_empty());
        let has_ret = mf.blocks.iter().any(|b| {
            b.instructions.iter().any(|i| matches!(i, MirInst::Ret { .. }))
        });
        assert!(has_ret, "lowered function must contain a ret instruction");
    }

    #[test]
    fn lower_add_constants() {
        use crate::hir::{HirFunction, HirParam, HirProgram};
        use crate::types::{AxiomType, IntegerKind, TypeContext};

        let mut type_ctx = TypeContext::new();
        let i64_tid = type_ctx.register(AxiomType::Integer(IntegerKind::I64));

        // fn add() -> i64 { return 1 + 2; }
        let c1 = HirNode {
            id: HirId(0),
            op: HirOp::IntConst(1),
            operands: vec![],
            ty: Some(i64_tid),
            span: None,
        };
        let c2 = HirNode {
            id: HirId(1),
            op: HirOp::IntConst(2),
            operands: vec![],
            ty: Some(i64_tid),
            span: None,
        };
        let add = HirNode {
            id: HirId(2),
            op: HirOp::Add,
            operands: vec![HirId(0), HirId(1)],
            ty: Some(i64_tid),
            span: None,
        };
        let ret = HirNode {
            id: HirId(3),
            op: HirOp::Return,
            operands: vec![HirId(2)],
            ty: None,
            span: None,
        };

        let func = HirFunction {
            name: "add".to_string(),
            params: vec![],
            body: vec![c1, c2, add, ret],
            return_type: Some(i64_tid),
            is_pub: false,
            span: None,
        };

        let program = HirProgram {
            functions: vec![func],
            structs: vec![],
            enums: vec![],
            type_context: type_ctx,
        };

        let mir = MirBuilder::lower_program(&program).unwrap();
        let mf = &mir.functions[0];

        // We expect: const 1, const 2, add, ret
        let insts = &mf.blocks[0].instructions;
        assert!(insts.len() >= 3, "expected at least const+const+add instructions");
        assert!(matches!(insts[0], MirInst::Const { value: MirConst::Int(1), .. }));
        assert!(matches!(insts[1], MirInst::Const { value: MirConst::Int(2), .. }));
        assert!(matches!(insts[2], MirInst::Add { .. }));
    }

    #[test]
    fn mir_program_display() {
        let program = MirProgram {
            functions: vec![
                MirFunction {
                    name: "main".to_string(),
                    params: vec![],
                    return_type: MirType::Void,
                    blocks: vec![MirBlock {
                        id: BlockId(0),
                        label: "entry_0".to_string(),
                        instructions: vec![MirInst::Ret { value: None }],
                        predecessors: vec![],
                        successors: vec![],
                    }],
                    entry_block: BlockId(0),
                    next_value: 0,
                },
            ],
        };

        let s = program.to_string();
        assert!(s.contains("fn main() -> void {"));
        assert!(s.contains("ret void"));
    }

    #[test]
    fn mir_error_display() {
        let err = MirError::new("something went wrong");
        assert_eq!(err.to_string(), "MIR error: something went wrong");
    }

    #[test]
    fn axiom_type_conversion() {
        assert_eq!(axiom_type_to_mir(&AxiomType::Integer(IntegerKind::I32)), MirType::I32);
        assert_eq!(axiom_type_to_mir(&AxiomType::Float(FloatKind::F64)), MirType::F64);
        assert_eq!(axiom_type_to_mir(&AxiomType::Bool), MirType::Bool);
        assert_eq!(axiom_type_to_mir(&AxiomType::Char), MirType::U32);
        assert_eq!(axiom_type_to_mir(&AxiomType::Simd(SimdKind::Vec4F)), MirType::Vec4F);
        assert_eq!(axiom_type_to_mir(&AxiomType::Unit), MirType::Void);
    }
}
