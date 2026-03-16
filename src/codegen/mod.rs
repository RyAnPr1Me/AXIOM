//! Code generation module -- MIR to x86-64 AT&T syntax assembly.
//!
//! Translates a [`MirProgram`] into a textual assembly file that can be
//! assembled with GNU `as`.  Only the **x86-64 / System V AMD64 ABI** target
//! is implemented; the other [`Target`] variants return an error.

use std::collections::HashMap;
use std::fmt;

use crate::mir::{
    BlockId, CmpOp, MirBlock, MirConst, MirFunction, MirInst, MirProgram, MirType, MirValue,
    VecOpKind,
};

// ---------------------------------------------------------------------------
// CodegenError
// ---------------------------------------------------------------------------

/// Error produced during code generation.
#[derive(Debug, Clone)]
pub struct CodegenError {
    pub message: String,
}

impl CodegenError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
        }
    }
}

impl fmt::Display for CodegenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "codegen error: {}", self.message)
    }
}

impl std::error::Error for CodegenError {}

// ---------------------------------------------------------------------------
// Target
// ---------------------------------------------------------------------------

/// Target architecture for code generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    X86_64,
    Aarch64,
    Wasm32,
}

// ---------------------------------------------------------------------------
// Register -- x86-64 register set
// ---------------------------------------------------------------------------

/// Physical x86-64 register.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Register {
    Rax,
    Rbx,
    Rcx,
    Rdx,
    Rsi,
    Rdi,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
    Xmm0,
    Xmm1,
    Xmm2,
    Xmm3,
    Xmm4,
    Xmm5,
    Xmm6,
    Xmm7,
    Xmm8,
    Xmm9,
    Xmm10,
    Xmm11,
    Xmm12,
    Xmm13,
    Xmm14,
    Xmm15,
}

impl Register {
    /// AT&T-syntax name for this register (64-bit GPR / SSE).
    pub fn name(&self) -> &'static str {
        match self {
            Register::Rax => "%rax",
            Register::Rbx => "%rbx",
            Register::Rcx => "%rcx",
            Register::Rdx => "%rdx",
            Register::Rsi => "%rsi",
            Register::Rdi => "%rdi",
            Register::R8 => "%r8",
            Register::R9 => "%r9",
            Register::R10 => "%r10",
            Register::R11 => "%r11",
            Register::R12 => "%r12",
            Register::R13 => "%r13",
            Register::R14 => "%r14",
            Register::R15 => "%r15",
            Register::Xmm0 => "%xmm0",
            Register::Xmm1 => "%xmm1",
            Register::Xmm2 => "%xmm2",
            Register::Xmm3 => "%xmm3",
            Register::Xmm4 => "%xmm4",
            Register::Xmm5 => "%xmm5",
            Register::Xmm6 => "%xmm6",
            Register::Xmm7 => "%xmm7",
            Register::Xmm8 => "%xmm8",
            Register::Xmm9 => "%xmm9",
            Register::Xmm10 => "%xmm10",
            Register::Xmm11 => "%xmm11",
            Register::Xmm12 => "%xmm12",
            Register::Xmm13 => "%xmm13",
            Register::Xmm14 => "%xmm14",
            Register::Xmm15 => "%xmm15",
        }
    }

    /// Whether this is a callee-saved (non-volatile) GPR.
    pub fn is_callee_saved(&self) -> bool {
        matches!(
            self,
            Register::Rbx
                | Register::R12
                | Register::R13
                | Register::R14
                | Register::R15
        )
    }

    /// Whether this is an SSE / vector register.
    pub fn is_xmm(&self) -> bool {
        matches!(
            self,
            Register::Xmm0
                | Register::Xmm1
                | Register::Xmm2
                | Register::Xmm3
                | Register::Xmm4
                | Register::Xmm5
                | Register::Xmm6
                | Register::Xmm7
                | Register::Xmm8
                | Register::Xmm9
                | Register::Xmm10
                | Register::Xmm11
                | Register::Xmm12
                | Register::Xmm13
                | Register::Xmm14
                | Register::Xmm15
        )
    }
}

impl fmt::Display for Register {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// Location -- where an SSA value lives
// ---------------------------------------------------------------------------

/// Runtime location of a value: register, stack slot, or immediate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Location {
    Reg(Register),
    /// Offset from `%rbp` (always negative for locals).
    Stack(i32),
    Immediate(i64),
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Location::Reg(r) => write!(f, "{}", r.name()),
            Location::Stack(off) => write!(f, "{}(%rbp)", off),
            Location::Immediate(v) => write!(f, "${}", v),
        }
    }
}

// ---------------------------------------------------------------------------
// RegAllocator -- simple linear-scan register allocator
// ---------------------------------------------------------------------------

/// Allocatable GPRs in priority order (caller-saved first, then callee-saved).
const GPR_POOL: &[Register] = &[
    Register::R10,
    Register::R11,
    Register::Rcx,
    Register::Rdx,
    Register::Rsi,
    Register::Rdi,
    Register::R8,
    Register::R9,
    Register::Rbx,
    Register::R12,
    Register::R13,
    Register::R14,
    Register::R15,
];

/// Allocatable XMM registers.
const XMM_POOL: &[Register] = &[
    Register::Xmm0,
    Register::Xmm1,
    Register::Xmm2,
    Register::Xmm3,
    Register::Xmm4,
    Register::Xmm5,
    Register::Xmm6,
    Register::Xmm7,
    Register::Xmm8,
    Register::Xmm9,
    Register::Xmm10,
    Register::Xmm11,
    Register::Xmm12,
    Register::Xmm13,
    Register::Xmm14,
    Register::Xmm15,
];

/// Simple linear-scan register allocator for one function.
#[derive(Debug, Clone)]
pub struct RegAllocator {
    /// Map from SSA value to location.
    locations: HashMap<MirValue, Location>,
    /// Set of GPRs currently in use.
    gpr_used: HashMap<Register, MirValue>,
    /// Set of XMM registers currently in use.
    xmm_used: HashMap<Register, MirValue>,
    /// Next stack offset (grows downward, always <= 0).
    stack_offset: i32,
    /// Callee-saved GPRs that have been touched.
    pub used_callee_saved: Vec<Register>,
}

impl RegAllocator {
    pub fn new() -> Self {
        Self {
            locations: HashMap::new(),
            gpr_used: HashMap::new(),
            xmm_used: HashMap::new(),
            stack_offset: 0,
            used_callee_saved: Vec::new(),
        }
    }

    /// Total bytes of local stack space consumed (positive number).
    pub fn frame_size(&self) -> i32 {
        -self.stack_offset
    }

    /// Return the location previously assigned to `value`, if any.
    pub fn get(&self, value: MirValue) -> Option<&Location> {
        self.locations.get(&value)
    }

    /// Allocate a location for `value` based on its MIR type.
    pub fn allocate(&mut self, value: MirValue, ty: &MirType) -> Location {
        if let Some(&loc) = self.locations.get(&value) {
            return loc;
        }

        let needs_xmm = matches!(
            ty,
            MirType::F32 | MirType::F64 | MirType::Vec4F | MirType::Vec8F | MirType::Vec16F
        );

        let loc = if needs_xmm {
            self.alloc_xmm(value, ty)
        } else {
            self.alloc_gpr(value, ty)
        };

        self.locations.insert(value, loc);
        loc
    }

    /// Free the resources held by `value`.
    pub fn free(&mut self, value: MirValue) {
        if let Some(loc) = self.locations.remove(&value) {
            match loc {
                Location::Reg(r) if r.is_xmm() => {
                    self.xmm_used.remove(&r);
                }
                Location::Reg(r) => {
                    self.gpr_used.remove(&r);
                }
                _ => {}
            }
        }
    }

    fn alloc_gpr(&mut self, value: MirValue, ty: &MirType) -> Location {
        for &reg in GPR_POOL {
            if !self.gpr_used.contains_key(&reg) {
                self.gpr_used.insert(reg, value);
                if reg.is_callee_saved() && !self.used_callee_saved.contains(&reg) {
                    self.used_callee_saved.push(reg);
                }
                return Location::Reg(reg);
            }
        }
        self.spill_to_stack(ty)
    }

    fn alloc_xmm(&mut self, value: MirValue, ty: &MirType) -> Location {
        for &reg in XMM_POOL {
            if !self.xmm_used.contains_key(&reg) {
                self.xmm_used.insert(reg, value);
                return Location::Reg(reg);
            }
        }
        self.spill_to_stack(ty)
    }

    fn spill_to_stack(&mut self, ty: &MirType) -> Location {
        let size = ty.size_bytes().max(8) as i32;
        let align = size;
        self.stack_offset -= size;
        if self.stack_offset % align != 0 {
            self.stack_offset -= align + (self.stack_offset % align);
        }
        Location::Stack(self.stack_offset)
    }
}

// ---------------------------------------------------------------------------
// InstructionSelector -- MIR instruction to assembly lines
// ---------------------------------------------------------------------------

/// Selects x86-64 instructions for each MIR instruction.
#[derive(Debug)]
pub struct InstructionSelector {
    allocator: RegAllocator,
    /// Accumulated assembly lines for the current block.
    lines: Vec<String>,
    /// Function name (used for label generation).
    func_name: String,
}

impl InstructionSelector {
    pub fn new(func_name: &str) -> Self {
        Self {
            allocator: RegAllocator::new(),
            lines: Vec::new(),
            func_name: func_name.to_owned(),
        }
    }

    /// Consume self and return the allocator (for frame-size queries).
    pub fn into_allocator(self) -> RegAllocator {
        self.allocator
    }

    /// Reference to the inner allocator.
    pub fn allocator(&self) -> &RegAllocator {
        &self.allocator
    }

    /// Mutable reference to the inner allocator.
    pub fn allocator_mut(&mut self) -> &mut RegAllocator {
        &mut self.allocator
    }

    /// Set up function parameter locations matching the System V AMD64 ABI.
    pub fn setup_params(&mut self, params: &[(MirValue, MirType)]) {
        const INT_ARG_REGS: &[Register] = &[
            Register::Rdi,
            Register::Rsi,
            Register::Rdx,
            Register::Rcx,
            Register::R8,
            Register::R9,
        ];

        const FP_ARG_REGS: &[Register] = &[
            Register::Xmm0,
            Register::Xmm1,
            Register::Xmm2,
            Register::Xmm3,
            Register::Xmm4,
            Register::Xmm5,
            Register::Xmm6,
            Register::Xmm7,
        ];

        let mut int_idx = 0usize;
        let mut fp_idx = 0usize;

        for (val, ty) in params {
            let is_fp = matches!(
                ty,
                MirType::F32 | MirType::F64 | MirType::Vec4F | MirType::Vec8F | MirType::Vec16F
            );

            if is_fp && fp_idx < FP_ARG_REGS.len() {
                let reg = FP_ARG_REGS[fp_idx];
                fp_idx += 1;
                self.allocator.xmm_used.insert(reg, *val);
                self.allocator.locations.insert(*val, Location::Reg(reg));
            } else if !is_fp && int_idx < INT_ARG_REGS.len() {
                let reg = INT_ARG_REGS[int_idx];
                int_idx += 1;
                self.allocator.gpr_used.insert(reg, *val);
                self.allocator.locations.insert(*val, Location::Reg(reg));
            } else {
                // Additional args come from the stack (caller pushed).
                let loc = self.allocator.allocate(*val, ty);
                self.allocator.locations.insert(*val, loc);
            }
        }
    }

    /// Generate a basic-block label.
    pub fn block_label(&self, block_id: BlockId) -> String {
        format!(".LBB_{}_{}", self.func_name, block_id.0)
    }

    /// Select instructions for every instruction in `block`.
    pub fn select_block(&mut self, block: &MirBlock) -> Vec<String> {
        self.lines.clear();
        for inst in &block.instructions {
            self.select_inst(inst);
        }
        self.lines.clone()
    }

    // -- per-instruction selection -----------------------------------------

    fn select_inst(&mut self, inst: &MirInst) {
        match inst {
            MirInst::Const { dst, ty, value } => self.sel_const(*dst, ty, value),

            MirInst::Add { dst, ty, lhs, rhs } => self.sel_binop("add", *dst, ty, *lhs, *rhs),
            MirInst::Sub { dst, ty, lhs, rhs } => self.sel_binop("sub", *dst, ty, *lhs, *rhs),
            MirInst::Mul { dst, ty, lhs, rhs } => self.sel_mul(*dst, ty, *lhs, *rhs),
            MirInst::Div { dst, ty, lhs, rhs } => self.sel_div(*dst, ty, *lhs, *rhs),
            MirInst::Mod { dst, ty, lhs, rhs } => self.sel_mod(*dst, ty, *lhs, *rhs),

            MirInst::And { dst, ty, lhs, rhs } => self.sel_binop("and", *dst, ty, *lhs, *rhs),
            MirInst::Or { dst, ty, lhs, rhs } => self.sel_binop("or", *dst, ty, *lhs, *rhs),
            MirInst::Xor { dst, ty, lhs, rhs } => self.sel_binop("xor", *dst, ty, *lhs, *rhs),
            MirInst::Shl { dst, ty, lhs, rhs } => self.sel_shift("shl", *dst, ty, *lhs, *rhs),
            MirInst::Shr { dst, ty, lhs, rhs } => self.sel_shift("shr", *dst, ty, *lhs, *rhs),

            MirInst::Neg { dst, ty, operand } => self.sel_neg(*dst, ty, *operand),
            MirInst::Not { dst, ty, operand } => self.sel_not(*dst, ty, *operand),

            MirInst::Cmp { dst, op, ty, lhs, rhs } => self.sel_cmp(*dst, *op, ty, *lhs, *rhs),

            MirInst::Load { dst, ty, addr } => self.sel_load(*dst, ty, *addr),
            MirInst::Store { ty, addr, value } => self.sel_store(ty, *addr, *value),

            MirInst::StackAlloc { dst, size, align } => {
                self.sel_stack_alloc(*dst, *size, *align);
            }

            MirInst::Br { cond, then_bb, else_bb } => self.sel_br(*cond, *then_bb, *else_bb),
            MirInst::Jump { target } => self.sel_jump(*target),
            MirInst::Ret { value } => self.sel_ret(value.as_ref()),

            MirInst::Call { dst, func, args } => self.sel_call(dst.as_ref(), func, args),

            MirInst::VecOp { dst, op, ty, operands } => {
                self.sel_vecop(*dst, *op, ty, operands);
            }

            MirInst::Cast { dst, from_ty, to_ty, value } => {
                self.sel_cast(*dst, from_ty, to_ty, *value);
            }

            MirInst::Gep { dst, base, indices } => self.sel_gep(*dst, *base, indices),

            MirInst::Phi { dst, ty, .. } => {
                // Phi nodes are resolved during SSA deconstruction;
                // allocate a location so later instructions can reference the value.
                self.allocator.allocate(*dst, ty);
            }
        }
    }

    // -- helpers -----------------------------------------------------------

    fn emit(&mut self, line: impl Into<String>) {
        self.lines.push(line.into());
    }

    /// Ensure `value` is in a register and return the register name.
    fn ensure_reg(&mut self, value: MirValue, ty: &MirType) -> String {
        let loc = self.allocator.allocate(value, ty);
        match loc {
            Location::Reg(r) => r.name().to_owned(),
            Location::Stack(off) => {
                let tmp = if is_float_or_vec(ty) { "%xmm15" } else { "%rax" };
                let mnemonic = move_mnemonic(ty);
                self.emit(format!("    {} {}(%rbp), {}", mnemonic, off, tmp));
                tmp.to_owned()
            }
            Location::Immediate(v) => {
                self.emit(format!("    movq ${}, %rax", v));
                "%rax".to_owned()
            }
        }
    }

    fn suffix(ty: &MirType) -> &'static str {
        match ty {
            MirType::I8 | MirType::U8 | MirType::Bool => "b",
            MirType::I16 | MirType::U16 => "w",
            MirType::I32 | MirType::U32 => "l",
            _ => "q",
        }
    }

    // -- const ------------------------------------------------------------

    fn sel_const(&mut self, dst: MirValue, ty: &MirType, value: &MirConst) {
        match value {
            MirConst::Int(v) => {
                let loc = self.allocator.allocate(dst, ty);
                match loc {
                    Location::Reg(r) => {
                        self.emit(format!("    movq ${}, {}", v, r.name()));
                    }
                    Location::Stack(off) => {
                        self.emit(format!("    movq ${}, %rax", v));
                        self.emit(format!("    movq %rax, {}(%rbp)", off));
                    }
                    Location::Immediate(_) => unreachable!(),
                }
            }
            MirConst::Float(v) => {
                let loc = self.allocator.allocate(dst, ty);
                let bits = v.to_bits() as i64;
                match loc {
                    Location::Reg(r) if r.is_xmm() => {
                        self.emit(format!("    movq ${}, %rax", bits));
                        self.emit(format!("    movq %rax, {}", r.name()));
                    }
                    Location::Reg(r) => {
                        self.emit(format!("    movq ${}, {}", bits, r.name()));
                    }
                    Location::Stack(off) => {
                        self.emit(format!("    movq ${}, %rax", bits));
                        self.emit(format!("    movq %rax, {}(%rbp)", off));
                    }
                    Location::Immediate(_) => unreachable!(),
                }
            }
            MirConst::Bool(b) => {
                let v: i64 = if *b { 1 } else { 0 };
                let loc = self.allocator.allocate(dst, ty);
                match loc {
                    Location::Reg(r) => {
                        self.emit(format!("    movq ${}, {}", v, r.name()));
                    }
                    Location::Stack(off) => {
                        self.emit(format!("    movq ${}, %rax", v));
                        self.emit(format!("    movq %rax, {}(%rbp)", off));
                    }
                    Location::Immediate(_) => unreachable!(),
                }
            }
            MirConst::Zero => {
                let loc = self.allocator.allocate(dst, ty);
                match loc {
                    Location::Reg(r) if r.is_xmm() => {
                        let n = r.name();
                        self.emit(format!("    xorps {}, {}", n, n));
                    }
                    Location::Reg(r) => {
                        let n = r.name();
                        self.emit(format!("    xorq {}, {}", n, n));
                    }
                    Location::Stack(off) => {
                        self.emit(format!("    movq $0, {}(%rbp)", off));
                    }
                    Location::Immediate(_) => unreachable!(),
                }
            }
        }
    }

    // -- binary integer / fp arithmetic -----------------------------------

    fn sel_binop(
        &mut self,
        op: &str,
        dst: MirValue,
        ty: &MirType,
        lhs: MirValue,
        rhs: MirValue,
    ) {
        if is_float_or_vec(ty) {
            self.sel_fp_binop(op, dst, ty, lhs, rhs);
            return;
        }

        let sfx = Self::suffix(ty);
        let lhs_r = self.ensure_reg(lhs, ty);
        let rhs_r = self.ensure_reg(rhs, ty);
        let dst_loc = self.allocator.allocate(dst, ty);

        match dst_loc {
            Location::Reg(rd) => {
                let rd_name = rd.name();
                if rd_name != lhs_r {
                    self.emit(format!("    movq {}, {}", lhs_r, rd_name));
                }
                self.emit(format!("    {}{} {}, {}", op, sfx, rhs_r, rd_name));
            }
            Location::Stack(off) => {
                self.emit(format!("    movq {}, %rax", lhs_r));
                self.emit(format!("    {}{} {}, %rax", op, sfx, rhs_r));
                self.emit(format!("    movq %rax, {}(%rbp)", off));
            }
            Location::Immediate(_) => unreachable!(),
        }
    }

    fn sel_fp_binop(
        &mut self,
        op: &str,
        dst: MirValue,
        ty: &MirType,
        lhs: MirValue,
        rhs: MirValue,
    ) {
        let (mnemonic, is_vec) = match (op, ty) {
            ("add", MirType::F32) => ("addss", false),
            ("add", MirType::F64) => ("addsd", false),
            ("sub", MirType::F32) => ("subss", false),
            ("sub", MirType::F64) => ("subsd", false),
            ("add", MirType::Vec4F) | ("add", MirType::Vec8F) | ("add", MirType::Vec16F) => ("vaddps", true),
            ("sub", MirType::Vec4F) | ("sub", MirType::Vec8F) | ("sub", MirType::Vec16F) => ("vsubps", true),
            _ => ("addsd", false),
        };

        let lhs_r = self.ensure_reg(lhs, ty);
        let rhs_r = self.ensure_reg(rhs, ty);
        let dst_loc = self.allocator.allocate(dst, ty);

        match dst_loc {
            Location::Reg(rd) => {
                let rd_name = rd.name();
                if is_vec {
                    self.emit(format!("    {} {}, {}, {}", mnemonic, rhs_r, lhs_r, rd_name));
                } else {
                    if rd_name != lhs_r {
                        self.emit(format!("    movsd {}, {}", lhs_r, rd_name));
                    }
                    self.emit(format!("    {} {}, {}", mnemonic, rhs_r, rd_name));
                }
            }
            Location::Stack(off) => {
                self.emit(format!("    movsd {}, %xmm15", lhs_r));
                self.emit(format!("    {} {}, %xmm15", mnemonic, rhs_r));
                self.emit(format!("    movsd %xmm15, {}(%rbp)", off));
            }
            Location::Immediate(_) => unreachable!(),
        }
    }

    // -- multiply ---------------------------------------------------------

    fn sel_mul(&mut self, dst: MirValue, ty: &MirType, lhs: MirValue, rhs: MirValue) {
        if is_float_or_vec(ty) {
            let mnemonic = match ty {
                MirType::F32 => "mulss",
                MirType::F64 => "mulsd",
                MirType::Vec4F | MirType::Vec8F | MirType::Vec16F => "vmulps",
                _ => "mulsd",
            };
            let lhs_r = self.ensure_reg(lhs, ty);
            let rhs_r = self.ensure_reg(rhs, ty);
            let dst_loc = self.allocator.allocate(dst, ty);
            match dst_loc {
                Location::Reg(rd) => {
                    let rd_name = rd.name();
                    if matches!(ty, MirType::Vec4F | MirType::Vec8F | MirType::Vec16F) {
                        self.emit(format!(
                            "    {} {}, {}, {}",
                            mnemonic, rhs_r, lhs_r, rd_name
                        ));
                    } else {
                        if rd_name != lhs_r {
                            self.emit(format!("    movsd {}, {}", lhs_r, rd_name));
                        }
                        self.emit(format!("    {} {}, {}", mnemonic, rhs_r, rd_name));
                    }
                }
                Location::Stack(off) => {
                    self.emit(format!("    movsd {}, %xmm15", lhs_r));
                    self.emit(format!("    {} {}, %xmm15", mnemonic, rhs_r));
                    self.emit(format!("    movsd %xmm15, {}(%rbp)", off));
                }
                Location::Immediate(_) => unreachable!(),
            }
            return;
        }

        let sfx = Self::suffix(ty);
        let lhs_r = self.ensure_reg(lhs, ty);
        let rhs_r = self.ensure_reg(rhs, ty);
        let dst_loc = self.allocator.allocate(dst, ty);

        match dst_loc {
            Location::Reg(rd) => {
                let rd_name = rd.name();
                if rd_name != lhs_r {
                    self.emit(format!("    movq {}, {}", lhs_r, rd_name));
                }
                self.emit(format!("    imul{} {}, {}", sfx, rhs_r, rd_name));
            }
            Location::Stack(off) => {
                self.emit(format!("    movq {}, %rax", lhs_r));
                self.emit(format!("    imul{} {}, %rax", sfx, rhs_r));
                self.emit(format!("    movq %rax, {}(%rbp)", off));
            }
            Location::Immediate(_) => unreachable!(),
        }
    }

    // -- divide -----------------------------------------------------------

    fn sel_div(&mut self, dst: MirValue, ty: &MirType, lhs: MirValue, rhs: MirValue) {
        if is_float_or_vec(ty) {
            let mnemonic = match ty {
                MirType::F32 => "divss",
                MirType::F64 => "divsd",
                MirType::Vec4F | MirType::Vec8F | MirType::Vec16F => "vdivps",
                _ => "divsd",
            };
            let lhs_r = self.ensure_reg(lhs, ty);
            let rhs_r = self.ensure_reg(rhs, ty);
            let dst_loc = self.allocator.allocate(dst, ty);
            match dst_loc {
                Location::Reg(rd) => {
                    let rd_name = rd.name();
                    if rd_name != lhs_r {
                        self.emit(format!("    movsd {}, {}", lhs_r, rd_name));
                    }
                    self.emit(format!("    {} {}, {}", mnemonic, rhs_r, rd_name));
                }
                Location::Stack(off) => {
                    self.emit(format!("    movsd {}, %xmm15", lhs_r));
                    self.emit(format!("    {} {}, %xmm15", mnemonic, rhs_r));
                    self.emit(format!("    movsd %xmm15, {}(%rbp)", off));
                }
                Location::Immediate(_) => unreachable!(),
            }
            return;
        }

        let lhs_r = self.ensure_reg(lhs, ty);
        let rhs_r = self.ensure_reg(rhs, ty);
        self.emit(format!("    movq {}, %rax", lhs_r));
        self.emit("    cqto".to_owned());
        self.emit(format!("    idivq {}", rhs_r));
        let dst_loc = self.allocator.allocate(dst, ty);
        match dst_loc {
            Location::Reg(rd) => {
                if rd != Register::Rax {
                    self.emit(format!("    movq %rax, {}", rd.name()));
                }
            }
            Location::Stack(off) => {
                self.emit(format!("    movq %rax, {}(%rbp)", off));
            }
            Location::Immediate(_) => unreachable!(),
        }
    }

    // -- modulo -----------------------------------------------------------

    fn sel_mod(&mut self, dst: MirValue, ty: &MirType, lhs: MirValue, rhs: MirValue) {
        let lhs_r = self.ensure_reg(lhs, ty);
        let rhs_r = self.ensure_reg(rhs, ty);
        self.emit(format!("    movq {}, %rax", lhs_r));
        self.emit("    cqto".to_owned());
        self.emit(format!("    idivq {}", rhs_r));
        let dst_loc = self.allocator.allocate(dst, ty);
        match dst_loc {
            Location::Reg(rd) => {
                if rd != Register::Rdx {
                    self.emit(format!("    movq %rdx, {}", rd.name()));
                }
            }
            Location::Stack(off) => {
                self.emit(format!("    movq %rdx, {}(%rbp)", off));
            }
            Location::Immediate(_) => unreachable!(),
        }
    }

    // -- shifts -----------------------------------------------------------

    fn sel_shift(
        &mut self,
        op: &str,
        dst: MirValue,
        ty: &MirType,
        lhs: MirValue,
        rhs: MirValue,
    ) {
        let sfx = Self::suffix(ty);
        let lhs_r = self.ensure_reg(lhs, ty);
        let rhs_r = self.ensure_reg(rhs, ty);
        if rhs_r != "%rcx" {
            self.emit(format!("    movq {}, %rcx", rhs_r));
        }
        let dst_loc = self.allocator.allocate(dst, ty);
        match dst_loc {
            Location::Reg(rd) => {
                let rd_name = rd.name();
                if rd_name != lhs_r {
                    self.emit(format!("    movq {}, {}", lhs_r, rd_name));
                }
                self.emit(format!("    {}{} %cl, {}", op, sfx, rd_name));
            }
            Location::Stack(off) => {
                self.emit(format!("    movq {}, %rax", lhs_r));
                self.emit(format!("    {}{} %cl, %rax", op, sfx));
                self.emit(format!("    movq %rax, {}(%rbp)", off));
            }
            Location::Immediate(_) => unreachable!(),
        }
    }

    // -- unary neg / not --------------------------------------------------

    fn sel_neg(&mut self, dst: MirValue, ty: &MirType, operand: MirValue) {
        let sfx = Self::suffix(ty);
        let src = self.ensure_reg(operand, ty);
        let dst_loc = self.allocator.allocate(dst, ty);
        match dst_loc {
            Location::Reg(rd) => {
                let rd_name = rd.name();
                if rd_name != src {
                    self.emit(format!("    movq {}, {}", src, rd_name));
                }
                self.emit(format!("    neg{} {}", sfx, rd_name));
            }
            Location::Stack(off) => {
                self.emit(format!("    movq {}, %rax", src));
                self.emit(format!("    neg{} %rax", sfx));
                self.emit(format!("    movq %rax, {}(%rbp)", off));
            }
            Location::Immediate(_) => unreachable!(),
        }
    }

    fn sel_not(&mut self, dst: MirValue, ty: &MirType, operand: MirValue) {
        let sfx = Self::suffix(ty);
        let src = self.ensure_reg(operand, ty);
        let dst_loc = self.allocator.allocate(dst, ty);
        match dst_loc {
            Location::Reg(rd) => {
                let rd_name = rd.name();
                if rd_name != src {
                    self.emit(format!("    movq {}, {}", src, rd_name));
                }
                self.emit(format!("    not{} {}", sfx, rd_name));
            }
            Location::Stack(off) => {
                self.emit(format!("    movq {}, %rax", src));
                self.emit(format!("    not{} %rax", sfx));
                self.emit(format!("    movq %rax, {}(%rbp)", off));
            }
            Location::Immediate(_) => unreachable!(),
        }
    }

    // -- compare ----------------------------------------------------------

    fn sel_cmp(
        &mut self,
        dst: MirValue,
        op: CmpOp,
        ty: &MirType,
        lhs: MirValue,
        rhs: MirValue,
    ) {
        let sfx = Self::suffix(ty);
        let lhs_r = self.ensure_reg(lhs, ty);
        let rhs_r = self.ensure_reg(rhs, ty);
        self.emit(format!("    cmp{} {}, {}", sfx, rhs_r, lhs_r));

        let setcc = match op {
            CmpOp::Eq => "sete",
            CmpOp::Ne => "setne",
            CmpOp::Lt => "setl",
            CmpOp::Le => "setle",
            CmpOp::Gt => "setg",
            CmpOp::Ge => "setge",
        };

        let dst_loc = self.allocator.allocate(dst, &MirType::Bool);
        match dst_loc {
            Location::Reg(rd) => {
                self.emit(format!("    {} %al", setcc));
                self.emit(format!("    movzbq %al, {}", rd.name()));
            }
            Location::Stack(off) => {
                self.emit(format!("    {} %al", setcc));
                self.emit("    movzbq %al, %rax".to_owned());
                self.emit(format!("    movq %rax, {}(%rbp)", off));
            }
            Location::Immediate(_) => unreachable!(),
        }
    }

    // -- memory -----------------------------------------------------------

    fn sel_load(&mut self, dst: MirValue, ty: &MirType, addr: MirValue) {
        let addr_r = self.ensure_reg(addr, &MirType::Ptr);
        let dst_loc = self.allocator.allocate(dst, ty);
        let mnemonic = move_mnemonic(ty);
        match dst_loc {
            Location::Reg(rd) => {
                self.emit(format!("    {} ({}), {}", mnemonic, addr_r, rd.name()));
            }
            Location::Stack(off) => {
                let tmp = if is_float_or_vec(ty) { "%xmm15" } else { "%rax" };
                self.emit(format!("    {} ({}), {}", mnemonic, addr_r, tmp));
                self.emit(format!("    {} {}, {}(%rbp)", mnemonic, tmp, off));
            }
            Location::Immediate(_) => unreachable!(),
        }
    }

    fn sel_store(&mut self, ty: &MirType, addr: MirValue, value: MirValue) {
        let val_r = self.ensure_reg(value, ty);
        let addr_r = self.ensure_reg(addr, &MirType::Ptr);
        let mnemonic = move_mnemonic(ty);
        self.emit(format!("    {} {}, ({})", mnemonic, val_r, addr_r));
    }

    // -- stack allocation -------------------------------------------------

    fn sel_stack_alloc(&mut self, dst: MirValue, size: usize, align: usize) {
        let align = align.max(8) as i32;
        let size = size.max(8) as i32;
        self.allocator.stack_offset -= size;
        if self.allocator.stack_offset % align != 0 {
            self.allocator.stack_offset -= align + (self.allocator.stack_offset % align);
        }
        let off = self.allocator.stack_offset;
        let dst_loc = self.allocator.allocate(dst, &MirType::Ptr);
        match dst_loc {
            Location::Reg(rd) => {
                self.emit(format!("    leaq {}(%rbp), {}", off, rd.name()));
            }
            Location::Stack(dst_off) => {
                self.emit(format!("    leaq {}(%rbp), %rax", off));
                self.emit(format!("    movq %rax, {}(%rbp)", dst_off));
            }
            Location::Immediate(_) => unreachable!(),
        }
    }

    // -- control flow -----------------------------------------------------

    fn sel_br(&mut self, cond: MirValue, then_bb: BlockId, else_bb: BlockId) {
        let cond_r = self.ensure_reg(cond, &MirType::Bool);
        let c = cond_r.clone();
        self.emit(format!("    testb {}, {}", c, c));
        self.emit(format!("    jne {}", self.block_label(then_bb)));
        self.emit(format!("    jmp {}", self.block_label(else_bb)));
    }

    fn sel_jump(&mut self, target: BlockId) {
        self.emit(format!("    jmp {}", self.block_label(target)));
    }

    fn sel_ret(&mut self, value: Option<&MirValue>) {
        if let Some(&val) = value {
            let loc = self
                .allocator
                .get(val)
                .copied()
                .unwrap_or(Location::Reg(Register::Rax));
            let is_fp = matches!(loc, Location::Reg(r) if r.is_xmm());
            if is_fp {
                let src = loc.to_string();
                if src != "%xmm0" {
                    self.emit(format!("    movsd {}, %xmm0", src));
                }
            } else {
                let src = loc.to_string();
                if src != "%rax" {
                    self.emit(format!("    movq {}, %rax", src));
                }
            }
        }
        self.emit("    # ret".to_owned());
    }

    // -- call (System V AMD64 ABI) ----------------------------------------

    fn sel_call(&mut self, dst: Option<&MirValue>, func: &str, args: &[MirValue]) {
        const INT_ARG_REGS: &[Register] = &[
            Register::Rdi,
            Register::Rsi,
            Register::Rdx,
            Register::Rcx,
            Register::R8,
            Register::R9,
        ];

        for (i, &arg) in args.iter().enumerate() {
            let src = self.ensure_reg(arg, &MirType::I64);
            if i < INT_ARG_REGS.len() {
                let tgt = INT_ARG_REGS[i].name();
                if src != tgt {
                    self.emit(format!("    movq {}, {}", src, tgt));
                }
            } else {
                self.emit(format!("    pushq {}", src));
            }
        }

        self.emit(format!("    callq {}", func));

        let stack_args = args.len().saturating_sub(INT_ARG_REGS.len());
        if stack_args > 0 {
            self.emit(format!("    addq ${}, %rsp", stack_args * 8));
        }

        if let Some(&d) = dst {
            let loc = self.allocator.allocate(d, &MirType::I64);
            match loc {
                Location::Reg(r) => {
                    if r != Register::Rax {
                        self.emit(format!("    movq %rax, {}", r.name()));
                    }
                }
                Location::Stack(off) => {
                    self.emit(format!("    movq %rax, {}(%rbp)", off));
                }
                Location::Immediate(_) => unreachable!(),
            }
        }
    }

    // -- SIMD / vector ops ------------------------------------------------

    fn sel_vecop(
        &mut self,
        dst: MirValue,
        op: VecOpKind,
        ty: &MirType,
        operands: &[MirValue],
    ) {
        let dst_loc = self.allocator.allocate(dst, ty);
        let dst_reg = match dst_loc {
            Location::Reg(r) => r.name().to_owned(),
            _ => "%xmm15".to_owned(),
        };

        match op {
            VecOpKind::Add | VecOpKind::Sub | VecOpKind::Mul | VecOpKind::Div => {
                let mnemonic = match op {
                    VecOpKind::Add => "vaddps",
                    VecOpKind::Sub => "vsubps",
                    VecOpKind::Mul => "vmulps",
                    VecOpKind::Div => "vdivps",
                    _ => unreachable!(),
                };
                if operands.len() >= 2 {
                    let src1 = self.ensure_reg(operands[0], ty);
                    let src2 = self.ensure_reg(operands[1], ty);
                    self.emit(format!(
                        "    {} {}, {}, {}",
                        mnemonic, src2, src1, dst_reg
                    ));
                }
            }
            VecOpKind::Load => {
                if let Some(&addr) = operands.first() {
                    let addr_r = self.ensure_reg(addr, &MirType::Ptr);
                    self.emit(format!("    vmovaps ({}), {}", addr_r, dst_reg));
                }
            }
            VecOpKind::Store => {
                if operands.len() >= 2 {
                    let val_r = self.ensure_reg(operands[0], ty);
                    let addr_r = self.ensure_reg(operands[1], &MirType::Ptr);
                    self.emit(format!("    vmovaps {}, ({})", val_r, addr_r));
                }
            }
            VecOpKind::Broadcast => {
                if let Some(&src) = operands.first() {
                    let src_r = self.ensure_reg(src, &MirType::F32);
                    self.emit(format!("    vbroadcastss {}, {}", src_r, dst_reg));
                }
            }
            VecOpKind::Shuffle => {
                if operands.len() >= 2 {
                    let src1 = self.ensure_reg(operands[0], ty);
                    let src2 = self.ensure_reg(operands[1], ty);
                    self.emit(format!(
                        "    vshufps $0, {}, {}, {}",
                        src2, src1, dst_reg
                    ));
                }
            }
            VecOpKind::Extract => {
                if operands.len() >= 2 {
                    let src = self.ensure_reg(operands[0], ty);
                    self.emit(format!("    vextractps $0, {}, {}", src, dst_reg));
                }
            }
            VecOpKind::Insert => {
                if operands.len() >= 2 {
                    let vec_r = self.ensure_reg(operands[0], ty);
                    let elem_r = self.ensure_reg(operands[1], &MirType::F32);
                    self.emit(format!(
                        "    vinsertps $0, {}, {}, {}",
                        elem_r, vec_r, dst_reg
                    ));
                }
            }
        }

        if let Location::Stack(off) = dst_loc {
            self.emit(format!("    vmovaps {}, {}(%rbp)", dst_reg, off));
        }
    }

    // -- cast -------------------------------------------------------------

    fn sel_cast(&mut self, dst: MirValue, from_ty: &MirType, to_ty: &MirType, value: MirValue) {
        let src = self.ensure_reg(value, from_ty);
        let dst_loc = self.allocator.allocate(dst, to_ty);
        let dst_name = match dst_loc {
            Location::Reg(r) => r.name().to_owned(),
            Location::Stack(_) => {
                if is_float_or_vec(to_ty) {
                    "%xmm15".to_owned()
                } else {
                    "%rax".to_owned()
                }
            }
            Location::Immediate(_) => unreachable!(),
        };

        match (from_ty, to_ty) {
            (MirType::I64, MirType::F64) | (MirType::I32, MirType::F64) => {
                self.emit(format!("    cvtsi2sdq {}, {}", src, dst_name));
            }
            (MirType::F64, MirType::I64) | (MirType::F64, MirType::I32) => {
                self.emit(format!("    cvttsd2siq {}, {}", src, dst_name));
            }
            (MirType::I64, MirType::F32) | (MirType::I32, MirType::F32) => {
                self.emit(format!("    cvtsi2ssq {}, {}", src, dst_name));
            }
            (MirType::F32, MirType::I64) | (MirType::F32, MirType::I32) => {
                self.emit(format!("    cvttss2siq {}, {}", src, dst_name));
            }
            (MirType::F32, MirType::F64) => {
                self.emit(format!("    cvtss2sd {}, {}", src, dst_name));
            }
            (MirType::F64, MirType::F32) => {
                self.emit(format!("    cvtsd2ss {}, {}", src, dst_name));
            }
            _ => {
                if src != dst_name {
                    self.emit(format!("    movq {}, {}", src, dst_name));
                }
            }
        }

        if let Location::Stack(off) = dst_loc {
            let mv = move_mnemonic(to_ty);
            self.emit(format!("    {} {}, {}(%rbp)", mv, dst_name, off));
        }
    }

    // -- GEP (pointer arithmetic) -----------------------------------------

    fn sel_gep(&mut self, dst: MirValue, base: MirValue, indices: &[MirValue]) {
        let base_r = self.ensure_reg(base, &MirType::Ptr);
        let dst_loc = self.allocator.allocate(dst, &MirType::Ptr);
        let dst_name = match dst_loc {
            Location::Reg(r) => r.name().to_owned(),
            _ => "%rax".to_owned(),
        };

        self.emit(format!("    movq {}, {}", base_r, dst_name));

        for idx in indices {
            let idx_r = self.ensure_reg(*idx, &MirType::I64);
            self.emit(format!("    leaq ({}, {}, 8), {}", dst_name, idx_r, dst_name));
        }

        if let Location::Stack(off) = dst_loc {
            self.emit(format!("    movq {}, {}(%rbp)", dst_name, off));
        }
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn is_float_or_vec(ty: &MirType) -> bool {
    matches!(
        ty,
        MirType::F32 | MirType::F64 | MirType::Vec4F | MirType::Vec8F | MirType::Vec16F
    )
}

fn move_mnemonic(ty: &MirType) -> &'static str {
    match ty {
        MirType::F32 => "movss",
        MirType::F64 => "movsd",
        MirType::Vec4F | MirType::Vec8F | MirType::Vec16F => "vmovaps",
        _ => "movq",
    }
}

// ---------------------------------------------------------------------------
// AsmEmitter -- assembles a complete function
// ---------------------------------------------------------------------------

/// Emits properly framed x86-64 AT&T-syntax assembly for a [`MirFunction`].
#[derive(Debug)]
pub struct AsmEmitter {
    func_name: String,
}

impl AsmEmitter {
    pub fn new(func_name: &str) -> Self {
        Self {
            func_name: func_name.to_owned(),
        }
    }

    /// Emit complete assembly text for `func`.
    pub fn emit_function(&mut self, func: &MirFunction) -> String {
        let mut selector = InstructionSelector::new(&func.name);
        selector.setup_params(&func.params);

        let mut block_asm: Vec<(BlockId, Vec<String>)> = Vec::new();
        for block in &func.blocks {
            let lines = selector.select_block(block);
            block_asm.push((block.id, lines));
        }

        let alloc = selector.into_allocator();
        let callee_saved = &alloc.used_callee_saved;

        // Frame size must be 16-byte aligned (after the callee-saved pushes
        // + the pushq %rbp already on the stack).
        let raw_frame = alloc.frame_size();
        let pushes = callee_saved.len() as i32 + 1; // +1 for %rbp
        let total = raw_frame + pushes * 8;
        let aligned = (total + 15) & !15;
        let frame_size = aligned - pushes * 8;

        let mut out = String::new();

        // -- prologue --
        out.push_str(&format!("{}:\n", self.func_name));
        out.push_str("    pushq %rbp\n");
        out.push_str("    movq %rsp, %rbp\n");

        for &r in callee_saved {
            out.push_str(&format!("    pushq {}\n", r.name()));
        }

        if frame_size > 0 {
            out.push_str(&format!("    subq ${}, %rsp\n", frame_size));
        }

        // -- body --
        for (bid, lines) in &block_asm {
            let label = format!(".LBB_{}_{}", func.name, bid.0);
            out.push_str(&format!("{}:\n", label));
            for line in lines {
                out.push_str(line);
                out.push('\n');
            }
        }

        // -- epilogue --
        out.push_str(&format!(".LBB_{}_epilogue:\n", func.name));
        if frame_size > 0 {
            out.push_str(&format!("    addq ${}, %rsp\n", frame_size));
        }
        for &r in callee_saved.iter().rev() {
            out.push_str(&format!("    popq {}\n", r.name()));
        }
        out.push_str("    popq %rbp\n");
        out.push_str("    retq\n");

        out
    }
}

// ---------------------------------------------------------------------------
// CodeGenerator -- top-level entry point
// ---------------------------------------------------------------------------

/// Main entry point: translates a full [`MirProgram`] to assembly text.
#[derive(Debug)]
pub struct CodeGenerator {
    target: Target,
}

impl CodeGenerator {
    pub fn new(target: Target) -> Self {
        Self { target }
    }

    /// Generate complete assembly text for the whole program.
    pub fn generate(&mut self, program: &MirProgram) -> Result<String, CodegenError> {
        if self.target != Target::X86_64 {
            return Err(CodegenError::new(format!(
                "unsupported target: {:?}",
                self.target
            )));
        }

        let mut asm = String::new();
        asm.push_str("    .text\n");

        for func in &program.functions {
            asm.push_str(&format!("    .globl {}\n", func.name));
        }
        asm.push('\n');

        for func in &program.functions {
            let func_asm = self.generate_function(func)?;
            asm.push_str(&func_asm);
            asm.push('\n');
        }

        Ok(asm)
    }

    /// Generate assembly text for a single function.
    pub fn generate_function(
        &mut self,
        func: &MirFunction,
    ) -> Result<String, CodegenError> {
        if self.target != Target::X86_64 {
            return Err(CodegenError::new(format!(
                "unsupported target: {:?}",
                self.target
            )));
        }

        let mut emitter = AsmEmitter::new(&func.name);
        Ok(emitter.emit_function(func))
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{BlockId, MirBlock, MirConst, MirFunction, MirInst, MirProgram, MirType, MirValue};

    fn make_func(name: &str, params: Vec<(MirValue, MirType)>, ret: MirType, insts: Vec<MirInst>) -> MirFunction {
        let block = MirBlock {
            id: BlockId(0),
            label: "entry".to_owned(),
            instructions: insts,
            predecessors: vec![],
            successors: vec![],
        };
        MirFunction {
            name: name.to_owned(),
            params,
            return_type: ret,
            blocks: vec![block],
            entry_block: BlockId(0),
            next_value: 100,
        }
    }

    // -- RegAllocator tests -----------------------------------------------

    #[test]
    fn reg_alloc_gpr_basic() {
        let mut alloc = RegAllocator::new();
        let loc = alloc.allocate(MirValue(0), &MirType::I64);
        assert!(matches!(loc, Location::Reg(_)));
        let loc2 = alloc.allocate(MirValue(0), &MirType::I64);
        assert_eq!(loc, loc2);
    }

    #[test]
    fn reg_alloc_xmm_for_float() {
        let mut alloc = RegAllocator::new();
        let loc = alloc.allocate(MirValue(0), &MirType::F64);
        match loc {
            Location::Reg(r) => assert!(r.is_xmm()),
            _ => panic!("expected XMM register for F64"),
        }
    }

    #[test]
    fn reg_alloc_spill_to_stack() {
        let mut alloc = RegAllocator::new();
        for i in 0..GPR_POOL.len() {
            let loc = alloc.allocate(MirValue(i), &MirType::I64);
            assert!(matches!(loc, Location::Reg(_)), "value {} should get a reg", i);
        }
        let spilled = alloc.allocate(MirValue(999), &MirType::I64);
        assert!(matches!(spilled, Location::Stack(_)), "should spill to stack");
    }

    #[test]
    fn reg_alloc_free_and_reuse() {
        let mut alloc = RegAllocator::new();
        let loc1 = alloc.allocate(MirValue(0), &MirType::I64);
        alloc.free(MirValue(0));
        let loc2 = alloc.allocate(MirValue(1), &MirType::I64);
        assert_eq!(loc1, loc2);
    }

    #[test]
    fn reg_alloc_callee_saved_tracking() {
        let mut alloc = RegAllocator::new();
        for i in 0..GPR_POOL.len() {
            alloc.allocate(MirValue(i), &MirType::I64);
        }
        assert!(!alloc.used_callee_saved.is_empty());
    }

    // -- InstructionSelector tests ----------------------------------------

    #[test]
    fn isel_const_int() {
        let mut sel = InstructionSelector::new("test");
        sel.select_inst(&MirInst::Const {
            dst: MirValue(0),
            ty: MirType::I64,
            value: MirConst::Int(42),
        });
        let asm = sel.lines.join("\n");
        assert!(asm.contains("$42"), "should contain immediate 42: {}", asm);
        assert!(asm.contains("mov"), "should contain a mov: {}", asm);
    }

    #[test]
    fn isel_const_bool() {
        let mut sel = InstructionSelector::new("test");
        sel.select_inst(&MirInst::Const {
            dst: MirValue(0),
            ty: MirType::Bool,
            value: MirConst::Bool(true),
        });
        let asm = sel.lines.join("\n");
        assert!(asm.contains("$1"), "true should become $1: {}", asm);
    }

    #[test]
    fn isel_const_zero() {
        let mut sel = InstructionSelector::new("test");
        sel.select_inst(&MirInst::Const {
            dst: MirValue(0),
            ty: MirType::I64,
            value: MirConst::Zero,
        });
        let asm = sel.lines.join("\n");
        assert!(asm.contains("xorq"), "zero const should use xorq: {}", asm);
    }

    #[test]
    fn isel_add_i64() {
        let mut sel = InstructionSelector::new("test");
        sel.allocator_mut().allocate(MirValue(0), &MirType::I64);
        sel.allocator_mut().allocate(MirValue(1), &MirType::I64);
        sel.select_inst(&MirInst::Add {
            dst: MirValue(2),
            ty: MirType::I64,
            lhs: MirValue(0),
            rhs: MirValue(1),
        });
        let asm = sel.lines.join("\n");
        assert!(asm.contains("addq"), "should contain addq: {}", asm);
    }

    #[test]
    fn isel_sub_i64() {
        let mut sel = InstructionSelector::new("test");
        sel.allocator_mut().allocate(MirValue(0), &MirType::I64);
        sel.allocator_mut().allocate(MirValue(1), &MirType::I64);
        sel.select_inst(&MirInst::Sub {
            dst: MirValue(2),
            ty: MirType::I64,
            lhs: MirValue(0),
            rhs: MirValue(1),
        });
        let asm = sel.lines.join("\n");
        assert!(asm.contains("subq"), "should contain subq: {}", asm);
    }

    #[test]
    fn isel_mul_i64() {
        let mut sel = InstructionSelector::new("test");
        sel.allocator_mut().allocate(MirValue(0), &MirType::I64);
        sel.allocator_mut().allocate(MirValue(1), &MirType::I64);
        sel.select_inst(&MirInst::Mul {
            dst: MirValue(2),
            ty: MirType::I64,
            lhs: MirValue(0),
            rhs: MirValue(1),
        });
        let asm = sel.lines.join("\n");
        assert!(asm.contains("imul"), "should contain imul: {}", asm);
    }

    #[test]
    fn isel_div_i64() {
        let mut sel = InstructionSelector::new("test");
        sel.allocator_mut().allocate(MirValue(0), &MirType::I64);
        sel.allocator_mut().allocate(MirValue(1), &MirType::I64);
        sel.select_inst(&MirInst::Div {
            dst: MirValue(2),
            ty: MirType::I64,
            lhs: MirValue(0),
            rhs: MirValue(1),
        });
        let asm = sel.lines.join("\n");
        assert!(asm.contains("idivq"), "should contain idivq: {}", asm);
        assert!(asm.contains("cqto"), "should sign-extend with cqto: {}", asm);
    }

    #[test]
    fn isel_cmp_eq() {
        let mut sel = InstructionSelector::new("test");
        sel.allocator_mut().allocate(MirValue(0), &MirType::I64);
        sel.allocator_mut().allocate(MirValue(1), &MirType::I64);
        sel.select_inst(&MirInst::Cmp {
            dst: MirValue(2),
            op: CmpOp::Eq,
            ty: MirType::I64,
            lhs: MirValue(0),
            rhs: MirValue(1),
        });
        let asm = sel.lines.join("\n");
        assert!(asm.contains("cmpq"), "should contain cmpq: {}", asm);
        assert!(asm.contains("sete"), "should contain sete: {}", asm);
    }

    #[test]
    fn isel_br() {
        let mut sel = InstructionSelector::new("test");
        sel.allocator_mut().allocate(MirValue(0), &MirType::Bool);
        sel.select_inst(&MirInst::Br {
            cond: MirValue(0),
            then_bb: BlockId(1),
            else_bb: BlockId(2),
        });
        let asm = sel.lines.join("\n");
        assert!(asm.contains("testb"), "should contain testb: {}", asm);
        assert!(asm.contains("jne"), "should contain jne: {}", asm);
        assert!(asm.contains(".LBB_test_1"), "should jump to then: {}", asm);
        assert!(asm.contains(".LBB_test_2"), "should jump to else: {}", asm);
    }

    #[test]
    fn isel_jump() {
        let mut sel = InstructionSelector::new("test");
        sel.select_inst(&MirInst::Jump {
            target: BlockId(3),
        });
        let asm = sel.lines.join("\n");
        assert!(asm.contains("jmp .LBB_test_3"), "should jmp: {}", asm);
    }

    #[test]
    fn isel_ret_value() {
        let mut sel = InstructionSelector::new("test");
        sel.allocator_mut().allocate(MirValue(0), &MirType::I64);
        sel.select_inst(&MirInst::Ret {
            value: Some(MirValue(0)),
        });
        let asm = sel.lines.join("\n");
        assert!(asm.contains("rax") || asm.contains("# ret"), "should handle ret: {}", asm);
    }

    #[test]
    fn isel_call() {
        let mut sel = InstructionSelector::new("test");
        sel.allocator_mut().allocate(MirValue(0), &MirType::I64);
        sel.select_inst(&MirInst::Call {
            dst: Some(MirValue(5)),
            func: "puts".to_owned(),
            args: vec![MirValue(0)],
        });
        let asm = sel.lines.join("\n");
        assert!(asm.contains("callq puts"), "should call puts: {}", asm);
        assert!(asm.contains("%rdi"), "first arg should go to rdi: {}", asm);
    }

    #[test]
    fn isel_load_store() {
        let mut sel = InstructionSelector::new("test");
        sel.allocator_mut().allocate(MirValue(0), &MirType::Ptr);
        sel.select_inst(&MirInst::Load {
            dst: MirValue(1),
            ty: MirType::I64,
            addr: MirValue(0),
        });
        let asm = sel.lines.join("\n");
        assert!(asm.contains("movq"), "load should use movq: {}", asm);
    }

    #[test]
    fn isel_vecop_add() {
        let mut sel = InstructionSelector::new("test");
        sel.allocator_mut().allocate(MirValue(0), &MirType::Vec4F);
        sel.allocator_mut().allocate(MirValue(1), &MirType::Vec4F);
        sel.select_inst(&MirInst::VecOp {
            dst: MirValue(2),
            op: VecOpKind::Add,
            ty: MirType::Vec4F,
            operands: vec![MirValue(0), MirValue(1)],
        });
        let asm = sel.lines.join("\n");
        assert!(asm.contains("vaddps"), "should use vaddps: {}", asm);
    }

    // -- Full function generation tests -----------------------------------

    #[test]
    fn gen_simple_add_function() {
        let func = make_func(
            "add",
            vec![
                (MirValue(0), MirType::I64),
                (MirValue(1), MirType::I64),
            ],
            MirType::I64,
            vec![
                MirInst::Add {
                    dst: MirValue(2),
                    ty: MirType::I64,
                    lhs: MirValue(0),
                    rhs: MirValue(1),
                },
                MirInst::Ret {
                    value: Some(MirValue(2)),
                },
            ],
        );

        let mut gen = CodeGenerator::new(Target::X86_64);
        let asm = gen.generate_function(&func).expect("codegen failed");

        assert!(asm.contains("add:"), "should contain function label");
        assert!(asm.contains("pushq %rbp"), "should have prologue push");
        assert!(asm.contains("movq %rsp, %rbp"), "should set up frame");
        assert!(asm.contains("addq"), "should contain addq");
        assert!(asm.contains("retq"), "should contain retq");
        assert!(asm.contains("popq %rbp"), "should restore rbp");
    }

    #[test]
    fn gen_full_program() {
        let func = make_func(
            "main",
            vec![],
            MirType::I64,
            vec![
                MirInst::Const {
                    dst: MirValue(0),
                    ty: MirType::I64,
                    value: MirConst::Int(0),
                },
                MirInst::Ret {
                    value: Some(MirValue(0)),
                },
            ],
        );

        let program = MirProgram {
            functions: vec![func],
        };

        let mut gen = CodeGenerator::new(Target::X86_64);
        let asm = gen.generate(&program).expect("codegen failed");

        assert!(asm.contains(".text"), "should have .text directive");
        assert!(asm.contains(".globl main"), "should have .globl main");
        assert!(asm.contains("main:"), "should have main label");
    }

    #[test]
    fn gen_unsupported_target_errors() {
        let program = MirProgram {
            functions: vec![],
        };

        let mut gen = CodeGenerator::new(Target::Aarch64);
        let result = gen.generate(&program);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .message
            .contains("unsupported target"));
    }

    #[test]
    fn gen_const_loading() {
        let func = make_func(
            "constants",
            vec![],
            MirType::I64,
            vec![
                MirInst::Const {
                    dst: MirValue(0),
                    ty: MirType::I64,
                    value: MirConst::Int(100),
                },
                MirInst::Const {
                    dst: MirValue(1),
                    ty: MirType::Bool,
                    value: MirConst::Bool(false),
                },
                MirInst::Const {
                    dst: MirValue(2),
                    ty: MirType::I64,
                    value: MirConst::Zero,
                },
                MirInst::Ret {
                    value: Some(MirValue(0)),
                },
            ],
        );

        let mut gen = CodeGenerator::new(Target::X86_64);
        let asm = gen.generate_function(&func).expect("codegen failed");

        assert!(asm.contains("$100"), "should load 100");
        assert!(asm.contains("$0"), "should load false as 0");
        assert!(asm.contains("xorq"), "should zero with xorq");
    }

    #[test]
    fn gen_function_with_branch() {
        let blocks = vec![
            MirBlock {
                id: BlockId(0),
                label: "entry".to_owned(),
                instructions: vec![
                    MirInst::Const {
                        dst: MirValue(0),
                        ty: MirType::Bool,
                        value: MirConst::Bool(true),
                    },
                    MirInst::Br {
                        cond: MirValue(0),
                        then_bb: BlockId(1),
                        else_bb: BlockId(2),
                    },
                ],
                predecessors: vec![],
                successors: vec![BlockId(1), BlockId(2)],
            },
            MirBlock {
                id: BlockId(1),
                label: "then".to_owned(),
                instructions: vec![
                    MirInst::Const {
                        dst: MirValue(1),
                        ty: MirType::I64,
                        value: MirConst::Int(1),
                    },
                    MirInst::Ret {
                        value: Some(MirValue(1)),
                    },
                ],
                predecessors: vec![BlockId(0)],
                successors: vec![],
            },
            MirBlock {
                id: BlockId(2),
                label: "else".to_owned(),
                instructions: vec![
                    MirInst::Const {
                        dst: MirValue(2),
                        ty: MirType::I64,
                        value: MirConst::Int(0),
                    },
                    MirInst::Ret {
                        value: Some(MirValue(2)),
                    },
                ],
                predecessors: vec![BlockId(0)],
                successors: vec![],
            },
        ];

        let func = MirFunction {
            name: "branch_test".to_owned(),
            params: vec![],
            return_type: MirType::I64,
            blocks,
            entry_block: BlockId(0),
            next_value: 10,
        };

        let mut gen = CodeGenerator::new(Target::X86_64);
        let asm = gen.generate_function(&func).expect("codegen failed");

        assert!(asm.contains("testb"), "should test condition");
        assert!(asm.contains("jne .LBB_branch_test_1"), "should branch to then");
        assert!(asm.contains("jmp .LBB_branch_test_2"), "should branch to else");
        assert!(asm.contains(".LBB_branch_test_1:"), "should have then label");
        assert!(asm.contains(".LBB_branch_test_2:"), "should have else label");
    }

    #[test]
    fn location_display() {
        assert_eq!(Location::Reg(Register::Rax).to_string(), "%rax");
        assert_eq!(Location::Stack(-16).to_string(), "-16(%rbp)");
        assert_eq!(Location::Immediate(42).to_string(), "$42");
    }

    #[test]
    fn register_display() {
        assert_eq!(Register::R13.to_string(), "%r13");
        assert_eq!(Register::Xmm5.to_string(), "%xmm5");
    }

    #[test]
    fn codegen_error_display() {
        let err = CodegenError::new("bad thing");
        assert_eq!(err.to_string(), "codegen error: bad thing");
    }
}
