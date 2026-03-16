//! High-Level Intermediate Representation (HIR).
//!
//! The HIR is a semantic, graph-style representation that sits between the
//! parser AST and the e-graph optimizer.  Every operation is assigned a unique
//! [`HirId`] so that later passes can reference nodes cheaply.

use std::collections::HashMap;
use std::fmt;

use crate::lexer::Span;
use crate::parser::{
    self, Ast, BinOp, ExprKind, ItemKind, StmtKind, UnaryOp, AssignOp,
};
use crate::types::{AxiomType, FloatKind, IntegerKind, TypeContext, TypeId};

// ---------------------------------------------------------------------------
// HirId
// ---------------------------------------------------------------------------

/// Unique identifier for a node in the HIR graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HirId(pub usize);

impl fmt::Display for HirId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// HirOp – every operation the HIR can express
// ---------------------------------------------------------------------------

/// An HIR operation.
#[derive(Debug, Clone, PartialEq)]
pub enum HirOp {
    // -- arithmetic ---------------------------------------------------------
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Neg,

    // -- comparison ---------------------------------------------------------
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,

    // -- logical ------------------------------------------------------------
    And,
    Or,
    Not,

    // -- bitwise ------------------------------------------------------------
    BitAnd,
    BitOr,
    BitXor,
    BitNot,
    Shl,
    Shr,

    // -- memory -------------------------------------------------------------
    /// Load from a memory location.
    Load,
    /// Store a value to a memory location.
    Store,
    /// Heap allocation.
    Alloc,
    /// Stack allocation.
    StackAlloc,

    // -- control flow -------------------------------------------------------
    /// Conditional branch: operands are [condition, then_target, else_target].
    Branch,
    /// Unconditional jump to a target node.
    Jump,
    /// Loop header — operands are [condition, body_start].
    Loop,
    /// Return from the current function; operand (if any) is the value.
    Return,

    // -- functions ----------------------------------------------------------
    /// Function call: first operand is the callee, rest are arguments.
    Call { name: String },
    /// Reference to the n-th parameter of the enclosing function.
    Param { index: usize },

    // -- constants ----------------------------------------------------------
    IntConst(i64),
    FloatConst(f64),
    BoolConst(bool),
    StringConst(String),

    // -- variables ----------------------------------------------------------
    /// Variable definition — binds a name to the single operand value.
    VarDef { name: String, mutable: bool },
    /// Reference to a previously defined variable.
    VarRef { name: String },

    // -- aggregate / composite ----------------------------------------------
    /// Array/slice index: operands are [base, index].
    Index,
    /// Slice: operands are [base, start, end].
    Slice,
    /// Field access: operand is the struct value.
    FieldAccess { field: String },
    /// Struct initialisation: operands are the field values in declaration
    /// order.
    StructInit { name: String },
    /// Array literal: operands are the element values.
    ArrayLiteral,

    // -- misc ---------------------------------------------------------------
    /// A sequence of nodes (block / scope).
    Block,
    /// No-op placeholder (e.g. `break` / `continue` markers).
    Nop,
    /// Break out of the nearest loop.
    Break,
    /// Continue to the next iteration of the nearest loop.
    Continue,
}

// ---------------------------------------------------------------------------
// HirNode
// ---------------------------------------------------------------------------

/// A single node in the HIR graph.
#[derive(Debug, Clone)]
pub struct HirNode {
    pub id: HirId,
    pub op: HirOp,
    /// Operand references (other HirIds this node consumes).
    pub operands: Vec<HirId>,
    /// Resolved type of this node (`None` until type-checking).
    pub ty: Option<TypeId>,
    /// Source location, carried through from the AST for diagnostics.
    pub span: Option<Span>,
}

// ---------------------------------------------------------------------------
// HirFunction / HirProgram
// ---------------------------------------------------------------------------

/// A function in HIR form.
#[derive(Debug, Clone)]
pub struct HirFunction {
    pub name: String,
    pub params: Vec<HirParam>,
    pub body: Vec<HirNode>,
    pub return_type: Option<TypeId>,
    pub is_pub: bool,
    pub span: Option<Span>,
}

/// A function parameter with its resolved type.
#[derive(Debug, Clone)]
pub struct HirParam {
    pub name: String,
    pub ty: TypeId,
}

/// A struct definition preserved in the HIR for later passes.
#[derive(Debug, Clone)]
pub struct HirStructDef {
    pub name: String,
    pub fields: Vec<(String, TypeId)>,
    pub span: Option<Span>,
}

/// An enum definition preserved in the HIR.
#[derive(Debug, Clone)]
pub struct HirEnumDef {
    pub name: String,
    pub variants: Vec<(String, Vec<TypeId>)>,
    pub span: Option<Span>,
}

/// The complete program in HIR form.
#[derive(Debug, Clone)]
pub struct HirProgram {
    pub functions: Vec<HirFunction>,
    pub structs: Vec<HirStructDef>,
    pub enums: Vec<HirEnumDef>,
    pub type_context: TypeContext,
}

// ---------------------------------------------------------------------------
// HirError
// ---------------------------------------------------------------------------

/// Error produced during AST → HIR lowering.
#[derive(Debug, Clone)]
pub struct HirError {
    pub message: String,
    pub span: Option<Span>,
}

impl fmt::Display for HirError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.span {
            Some(sp) => write!(f, "[{}:{}] {}", sp.line, sp.col, self.message),
            None => write!(f, "{}", self.message),
        }
    }
}

impl std::error::Error for HirError {}

impl HirError {
    fn new(message: impl Into<String>, span: Option<Span>) -> Self {
        Self { message: message.into(), span }
    }
}

// ---------------------------------------------------------------------------
// AstLowering – converts parser AST into HIR
// ---------------------------------------------------------------------------

/// Converts a parsed [`Ast`] into a [`HirProgram`].
pub struct AstLowering {
    next_id: usize,
    type_ctx: TypeContext,
    /// Stack of scopes, each mapping variable name → HirId of its definition.
    scopes: Vec<HashMap<String, HirId>>,
}

impl AstLowering {
    pub fn new() -> Self {
        Self {
            next_id: 0,
            type_ctx: TypeContext::new(),
            scopes: vec![HashMap::new()],
        }
    }

    /// Lower an entire AST into a HIR program.
    pub fn lower(ast: &Ast) -> Result<HirProgram, HirError> {
        let mut lowering = AstLowering::new();

        let mut functions = Vec::new();
        let mut structs = Vec::new();
        let mut enums = Vec::new();

        for item in &ast.items {
            match &item.kind {
                ItemKind::Function {
                    name,
                    params,
                    return_type,
                    body,
                    is_pub,
                    generics: _,
                    annotations: _,
                } => {
                    let func =
                        lowering.lower_function(name, params, return_type, body, *is_pub, item.span)?;
                    functions.push(func);
                }
                ItemKind::Struct {
                    name,
                    fields,
                    is_pub: _,
                    generics: _,
                } => {
                    let def = lowering.lower_struct_def(name, fields, item.span)?;
                    structs.push(def);
                }
                ItemKind::Enum {
                    name,
                    variants,
                    is_pub: _,
                    generics: _,
                } => {
                    let def = lowering.lower_enum_def(name, variants, item.span)?;
                    enums.push(def);
                }
                ItemKind::Import { .. } => {
                    // Imports are resolved before HIR; skip.
                }
            }
        }

        Ok(HirProgram {
            functions,
            structs,
            enums,
            type_context: lowering.type_ctx,
        })
    }

    // -- id allocation ------------------------------------------------------

    fn alloc_id(&mut self) -> HirId {
        let id = HirId(self.next_id);
        self.next_id += 1;
        id
    }

    fn make_node(
        &mut self,
        op: HirOp,
        operands: Vec<HirId>,
        span: Option<Span>,
    ) -> HirNode {
        let id = self.alloc_id();
        HirNode { id, op, operands, ty: None, span }
    }

    // -- scope management ---------------------------------------------------

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn define_var(&mut self, name: &str, id: HirId) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), id);
        }
    }

    fn lookup_var(&self, name: &str) -> Option<HirId> {
        for scope in self.scopes.iter().rev() {
            if let Some(&id) = scope.get(name) {
                return Some(id);
            }
        }
        None
    }

    // -- type lowering (syntactic → semantic) -------------------------------

    fn lower_type(&mut self, ty: &parser::Type) -> TypeId {
        match ty {
            parser::Type::Named(name) => self.lower_named_type(name),
            parser::Type::Array { size, elem } => {
                let elem_id = self.lower_type(elem);
                match size {
                    Some(len) => self.type_ctx.register(AxiomType::Array {
                        element: elem_id,
                        length: *len,
                    }),
                    None => self.type_ctx.register(AxiomType::Slice {
                        element: elem_id,
                    }),
                }
            }
            parser::Type::Ref { mutable, inner } => {
                let inner_id = self.lower_type(inner);
                self.type_ctx.register(AxiomType::Reference {
                    mutable: *mutable,
                    inner: inner_id,
                })
            }
            parser::Type::Generic { name, args } => {
                // Register the base type; generics are monomorphised later.
                let _ = args; // preserved for future generic support
                self.lower_named_type(name)
            }
        }
    }

    fn lower_named_type(&mut self, name: &str) -> TypeId {
        match name {
            "i8" => self.type_ctx.register(AxiomType::Integer(IntegerKind::I8)),
            "i16" => self.type_ctx.register(AxiomType::Integer(IntegerKind::I16)),
            "i32" => self.type_ctx.register(AxiomType::Integer(IntegerKind::I32)),
            "i64" => self.type_ctx.register(AxiomType::Integer(IntegerKind::I64)),
            "u8" => self.type_ctx.register(AxiomType::Integer(IntegerKind::U8)),
            "u16" => self.type_ctx.register(AxiomType::Integer(IntegerKind::U16)),
            "u32" => self.type_ctx.register(AxiomType::Integer(IntegerKind::U32)),
            "u64" => self.type_ctx.register(AxiomType::Integer(IntegerKind::U64)),
            "f32" => self.type_ctx.register(AxiomType::Float(FloatKind::F32)),
            "f64" => self.type_ctx.register(AxiomType::Float(FloatKind::F64)),
            "bool" => self.type_ctx.register(AxiomType::Bool),
            "char" => self.type_ctx.register(AxiomType::Char),
            other => {
                // Treat as a struct / enum / generic param by name.
                self.type_ctx.register(AxiomType::GenericParam {
                    name: other.to_string(),
                })
            }
        }
    }

    // -- top-level items ----------------------------------------------------

    fn lower_function(
        &mut self,
        name: &str,
        params: &[parser::Param],
        return_type: &Option<parser::Type>,
        body: &[parser::Stmt],
        is_pub: bool,
        span: Span,
    ) -> Result<HirFunction, HirError> {
        self.push_scope();

        // Lower parameters.
        let mut hir_params = Vec::with_capacity(params.len());
        for (i, param) in params.iter().enumerate() {
            let ty = self.lower_type(&param.ty);
            let node = self.make_node(
                HirOp::Param { index: i },
                vec![],
                Some(param.span),
            );
            self.define_var(&param.name, node.id);
            hir_params.push(HirParam {
                name: param.name.clone(),
                ty,
            });
        }

        // Lower body.
        let mut nodes = Vec::new();
        for stmt in body {
            let lowered = self.lower_stmt(stmt)?;
            nodes.extend(lowered);
        }

        let ret_ty = return_type.as_ref().map(|t| self.lower_type(t));

        self.pop_scope();

        Ok(HirFunction {
            name: name.to_string(),
            params: hir_params,
            body: nodes,
            return_type: ret_ty,
            is_pub,
            span: Some(span),
        })
    }

    fn lower_struct_def(
        &mut self,
        name: &str,
        fields: &[parser::Field],
        span: Span,
    ) -> Result<HirStructDef, HirError> {
        let hir_fields = fields
            .iter()
            .map(|field| {
                let ty = self.lower_type(&field.ty);
                (field.name.clone(), ty)
            })
            .collect();
        Ok(HirStructDef {
            name: name.to_string(),
            fields: hir_fields,
            span: Some(span),
        })
    }

    fn lower_enum_def(
        &mut self,
        name: &str,
        variants: &[parser::Variant],
        span: Span,
    ) -> Result<HirEnumDef, HirError> {
        let hir_variants = variants
            .iter()
            .map(|variant| {
                let field_tys: Vec<TypeId> =
                    variant.fields.iter().map(|t| self.lower_type(t)).collect();
                (variant.name.clone(), field_tys)
            })
            .collect();
        Ok(HirEnumDef {
            name: name.to_string(),
            variants: hir_variants,
            span: Some(span),
        })
    }

    // -- statements ---------------------------------------------------------

    fn lower_stmt(
        &mut self,
        stmt: &parser::Stmt,
    ) -> Result<Vec<HirNode>, HirError> {
        let span = Some(stmt.span);
        match &stmt.kind {
            StmtKind::VarDecl { name, ty, value, mutable } => {
                let (mut nodes, val_id) = self.lower_expr(value)?;
                let var_node = self.make_node(
                    HirOp::VarDef { name: name.clone(), mutable: *mutable },
                    vec![val_id],
                    span,
                );
                let var_id = var_node.id;

                // Attach declared type if present.
                let mut node = var_node;
                if let Some(t) = ty {
                    node.ty = Some(self.lower_type(t));
                }

                self.define_var(name, var_id);
                nodes.push(node);
                Ok(nodes)
            }

            StmtKind::Assignment { target, op, value } => {
                self.lower_assignment(target, op, value, span)
            }

            StmtKind::Return(maybe_expr) => {
                match maybe_expr {
                    Some(expr) => {
                        let (mut nodes, val_id) = self.lower_expr(expr)?;
                        let ret = self.make_node(HirOp::Return, vec![val_id], span);
                        nodes.push(ret);
                        Ok(nodes)
                    }
                    None => {
                        let ret = self.make_node(HirOp::Return, vec![], span);
                        Ok(vec![ret])
                    }
                }
            }

            StmtKind::If { condition, body, elif_branches, else_body } => {
                self.lower_if(condition, body, elif_branches, else_body, span)
            }

            StmtKind::For { var, iter, body } => {
                self.lower_for(var, iter, body, span)
            }

            StmtKind::While { condition, body } => {
                self.lower_while(condition, body, span)
            }

            StmtKind::Break => {
                let node = self.make_node(HirOp::Break, vec![], span);
                Ok(vec![node])
            }

            StmtKind::Continue => {
                let node = self.make_node(HirOp::Continue, vec![], span);
                Ok(vec![node])
            }

            StmtKind::ExprStmt(expr) => {
                let (nodes, _last) = self.lower_expr(expr)?;
                Ok(nodes)
            }
        }
    }

    // -- assignment lowering ------------------------------------------------

    fn lower_assignment(
        &mut self,
        target: &parser::Expr,
        op: &AssignOp,
        value: &parser::Expr,
        span: Option<Span>,
    ) -> Result<Vec<HirNode>, HirError> {
        let (mut nodes, val_id) = self.lower_expr(value)?;

        // For compound assignment (+=, etc.) synthesise the binary op first.
        let effective_val = match op {
            AssignOp::Assign => val_id,
            compound => {
                let (mut target_nodes, target_id) = self.lower_expr(target)?;
                nodes.append(&mut target_nodes);
                let bin_op = match compound {
                    AssignOp::AddAssign => HirOp::Add,
                    AssignOp::SubAssign => HirOp::Sub,
                    AssignOp::MulAssign => HirOp::Mul,
                    AssignOp::DivAssign => HirOp::Div,
                    AssignOp::Assign => unreachable!(),
                };
                let bin = self.make_node(bin_op, vec![target_id, val_id], span);
                let id = bin.id;
                nodes.push(bin);
                id
            }
        };

        // Emit a Store to the target location.
        let (mut target_nodes, target_id) = self.lower_expr(target)?;
        nodes.append(&mut target_nodes);
        let store = self.make_node(HirOp::Store, vec![target_id, effective_val], span);
        nodes.push(store);
        Ok(nodes)
    }

    // -- control-flow lowering ----------------------------------------------

    fn lower_if(
        &mut self,
        condition: &parser::Expr,
        body: &[parser::Stmt],
        elif_branches: &[(parser::Expr, Vec<parser::Stmt>)],
        else_body: &Option<Vec<parser::Stmt>>,
        span: Option<Span>,
    ) -> Result<Vec<HirNode>, HirError> {
        let (mut nodes, cond_id) = self.lower_expr(condition)?;

        // Lower then-body into a Block node.
        self.push_scope();
        let mut then_nodes = Vec::new();
        for s in body {
            then_nodes.extend(self.lower_stmt(s)?);
        }
        self.pop_scope();

        let then_ids: Vec<HirId> = then_nodes.iter().map(|n| n.id).collect();
        nodes.extend(then_nodes);
        let then_block = self.make_node(HirOp::Block, then_ids, span);
        let then_id = then_block.id;
        nodes.push(then_block);

        // Lower elif chains recursively as nested Branch nodes.
        let else_id = self.lower_elif_chain(elif_branches, else_body, span, &mut nodes)?;

        let branch = self.make_node(
            HirOp::Branch,
            match else_id {
                Some(eid) => vec![cond_id, then_id, eid],
                None => vec![cond_id, then_id],
            },
            span,
        );
        nodes.push(branch);
        Ok(nodes)
    }

    fn lower_elif_chain(
        &mut self,
        elifs: &[(parser::Expr, Vec<parser::Stmt>)],
        else_body: &Option<Vec<parser::Stmt>>,
        span: Option<Span>,
        nodes: &mut Vec<HirNode>,
    ) -> Result<Option<HirId>, HirError> {
        if elifs.is_empty() {
            // Terminal else block (if any).
            if let Some(stmts) = else_body {
                self.push_scope();
                let mut else_nodes = Vec::new();
                for s in stmts {
                    else_nodes.extend(self.lower_stmt(s)?);
                }
                self.pop_scope();

                let else_ids: Vec<HirId> = else_nodes.iter().map(|n| n.id).collect();
                nodes.extend(else_nodes);
                let block = self.make_node(HirOp::Block, else_ids, span);
                let id = block.id;
                nodes.push(block);
                return Ok(Some(id));
            }
            return Ok(None);
        }

        let (cond_expr, body) = &elifs[0];
        let (mut cond_nodes, cond_id) = self.lower_expr(cond_expr)?;
        nodes.append(&mut cond_nodes);

        self.push_scope();
        let mut then_nodes = Vec::new();
        for s in body {
            then_nodes.extend(self.lower_stmt(s)?);
        }
        self.pop_scope();

        let then_ids: Vec<HirId> = then_nodes.iter().map(|n| n.id).collect();
        nodes.extend(then_nodes);
        let then_block = self.make_node(HirOp::Block, then_ids, span);
        let then_id = then_block.id;
        nodes.push(then_block);

        let else_id =
            self.lower_elif_chain(&elifs[1..], else_body, span, nodes)?;

        let branch = self.make_node(
            HirOp::Branch,
            match else_id {
                Some(eid) => vec![cond_id, then_id, eid],
                None => vec![cond_id, then_id],
            },
            span,
        );
        let branch_id = branch.id;
        nodes.push(branch);
        Ok(Some(branch_id))
    }

    fn lower_while(
        &mut self,
        condition: &parser::Expr,
        body: &[parser::Stmt],
        span: Option<Span>,
    ) -> Result<Vec<HirNode>, HirError> {
        let (mut nodes, cond_id) = self.lower_expr(condition)?;

        self.push_scope();
        let mut body_nodes = Vec::new();
        for s in body {
            body_nodes.extend(self.lower_stmt(s)?);
        }
        self.pop_scope();

        let body_ids: Vec<HirId> = body_nodes.iter().map(|n| n.id).collect();
        nodes.extend(body_nodes);
        let body_block = self.make_node(HirOp::Block, body_ids, span);
        let body_id = body_block.id;
        nodes.push(body_block);

        let loop_node = self.make_node(HirOp::Loop, vec![cond_id, body_id], span);
        nodes.push(loop_node);
        Ok(nodes)
    }

    fn lower_for(
        &mut self,
        var: &str,
        iter: &parser::Expr,
        body: &[parser::Stmt],
        span: Option<Span>,
    ) -> Result<Vec<HirNode>, HirError> {
        // Desugar `for var in iter` into a Loop with an implicit iterator.
        let (mut nodes, iter_id) = self.lower_expr(iter)?;

        self.push_scope();

        // Bind the loop variable.
        let var_node = self.make_node(
            HirOp::VarDef { name: var.to_string(), mutable: true },
            vec![iter_id],
            span,
        );
        let var_id = var_node.id;
        nodes.push(var_node);
        self.define_var(var, var_id);

        let mut body_nodes = Vec::new();
        for s in body {
            body_nodes.extend(self.lower_stmt(s)?);
        }
        self.pop_scope();

        let body_ids: Vec<HirId> = body_nodes.iter().map(|n| n.id).collect();
        nodes.extend(body_nodes);
        let body_block = self.make_node(HirOp::Block, body_ids, span);
        let body_id = body_block.id;
        nodes.push(body_block);

        // The condition is implicit (iterator exhaustion); use the iter id.
        let loop_node = self.make_node(HirOp::Loop, vec![iter_id, body_id], span);
        nodes.push(loop_node);
        Ok(nodes)
    }

    // -- expression lowering ------------------------------------------------

    /// Lower an expression, returning the nodes produced and the [`HirId`] of
    /// the result value.
    fn lower_expr(
        &mut self,
        expr: &parser::Expr,
    ) -> Result<(Vec<HirNode>, HirId), HirError> {
        let span = Some(expr.span);
        match &expr.kind {
            // -- literals ---------------------------------------------------
            ExprKind::IntLiteral(v) => {
                let node = self.make_node(HirOp::IntConst(*v), vec![], span);
                let id = node.id;
                Ok((vec![node], id))
            }
            ExprKind::FloatLiteral(v) => {
                let node = self.make_node(HirOp::FloatConst(*v), vec![], span);
                let id = node.id;
                Ok((vec![node], id))
            }
            ExprKind::BoolLiteral(v) => {
                let node = self.make_node(HirOp::BoolConst(*v), vec![], span);
                let id = node.id;
                Ok((vec![node], id))
            }
            ExprKind::StringLiteral(v) => {
                let node =
                    self.make_node(HirOp::StringConst(v.clone()), vec![], span);
                let id = node.id;
                Ok((vec![node], id))
            }
            ExprKind::CharLiteral(c) => {
                let node =
                    self.make_node(HirOp::IntConst(*c as i64), vec![], span);
                let id = node.id;
                Ok((vec![node], id))
            }

            // -- identifiers ------------------------------------------------
            ExprKind::Identifier(name) => {
                let node = self.make_node(
                    HirOp::VarRef { name: name.clone() },
                    vec![],
                    span,
                );
                let id = node.id;
                Ok((vec![node], id))
            }

            // -- binary ops -------------------------------------------------
            ExprKind::BinaryOp { op, lhs, rhs } => {
                let (mut nodes, lhs_id) = self.lower_expr(lhs)?;
                let (rhs_nodes, rhs_id) = self.lower_expr(rhs)?;
                nodes.extend(rhs_nodes);

                let hir_op = match op {
                    BinOp::Add => HirOp::Add,
                    BinOp::Sub => HirOp::Sub,
                    BinOp::Mul => HirOp::Mul,
                    BinOp::Div => HirOp::Div,
                    BinOp::Mod => HirOp::Mod,
                    BinOp::Eq => HirOp::Eq,
                    BinOp::NotEq => HirOp::Ne,
                    BinOp::Lt => HirOp::Lt,
                    BinOp::Gt => HirOp::Gt,
                    BinOp::LtEq => HirOp::Le,
                    BinOp::GtEq => HirOp::Ge,
                    BinOp::And => HirOp::And,
                    BinOp::Or => HirOp::Or,
                    BinOp::BitAnd => HirOp::BitAnd,
                    BinOp::BitOr => HirOp::BitOr,
                    BinOp::BitXor => HirOp::BitXor,
                    BinOp::Shl => HirOp::Shl,
                    BinOp::Shr => HirOp::Shr,
                };

                let node = self.make_node(hir_op, vec![lhs_id, rhs_id], span);
                let id = node.id;
                nodes.push(node);
                Ok((nodes, id))
            }

            // -- unary ops --------------------------------------------------
            ExprKind::UnaryOp { op, operand } => {
                let (mut nodes, operand_id) = self.lower_expr(operand)?;
                let hir_op = match op {
                    UnaryOp::Neg => HirOp::Neg,
                    UnaryOp::Not => HirOp::Not,
                    UnaryOp::BitNot => HirOp::BitNot,
                };
                let node = self.make_node(hir_op, vec![operand_id], span);
                let id = node.id;
                nodes.push(node);
                Ok((nodes, id))
            }

            // -- function calls ---------------------------------------------
            ExprKind::Call { callee, args } => {
                let name = match &callee.kind {
                    ExprKind::Identifier(n) => n.clone(),
                    _ => "<indirect>".to_string(),
                };

                let mut nodes = Vec::new();
                let mut arg_ids = Vec::new();
                for arg in args {
                    let (arg_nodes, arg_id) = self.lower_expr(arg)?;
                    nodes.extend(arg_nodes);
                    arg_ids.push(arg_id);
                }

                let call = self.make_node(
                    HirOp::Call { name },
                    arg_ids,
                    span,
                );
                let id = call.id;
                nodes.push(call);
                Ok((nodes, id))
            }

            // -- method calls (desugar to Call) -----------------------------
            ExprKind::MethodCall { object, method, args } => {
                let (mut nodes, obj_id) = self.lower_expr(object)?;
                let mut arg_ids = vec![obj_id];
                for arg in args {
                    let (arg_nodes, arg_id) = self.lower_expr(arg)?;
                    nodes.extend(arg_nodes);
                    arg_ids.push(arg_id);
                }
                let call = self.make_node(
                    HirOp::Call { name: method.clone() },
                    arg_ids,
                    span,
                );
                let id = call.id;
                nodes.push(call);
                Ok((nodes, id))
            }

            // -- field access -----------------------------------------------
            ExprKind::FieldAccess { object, field } => {
                let (mut nodes, obj_id) = self.lower_expr(object)?;
                let node = self.make_node(
                    HirOp::FieldAccess { field: field.clone() },
                    vec![obj_id],
                    span,
                );
                let id = node.id;
                nodes.push(node);
                Ok((nodes, id))
            }

            // -- indexing ---------------------------------------------------
            ExprKind::Index { object, index } => {
                let (mut nodes, obj_id) = self.lower_expr(object)?;
                let (idx_nodes, idx_id) = self.lower_expr(index)?;
                nodes.extend(idx_nodes);
                let node = self.make_node(HirOp::Index, vec![obj_id, idx_id], span);
                let id = node.id;
                nodes.push(node);
                Ok((nodes, id))
            }

            // -- ranges (lower as a Call to a built-in) ---------------------
            ExprKind::Range { start, end } => {
                let (mut nodes, start_id) = self.lower_expr(start)?;
                let (end_nodes, end_id) = self.lower_expr(end)?;
                nodes.extend(end_nodes);
                let node = self.make_node(HirOp::Slice, vec![start_id, end_id], span);
                let id = node.id;
                nodes.push(node);
                Ok((nodes, id))
            }

            // -- new expression (heap alloc + struct init) ------------------
            ExprKind::New { ty, args } => {
                let mut nodes = Vec::new();
                let mut arg_ids = Vec::new();
                for arg in args {
                    let (arg_nodes, arg_id) = self.lower_expr(arg)?;
                    nodes.extend(arg_nodes);
                    arg_ids.push(arg_id);
                }
                let type_name = match ty {
                    parser::Type::Named(n) => n.clone(),
                    parser::Type::Generic { name, .. } => name.clone(),
                    _ => "anon".to_string(),
                };
                let init = self.make_node(
                    HirOp::StructInit { name: type_name },
                    arg_ids.clone(),
                    span,
                );
                let init_id = init.id;
                nodes.push(init);

                let alloc = self.make_node(HirOp::Alloc, vec![init_id], span);
                let alloc_id = alloc.id;
                nodes.push(alloc);
                Ok((nodes, alloc_id))
            }

            // -- array literal ----------------------------------------------
            ExprKind::ArrayLiteral(elems) => {
                let mut nodes = Vec::new();
                let mut elem_ids = Vec::new();
                for elem in elems {
                    let (elem_nodes, elem_id) = self.lower_expr(elem)?;
                    nodes.extend(elem_nodes);
                    elem_ids.push(elem_id);
                }
                let arr = self.make_node(HirOp::ArrayLiteral, elem_ids, span);
                let id = arr.id;
                nodes.push(arr);
                Ok((nodes, id))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Span;
    use crate::parser::{Ast, Expr, ExprKind, Item, ItemKind, Param, Stmt, StmtKind};

    fn dummy_span() -> Span {
        Span::new(1, 1)
    }

    fn int_expr(v: i64) -> Expr {
        Expr { kind: ExprKind::IntLiteral(v), span: dummy_span() }
    }

    fn ident_expr(name: &str) -> Expr {
        Expr { kind: ExprKind::Identifier(name.to_string()), span: dummy_span() }
    }

    fn make_fn_item(
        name: &str,
        params: Vec<Param>,
        ret: Option<parser::Type>,
        body: Vec<Stmt>,
    ) -> Item {
        Item {
            kind: ItemKind::Function {
                name: name.to_string(),
                generics: vec![],
                params,
                return_type: ret,
                body,
                annotations: vec![],
                is_pub: false,
            },
            span: dummy_span(),
        }
    }

    #[test]
    fn lower_empty_program() {
        let ast = Ast { items: vec![] };
        let prog = AstLowering::lower(&ast).unwrap();
        assert!(prog.functions.is_empty());
        assert!(prog.structs.is_empty());
        assert!(prog.enums.is_empty());
    }

    #[test]
    fn lower_simple_function() {
        let body = vec![Stmt {
            kind: StmtKind::Return(Some(int_expr(42))),
            span: dummy_span(),
        }];
        let ast = Ast {
            items: vec![make_fn_item(
                "main",
                vec![],
                Some(parser::Type::Named("i32".to_string())),
                body,
            )],
        };
        let prog = AstLowering::lower(&ast).unwrap();
        assert_eq!(prog.functions.len(), 1);
        assert_eq!(prog.functions[0].name, "main");
        assert!(prog.functions[0].return_type.is_some());

        // Body should contain an IntConst and a Return.
        let ops: Vec<_> = prog.functions[0].body.iter().map(|n| &n.op).collect();
        assert!(matches!(ops[0], HirOp::IntConst(42)));
        assert!(matches!(ops[1], HirOp::Return));
    }

    #[test]
    fn lower_binary_op() {
        let add = Expr {
            kind: ExprKind::BinaryOp {
                op: BinOp::Add,
                lhs: Box::new(int_expr(1)),
                rhs: Box::new(int_expr(2)),
            },
            span: dummy_span(),
        };
        let body = vec![Stmt {
            kind: StmtKind::Return(Some(add)),
            span: dummy_span(),
        }];
        let ast = Ast {
            items: vec![make_fn_item("add", vec![], None, body)],
        };
        let prog = AstLowering::lower(&ast).unwrap();
        let ops: Vec<_> = prog.functions[0].body.iter().map(|n| &n.op).collect();
        // IntConst(1), IntConst(2), Add, Return
        assert_eq!(ops.len(), 4);
        assert!(matches!(ops[2], HirOp::Add));
    }

    #[test]
    fn lower_var_decl() {
        let body = vec![Stmt {
            kind: StmtKind::VarDecl {
                name: "x".to_string(),
                ty: Some(parser::Type::Named("i32".to_string())),
                value: int_expr(10),
                mutable: false,
            },
            span: dummy_span(),
        }];
        let ast = Ast {
            items: vec![make_fn_item("f", vec![], None, body)],
        };
        let prog = AstLowering::lower(&ast).unwrap();
        let ops: Vec<_> = prog.functions[0].body.iter().map(|n| &n.op).collect();
        // IntConst(10), VarDef { name: "x" }
        assert_eq!(ops.len(), 2);
        assert!(matches!(ops[1], HirOp::VarDef { ref name, .. } if name == "x"));
        // The VarDef should have a type annotation.
        assert!(prog.functions[0].body[1].ty.is_some());
    }

    #[test]
    fn lower_if_else() {
        let cond = Expr {
            kind: ExprKind::BoolLiteral(true),
            span: dummy_span(),
        };
        let then_body = vec![Stmt {
            kind: StmtKind::Return(Some(int_expr(1))),
            span: dummy_span(),
        }];
        let else_body = vec![Stmt {
            kind: StmtKind::Return(Some(int_expr(2))),
            span: dummy_span(),
        }];
        let body = vec![Stmt {
            kind: StmtKind::If {
                condition: cond,
                body: then_body,
                elif_branches: vec![],
                else_body: Some(else_body),
            },
            span: dummy_span(),
        }];
        let ast = Ast {
            items: vec![make_fn_item("f", vec![], None, body)],
        };
        let prog = AstLowering::lower(&ast).unwrap();
        // Should contain at least a BoolConst, some blocks, and a Branch.
        let has_branch = prog.functions[0]
            .body
            .iter()
            .any(|n| matches!(n.op, HirOp::Branch));
        assert!(has_branch);
    }

    #[test]
    fn lower_while_loop() {
        let cond = Expr {
            kind: ExprKind::BoolLiteral(true),
            span: dummy_span(),
        };
        let body = vec![Stmt {
            kind: StmtKind::While {
                condition: cond,
                body: vec![Stmt {
                    kind: StmtKind::Break,
                    span: dummy_span(),
                }],
            },
            span: dummy_span(),
        }];
        let ast = Ast {
            items: vec![make_fn_item("f", vec![], None, body)],
        };
        let prog = AstLowering::lower(&ast).unwrap();
        let has_loop = prog.functions[0]
            .body
            .iter()
            .any(|n| matches!(n.op, HirOp::Loop));
        assert!(has_loop);
    }

    #[test]
    fn unique_hir_ids() {
        let body = vec![
            Stmt {
                kind: StmtKind::ExprStmt(int_expr(1)),
                span: dummy_span(),
            },
            Stmt {
                kind: StmtKind::ExprStmt(int_expr(2)),
                span: dummy_span(),
            },
        ];
        let ast = Ast {
            items: vec![make_fn_item("f", vec![], None, body)],
        };
        let prog = AstLowering::lower(&ast).unwrap();
        let ids: Vec<HirId> = prog.functions[0].body.iter().map(|n| n.id).collect();
        // All ids must be unique.
        let mut seen = std::collections::HashSet::new();
        for id in &ids {
            assert!(seen.insert(id), "duplicate HirId: {id}");
        }
    }

    #[test]
    fn lower_function_call() {
        let call = Expr {
            kind: ExprKind::Call {
                callee: Box::new(ident_expr("foo")),
                args: vec![int_expr(1), int_expr(2)],
            },
            span: dummy_span(),
        };
        let body = vec![Stmt {
            kind: StmtKind::ExprStmt(call),
            span: dummy_span(),
        }];
        let ast = Ast {
            items: vec![make_fn_item("main", vec![], None, body)],
        };
        let prog = AstLowering::lower(&ast).unwrap();
        let has_call = prog.functions[0]
            .body
            .iter()
            .any(|n| matches!(&n.op, HirOp::Call { name } if name == "foo"));
        assert!(has_call);
    }

    #[test]
    fn lower_struct_def() {
        use crate::parser::{Field, Type};
        let ast = Ast {
            items: vec![Item {
                kind: ItemKind::Struct {
                    name: "Point".to_string(),
                    generics: vec![],
                    fields: vec![
                        Field { name: "x".to_string(), ty: Type::Named("f64".into()), span: dummy_span() },
                        Field { name: "y".to_string(), ty: Type::Named("f64".into()), span: dummy_span() },
                    ],
                    is_pub: true,
                },
                span: dummy_span(),
            }],
        };
        let prog = AstLowering::lower(&ast).unwrap();
        assert_eq!(prog.structs.len(), 1);
        assert_eq!(prog.structs[0].name, "Point");
        assert_eq!(prog.structs[0].fields.len(), 2);
    }
}