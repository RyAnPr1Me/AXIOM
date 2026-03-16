/// Parser for the Axiom programming language.
///
/// Transforms a token stream produced by the lexer into an abstract syntax tree
/// (AST). Uses recursive-descent parsing with Pratt-style operator precedence
/// for expressions. Indentation-based blocks are handled via INDENT / DEDENT
/// tokens emitted by the lexer.

use crate::lexer::{Span, Token, TokenKind};
use std::fmt;

// ---------------------------------------------------------------------------
// Parse error
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
    pub span: Span,
}

impl ParseError {
    pub fn new(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Parse error at {}: {}", self.span, self.message)
    }
}

impl std::error::Error for ParseError {}

// ---------------------------------------------------------------------------
// AST – types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    /// A simple named type, e.g. `i32`, `Vec3`.
    Named(String),
    /// An array type with a fixed size, e.g. `[10]i32`.
    Array {
        size: Option<u64>,
        elem: Box<Type>,
    },
    /// A reference type: `&T` or `&mut T`.
    Ref {
        mutable: bool,
        inner: Box<Type>,
    },
    /// A generic type: `Option[T]`, `HashMap[K, V]`.
    Generic {
        name: String,
        args: Vec<Type>,
    },
}

// ---------------------------------------------------------------------------
// AST – expressions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    NotEq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    And,
    Or,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    Neg,
    Not,
    BitNot,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AssignOp {
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    /// Integer literal.
    IntLiteral(i64),
    /// Floating-point literal.
    FloatLiteral(f64),
    /// Boolean literal.
    BoolLiteral(bool),
    /// String literal (contents after escape processing).
    StringLiteral(String),
    /// Character literal.
    CharLiteral(char),
    /// Identifier reference.
    Identifier(String),
    /// Binary operation: `lhs op rhs`.
    BinaryOp {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    /// Unary operation: `op expr`.
    UnaryOp {
        op: UnaryOp,
        operand: Box<Expr>,
    },
    /// Function call: `callee(args)`.
    Call {
        callee: Box<Expr>,
        args: Vec<Expr>,
    },
    /// Method call: `object.method(args)`.
    MethodCall {
        object: Box<Expr>,
        method: String,
        args: Vec<Expr>,
    },
    /// Field access: `object.field`.
    FieldAccess {
        object: Box<Expr>,
        field: String,
    },
    /// Index access: `object[index]`.
    Index {
        object: Box<Expr>,
        index: Box<Expr>,
    },
    /// Range expression: `start..end`.
    Range {
        start: Box<Expr>,
        end: Box<Expr>,
    },
    /// New expression: `new Type(args)`.
    New {
        ty: Type,
        args: Vec<Expr>,
    },
    /// Array literal: `[a, b, c]`.
    ArrayLiteral(Vec<Expr>),
}

// ---------------------------------------------------------------------------
// AST – statements
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StmtKind {
    /// Variable declaration with explicit type: `name: type = expr`.
    VarDecl {
        name: String,
        ty: Option<Type>,
        value: Expr,
        mutable: bool,
    },
    /// Assignment: `target = expr`, `target += expr`, etc.
    Assignment {
        target: Expr,
        op: AssignOp,
        value: Expr,
    },
    /// Return statement: `return expr` or bare `return`.
    Return(Option<Expr>),
    /// If / elif / else chain.
    If {
        condition: Expr,
        body: Vec<Stmt>,
        elif_branches: Vec<(Expr, Vec<Stmt>)>,
        else_body: Option<Vec<Stmt>>,
    },
    /// For-in loop: `for x in expr:` (range or iterable).
    For {
        var: String,
        iter: Expr,
        body: Vec<Stmt>,
    },
    /// While loop: `while cond:`.
    While {
        condition: Expr,
        body: Vec<Stmt>,
    },
    /// Break statement.
    Break,
    /// Continue statement.
    Continue,
    /// Bare expression used as a statement.
    ExprStmt(Expr),
}

// ---------------------------------------------------------------------------
// AST – top-level items
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub name: String,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    pub name: String,
    pub fields: Vec<Type>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Item {
    pub kind: ItemKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ItemKind {
    /// Function definition.
    Function {
        name: String,
        generics: Vec<String>,
        params: Vec<Param>,
        return_type: Option<Type>,
        body: Vec<Stmt>,
        annotations: Vec<String>,
        is_pub: bool,
    },
    /// Struct definition.
    Struct {
        name: String,
        generics: Vec<String>,
        fields: Vec<Field>,
        is_pub: bool,
    },
    /// Enum definition.
    Enum {
        name: String,
        generics: Vec<String>,
        variants: Vec<Variant>,
        is_pub: bool,
    },
    /// Import declaration.
    Import {
        path: String,
    },
}

// ---------------------------------------------------------------------------
// AST – program root
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct Ast {
    pub items: Vec<Item>,
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    /// Parse a full program from the token stream.
    pub fn parse(tokens: Vec<Token>) -> Result<Ast, ParseError> {
        let mut parser = Parser { tokens, pos: 0 };
        let ast = parser.parse_program()?;
        Ok(ast)
    }

    // -- cursor helpers -----------------------------------------------------

    fn peek(&self) -> &Token {
        &self.tokens[self.pos]
    }

    fn peek_kind(&self) -> &TokenKind {
        &self.tokens[self.pos].kind
    }

    fn at(&self, kind: &TokenKind) -> bool {
        self.peek_kind() == kind
    }

    fn advance(&mut self) -> Token {
        let tok = self.tokens[self.pos].clone();
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        tok
    }

    fn expect(&mut self, kind: &TokenKind) -> Result<Token, ParseError> {
        if self.at(kind) {
            Ok(self.advance())
        } else {
            Err(self.error(format!("expected {:?}, found {:?}", kind, self.peek_kind())))
        }
    }

    fn error(&self, message: impl Into<String>) -> ParseError {
        ParseError::new(message, self.peek().span)
    }

    fn skip_newlines(&mut self) {
        while self.at(&TokenKind::Newline) {
            self.advance();
        }
    }

    fn expect_newline_or_eof(&mut self) -> Result<(), ParseError> {
        if self.at(&TokenKind::Newline) || self.at(&TokenKind::Eof) || self.at(&TokenKind::Dedent) {
            if self.at(&TokenKind::Newline) {
                self.advance();
            }
            Ok(())
        } else {
            Err(self.error(format!("expected newline, found {:?}", self.peek_kind())))
        }
    }

    // -- program ------------------------------------------------------------

    fn parse_program(&mut self) -> Result<Ast, ParseError> {
        let mut items = Vec::new();
        self.skip_newlines();
        while !self.at(&TokenKind::Eof) {
            items.push(self.parse_item()?);
            self.skip_newlines();
        }
        Ok(Ast { items })
    }

    // -- items --------------------------------------------------------------

    fn parse_item(&mut self) -> Result<Item, ParseError> {
        let mut annotations = Vec::new();
        let mut is_pub = false;

        // Collect leading annotations.
        while self.at(&TokenKind::Annotation) {
            let tok = self.advance();
            annotations.push(tok.text);
            self.skip_newlines();
        }

        // Optional `pub`.
        if self.at(&TokenKind::Pub) {
            is_pub = true;
            self.advance();
        }

        let span = self.peek().span;

        match self.peek_kind() {
            TokenKind::Fn => self.parse_function(annotations, is_pub, span),
            TokenKind::Struct => self.parse_struct(is_pub, span),
            TokenKind::Enum => self.parse_enum(is_pub, span),
            TokenKind::Import => self.parse_import(span),
            _ => Err(self.error(format!(
                "expected top-level item (fn, struct, enum, import), found {:?}",
                self.peek_kind()
            ))),
        }
    }

    // -- function -----------------------------------------------------------

    fn parse_function(
        &mut self,
        annotations: Vec<String>,
        is_pub: bool,
        span: Span,
    ) -> Result<Item, ParseError> {
        self.expect(&TokenKind::Fn)?;
        let name_tok = self.expect(&TokenKind::Identifier)?;
        let name = name_tok.text;

        // Optional generics: `[T, U]`.
        let generics = if self.at(&TokenKind::LBracket) {
            self.parse_generic_params()?
        } else {
            Vec::new()
        };

        // Parameter list.
        self.expect(&TokenKind::LParen)?;
        let params = self.parse_params()?;
        self.expect(&TokenKind::RParen)?;

        // Optional return type.
        let return_type = if self.at(&TokenKind::Arrow) {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };

        // Colon + block body.
        self.expect(&TokenKind::Colon)?;
        let body = self.parse_block()?;

        Ok(Item {
            kind: ItemKind::Function {
                name,
                generics,
                params,
                return_type,
                body,
                annotations,
                is_pub,
            },
            span,
        })
    }

    fn parse_generic_params(&mut self) -> Result<Vec<String>, ParseError> {
        self.expect(&TokenKind::LBracket)?;
        let mut params = Vec::new();
        if !self.at(&TokenKind::RBracket) {
            params.push(self.expect(&TokenKind::Identifier)?.text);
            while self.at(&TokenKind::Comma) {
                self.advance();
                params.push(self.expect(&TokenKind::Identifier)?.text);
            }
        }
        self.expect(&TokenKind::RBracket)?;
        Ok(params)
    }

    fn parse_params(&mut self) -> Result<Vec<Param>, ParseError> {
        let mut params = Vec::new();

        // Handle `self` / `&self` as first parameter.
        if self.at(&TokenKind::SelfKw) || self.at(&TokenKind::Ampersand) {
            let span = self.peek().span;
            let mut mutable = false;
            if self.at(&TokenKind::Ampersand) {
                self.advance();
                if self.at(&TokenKind::Mut) {
                    mutable = true;
                    self.advance();
                }
            }
            self.expect(&TokenKind::SelfKw)?;
            let self_ty = Type::Named("Self".to_string());
            let ty = if mutable {
                Type::Ref {
                    mutable: true,
                    inner: Box::new(self_ty),
                }
            } else {
                self_ty
            };
            params.push(Param {
                name: "self".to_string(),
                ty,
                span,
            });
            if self.at(&TokenKind::Comma) {
                self.advance();
            } else {
                return Ok(params);
            }
        }

        if !self.at(&TokenKind::RParen) {
            params.push(self.parse_param()?);
            while self.at(&TokenKind::Comma) {
                self.advance();
                if self.at(&TokenKind::RParen) {
                    break; // trailing comma
                }
                params.push(self.parse_param()?);
            }
        }
        Ok(params)
    }

    fn parse_param(&mut self) -> Result<Param, ParseError> {
        let span = self.peek().span;
        let name = self.expect(&TokenKind::Identifier)?.text;
        self.expect(&TokenKind::Colon)?;
        let ty = self.parse_type()?;
        Ok(Param { name, ty, span })
    }

    // -- struct -------------------------------------------------------------

    fn parse_struct(&mut self, is_pub: bool, span: Span) -> Result<Item, ParseError> {
        self.expect(&TokenKind::Struct)?;
        let name = self.expect(&TokenKind::Identifier)?.text;

        let generics = if self.at(&TokenKind::LBracket) {
            self.parse_generic_params()?
        } else {
            Vec::new()
        };

        self.expect(&TokenKind::Colon)?;
        let fields = self.parse_struct_fields()?;

        Ok(Item {
            kind: ItemKind::Struct {
                name,
                generics,
                fields,
                is_pub,
            },
            span,
        })
    }

    fn parse_struct_fields(&mut self) -> Result<Vec<Field>, ParseError> {
        self.expect_newline_or_eof()?;
        self.skip_newlines();
        self.expect(&TokenKind::Indent)?;

        let mut fields = Vec::new();
        self.skip_newlines();
        while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
            let fspan = self.peek().span;
            let fname = self.expect(&TokenKind::Identifier)?.text;
            self.expect(&TokenKind::Colon)?;
            let fty = self.parse_type()?;
            fields.push(Field {
                name: fname,
                ty: fty,
                span: fspan,
            });
            self.expect_newline_or_eof()?;
            self.skip_newlines();
        }
        if self.at(&TokenKind::Dedent) {
            self.advance();
        }
        Ok(fields)
    }

    // -- enum ---------------------------------------------------------------

    fn parse_enum(&mut self, is_pub: bool, span: Span) -> Result<Item, ParseError> {
        self.expect(&TokenKind::Enum)?;
        let name = self.expect(&TokenKind::Identifier)?.text;

        let generics = if self.at(&TokenKind::LBracket) {
            self.parse_generic_params()?
        } else {
            Vec::new()
        };

        self.expect(&TokenKind::Colon)?;
        let variants = self.parse_enum_variants()?;

        Ok(Item {
            kind: ItemKind::Enum {
                name,
                generics,
                variants,
                is_pub,
            },
            span,
        })
    }

    fn parse_enum_variants(&mut self) -> Result<Vec<Variant>, ParseError> {
        self.expect_newline_or_eof()?;
        self.skip_newlines();
        self.expect(&TokenKind::Indent)?;

        let mut variants = Vec::new();
        self.skip_newlines();
        while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
            let vspan = self.peek().span;
            let vname = self.expect(&TokenKind::Identifier)?.text;

            let fields = if self.at(&TokenKind::LParen) {
                self.advance();
                let mut tys = Vec::new();
                if !self.at(&TokenKind::RParen) {
                    tys.push(self.parse_type()?);
                    while self.at(&TokenKind::Comma) {
                        self.advance();
                        if self.at(&TokenKind::RParen) {
                            break;
                        }
                        tys.push(self.parse_type()?);
                    }
                }
                self.expect(&TokenKind::RParen)?;
                tys
            } else {
                Vec::new()
            };

            variants.push(Variant {
                name: vname,
                fields,
                span: vspan,
            });
            self.expect_newline_or_eof()?;
            self.skip_newlines();
        }
        if self.at(&TokenKind::Dedent) {
            self.advance();
        }
        Ok(variants)
    }

    // -- import -------------------------------------------------------------

    fn parse_import(&mut self, span: Span) -> Result<Item, ParseError> {
        self.expect(&TokenKind::Import)?;
        let mut path = self.expect(&TokenKind::Identifier)?.text;
        while self.at(&TokenKind::Dot) {
            self.advance();
            let seg = self.expect(&TokenKind::Identifier)?.text;
            path.push('.');
            path.push_str(&seg);
        }
        self.expect_newline_or_eof()?;
        Ok(Item {
            kind: ItemKind::Import { path },
            span,
        })
    }

    // -- block (INDENT body DEDENT) -----------------------------------------

    fn parse_block(&mut self) -> Result<Vec<Stmt>, ParseError> {
        self.expect_newline_or_eof()?;
        self.skip_newlines();
        self.expect(&TokenKind::Indent)?;

        let mut stmts = Vec::new();
        self.skip_newlines();
        while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
            stmts.push(self.parse_stmt()?);
            self.skip_newlines();
        }
        if self.at(&TokenKind::Dedent) {
            self.advance();
        }
        Ok(stmts)
    }

    // -- statements ---------------------------------------------------------

    fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        let span = self.peek().span;

        match self.peek_kind() {
            TokenKind::Return => self.parse_return(span),
            TokenKind::If => self.parse_if(span),
            TokenKind::For => self.parse_for(span),
            TokenKind::While => self.parse_while(span),
            TokenKind::Break => {
                self.advance();
                self.expect_newline_or_eof()?;
                Ok(Stmt { kind: StmtKind::Break, span })
            }
            TokenKind::Continue => {
                self.advance();
                self.expect_newline_or_eof()?;
                Ok(Stmt { kind: StmtKind::Continue, span })
            }
            TokenKind::Let => self.parse_let_decl(span),
            // `identifier : type = expr`  or  `identifier := expr`  (declaration)
            // `expr = expr` / `expr += expr`  (assignment)
            // Otherwise expression-statement.
            _ => self.parse_expr_or_assign(span),
        }
    }

    fn parse_return(&mut self, span: Span) -> Result<Stmt, ParseError> {
        self.advance(); // consume `return`
        if self.at(&TokenKind::Newline) || self.at(&TokenKind::Eof) || self.at(&TokenKind::Dedent) {
            self.expect_newline_or_eof()?;
            return Ok(Stmt {
                kind: StmtKind::Return(None),
                span,
            });
        }
        let expr = self.parse_expr()?;
        self.expect_newline_or_eof()?;
        Ok(Stmt {
            kind: StmtKind::Return(Some(expr)),
            span,
        })
    }

    fn parse_if(&mut self, span: Span) -> Result<Stmt, ParseError> {
        self.advance(); // consume `if`
        let condition = self.parse_expr()?;
        self.expect(&TokenKind::Colon)?;
        let body = self.parse_block()?;

        let mut elif_branches = Vec::new();
        let mut else_body = None;

        self.skip_newlines();
        while self.at(&TokenKind::Elif) {
            self.advance();
            let elif_cond = self.parse_expr()?;
            self.expect(&TokenKind::Colon)?;
            let elif_body = self.parse_block()?;
            elif_branches.push((elif_cond, elif_body));
            self.skip_newlines();
        }

        if self.at(&TokenKind::Else) {
            self.advance();
            self.expect(&TokenKind::Colon)?;
            else_body = Some(self.parse_block()?);
        }

        Ok(Stmt {
            kind: StmtKind::If {
                condition,
                body,
                elif_branches,
                else_body,
            },
            span,
        })
    }

    fn parse_for(&mut self, span: Span) -> Result<Stmt, ParseError> {
        self.advance(); // consume `for`
        let var = self.expect(&TokenKind::Identifier)?.text;
        self.expect(&TokenKind::In)?;
        let iter = self.parse_expr()?;
        self.expect(&TokenKind::Colon)?;
        let body = self.parse_block()?;
        Ok(Stmt {
            kind: StmtKind::For { var, iter, body },
            span,
        })
    }

    fn parse_while(&mut self, span: Span) -> Result<Stmt, ParseError> {
        self.advance(); // consume `while`
        let condition = self.parse_expr()?;
        self.expect(&TokenKind::Colon)?;
        let body = self.parse_block()?;
        Ok(Stmt {
            kind: StmtKind::While { condition, body },
            span,
        })
    }

    fn parse_let_decl(&mut self, span: Span) -> Result<Stmt, ParseError> {
        self.advance(); // consume `let`
        let mutable = if self.at(&TokenKind::Mut) {
            self.advance();
            true
        } else {
            false
        };
        let name = self.expect(&TokenKind::Identifier)?.text;

        // `name: type = expr`  or  `name := expr`
        if self.at(&TokenKind::ColonEq) {
            self.advance();
            let value = self.parse_expr()?;
            self.expect_newline_or_eof()?;
            return Ok(Stmt {
                kind: StmtKind::VarDecl {
                    name,
                    ty: None,
                    value,
                    mutable,
                },
                span,
            });
        }

        self.expect(&TokenKind::Colon)?;
        let ty = self.parse_type()?;
        self.expect(&TokenKind::Eq)?;
        let value = self.parse_expr()?;
        self.expect_newline_or_eof()?;
        Ok(Stmt {
            kind: StmtKind::VarDecl {
                name,
                ty: Some(ty),
                value,
                mutable,
            },
            span,
        })
    }

    /// Parse something that starts with an expression – could be an
    /// assignment, variable declaration (without `let`), or bare expression.
    fn parse_expr_or_assign(&mut self, span: Span) -> Result<Stmt, ParseError> {
        // Look-ahead for `ident : type = expr` (declaration without `let`)
        // or `ident := expr`.
        if matches!(self.peek_kind(), TokenKind::Identifier) {
            let saved = self.pos;
            let name = self.advance().text;

            if self.at(&TokenKind::ColonEq) {
                self.advance();
                let value = self.parse_expr()?;
                self.expect_newline_or_eof()?;
                return Ok(Stmt {
                    kind: StmtKind::VarDecl {
                        name,
                        ty: None,
                        value,
                        mutable: false,
                    },
                    span,
                });
            }

            if self.at(&TokenKind::Colon) && !self.is_block_colon() {
                self.advance();
                let ty = self.parse_type()?;
                self.expect(&TokenKind::Eq)?;
                let value = self.parse_expr()?;
                self.expect_newline_or_eof()?;
                return Ok(Stmt {
                    kind: StmtKind::VarDecl {
                        name,
                        ty: Some(ty),
                        value,
                        mutable: false,
                    },
                    span,
                });
            }

            // Not a declaration – backtrack and parse as expression.
            self.pos = saved;
        }

        let expr = self.parse_expr()?;

        // Check for assignment operators.
        let op = match self.peek_kind() {
            TokenKind::Eq => Some(AssignOp::Assign),
            TokenKind::PlusEq => Some(AssignOp::AddAssign),
            TokenKind::MinusEq => Some(AssignOp::SubAssign),
            TokenKind::StarEq => Some(AssignOp::MulAssign),
            TokenKind::SlashEq => Some(AssignOp::DivAssign),
            _ => None,
        };

        if let Some(op) = op {
            self.advance();
            let value = self.parse_expr()?;
            self.expect_newline_or_eof()?;
            return Ok(Stmt {
                kind: StmtKind::Assignment {
                    target: expr,
                    op,
                    value,
                },
                span,
            });
        }

        self.expect_newline_or_eof()?;
        Ok(Stmt {
            kind: StmtKind::ExprStmt(expr),
            span,
        })
    }

    /// Check if the current `:` introduces a block (i.e. is followed by
    /// Newline then Indent or directly by Indent). This disambiguates
    /// `name: type = ...` from `if cond:` blocks when used within
    /// `parse_expr_or_assign`.
    fn is_block_colon(&self) -> bool {
        let mut ahead = self.pos + 1;
        while ahead < self.tokens.len() && self.tokens[ahead].kind == TokenKind::Newline {
            ahead += 1;
        }
        ahead < self.tokens.len() && self.tokens[ahead].kind == TokenKind::Indent
    }

    // -- type parsing -------------------------------------------------------

    fn parse_type(&mut self) -> Result<Type, ParseError> {
        // Reference: `&T` or `&mut T`
        if self.at(&TokenKind::Ampersand) {
            self.advance();
            let mutable = if self.at(&TokenKind::Mut) {
                self.advance();
                true
            } else {
                false
            };
            let inner = self.parse_type()?;
            return Ok(Type::Ref {
                mutable,
                inner: Box::new(inner),
            });
        }

        // Array: `[size]elem` or `[]elem`
        if self.at(&TokenKind::LBracket) {
            self.advance();
            let size = if self.at(&TokenKind::IntLiteral) {
                let tok = self.advance();
                let value = parse_int_literal(&tok.text).map_err(|e| self.error(e))?;
                if value < 0 {
                    return Err(self.error("array size must be non-negative"));
                }
                Some(value as u64)
            } else {
                None
            };
            self.expect(&TokenKind::RBracket)?;
            let elem = self.parse_type()?;
            return Ok(Type::Array {
                size,
                elem: Box::new(elem),
            });
        }

        // Named type (possibly generic).
        let name = self.expect(&TokenKind::Identifier)?.text;
        if self.at(&TokenKind::LBracket) {
            self.advance();
            let mut args = Vec::new();
            if !self.at(&TokenKind::RBracket) {
                args.push(self.parse_type()?);
                while self.at(&TokenKind::Comma) {
                    self.advance();
                    if self.at(&TokenKind::RBracket) {
                        break;
                    }
                    args.push(self.parse_type()?);
                }
            }
            self.expect(&TokenKind::RBracket)?;
            Ok(Type::Generic { name, args })
        } else {
            Ok(Type::Named(name))
        }
    }

    // -- expression parsing (Pratt / precedence-climbing) -------------------

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_or_expr()
    }

    // Precedence levels (lowest to highest):
    //  ||
    //  &&
    //  | (bitwise or)
    //  ^ (bitwise xor)
    //  & (bitwise and)
    //  == !=
    //  < > <= >=
    //  << >>
    //  + -
    //  * / %
    //  unary (- ! ~)
    //  postfix (call, index, field)
    //  primary

    fn parse_or_expr(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_and_expr()?;
        while self.at(&TokenKind::PipePipe) {
            self.advance();
            let rhs = self.parse_and_expr()?;
            let span = lhs.span;
            lhs = Expr {
                kind: ExprKind::BinaryOp {
                    op: BinOp::Or,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                span,
            };
        }
        Ok(lhs)
    }

    fn parse_and_expr(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_bitor_expr()?;
        while self.at(&TokenKind::AmpAmp) {
            self.advance();
            let rhs = self.parse_bitor_expr()?;
            let span = lhs.span;
            lhs = Expr {
                kind: ExprKind::BinaryOp {
                    op: BinOp::And,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                span,
            };
        }
        Ok(lhs)
    }

    fn parse_bitor_expr(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_bitxor_expr()?;
        while self.at(&TokenKind::Pipe) {
            self.advance();
            let rhs = self.parse_bitxor_expr()?;
            let span = lhs.span;
            lhs = Expr {
                kind: ExprKind::BinaryOp {
                    op: BinOp::BitOr,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                span,
            };
        }
        Ok(lhs)
    }

    fn parse_bitxor_expr(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_bitand_expr()?;
        while self.at(&TokenKind::Caret) {
            self.advance();
            let rhs = self.parse_bitand_expr()?;
            let span = lhs.span;
            lhs = Expr {
                kind: ExprKind::BinaryOp {
                    op: BinOp::BitXor,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                span,
            };
        }
        Ok(lhs)
    }

    fn parse_bitand_expr(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_equality_expr()?;
        while self.at(&TokenKind::Ampersand) {
            self.advance();
            let rhs = self.parse_equality_expr()?;
            let span = lhs.span;
            lhs = Expr {
                kind: ExprKind::BinaryOp {
                    op: BinOp::BitAnd,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                span,
            };
        }
        Ok(lhs)
    }

    fn parse_equality_expr(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_comparison_expr()?;
        loop {
            let op = match self.peek_kind() {
                TokenKind::EqEq => BinOp::Eq,
                TokenKind::NotEq => BinOp::NotEq,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_comparison_expr()?;
            let span = lhs.span;
            lhs = Expr {
                kind: ExprKind::BinaryOp {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                span,
            };
        }
        Ok(lhs)
    }

    fn parse_comparison_expr(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_shift_expr()?;
        loop {
            let op = match self.peek_kind() {
                TokenKind::Lt => BinOp::Lt,
                TokenKind::Gt => BinOp::Gt,
                TokenKind::LtEq => BinOp::LtEq,
                TokenKind::GtEq => BinOp::GtEq,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_shift_expr()?;
            let span = lhs.span;
            lhs = Expr {
                kind: ExprKind::BinaryOp {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                span,
            };
        }
        Ok(lhs)
    }

    fn parse_shift_expr(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_additive_expr()?;
        loop {
            let op = match self.peek_kind() {
                TokenKind::Shl => BinOp::Shl,
                TokenKind::Shr => BinOp::Shr,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_additive_expr()?;
            let span = lhs.span;
            lhs = Expr {
                kind: ExprKind::BinaryOp {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                span,
            };
        }
        Ok(lhs)
    }

    fn parse_additive_expr(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_multiplicative_expr()?;
        loop {
            let op = match self.peek_kind() {
                TokenKind::Plus => BinOp::Add,
                TokenKind::Minus => BinOp::Sub,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_multiplicative_expr()?;
            let span = lhs.span;
            lhs = Expr {
                kind: ExprKind::BinaryOp {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                span,
            };
        }
        Ok(lhs)
    }

    fn parse_multiplicative_expr(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_unary_expr()?;
        loop {
            let op = match self.peek_kind() {
                TokenKind::Star => BinOp::Mul,
                TokenKind::Slash => BinOp::Div,
                TokenKind::Percent => BinOp::Mod,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_unary_expr()?;
            let span = lhs.span;
            lhs = Expr {
                kind: ExprKind::BinaryOp {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                span,
            };
        }
        Ok(lhs)
    }

    fn parse_unary_expr(&mut self) -> Result<Expr, ParseError> {
        let span = self.peek().span;
        match self.peek_kind() {
            TokenKind::Minus => {
                self.advance();
                let operand = self.parse_unary_expr()?;
                Ok(Expr {
                    kind: ExprKind::UnaryOp {
                        op: UnaryOp::Neg,
                        operand: Box::new(operand),
                    },
                    span,
                })
            }
            TokenKind::Bang => {
                self.advance();
                let operand = self.parse_unary_expr()?;
                Ok(Expr {
                    kind: ExprKind::UnaryOp {
                        op: UnaryOp::Not,
                        operand: Box::new(operand),
                    },
                    span,
                })
            }
            TokenKind::Tilde => {
                self.advance();
                let operand = self.parse_unary_expr()?;
                Ok(Expr {
                    kind: ExprKind::UnaryOp {
                        op: UnaryOp::BitNot,
                        operand: Box::new(operand),
                    },
                    span,
                })
            }
            _ => self.parse_postfix_expr(),
        }
    }

    fn parse_postfix_expr(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_primary_expr()?;

        loop {
            match self.peek_kind() {
                // Call: expr(args)
                TokenKind::LParen => {
                    self.advance();
                    let args = self.parse_call_args()?;
                    self.expect(&TokenKind::RParen)?;
                    let span = expr.span;
                    expr = Expr {
                        kind: ExprKind::Call {
                            callee: Box::new(expr),
                            args,
                        },
                        span,
                    };
                }
                // Index: expr[index]
                TokenKind::LBracket => {
                    self.advance();
                    let index = self.parse_expr()?;
                    self.expect(&TokenKind::RBracket)?;
                    let span = expr.span;
                    expr = Expr {
                        kind: ExprKind::Index {
                            object: Box::new(expr),
                            index: Box::new(index),
                        },
                        span,
                    };
                }
                // Field or method: expr.ident  /  expr.ident(args)
                TokenKind::Dot => {
                    self.advance();
                    // Check for `..` – this is a range operator in disguise
                    // (`a.b..c` is parsed as range on field access).
                    let field_tok = self.expect(&TokenKind::Identifier)?;
                    if self.at(&TokenKind::LParen) {
                        self.advance();
                        let args = self.parse_call_args()?;
                        self.expect(&TokenKind::RParen)?;
                        let span = expr.span;
                        expr = Expr {
                            kind: ExprKind::MethodCall {
                                object: Box::new(expr),
                                method: field_tok.text,
                                args,
                            },
                            span,
                        };
                    } else {
                        let span = expr.span;
                        expr = Expr {
                            kind: ExprKind::FieldAccess {
                                object: Box::new(expr),
                                field: field_tok.text,
                            },
                            span,
                        };
                    }
                }
                // Range: expr..expr
                TokenKind::DotDot => {
                    self.advance();
                    let rhs = self.parse_additive_expr()?;
                    let span = expr.span;
                    expr = Expr {
                        kind: ExprKind::Range {
                            start: Box::new(expr),
                            end: Box::new(rhs),
                        },
                        span,
                    };
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_call_args(&mut self) -> Result<Vec<Expr>, ParseError> {
        let mut args = Vec::new();
        if !self.at(&TokenKind::RParen) {
            args.push(self.parse_expr()?);
            while self.at(&TokenKind::Comma) {
                self.advance();
                if self.at(&TokenKind::RParen) {
                    break;
                }
                args.push(self.parse_expr()?);
            }
        }
        Ok(args)
    }

    fn parse_primary_expr(&mut self) -> Result<Expr, ParseError> {
        let span = self.peek().span;

        match self.peek_kind().clone() {
            TokenKind::IntLiteral => {
                let tok = self.advance();
                let value = parse_int_literal(&tok.text).map_err(|e| self.error(e))?;
                Ok(Expr {
                    kind: ExprKind::IntLiteral(value),
                    span,
                })
            }
            TokenKind::FloatLiteral => {
                let tok = self.advance();
                let value: f64 = tok
                    .text
                    .replace('_', "")
                    .parse()
                    .map_err(|_| self.error("invalid float literal"))?;
                Ok(Expr {
                    kind: ExprKind::FloatLiteral(value),
                    span,
                })
            }
            TokenKind::BoolTrue => {
                self.advance();
                Ok(Expr {
                    kind: ExprKind::BoolLiteral(true),
                    span,
                })
            }
            TokenKind::BoolFalse => {
                self.advance();
                Ok(Expr {
                    kind: ExprKind::BoolLiteral(false),
                    span,
                })
            }
            TokenKind::StringLiteral => {
                let tok = self.advance();
                Ok(Expr {
                    kind: ExprKind::StringLiteral(tok.text),
                    span,
                })
            }
            TokenKind::CharLiteral => {
                let tok = self.advance();
                let mut chars = tok.text.chars();
                let ch = chars
                    .next()
                    .ok_or_else(|| self.error("empty char literal"))?;
                if chars.next().is_some() {
                    return Err(self.error("char literal must contain exactly one character"));
                }
                Ok(Expr {
                    kind: ExprKind::CharLiteral(ch),
                    span,
                })
            }
            TokenKind::Identifier => {
                let tok = self.advance();
                Ok(Expr {
                    kind: ExprKind::Identifier(tok.text),
                    span,
                })
            }
            TokenKind::SelfKw => {
                self.advance();
                Ok(Expr {
                    kind: ExprKind::Identifier("self".to_string()),
                    span,
                })
            }
            TokenKind::New => self.parse_new_expr(span),
            // Parenthesised expression.
            TokenKind::LParen => {
                self.advance();
                let expr = self.parse_expr()?;
                self.expect(&TokenKind::RParen)?;
                Ok(expr)
            }
            // Array literal: `[a, b, c]`.
            TokenKind::LBracket => {
                self.advance();
                let mut elems = Vec::new();
                if !self.at(&TokenKind::RBracket) {
                    elems.push(self.parse_expr()?);
                    while self.at(&TokenKind::Comma) {
                        self.advance();
                        if self.at(&TokenKind::RBracket) {
                            break;
                        }
                        elems.push(self.parse_expr()?);
                    }
                }
                self.expect(&TokenKind::RBracket)?;
                Ok(Expr {
                    kind: ExprKind::ArrayLiteral(elems),
                    span,
                })
            }
            _ => Err(self.error(format!(
                "expected expression, found {:?}",
                self.peek_kind()
            ))),
        }
    }

    fn parse_new_expr(&mut self, span: Span) -> Result<Expr, ParseError> {
        self.advance(); // consume `new`
        let ty = self.parse_type()?;
        self.expect(&TokenKind::LParen)?;
        let args = self.parse_call_args()?;
        self.expect(&TokenKind::RParen)?;
        Ok(Expr {
            kind: ExprKind::New { ty, args },
            span,
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse an integer literal string (decimal, hex, or binary) into an `i64`.
fn parse_int_literal(text: &str) -> Result<i64, String> {
    let clean: String = text.replace('_', "");
    if let Some(hex) = clean.strip_prefix("0x").or_else(|| clean.strip_prefix("0X")) {
        i64::from_str_radix(hex, 16).map_err(|e| format!("invalid hex literal: {e}"))
    } else if let Some(bin) = clean.strip_prefix("0b").or_else(|| clean.strip_prefix("0B")) {
        i64::from_str_radix(bin, 2).map_err(|e| format!("invalid binary literal: {e}"))
    } else {
        clean
            .parse::<i64>()
            .map_err(|e| format!("invalid integer literal: {e}"))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;

    /// Helper – lex + parse, returning the AST or panicking on error.
    fn parse(src: &str) -> Ast {
        let tokens = Lexer::tokenize(src).expect("lexer failed");
        Parser::parse(tokens).expect("parser failed")
    }

    /// Helper – lex + parse, expecting an error.
    fn parse_err(src: &str) -> ParseError {
        let tokens = Lexer::tokenize(src).expect("lexer failed");
        Parser::parse(tokens).expect_err("expected parse error")
    }

    // -- imports ------------------------------------------------------------

    #[test]
    fn parse_import() {
        let ast = parse("import math\n");
        assert_eq!(ast.items.len(), 1);
        match &ast.items[0].kind {
            ItemKind::Import { path } => assert_eq!(path, "math"),
            other => panic!("expected Import, got {other:?}"),
        }
    }

    #[test]
    fn parse_dotted_import() {
        let ast = parse("import std.collections\n");
        assert_eq!(ast.items.len(), 1);
        match &ast.items[0].kind {
            ItemKind::Import { path } => assert_eq!(path, "std.collections"),
            other => panic!("expected Import, got {other:?}"),
        }
    }

    // -- functions ----------------------------------------------------------

    #[test]
    fn parse_empty_function() {
        let ast = parse("fn main():\n    return\n");
        assert_eq!(ast.items.len(), 1);
        match &ast.items[0].kind {
            ItemKind::Function { name, body, .. } => {
                assert_eq!(name, "main");
                assert_eq!(body.len(), 1);
                assert!(matches!(body[0].kind, StmtKind::Return(None)));
            }
            other => panic!("expected Function, got {other:?}"),
        }
    }

    #[test]
    fn parse_function_with_params_and_return() {
        let ast = parse("fn add(a: i32, b: i32) -> i32:\n    return a + b\n");
        match &ast.items[0].kind {
            ItemKind::Function {
                name,
                params,
                return_type,
                body,
                ..
            } => {
                assert_eq!(name, "add");
                assert_eq!(params.len(), 2);
                assert_eq!(params[0].name, "a");
                assert_eq!(params[1].name, "b");
                assert!(return_type.is_some());
                assert_eq!(body.len(), 1);
            }
            other => panic!("expected Function, got {other:?}"),
        }
    }

    #[test]
    fn parse_generic_function() {
        let ast = parse("fn identity[T](x: T) -> T:\n    return x\n");
        match &ast.items[0].kind {
            ItemKind::Function {
                name, generics, ..
            } => {
                assert_eq!(name, "identity");
                assert_eq!(generics, &["T"]);
            }
            other => panic!("expected Function, got {other:?}"),
        }
    }

    #[test]
    fn parse_annotated_function() {
        let ast = parse("@inline\nfn fast() -> i32:\n    return 42\n");
        match &ast.items[0].kind {
            ItemKind::Function { annotations, .. } => {
                assert_eq!(annotations, &["@inline"]);
            }
            other => panic!("expected Function, got {other:?}"),
        }
    }

    #[test]
    fn parse_pub_function() {
        let ast = parse("pub fn api() -> i32:\n    return 1\n");
        match &ast.items[0].kind {
            ItemKind::Function { is_pub, name, .. } => {
                assert!(is_pub);
                assert_eq!(name, "api");
            }
            other => panic!("expected Function, got {other:?}"),
        }
    }

    // -- structs ------------------------------------------------------------

    #[test]
    fn parse_struct() {
        let src = "struct Vec3:\n    x: f64\n    y: f64\n    z: f64\n";
        let ast = parse(src);
        match &ast.items[0].kind {
            ItemKind::Struct { name, fields, .. } => {
                assert_eq!(name, "Vec3");
                assert_eq!(fields.len(), 3);
                assert_eq!(fields[0].name, "x");
                assert_eq!(fields[2].name, "z");
            }
            other => panic!("expected Struct, got {other:?}"),
        }
    }

    #[test]
    fn parse_generic_struct() {
        let src = "struct Pair[A, B]:\n    first: A\n    second: B\n";
        let ast = parse(src);
        match &ast.items[0].kind {
            ItemKind::Struct {
                name, generics, ..
            } => {
                assert_eq!(name, "Pair");
                assert_eq!(generics, &["A", "B"]);
            }
            other => panic!("expected Struct, got {other:?}"),
        }
    }

    // -- enums --------------------------------------------------------------

    #[test]
    fn parse_enum() {
        let src = "enum Color:\n    Red\n    Green\n    Blue\n";
        let ast = parse(src);
        match &ast.items[0].kind {
            ItemKind::Enum { name, variants, .. } => {
                assert_eq!(name, "Color");
                assert_eq!(variants.len(), 3);
                assert_eq!(variants[0].name, "Red");
            }
            other => panic!("expected Enum, got {other:?}"),
        }
    }

    #[test]
    fn parse_enum_with_data() {
        let src = "enum Option[T]:\n    Some(T)\n    None\n";
        let ast = parse(src);
        match &ast.items[0].kind {
            ItemKind::Enum {
                name,
                generics,
                variants,
                ..
            } => {
                assert_eq!(name, "Option");
                assert_eq!(generics, &["T"]);
                assert_eq!(variants.len(), 2);
                assert_eq!(variants[0].fields.len(), 1);
                assert_eq!(variants[1].fields.len(), 0);
            }
            other => panic!("expected Enum, got {other:?}"),
        }
    }

    // -- statements ---------------------------------------------------------

    #[test]
    fn parse_var_decl_typed() {
        let ast = parse("fn f():\n    x: i32 = 10\n");
        let body = fn_body(&ast);
        match &body[0].kind {
            StmtKind::VarDecl {
                name,
                ty: Some(ty),
                ..
            } => {
                assert_eq!(name, "x");
                assert_eq!(*ty, Type::Named("i32".into()));
            }
            other => panic!("expected VarDecl, got {other:?}"),
        }
    }

    #[test]
    fn parse_var_decl_inferred() {
        let ast = parse("fn f():\n    x := 10\n");
        let body = fn_body(&ast);
        match &body[0].kind {
            StmtKind::VarDecl { name, ty, .. } => {
                assert_eq!(name, "x");
                assert!(ty.is_none());
            }
            other => panic!("expected VarDecl, got {other:?}"),
        }
    }

    #[test]
    fn parse_let_decl() {
        let ast = parse("fn f():\n    let mut y: f64 = 3.14\n");
        let body = fn_body(&ast);
        match &body[0].kind {
            StmtKind::VarDecl {
                name, mutable, ty, ..
            } => {
                assert_eq!(name, "y");
                assert!(mutable);
                assert!(ty.is_some());
            }
            other => panic!("expected VarDecl, got {other:?}"),
        }
    }

    #[test]
    fn parse_assignment() {
        let ast = parse("fn f():\n    x = 5\n");
        let body = fn_body(&ast);
        assert!(matches!(body[0].kind, StmtKind::Assignment { .. }));
    }

    #[test]
    fn parse_compound_assignment() {
        let ast = parse("fn f():\n    x += 1\n");
        let body = fn_body(&ast);
        match &body[0].kind {
            StmtKind::Assignment { op, .. } => assert_eq!(*op, AssignOp::AddAssign),
            other => panic!("expected Assignment, got {other:?}"),
        }
    }

    #[test]
    fn parse_if_elif_else() {
        let src = "fn f():\n    if x > 0:\n        return 1\n    elif x == 0:\n        return 0\n    else:\n        return -1\n";
        let ast = parse(src);
        let body = fn_body(&ast);
        match &body[0].kind {
            StmtKind::If {
                elif_branches,
                else_body,
                ..
            } => {
                assert_eq!(elif_branches.len(), 1);
                assert!(else_body.is_some());
            }
            other => panic!("expected If, got {other:?}"),
        }
    }

    #[test]
    fn parse_for_loop() {
        let src = "fn f():\n    for i in 0..10:\n        x = i\n";
        let ast = parse(src);
        let body = fn_body(&ast);
        match &body[0].kind {
            StmtKind::For { var, iter, body } => {
                assert_eq!(var, "i");
                assert!(matches!(iter.kind, ExprKind::Range { .. }));
                assert_eq!(body.len(), 1);
            }
            other => panic!("expected For, got {other:?}"),
        }
    }

    #[test]
    fn parse_while_loop() {
        let src = "fn f():\n    while x > 0:\n        x -= 1\n";
        let ast = parse(src);
        let body = fn_body(&ast);
        assert!(matches!(body[0].kind, StmtKind::While { .. }));
    }

    #[test]
    fn parse_break_continue() {
        let src = "fn f():\n    break\n    continue\n";
        let ast = parse(src);
        let body = fn_body(&ast);
        assert!(matches!(body[0].kind, StmtKind::Break));
        assert!(matches!(body[1].kind, StmtKind::Continue));
    }

    // -- expressions --------------------------------------------------------

    #[test]
    fn parse_arithmetic_precedence() {
        // `1 + 2 * 3` should parse as `1 + (2 * 3)`.
        let ast = parse("fn f():\n    return 1 + 2 * 3\n");
        let body = fn_body(&ast);
        if let StmtKind::Return(Some(expr)) = &body[0].kind {
            if let ExprKind::BinaryOp { op, rhs, .. } = &expr.kind {
                assert_eq!(*op, BinOp::Add);
                assert!(matches!(rhs.kind, ExprKind::BinaryOp { op: BinOp::Mul, .. }));
            } else {
                panic!("expected BinaryOp, got {expr:?}");
            }
        } else {
            panic!("expected return stmt");
        }
    }

    #[test]
    fn parse_unary() {
        let ast = parse("fn f():\n    return -x\n");
        let body = fn_body(&ast);
        if let StmtKind::Return(Some(expr)) = &body[0].kind {
            assert!(matches!(
                expr.kind,
                ExprKind::UnaryOp {
                    op: UnaryOp::Neg,
                    ..
                }
            ));
        }
    }

    #[test]
    fn parse_function_call() {
        let ast = parse("fn f():\n    foo(1, 2)\n");
        let body = fn_body(&ast);
        if let StmtKind::ExprStmt(expr) = &body[0].kind {
            if let ExprKind::Call { args, .. } = &expr.kind {
                assert_eq!(args.len(), 2);
            } else {
                panic!("expected Call");
            }
        }
    }

    #[test]
    fn parse_method_call() {
        let ast = parse("fn f():\n    obj.method(1)\n");
        let body = fn_body(&ast);
        if let StmtKind::ExprStmt(expr) = &body[0].kind {
            assert!(matches!(expr.kind, ExprKind::MethodCall { .. }));
        }
    }

    #[test]
    fn parse_field_access() {
        let ast = parse("fn f():\n    obj.field\n");
        let body = fn_body(&ast);
        if let StmtKind::ExprStmt(expr) = &body[0].kind {
            if let ExprKind::FieldAccess { field, .. } = &expr.kind {
                assert_eq!(field, "field");
            } else {
                panic!("expected FieldAccess");
            }
        }
    }

    #[test]
    fn parse_index_access() {
        let ast = parse("fn f():\n    arr[0]\n");
        let body = fn_body(&ast);
        if let StmtKind::ExprStmt(expr) = &body[0].kind {
            assert!(matches!(expr.kind, ExprKind::Index { .. }));
        }
    }

    #[test]
    fn parse_new_expr() {
        let ast = parse("fn f():\n    new Vec3(1, 2, 3)\n");
        let body = fn_body(&ast);
        if let StmtKind::ExprStmt(expr) = &body[0].kind {
            if let ExprKind::New { ty, args } = &expr.kind {
                assert_eq!(*ty, Type::Named("Vec3".into()));
                assert_eq!(args.len(), 3);
            } else {
                panic!("expected New");
            }
        }
    }

    #[test]
    fn parse_array_literal() {
        let ast = parse("fn f():\n    return [1, 2, 3]\n");
        let body = fn_body(&ast);
        if let StmtKind::Return(Some(expr)) = &body[0].kind {
            if let ExprKind::ArrayLiteral(elems) = &expr.kind {
                assert_eq!(elems.len(), 3);
            } else {
                panic!("expected ArrayLiteral");
            }
        }
    }

    #[test]
    fn parse_bool_literal() {
        let ast = parse("fn f():\n    return true\n");
        let body = fn_body(&ast);
        if let StmtKind::Return(Some(expr)) = &body[0].kind {
            assert!(matches!(expr.kind, ExprKind::BoolLiteral(true)));
        }
    }

    #[test]
    fn parse_string_literal() {
        let ast = parse("fn f():\n    return \"hello\"\n");
        let body = fn_body(&ast);
        if let StmtKind::Return(Some(expr)) = &body[0].kind {
            assert!(matches!(expr.kind, ExprKind::StringLiteral(_)));
        }
    }

    #[test]
    fn parse_parenthesised_expr() {
        let ast = parse("fn f():\n    return (1 + 2) * 3\n");
        let body = fn_body(&ast);
        if let StmtKind::Return(Some(expr)) = &body[0].kind {
            assert!(matches!(expr.kind, ExprKind::BinaryOp { op: BinOp::Mul, .. }));
        }
    }

    #[test]
    fn parse_range_expr() {
        let ast = parse("fn f():\n    return 0..10\n");
        let body = fn_body(&ast);
        if let StmtKind::Return(Some(expr)) = &body[0].kind {
            assert!(matches!(expr.kind, ExprKind::Range { .. }));
        }
    }

    // -- types --------------------------------------------------------------

    #[test]
    fn parse_ref_type() {
        let ast = parse("fn f(x: &i32):\n    return\n");
        match &ast.items[0].kind {
            ItemKind::Function { params, .. } => {
                assert_eq!(
                    params[0].ty,
                    Type::Ref {
                        mutable: false,
                        inner: Box::new(Type::Named("i32".into()))
                    }
                );
            }
            _ => panic!("expected function"),
        }
    }

    #[test]
    fn parse_mut_ref_type() {
        let ast = parse("fn f(x: &mut i32):\n    return\n");
        match &ast.items[0].kind {
            ItemKind::Function { params, .. } => {
                assert_eq!(
                    params[0].ty,
                    Type::Ref {
                        mutable: true,
                        inner: Box::new(Type::Named("i32".into()))
                    }
                );
            }
            _ => panic!("expected function"),
        }
    }

    #[test]
    fn parse_array_type() {
        let ast = parse("fn f(x: [10]i32):\n    return\n");
        match &ast.items[0].kind {
            ItemKind::Function { params, .. } => {
                assert_eq!(
                    params[0].ty,
                    Type::Array {
                        size: Some(10),
                        elem: Box::new(Type::Named("i32".into()))
                    }
                );
            }
            _ => panic!("expected function"),
        }
    }

    #[test]
    fn parse_slice_type() {
        let ast = parse("fn f(x: []i32):\n    return\n");
        match &ast.items[0].kind {
            ItemKind::Function { params, .. } => {
                assert_eq!(
                    params[0].ty,
                    Type::Array {
                        size: None,
                        elem: Box::new(Type::Named("i32".into()))
                    }
                );
            }
            _ => panic!("expected function"),
        }
    }

    #[test]
    fn parse_generic_type() {
        let ast = parse("fn f(x: Option[i32]):\n    return\n");
        match &ast.items[0].kind {
            ItemKind::Function { params, .. } => {
                assert_eq!(
                    params[0].ty,
                    Type::Generic {
                        name: "Option".into(),
                        args: vec![Type::Named("i32".into())]
                    }
                );
            }
            _ => panic!("expected function"),
        }
    }

    // -- error cases --------------------------------------------------------

    #[test]
    fn error_on_unexpected_token() {
        let err = parse_err("fn 123():\n    return\n");
        assert!(err.message.contains("expected"));
    }

    #[test]
    fn error_on_missing_colon() {
        let err = parse_err("fn f()\n    return\n");
        assert!(err.message.contains("expected"));
    }

    // -- multiple items -----------------------------------------------------

    #[test]
    fn parse_multiple_items() {
        let src = "import math\n\nstruct Point:\n    x: f64\n    y: f64\n\nfn distance(a: Point, b: Point) -> f64:\n    return 0\n";
        let ast = parse(src);
        assert_eq!(ast.items.len(), 3);
        assert!(matches!(ast.items[0].kind, ItemKind::Import { .. }));
        assert!(matches!(ast.items[1].kind, ItemKind::Struct { .. }));
        assert!(matches!(ast.items[2].kind, ItemKind::Function { .. }));
    }

    // -- helpers for tests --------------------------------------------------

    fn fn_body(ast: &Ast) -> &[Stmt] {
        match &ast.items[0].kind {
            ItemKind::Function { body, .. } => body,
            other => panic!("expected Function, got {other:?}"),
        }
    }
}