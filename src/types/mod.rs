/// Type system for the Axiom programming language.
///
/// Defines the concrete types used during type checking and compilation.
/// The parser produces syntactic `Type` nodes; this module resolves them into
/// fully-qualified `AxiomType` values managed by a `TypeContext`.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// TypeId – lightweight handle into the type table
// ---------------------------------------------------------------------------

/// Lightweight, copy-cheap handle that identifies a type inside a
/// [`TypeContext`].  Two `TypeId`s are equal iff they refer to the same slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeId(pub usize);

// ---------------------------------------------------------------------------
// AxiomType – the full type representation
// ---------------------------------------------------------------------------

/// Integer width / signedness descriptor.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IntegerKind {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

/// Floating-point width descriptor.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FloatKind {
    F32,
    F64,
}

/// SIMD / vector lane count for `f32` vectors.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SimdKind {
    /// 4-lane `f32` vector (`vec4f`).
    Vec4F,
    /// 8-lane `f32` vector (`vec8f`).
    Vec8F,
    /// 16-lane `f32` vector (`vec16f`).
    Vec16F,
}

/// A single field inside a [`AxiomType::Struct`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructField {
    pub name: String,
    pub ty: TypeId,
}

/// A single variant of an [`AxiomType::Enum`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EnumVariant {
    pub name: String,
    /// `None` for unit variants, `Some` for variants carrying data.
    pub data: Option<Vec<TypeId>>,
}

/// Concrete type representation used throughout type checking and code
/// generation.
///
/// Every distinct Axiom type maps to exactly one `AxiomType` value inside a
/// [`TypeContext`].  Generic types are monomorphized before reaching
/// back-end passes, so `GenericParam` only appears during early analysis.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AxiomType {
    // -- primitives ---------------------------------------------------------
    /// Signed or unsigned integer type.
    Integer(IntegerKind),
    /// IEEE-754 floating-point type.
    Float(FloatKind),
    /// Boolean (`true` / `false`).
    Bool,
    /// Unicode scalar value.
    Char,

    // -- SIMD / vector ------------------------------------------------------
    /// Fixed-width SIMD vector of `f32` lanes.
    Simd(SimdKind),

    // -- composite ----------------------------------------------------------
    /// Fixed-size array `[N]T` where the length is known at compile time.
    Array {
        element: TypeId,
        length: u64,
    },
    /// Dynamically-sized slice `[]T`.
    Slice {
        element: TypeId,
    },
    /// Named struct with ordered fields.
    Struct {
        name: String,
        fields: Vec<StructField>,
    },
    /// Named enum with variants that may carry data.
    Enum {
        name: String,
        variants: Vec<EnumVariant>,
    },

    // -- callable -----------------------------------------------------------
    /// Function signature: `(param_types) -> return_type`.
    Function {
        params: Vec<TypeId>,
        return_type: TypeId,
    },

    // -- indirection --------------------------------------------------------
    /// Immutable (`&T`) or mutable (`&mut T`) borrow.
    Reference {
        mutable: bool,
        inner: TypeId,
    },
    /// Heap-allocated pointer created via `new`.
    Pointer {
        inner: TypeId,
    },

    // -- generics -----------------------------------------------------------
    /// Unresolved generic type parameter (e.g. `T`).
    GenericParam {
        name: String,
    },

    // -- special ------------------------------------------------------------
    /// The unit type, returned by functions with no explicit return.
    Unit,
}

// ---------------------------------------------------------------------------
// AxiomType – convenience query helpers
// ---------------------------------------------------------------------------

impl AxiomType {
    /// Returns `true` for any numeric type (integer, float, or SIMD).
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            AxiomType::Integer(_) | AxiomType::Float(_) | AxiomType::Simd(_)
        )
    }

    /// Returns `true` for integer types only.
    pub fn is_integer(&self) -> bool {
        matches!(self, AxiomType::Integer(_))
    }

    /// Returns `true` for floating-point types only.
    pub fn is_float(&self) -> bool {
        matches!(self, AxiomType::Float(_))
    }

    /// Returns `true` for signed integer types.
    pub fn is_signed(&self) -> bool {
        matches!(
            self,
            AxiomType::Integer(
                IntegerKind::I8
                    | IntegerKind::I16
                    | IntegerKind::I32
                    | IntegerKind::I64
            )
        )
    }

    /// Returns `true` for unsigned integer types.
    pub fn is_unsigned(&self) -> bool {
        matches!(
            self,
            AxiomType::Integer(
                IntegerKind::U8
                    | IntegerKind::U16
                    | IntegerKind::U32
                    | IntegerKind::U64
            )
        )
    }

    /// Returns `true` for SIMD / vector types.
    pub fn is_simd(&self) -> bool {
        matches!(self, AxiomType::Simd(_))
    }

    /// Returns the size in bytes of the type, or `0` for unsized /
    /// variable-size types (slices, generic params, unit).
    pub fn size_bytes(&self) -> usize {
        match self {
            AxiomType::Integer(k) => match k {
                IntegerKind::I8 | IntegerKind::U8 => 1,
                IntegerKind::I16 | IntegerKind::U16 => 2,
                IntegerKind::I32 | IntegerKind::U32 => 4,
                IntegerKind::I64 | IntegerKind::U64 => 8,
            },
            AxiomType::Float(k) => match k {
                FloatKind::F32 => 4,
                FloatKind::F64 => 8,
            },
            AxiomType::Bool => 1,
            AxiomType::Char => 4, // Unicode scalar → 4 bytes (like Rust)
            AxiomType::Simd(k) => match k {
                SimdKind::Vec4F => 16,  // 4 × f32
                SimdKind::Vec8F => 32,  // 8 × f32
                SimdKind::Vec16F => 64, // 16 × f32
            },
            AxiomType::Reference { .. } | AxiomType::Pointer { .. } => 8,
            AxiomType::Unit => 0,
            // Compound / unsized types return 0; callers should compute
            // layout through the TypeContext for these.
            _ => 0,
        }
    }

    /// Returns the required alignment in bytes, or `1` for types whose
    /// alignment is not statically known.
    pub fn alignment(&self) -> usize {
        match self {
            AxiomType::Integer(k) => match k {
                IntegerKind::I8 | IntegerKind::U8 => 1,
                IntegerKind::I16 | IntegerKind::U16 => 2,
                IntegerKind::I32 | IntegerKind::U32 => 4,
                IntegerKind::I64 | IntegerKind::U64 => 8,
            },
            AxiomType::Float(k) => match k {
                FloatKind::F32 => 4,
                FloatKind::F64 => 8,
            },
            AxiomType::Bool => 1,
            AxiomType::Char => 4,
            AxiomType::Simd(k) => match k {
                SimdKind::Vec4F => 16,
                SimdKind::Vec8F => 32,
                SimdKind::Vec16F => 64,
            },
            AxiomType::Reference { .. } | AxiomType::Pointer { .. } => 8,
            AxiomType::Unit => 1,
            _ => 1,
        }
    }
}

// ---------------------------------------------------------------------------
// TypeContext – central type registry
// ---------------------------------------------------------------------------

/// Central registry for all types encountered during compilation.
///
/// Types are interned: each unique `AxiomType` is stored exactly once, and
/// subsequent registrations of an identical type return the same [`TypeId`].
///
/// # Examples
///
/// ```
/// use axiom::types::{TypeContext, AxiomType, IntegerKind};
///
/// let mut ctx = TypeContext::new();
/// let id1 = ctx.register(AxiomType::Integer(IntegerKind::I32));
/// let id2 = ctx.register(AxiomType::Integer(IntegerKind::I32));
/// assert_eq!(id1, id2);
/// ```
#[derive(Debug, Clone)]
pub struct TypeContext {
    /// All registered types, indexed by `TypeId`.
    types: Vec<AxiomType>,
    /// Reverse map for interning: `AxiomType → TypeId`.
    intern: HashMap<AxiomType, TypeId>,
}

impl TypeContext {
    /// Creates a new, empty type context.
    pub fn new() -> Self {
        Self {
            types: Vec::new(),
            intern: HashMap::new(),
        }
    }

    /// Registers a type and returns its [`TypeId`].
    ///
    /// If an identical type has already been registered, the existing id is
    /// returned (interning).
    pub fn register(&mut self, ty: AxiomType) -> TypeId {
        if let Some(&id) = self.intern.get(&ty) {
            return id;
        }
        let id = TypeId(self.types.len());
        self.intern.insert(ty.clone(), id);
        self.types.push(ty);
        id
    }

    /// Resolves a [`TypeId`] back to its [`AxiomType`].
    ///
    /// # Panics
    ///
    /// Panics if `id` does not correspond to a registered type.
    pub fn resolve(&self, id: TypeId) -> &AxiomType {
        &self.types[id.0]
    }

    /// Returns `true` if both ids refer to the same concrete type.
    pub fn types_equal(&self, a: TypeId, b: TypeId) -> bool {
        a == b
    }

    /// Returns `true` if the type identified by `id` is numeric.
    pub fn is_numeric(&self, id: TypeId) -> bool {
        self.resolve(id).is_numeric()
    }

    /// Returns `true` if the type identified by `id` is an integer.
    pub fn is_integer(&self, id: TypeId) -> bool {
        self.resolve(id).is_integer()
    }

    /// Returns `true` if the type identified by `id` is a float.
    pub fn is_float(&self, id: TypeId) -> bool {
        self.resolve(id).is_float()
    }

    /// Returns `true` if the type identified by `id` is a signed integer.
    pub fn is_signed(&self, id: TypeId) -> bool {
        self.resolve(id).is_signed()
    }

    /// Returns `true` if the type identified by `id` is an unsigned integer.
    pub fn is_unsigned(&self, id: TypeId) -> bool {
        self.resolve(id).is_unsigned()
    }

    /// Returns `true` if the type identified by `id` is a SIMD vector.
    pub fn is_simd(&self, id: TypeId) -> bool {
        self.resolve(id).is_simd()
    }

    /// Returns the size in bytes of the type, or `0` for unsized types.
    pub fn size_of(&self, id: TypeId) -> usize {
        let ty = self.resolve(id);
        match ty {
            AxiomType::Array { element, length } => {
                let elem_size = self.size_of(*element);
                elem_size * (*length as usize)
            }
            other => other.size_bytes(),
        }
    }

    /// Returns the alignment in bytes of the type.
    pub fn align_of(&self, id: TypeId) -> usize {
        let ty = self.resolve(id);
        match ty {
            AxiomType::Array { element, .. } => self.align_of(*element),
            other => other.alignment(),
        }
    }

    /// Monomorphizes a type by substituting generic parameters according to
    /// the provided mapping.
    ///
    /// Returns a new [`TypeId`] with all `GenericParam` nodes replaced by
    /// their concrete counterparts.  Types that contain no generic parameters
    /// are returned unchanged.
    pub fn monomorphize(
        &mut self,
        id: TypeId,
        substitutions: &HashMap<String, TypeId>,
    ) -> TypeId {
        let ty = self.resolve(id).clone();
        match ty {
            AxiomType::GenericParam { ref name } => {
                substitutions.get(name).copied().unwrap_or(id)
            }
            AxiomType::Array { element, length } => {
                let new_elem = self.monomorphize(element, substitutions);
                self.register(AxiomType::Array {
                    element: new_elem,
                    length,
                })
            }
            AxiomType::Slice { element } => {
                let new_elem = self.monomorphize(element, substitutions);
                self.register(AxiomType::Slice { element: new_elem })
            }
            AxiomType::Reference { mutable, inner } => {
                let new_inner = self.monomorphize(inner, substitutions);
                self.register(AxiomType::Reference {
                    mutable,
                    inner: new_inner,
                })
            }
            AxiomType::Pointer { inner } => {
                let new_inner = self.monomorphize(inner, substitutions);
                self.register(AxiomType::Pointer { inner: new_inner })
            }
            AxiomType::Function {
                params,
                return_type,
            } => {
                let new_params: Vec<TypeId> = params
                    .iter()
                    .map(|&p| self.monomorphize(p, substitutions))
                    .collect();
                let new_ret = self.monomorphize(return_type, substitutions);
                self.register(AxiomType::Function {
                    params: new_params,
                    return_type: new_ret,
                })
            }
            AxiomType::Struct { name, fields } => {
                let new_fields: Vec<StructField> = fields
                    .iter()
                    .map(|f| StructField {
                        name: f.name.clone(),
                        ty: self.monomorphize(f.ty, substitutions),
                    })
                    .collect();
                self.register(AxiomType::Struct {
                    name,
                    fields: new_fields,
                })
            }
            AxiomType::Enum { name, variants } => {
                let new_variants: Vec<EnumVariant> = variants
                    .iter()
                    .map(|v| EnumVariant {
                        name: v.name.clone(),
                        data: v.data.as_ref().map(|types| {
                            types
                                .iter()
                                .map(|&t| self.monomorphize(t, substitutions))
                                .collect()
                        }),
                    })
                    .collect();
                self.register(AxiomType::Enum {
                    name,
                    variants: new_variants,
                })
            }
            // Leaf types with no generic sub-parts are returned as-is.
            _ => id,
        }
    }

    /// Returns the total number of registered types.
    pub fn len(&self) -> usize {
        self.types.len()
    }

    /// Returns `true` if no types have been registered.
    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }
}

impl Default for TypeContext {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Display implementations
// ---------------------------------------------------------------------------

impl std::fmt::Display for AxiomType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AxiomType::Integer(k) => match k {
                IntegerKind::I8 => write!(f, "i8"),
                IntegerKind::I16 => write!(f, "i16"),
                IntegerKind::I32 => write!(f, "i32"),
                IntegerKind::I64 => write!(f, "i64"),
                IntegerKind::U8 => write!(f, "u8"),
                IntegerKind::U16 => write!(f, "u16"),
                IntegerKind::U32 => write!(f, "u32"),
                IntegerKind::U64 => write!(f, "u64"),
            },
            AxiomType::Float(k) => match k {
                FloatKind::F32 => write!(f, "f32"),
                FloatKind::F64 => write!(f, "f64"),
            },
            AxiomType::Bool => write!(f, "bool"),
            AxiomType::Char => write!(f, "char"),
            AxiomType::Simd(k) => match k {
                SimdKind::Vec4F => write!(f, "vec4f"),
                SimdKind::Vec8F => write!(f, "vec8f"),
                SimdKind::Vec16F => write!(f, "vec16f"),
            },
            AxiomType::Array { length, .. } => write!(f, "[{length}]T"),
            AxiomType::Slice { .. } => write!(f, "[]T"),
            AxiomType::Struct { name, .. } => write!(f, "{name}"),
            AxiomType::Enum { name, .. } => write!(f, "{name}"),
            AxiomType::Function { params, .. } => {
                write!(f, "fn(")?;
                for (i, _) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "_")?;
                }
                write!(f, ") -> _")
            }
            AxiomType::Reference { mutable, .. } => {
                if *mutable {
                    write!(f, "&mut T")
                } else {
                    write!(f, "&T")
                }
            }
            AxiomType::Pointer { .. } => write!(f, "*T"),
            AxiomType::GenericParam { name } => write!(f, "{name}"),
            AxiomType::Unit => write!(f, "()"),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_resolve() {
        let mut ctx = TypeContext::new();
        let id = ctx.register(AxiomType::Integer(IntegerKind::I32));
        assert_eq!(ctx.resolve(id), &AxiomType::Integer(IntegerKind::I32));
    }

    #[test]
    fn interning_returns_same_id() {
        let mut ctx = TypeContext::new();
        let a = ctx.register(AxiomType::Bool);
        let b = ctx.register(AxiomType::Bool);
        assert_eq!(a, b);
        assert_eq!(ctx.len(), 1);
    }

    #[test]
    fn distinct_types_get_distinct_ids() {
        let mut ctx = TypeContext::new();
        let a = ctx.register(AxiomType::Integer(IntegerKind::I32));
        let b = ctx.register(AxiomType::Integer(IntegerKind::U32));
        assert_ne!(a, b);
    }

    #[test]
    fn numeric_predicates() {
        assert!(AxiomType::Integer(IntegerKind::I64).is_numeric());
        assert!(AxiomType::Float(FloatKind::F64).is_numeric());
        assert!(AxiomType::Simd(SimdKind::Vec4F).is_numeric());
        assert!(!AxiomType::Bool.is_numeric());
    }

    #[test]
    fn signedness() {
        assert!(AxiomType::Integer(IntegerKind::I32).is_signed());
        assert!(!AxiomType::Integer(IntegerKind::U32).is_signed());
        assert!(AxiomType::Integer(IntegerKind::U64).is_unsigned());
    }

    #[test]
    fn simd_sizes() {
        assert_eq!(AxiomType::Simd(SimdKind::Vec4F).size_bytes(), 16);
        assert_eq!(AxiomType::Simd(SimdKind::Vec8F).size_bytes(), 32);
        assert_eq!(AxiomType::Simd(SimdKind::Vec16F).size_bytes(), 64);
    }

    #[test]
    fn alignment_values() {
        assert_eq!(AxiomType::Integer(IntegerKind::I8).alignment(), 1);
        assert_eq!(AxiomType::Integer(IntegerKind::I32).alignment(), 4);
        assert_eq!(AxiomType::Simd(SimdKind::Vec16F).alignment(), 64);
    }

    #[test]
    fn context_delegates() {
        let mut ctx = TypeContext::new();
        let id = ctx.register(AxiomType::Float(FloatKind::F64));
        assert!(ctx.is_numeric(id));
        assert!(ctx.is_float(id));
        assert!(!ctx.is_integer(id));
    }

    #[test]
    fn array_size_through_context() {
        let mut ctx = TypeContext::new();
        let elem = ctx.register(AxiomType::Integer(IntegerKind::I32));
        let arr = ctx.register(AxiomType::Array {
            element: elem,
            length: 10,
        });
        assert_eq!(ctx.size_of(arr), 40);
        assert_eq!(ctx.align_of(arr), 4);
    }

    #[test]
    fn monomorphize_generic() {
        let mut ctx = TypeContext::new();
        let t_param = ctx.register(AxiomType::GenericParam {
            name: "T".to_string(),
        });
        let i32_ty = ctx.register(AxiomType::Integer(IntegerKind::I32));

        let mut subs = HashMap::new();
        subs.insert("T".to_string(), i32_ty);

        let result = ctx.monomorphize(t_param, &subs);
        assert_eq!(result, i32_ty);
    }

    #[test]
    fn monomorphize_array_of_generic() {
        let mut ctx = TypeContext::new();
        let t_param = ctx.register(AxiomType::GenericParam {
            name: "T".to_string(),
        });
        let arr = ctx.register(AxiomType::Array {
            element: t_param,
            length: 5,
        });
        let f64_ty = ctx.register(AxiomType::Float(FloatKind::F64));

        let mut subs = HashMap::new();
        subs.insert("T".to_string(), f64_ty);

        let result = ctx.monomorphize(arr, &subs);
        let expected = ctx.register(AxiomType::Array {
            element: f64_ty,
            length: 5,
        });
        assert_eq!(result, expected);
    }

    #[test]
    fn monomorphize_function_type() {
        let mut ctx = TypeContext::new();
        let t = ctx.register(AxiomType::GenericParam {
            name: "T".to_string(),
        });
        let func = ctx.register(AxiomType::Function {
            params: vec![t, t],
            return_type: t,
        });
        let i64_ty = ctx.register(AxiomType::Integer(IntegerKind::I64));

        let mut subs = HashMap::new();
        subs.insert("T".to_string(), i64_ty);

        let result = ctx.monomorphize(func, &subs);
        let expected = ctx.register(AxiomType::Function {
            params: vec![i64_ty, i64_ty],
            return_type: i64_ty,
        });
        assert_eq!(result, expected);
    }

    #[test]
    fn display_primitives() {
        assert_eq!(AxiomType::Integer(IntegerKind::I32).to_string(), "i32");
        assert_eq!(AxiomType::Float(FloatKind::F64).to_string(), "f64");
        assert_eq!(AxiomType::Bool.to_string(), "bool");
        assert_eq!(AxiomType::Unit.to_string(), "()");
        assert_eq!(AxiomType::Simd(SimdKind::Vec8F).to_string(), "vec8f");
    }

    #[test]
    fn types_equal_via_context() {
        let mut ctx = TypeContext::new();
        let a = ctx.register(AxiomType::Char);
        let b = ctx.register(AxiomType::Char);
        assert!(ctx.types_equal(a, b));
    }

    #[test]
    fn default_context_is_empty() {
        let ctx = TypeContext::default();
        assert!(ctx.is_empty());
        assert_eq!(ctx.len(), 0);
    }

    #[test]
    fn monomorphize_leaves_concrete_unchanged() {
        let mut ctx = TypeContext::new();
        let i32_ty = ctx.register(AxiomType::Integer(IntegerKind::I32));
        let subs = HashMap::new();
        let result = ctx.monomorphize(i32_ty, &subs);
        assert_eq!(result, i32_ty);
    }
}