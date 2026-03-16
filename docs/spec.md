# Axiom Language Specification

> **Version 0.1 – Draft**

## 1 Overview

Axiom is a statically compiled, systems-level programming language that combines
Python-like indentation syntax with zero-cost abstractions, explicit memory
control, and aggressive compiler optimizations powered by equality saturation
(e-graphs).

### 1.1 Design Goals

| Goal | Description |
|------|-------------|
| **Performance** | Match or exceed hand-tuned C/C++ on numerical workloads. |
| **Ergonomics** | Python-style syntax – no braces, no semicolons. |
| **Safety** | Static types, borrow-aware memory model, bounds checking. |
| **Optimisation** | E-graph / equality-saturation rewriting at the HIR level. |
| **Portability** | Targets x86-64 (primary), AArch64, and WebAssembly. |

---

## 2 Syntax

### 2.1 Basics

* **Indentation-based blocks** – 4-space indent opens a block; dedent closes it.
* **Comments** – line comments start with `#`.
* **Identifiers** – `[a-zA-Z_][a-zA-Z0-9_]*`.
* **Statement separator** – newline; no semicolons.

### 2.2 Literals

| Kind | Examples |
|------|----------|
| Integer | `42`, `0xFF`, `0b1010`, `1_000_000` |
| Float | `3.14`, `1e-6`, `6.022e23` |
| Boolean | `true`, `false` |
| String | `"hello"`, `"line\n"` |

---

## 3 Type System

### 3.1 Primitive Types

| Type | Width | Description |
|------|-------|-------------|
| `i8`..`i64` | 8–64 bits | Signed integers |
| `u8`..`u64` | 8–64 bits | Unsigned integers |
| `f32`, `f64` | 32/64 bits | IEEE 754 floating-point |
| `bool` | 1 byte | Boolean |
| `str` | varies | UTF-8 string (immutable) |
| `unit` | 0 | No value (void equivalent) |

### 3.2 Compound Types

```
# Fixed-size array
let a: [4]f64 = [1.0, 2.0, 3.0, 4.0]

# Dynamic slice
let s: []i32 = vec_new()

# Struct
struct Vec3:
    x: f64
    y: f64
    z: f64

# Enum
enum Shape:
    Circle(f64)
    Rect(f64, f64)
```

### 3.3 Generics

```
fn max<T: Ord>(a: T, b: T) -> T:
    if a > b:
        return a
    return b
```

---

## 4 Memory Model

* **Stack by default** – local variables and small structs live on the stack.
* **Explicit heap** – `memory.alloc(size)` / `memory.dealloc(ptr)`.
* **Borrowing** – references use `&` (immutable) and `&mut` (mutable).
  The compiler enforces single-writer / multiple-reader at compile time.

```
fn sum(data: &[]f64) -> f64:
    s := 0.0
    for x in data:
        s += x
    return s
```

---

## 5 Control Flow

### 5.1 Conditional

```
if x > 0:
    println("positive")
elif x == 0:
    println("zero")
else:
    println("negative")
```

### 5.2 Loops

```
# Range-based for
for i in 0..n:
    process(i)

# While
while cond:
    step()
```

### 5.3 Match

```
match shape:
    Circle(r):
        return pi() * r * r
    Rect(w, h):
        return w * h
```

---

## 6 Functions

```
fn add(a: f64, b: f64) -> f64:
    return a + b
```

### 6.1 Annotations

| Annotation | Meaning |
|------------|---------|
| `@inline` | Hint to always inline the function |
| `@pure` | Function has no side effects |
| `@simd` | Enable auto-vectorization for this function |

```
@pure
@simd
fn dot(a: []f64, b: []f64) -> f64:
    sum := 0.0
    for i in 0..len(a):
        sum += a[i] * b[i]
    return sum
```

---

## 7 Modules and Imports

```
import math
import io.{println, read_line}

# Qualified access
let angle = math.atan2(y, x)
```

Every `.ax` file is a module. The standard library modules (`math`, `io`,
`memory`, `simd`, `collections`, `threads`, `filesystem`) are implicitly
available.

---

## 8 Standard Library

| Module | Purpose | Key functions |
|--------|---------|---------------|
| `math` | Numeric operations | `abs`, `sqrt`, `sin`, `cos`, `pow`, `log`, `min`, `max`, `clamp` |
| `io` | Input / output | `print`, `println`, `read_line`, `read_file`, `write_file` |
| `memory` | Heap management | `alloc`, `dealloc`, `copy`, `zero`, `size_of` |
| `simd` | SIMD intrinsics | `vec4f_add`, `vec8f_mul`, `dot4`, `dot8` |
| `collections` | Data structures | `vec_new`, `vec_push`, `map_new`, `map_insert` |
| `threads` | Concurrency | `spawn`, `join`, `mutex_new`, `channel_new` |
| `filesystem` | File system | `open`, `close`, `read`, `write`, `exists`, `mkdir` |

---

## 9 Compiler Pipeline

The Axiom compiler processes source code through **six major stages**:

```
Source (.ax)
  │
  ▼
┌─────────────┐
│  1. Lexer   │  Tokenizes source → Token stream
└─────┬───────┘
      ▼
┌─────────────┐
│  2. Parser  │  Token stream → AST
└─────┬───────┘
      ▼
┌─────────────┐
│  3. HIR     │  AST → typed, graph-style HIR
│   Lowering  │  with unique HirId per node
└─────┬───────┘
      ▼
┌─────────────────────────────────────────────┐
│  4. Optimiser                               │
│  ┌──────────────────┐  ┌──────────────────┐ │
│  │ Peephole rules   │  │ Loop transforms  │ │
│  │ (strength reduce,│  │ (unroll, fuse,   │ │
│  │  const fold)     │  │  tile, vectorize)│ │
│  └──────────────────┘  └──────────────────┘ │
│  ┌──────────────────────────────────────┐   │
│  │ E-graph / equality-saturation pass   │   │
│  │ (algebraic rewrites, cost-guided     │   │
│  │  extraction)                         │   │
│  └──────────────────────────────────────┘   │
│  ┌──────────────────┐                       │
│  │ Superoptimizer   │  (O3 only)            │
│  └──────────────────┘                       │
└─────────────┬───────────────────────────────┘
              ▼
┌─────────────┐
│  5. MIR     │  HIR → SSA-based MIR with
│   Lowering  │  machine-level types & blocks
└─────┬───────┘
      ▼
┌─────────────┐
│  6. Codegen │  MIR → x86-64 AT&T assembly
└─────┬───────┘
      ▼
   out.s
```

### 9.1 Optimisation Levels

| Flag | Passes |
|------|--------|
| `-O0` | No optimizations |
| `-O1` | Peephole rules only |
| `-O2` | Peephole + loop transforms + e-graph rewriting |
| `-O3` | All of the above + superoptimization on small blocks |

---

## 10 Example Programs

### Hello World

```axiom
fn main():
    println("Hello, Axiom!")
```

### Fibonacci

```axiom
fn fib(n: i64) -> i64:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

fn main():
    for i in 0..20:
        println(fib(i))
```

### Matrix Multiply

```axiom
fn matmul(a: []f64, b: []f64, c: []f64, n: i64):
    for i in 0..n:
        for j in 0..n:
            sum := 0.0
            for k in 0..n:
                sum += a[i * n + k] * b[k * n + j]
            c[i * n + j] = sum
```
