# AXIOM

**A statically compiled, Python-syntax systems language with equality-saturation optimization.**

Axiom combines the readability of Python with the performance of C — no braces,
no semicolons, just fast code.

## Key Features

- **Python-like syntax** — indentation-based blocks, clean and readable.
- **Static typing** — full type inference with explicit annotations where it matters.
- **E-graph optimization** — equality-saturation rewriting discovers algebraically optimal code.
- **Auto-vectorization** — SIMD intrinsics and loop vectorization built into the optimizer.
- **Explicit memory control** — stack by default, explicit heap, borrow-aware safety.
- **Multi-target** — x86-64 (primary), AArch64, and WebAssembly backends.

## Quick Start

```bash
# Build the compiler
cargo build --release

# Compile an Axiom source file
./target/release/axiom examples/hello.ax -o hello.s

# Dump intermediate representations
./target/release/axiom examples/fibonacci.ax --emit tokens
./target/release/axiom examples/fibonacci.ax --emit ast
./target/release/axiom examples/fibonacci.ax --emit hir
./target/release/axiom examples/fibonacci.ax --emit mir
./target/release/axiom examples/fibonacci.ax --emit asm

# Set optimization level (0–3, default 2)
./target/release/axiom examples/matrix.ax --opt-level 3
```

## Example

```python
fn fib(n: i64) -> i64:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

fn main():
    for i in 0..20:
        println(fib(i))
```

## Compiler Pipeline

```
Source (.ax)
  │
  ▼
┌─────────────┐
│  1. Lexer   │  Tokenize source → Token stream
└─────┬───────┘
      ▼
┌─────────────┐
│  2. Parser  │  Token stream → AST
└─────┬───────┘
      ▼
┌─────────────┐
│  3. HIR     │  AST → typed graph IR (HirId per node)
└─────┬───────┘
      ▼
┌─────────────────────────────────────────────┐
│  4. Optimizer                               │
│     • Peephole rewrites (strength reduce,   │
│       constant folding)                     │
│     • Loop transforms (unroll, fuse, tile)  │
│     • E-graph / equality saturation         │
│     • Superoptimizer (O3 only)              │
└─────────────┬───────────────────────────────┘
              ▼
┌─────────────┐
│  5. MIR     │  HIR → SSA-based mid-level IR
└─────┬───────┘
      ▼
┌─────────────┐
│  6. Codegen │  MIR → x86-64 AT&T assembly
└─────────────┘
```

## Building from Source

```bash
# Prerequisites: Rust 1.70+
cargo build            # debug build
cargo build --release  # optimized build
```

## Running Tests

```bash
cargo test
```

## Project Structure

```
src/
├── lib.rs           # Crate root – re-exports all modules
├── main.rs          # CLI entry point
├── lexer/           # Tokenizer (INDENT/DEDENT, keywords, operators)
├── parser/          # Recursive-descent parser → AST
├── types/           # Type system (TypeContext, AxiomType, TypeId)
├── hir/             # High-level IR with unique HirId nodes
├── egraph/          # Equality-saturation optimizer (egg-based)
├── optimizer/       # Peephole, loop, and superoptimization passes
├── mir/             # Mid-level SSA IR with basic blocks
├── codegen/         # x86-64 assembly code generation
└── stdlib/          # Standard library function stubs

examples/            # Example .ax programs
docs/                # Language specification
```

## License

MIT