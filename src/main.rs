use std::env;
use std::fs;
use std::process;

use axiom::codegen::{CodeGenerator, Target};
use axiom::hir::AstLowering;
use axiom::lexer::Lexer;
use axiom::mir::MirBuilder;
use axiom::optimizer::{OptLevel, OptimizationPipeline};
use axiom::parser::Parser;

fn usage() {
    eprintln!(
        "\
Axiom compiler – a statically compiled, Python-syntax systems language

USAGE:
    axiom <input.ax> [OPTIONS]

OPTIONS:
    --emit <stage>       Dump an intermediate representation and exit.
                         Stages: tokens, ast, hir, mir, asm
    --opt-level <N>      Optimization level 0–3 (default: 2)
    -o <output>          Output file (default: out.s)
    --help               Show this help message"
    );
}

#[derive(Debug)]
struct Args {
    input: String,
    emit: Option<String>,
    opt_level: OptLevel,
    output: String,
}

fn parse_args() -> Option<Args> {
    let args: Vec<String> = env::args().skip(1).collect();

    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        return None;
    }

    let mut input: Option<String> = None;
    let mut emit: Option<String> = None;
    let mut opt_level = OptLevel::O2;
    let mut output = String::from("out.s");

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--emit" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --emit requires an argument");
                    process::exit(1);
                }
                let stage = &args[i];
                match stage.as_str() {
                    "tokens" | "ast" | "hir" | "mir" | "asm" => {
                        emit = Some(stage.clone());
                    }
                    _ => {
                        eprintln!("error: unknown emit stage '{stage}'. Expected: tokens, ast, hir, mir, asm");
                        process::exit(1);
                    }
                }
            }
            "--opt-level" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --opt-level requires an argument");
                    process::exit(1);
                }
                opt_level = match args[i].as_str() {
                    "0" => OptLevel::O0,
                    "1" => OptLevel::O1,
                    "2" => OptLevel::O2,
                    "3" => OptLevel::O3,
                    other => {
                        eprintln!("error: invalid optimization level '{other}'. Expected 0–3");
                        process::exit(1);
                    }
                };
            }
            "-o" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: -o requires an argument");
                    process::exit(1);
                }
                output = args[i].clone();
            }
            arg if arg.starts_with('-') => {
                eprintln!("error: unknown option '{arg}'");
                process::exit(1);
            }
            _ => {
                if input.is_some() {
                    eprintln!("error: multiple input files are not supported");
                    process::exit(1);
                }
                input = Some(args[i].clone());
            }
        }
        i += 1;
    }

    let input = match input {
        Some(p) => p,
        None => {
            eprintln!("error: no input file specified");
            return None;
        }
    };

    Some(Args { input, emit, opt_level, output })
}

fn main() {
    let args = match parse_args() {
        Some(a) => a,
        None => {
            usage();
            process::exit(1);
        }
    };

    // -- read source --------------------------------------------------------
    let source = match fs::read_to_string(&args.input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read '{}': {e}", args.input);
            process::exit(1);
        }
    };

    // -- lex ----------------------------------------------------------------
    eprintln!("[1/6] Lexing ...");
    let tokens = match Lexer::tokenize(&source) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Lex error: {e}");
            process::exit(1);
        }
    };
    eprintln!("       {} tokens", tokens.len());

    if args.emit.as_deref() == Some("tokens") {
        for tok in &tokens {
            println!("{:?}", tok);
        }
        return;
    }

    // -- parse --------------------------------------------------------------
    eprintln!("[2/6] Parsing ...");
    let ast = match Parser::parse(tokens) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Parse error: {e}");
            process::exit(1);
        }
    };
    eprintln!("       {} top-level items", ast.items.len());

    if args.emit.as_deref() == Some("ast") {
        println!("{:#?}", ast);
        return;
    }

    // -- lower to HIR -------------------------------------------------------
    eprintln!("[3/6] Lowering to HIR ...");
    let mut hir_program = match AstLowering::lower(&ast) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("HIR lowering error: {e}");
            process::exit(1);
        }
    };
    eprintln!("       {} functions", hir_program.functions.len());

    if args.emit.as_deref() == Some("hir") {
        println!("{:#?}", hir_program);
        return;
    }

    // -- optimize -----------------------------------------------------------
    eprintln!("[4/6] Optimizing (level {:?}) ...", args.opt_level);
    let pipeline = OptimizationPipeline::new(args.opt_level);
    pipeline.optimize(&mut hir_program);

    // -- lower to MIR -------------------------------------------------------
    eprintln!("[5/6] Lowering to MIR ...");
    let mir_program = match MirBuilder::lower_program(&hir_program) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("MIR lowering error: {e}");
            process::exit(1);
        }
    };
    eprintln!("       {} MIR functions", mir_program.functions.len());

    if args.emit.as_deref() == Some("mir") {
        println!("{:#?}", mir_program);
        return;
    }

    // -- codegen ------------------------------------------------------------
    eprintln!("[6/6] Generating x86-64 assembly ...");
    let mut codegen = CodeGenerator::new(Target::X86_64);
    let asm = match codegen.generate(&mir_program) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Codegen error: {e}");
            process::exit(1);
        }
    };

    if args.emit.as_deref() == Some("asm") {
        println!("{asm}");
        return;
    }

    // -- write output -------------------------------------------------------
    if let Err(e) = fs::write(&args.output, &asm) {
        eprintln!("error: cannot write '{}': {e}", args.output);
        process::exit(1);
    }
    eprintln!("       Wrote {} bytes to {}", asm.len(), args.output);
}
