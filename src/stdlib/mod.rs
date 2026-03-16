//! Standard library module definitions for the Axiom language.
//!
//! These represent built-in functions and types available to all Axiom programs.
//! The [`StdLib`] registry is queried during semantic analysis to resolve calls
//! to standard library functions and validate their signatures.

/// A standard library module with its functions.
#[derive(Debug, Clone)]
pub struct StdModule {
    pub name: String,
    pub functions: Vec<StdFunction>,
}

/// A standard library function signature.
#[derive(Debug, Clone)]
pub struct StdFunction {
    pub name: String,
    /// `(param_name, type_name)` pairs.
    pub params: Vec<(String, String)>,
    pub return_type: String,
    pub description: String,
}

/// Registry of all standard library modules.
#[derive(Debug, Clone)]
pub struct StdLib {
    pub modules: Vec<StdModule>,
}

impl StdLib {
    /// Create a new standard library registry with all built-in modules.
    pub fn new() -> Self {
        Self {
            modules: vec![
                Self::math_module(),
                Self::io_module(),
                Self::memory_module(),
                Self::simd_module(),
                Self::collections_module(),
                Self::threads_module(),
                Self::filesystem_module(),
            ],
        }
    }

    /// Look up a function by module and function name.
    pub fn lookup_function(&self, module: &str, name: &str) -> Option<&StdFunction> {
        self.lookup_module(module)
            .and_then(|m| m.functions.iter().find(|f| f.name == name))
    }

    /// Look up a module by name.
    pub fn lookup_module(&self, name: &str) -> Option<&StdModule> {
        self.modules.iter().find(|m| m.name == name)
    }

    // -- module builders ----------------------------------------------------

    fn f(
        name: &str,
        params: &[(&str, &str)],
        ret: &str,
        desc: &str,
    ) -> StdFunction {
        StdFunction {
            name: name.to_string(),
            params: params.iter().map(|(n, t)| (n.to_string(), t.to_string())).collect(),
            return_type: ret.to_string(),
            description: desc.to_string(),
        }
    }

    fn math_module() -> StdModule {
        StdModule {
            name: "math".to_string(),
            functions: vec![
                Self::f("abs",   &[("x", "f64")],                "f64", "Absolute value"),
                Self::f("sqrt",  &[("x", "f64")],                "f64", "Square root"),
                Self::f("sin",   &[("x", "f64")],                "f64", "Sine"),
                Self::f("cos",   &[("x", "f64")],                "f64", "Cosine"),
                Self::f("tan",   &[("x", "f64")],                "f64", "Tangent"),
                Self::f("asin",  &[("x", "f64")],                "f64", "Inverse sine"),
                Self::f("acos",  &[("x", "f64")],                "f64", "Inverse cosine"),
                Self::f("atan",  &[("x", "f64")],                "f64", "Inverse tangent"),
                Self::f("atan2", &[("y", "f64"), ("x", "f64")],  "f64", "Two-argument inverse tangent"),
                Self::f("pow",   &[("base", "f64"), ("exp", "f64")], "f64", "Exponentiation"),
                Self::f("exp",   &[("x", "f64")],                "f64", "e raised to the power x"),
                Self::f("log",   &[("x", "f64")],                "f64", "Natural logarithm"),
                Self::f("log2",  &[("x", "f64")],                "f64", "Base-2 logarithm"),
                Self::f("log10", &[("x", "f64")],                "f64", "Base-10 logarithm"),
                Self::f("floor", &[("x", "f64")],                "f64", "Floor"),
                Self::f("ceil",  &[("x", "f64")],                "f64", "Ceiling"),
                Self::f("round", &[("x", "f64")],                "f64", "Round to nearest integer"),
                Self::f("min",   &[("a", "f64"), ("b", "f64")],  "f64", "Minimum of two values"),
                Self::f("max",   &[("a", "f64"), ("b", "f64")],  "f64", "Maximum of two values"),
                Self::f("clamp", &[("x", "f64"), ("lo", "f64"), ("hi", "f64")], "f64", "Clamp x between lo and hi"),
                Self::f("pi",    &[],                             "f64", "The constant π"),
                Self::f("e",     &[],                             "f64", "Euler's number e"),
            ],
        }
    }

    fn io_module() -> StdModule {
        StdModule {
            name: "io".to_string(),
            functions: vec![
                Self::f("print",     &[("s", "str")],                      "unit", "Print a string to stdout"),
                Self::f("println",   &[("s", "str")],                      "unit", "Print a string followed by newline"),
                Self::f("eprint",    &[("s", "str")],                      "unit", "Print to stderr"),
                Self::f("eprintln",  &[("s", "str")],                      "unit", "Print to stderr with newline"),
                Self::f("read_line", &[],                                   "str",  "Read a line from stdin"),
                Self::f("read_file", &[("path", "str")],                   "str",  "Read entire file to string"),
                Self::f("write_file",&[("path", "str"), ("data", "str")],  "unit", "Write string to file"),
            ],
        }
    }

    fn memory_module() -> StdModule {
        StdModule {
            name: "memory".to_string(),
            functions: vec![
                Self::f("alloc",    &[("size", "u64")],                                "ptr",  "Allocate size bytes on the heap"),
                Self::f("dealloc",  &[("p", "ptr")],                                   "unit", "Deallocate heap memory"),
                Self::f("copy",     &[("dst", "ptr"), ("src", "ptr"), ("n", "u64")],   "unit", "Copy n bytes from src to dst"),
                Self::f("zero",     &[("p", "ptr"), ("n", "u64")],                     "unit", "Zero n bytes starting at p"),
                Self::f("size_of",  &[("ty", "type")],                                 "u64",  "Size of a type in bytes"),
                Self::f("align_of", &[("ty", "type")],                                 "u64",  "Alignment of a type in bytes"),
            ],
        }
    }

    fn simd_module() -> StdModule {
        StdModule {
            name: "simd".to_string(),
            functions: vec![
                Self::f("vec4f_add",   &[("a", "vec4f"), ("b", "vec4f")],   "vec4f", "Add two 4×f32 vectors"),
                Self::f("vec4f_mul",   &[("a", "vec4f"), ("b", "vec4f")],   "vec4f", "Multiply two 4×f32 vectors"),
                Self::f("vec4f_load",  &[("p", "ptr")],                     "vec4f", "Load 4×f32 from aligned pointer"),
                Self::f("vec4f_store", &[("p", "ptr"), ("v", "vec4f")],     "unit",  "Store 4×f32 to aligned pointer"),
                Self::f("vec8f_add",   &[("a", "vec8f"), ("b", "vec8f")],   "vec8f", "Add two 8×f32 vectors"),
                Self::f("vec8f_mul",   &[("a", "vec8f"), ("b", "vec8f")],   "vec8f", "Multiply two 8×f32 vectors"),
                Self::f("dot4",        &[("a", "vec4f"), ("b", "vec4f")],   "f32",   "Dot product of two 4×f32 vectors"),
                Self::f("dot8",        &[("a", "vec8f"), ("b", "vec8f")],   "f32",   "Dot product of two 8×f32 vectors"),
            ],
        }
    }

    fn collections_module() -> StdModule {
        StdModule {
            name: "collections".to_string(),
            functions: vec![
                Self::f("vec_new",    &[],                                        "vec",  "Create a new empty vector"),
                Self::f("vec_push",   &[("v", "vec"), ("val", "any")],            "unit", "Push a value onto the vector"),
                Self::f("vec_pop",    &[("v", "vec")],                            "any",  "Pop the last element"),
                Self::f("vec_len",    &[("v", "vec")],                            "u64",  "Return the length of the vector"),
                Self::f("vec_get",    &[("v", "vec"), ("idx", "u64")],            "any",  "Get element at index"),
                Self::f("vec_set",    &[("v", "vec"), ("idx", "u64"), ("val", "any")], "unit", "Set element at index"),
                Self::f("map_new",    &[],                                        "map",  "Create a new empty hash map"),
                Self::f("map_insert", &[("m", "map"), ("key", "any"), ("val", "any")], "unit", "Insert a key-value pair"),
                Self::f("map_get",    &[("m", "map"), ("key", "any")],            "any",  "Look up a value by key"),
                Self::f("map_remove", &[("m", "map"), ("key", "any")],            "unit", "Remove a key-value pair"),
            ],
        }
    }

    fn threads_module() -> StdModule {
        StdModule {
            name: "threads".to_string(),
            functions: vec![
                Self::f("spawn",         &[("func", "fn")],              "handle", "Spawn a new thread running func"),
                Self::f("join",          &[("h", "handle")],             "unit",   "Wait for a thread to finish"),
                Self::f("mutex_new",     &[],                            "mutex",  "Create a new mutex"),
                Self::f("mutex_lock",    &[("m", "mutex")],             "unit",   "Acquire the mutex"),
                Self::f("mutex_unlock",  &[("m", "mutex")],             "unit",   "Release the mutex"),
                Self::f("channel_new",   &[],                            "channel","Create a new channel pair"),
                Self::f("channel_send",  &[("ch", "channel"), ("val", "any")], "unit", "Send a value on the channel"),
                Self::f("channel_recv",  &[("ch", "channel")],          "any",    "Receive a value from the channel"),
            ],
        }
    }

    fn filesystem_module() -> StdModule {
        StdModule {
            name: "filesystem".to_string(),
            functions: vec![
                Self::f("open",     &[("path", "str"), ("mode", "str")], "fd",   "Open a file, returns descriptor"),
                Self::f("close",    &[("f", "fd")],                      "unit", "Close a file descriptor"),
                Self::f("read",     &[("f", "fd"), ("n", "u64")],        "str",  "Read up to n bytes"),
                Self::f("write",    &[("f", "fd"), ("data", "str")],     "u64",  "Write data, return bytes written"),
                Self::f("exists",   &[("path", "str")],                  "bool", "Check whether a path exists"),
                Self::f("mkdir",    &[("path", "str")],                  "unit", "Create a directory"),
                Self::f("remove",   &[("path", "str")],                  "unit", "Remove a file or directory"),
                Self::f("list_dir", &[("path", "str")],                  "vec",  "List entries in a directory"),
            ],
        }
    }
}

impl Default for StdLib {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_modules_present() {
        let stdlib = StdLib::new();
        let expected = ["math", "io", "memory", "simd", "collections", "threads", "filesystem"];
        for name in &expected {
            assert!(stdlib.lookup_module(name).is_some(), "missing module: {name}");
        }
        assert_eq!(stdlib.modules.len(), expected.len());
    }

    #[test]
    fn math_function_lookup() {
        let stdlib = StdLib::new();
        let sqrt = stdlib.lookup_function("math", "sqrt").expect("sqrt not found");
        assert_eq!(sqrt.return_type, "f64");
        assert_eq!(sqrt.params.len(), 1);
        assert_eq!(sqrt.params[0].0, "x");
    }

    #[test]
    fn math_all_functions() {
        let stdlib = StdLib::new();
        let math = stdlib.lookup_module("math").unwrap();
        let names: Vec<&str> = math.functions.iter().map(|f| f.name.as_str()).collect();
        for expected in &[
            "abs", "sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
            "pow", "exp", "log", "log2", "log10", "floor", "ceil", "round",
            "min", "max", "clamp", "pi", "e",
        ] {
            assert!(names.contains(expected), "math missing: {expected}");
        }
    }

    #[test]
    fn io_functions() {
        let stdlib = StdLib::new();
        for name in &["print", "println", "eprint", "eprintln", "read_line", "read_file", "write_file"] {
            assert!(stdlib.lookup_function("io", name).is_some(), "io missing: {name}");
        }
    }

    #[test]
    fn memory_functions() {
        let stdlib = StdLib::new();
        for name in &["alloc", "dealloc", "copy", "zero", "size_of", "align_of"] {
            assert!(stdlib.lookup_function("memory", name).is_some(), "memory missing: {name}");
        }
    }

    #[test]
    fn simd_functions() {
        let stdlib = StdLib::new();
        for name in &["vec4f_add", "vec4f_mul", "vec4f_load", "vec4f_store",
                       "vec8f_add", "vec8f_mul", "dot4", "dot8"] {
            assert!(stdlib.lookup_function("simd", name).is_some(), "simd missing: {name}");
        }
    }

    #[test]
    fn collections_functions() {
        let stdlib = StdLib::new();
        for name in &["vec_new", "vec_push", "vec_pop", "vec_len", "vec_get", "vec_set",
                       "map_new", "map_insert", "map_get", "map_remove"] {
            assert!(stdlib.lookup_function("collections", name).is_some(), "collections missing: {name}");
        }
    }

    #[test]
    fn threads_functions() {
        let stdlib = StdLib::new();
        for name in &["spawn", "join", "mutex_new", "mutex_lock", "mutex_unlock",
                       "channel_new", "channel_send", "channel_recv"] {
            assert!(stdlib.lookup_function("threads", name).is_some(), "threads missing: {name}");
        }
    }

    #[test]
    fn filesystem_functions() {
        let stdlib = StdLib::new();
        for name in &["open", "close", "read", "write", "exists", "mkdir", "remove", "list_dir"] {
            assert!(stdlib.lookup_function("filesystem", name).is_some(), "filesystem missing: {name}");
        }
    }

    #[test]
    fn lookup_nonexistent_module() {
        let stdlib = StdLib::new();
        assert!(stdlib.lookup_module("nope").is_none());
    }

    #[test]
    fn lookup_nonexistent_function() {
        let stdlib = StdLib::new();
        assert!(stdlib.lookup_function("math", "nope").is_none());
        assert!(stdlib.lookup_function("nope", "abs").is_none());
    }

    #[test]
    fn clamp_has_three_params() {
        let stdlib = StdLib::new();
        let clamp = stdlib.lookup_function("math", "clamp").unwrap();
        assert_eq!(clamp.params.len(), 3);
    }

    #[test]
    fn default_trait() {
        let stdlib = StdLib::default();
        assert!(!stdlib.modules.is_empty());
    }
}
