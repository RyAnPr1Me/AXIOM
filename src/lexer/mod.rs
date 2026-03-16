/// Lexer for the Axiom programming language.
///
/// Tokenizes source code into a stream of tokens including support for
/// indentation-based blocks (INDENT/DEDENT), Python-style comments (#),
/// and all Axiom operators, keywords, and literals.

use std::fmt;

// ---------------------------------------------------------------------------
// Span
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub line: usize,
    pub col: usize,
}

impl Span {
    pub fn new(line: usize, col: usize) -> Self {
        Self { line, col }
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}

// ---------------------------------------------------------------------------
// LexError
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct LexError {
    pub message: String,
    pub span: Span,
}

impl LexError {
    pub fn new(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
        }
    }
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Lex error at {}: {}", self.span, self.message)
    }
}

impl std::error::Error for LexError {}

// ---------------------------------------------------------------------------
// TokenKind
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Literals
    Identifier,
    IntLiteral,
    FloatLiteral,
    StringLiteral,
    CharLiteral,
    BoolTrue,
    BoolFalse,

    // Keywords
    Fn,
    Return,
    If,
    Elif,
    Else,
    For,
    While,
    In,
    Let,
    Mut,
    Struct,
    Enum,
    Import,
    New,
    Break,
    Continue,
    Match,
    Pub,
    SelfKw,

    // Annotations
    Annotation,

    // Operators – arithmetic
    Plus,
    Minus,
    Star,
    Slash,
    Percent,

    // Operators – comparison
    EqEq,
    NotEq,
    Lt,
    Gt,
    LtEq,
    GtEq,

    // Operators – assignment
    Eq,
    ColonEq,
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,

    // Operators – bitwise
    Ampersand,
    Pipe,
    Caret,
    Tilde,
    Shl,
    Shr,

    // Operators – logical
    AmpAmp,
    PipePipe,
    Bang,

    // Operators – misc
    DotDot,
    Arrow,

    // Delimiters
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Colon,
    Comma,
    Dot,

    // Layout
    Newline,
    Indent,
    Dedent,

    // End of file
    Eof,
}

// ---------------------------------------------------------------------------
// Token
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    pub text: String,
}

impl Token {
    pub fn new(kind: TokenKind, span: Span, text: impl Into<String>) -> Self {
        Self {
            kind,
            span,
            text: text.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

pub struct Lexer<'src> {
    src: &'src [u8],
    pos: usize,
    line: usize,
    col: usize,
    /// Stack of indentation levels (measured in columns). Always starts with 0.
    indent_stack: Vec<usize>,
    /// Pending tokens that were generated during indent/dedent processing and
    /// need to be emitted before the next real token.
    pending: Vec<Token>,
    /// Whether we are at the beginning of a logical line (for indent tracking).
    at_line_start: bool,
}

impl<'src> Lexer<'src> {
    // -- public API ---------------------------------------------------------

    /// Tokenize the full input string, returning a vector of tokens or a lex
    /// error on the first problem encountered.
    pub fn tokenize(input: &str) -> Result<Vec<Token>, LexError> {
        let mut lexer = Lexer {
            src: input.as_bytes(),
            pos: 0,
            line: 1,
            col: 1,
            indent_stack: vec![0],
            pending: Vec::new(),
            at_line_start: true,
        };

        let mut tokens: Vec<Token> = Vec::new();

        loop {
            let tok = lexer.next_token()?;
            let is_eof = tok.kind == TokenKind::Eof;
            tokens.push(tok);
            if is_eof {
                break;
            }
        }

        Ok(tokens)
    }

    // -- cursor helpers -----------------------------------------------------

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn peek_ahead(&self, offset: usize) -> Option<u8> {
        self.src.get(self.pos + offset).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let ch = self.peek()?;
        self.pos += 1;
        if ch == b'\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(ch)
    }

    fn span(&self) -> Span {
        Span::new(self.line, self.col)
    }

    fn err(&self, msg: impl Into<String>) -> LexError {
        LexError::new(msg, self.span())
    }

    fn at_end(&self) -> bool {
        self.pos >= self.src.len()
    }

    // -- indentation --------------------------------------------------------

    /// Measure the leading whitespace of the current line and emit
    /// INDENT / DEDENT tokens as appropriate.  Should only be called when
    /// `self.at_line_start` is true.
    fn handle_indentation(&mut self) -> Result<(), LexError> {
        let indent_span = Span::new(self.line, 1);

        // Count leading whitespace. Each space counts as 1 column; each tab
        // counts as 4 columns.  The Axiom style guide mandates spaces only,
        // but tabs are handled gracefully.
        let mut level: usize = 0;
        while self.peek() == Some(b' ') {
            self.advance();
            level += 1;
        }
        while self.peek() == Some(b'\t') {
            self.advance();
            level += 4;
        }

        // Skip blank lines and comment-only lines – they don't affect indentation.
        match self.peek() {
            Some(b'\n') | Some(b'#') | None => {
                self.at_line_start = false;
                return Ok(());
            }
            Some(b'\r') => {
                if self.peek_ahead(1) == Some(b'\n') {
                    self.at_line_start = false;
                    return Ok(());
                }
            }
            _ => {}
        }

        let current = *self.indent_stack.last().unwrap();

        if level > current {
            self.indent_stack.push(level);
            self.pending.push(Token::new(
                TokenKind::Indent,
                indent_span,
                "",
            ));
        } else {
            while level < *self.indent_stack.last().unwrap() {
                self.indent_stack.pop();
                if level > *self.indent_stack.last().unwrap() {
                    return Err(LexError::new(
                        "unindent does not match any outer indentation level",
                        indent_span,
                    ));
                }
                self.pending.push(Token::new(
                    TokenKind::Dedent,
                    indent_span,
                    "",
                ));
            }
        }

        self.at_line_start = false;
        Ok(())
    }

    // -- main dispatch ------------------------------------------------------

    fn next_token(&mut self) -> Result<Token, LexError> {
        // Return any pending INDENT/DEDENT tokens before lexing new tokens.
        if !self.pending.is_empty() {
            return Ok(self.pending.remove(0));
        }

        // Handle indentation at the start of a line.
        if self.at_line_start {
            self.handle_indentation()?;
            // If indentation processing produced tokens, return the first one.
            if !self.pending.is_empty() {
                return Ok(self.pending.remove(0));
            }
        }

        // Skip spaces / tabs within a line (but not newlines).
        self.skip_inline_whitespace();

        // Skip comments.
        if self.peek() == Some(b'#') {
            self.skip_comment();
        }

        if self.at_end() {
            // Emit remaining DEDENTs.
            let span = self.span();
            while self.indent_stack.len() > 1 {
                self.indent_stack.pop();
                self.pending.push(Token::new(TokenKind::Dedent, span, ""));
            }
            if !self.pending.is_empty() {
                return Ok(self.pending.remove(0));
            }
            return Ok(Token::new(TokenKind::Eof, span, ""));
        }

        let ch = self.peek().unwrap();
        let span = self.span();

        match ch {
            // Newlines
            b'\n' => {
                self.advance();
                self.at_line_start = true;
                Ok(Token::new(TokenKind::Newline, span, "\n"))
            }
            b'\r' => {
                self.advance();
                if self.peek() == Some(b'\n') {
                    self.advance();
                }
                self.at_line_start = true;
                Ok(Token::new(TokenKind::Newline, span, "\n"))
            }

            // String literals
            b'"' => self.lex_string(),

            // Char literals
            b'\'' => self.lex_char(),

            // Annotations
            b'@' => self.lex_annotation(),

            // Numbers
            b'0'..=b'9' => self.lex_number(),

            // Identifiers / keywords
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => self.lex_identifier(),

            // Operators & delimiters
            _ => self.lex_operator_or_delimiter(),
        }
    }

    // -- inline whitespace / comments ---------------------------------------

    fn skip_inline_whitespace(&mut self) {
        while let Some(b' ' | b'\t') = self.peek() {
            self.advance();
        }
    }

    fn skip_comment(&mut self) {
        while let Some(ch) = self.peek() {
            if ch == b'\n' {
                break;
            }
            self.advance();
        }
    }

    // -- strings ------------------------------------------------------------

    fn lex_string(&mut self) -> Result<Token, LexError> {
        let span = self.span();
        self.advance(); // consume opening "
        let mut value = String::new();

        loop {
            match self.peek() {
                None | Some(b'\n') => {
                    return Err(LexError::new("unterminated string literal", span));
                }
                Some(b'"') => {
                    self.advance();
                    break;
                }
                Some(b'\\') => {
                    self.advance();
                    match self.peek() {
                        Some(b'n') => { self.advance(); value.push('\n'); }
                        Some(b't') => { self.advance(); value.push('\t'); }
                        Some(b'r') => { self.advance(); value.push('\r'); }
                        Some(b'\\') => { self.advance(); value.push('\\'); }
                        Some(b'"') => { self.advance(); value.push('"'); }
                        Some(b'0') => { self.advance(); value.push('\0'); }
                        Some(c) => {
                            return Err(LexError::new(
                                format!("unknown escape sequence: \\{}", c as char),
                                self.span(),
                            ));
                        }
                        None => {
                            return Err(LexError::new("unterminated string literal", span));
                        }
                    }
                }
                Some(c) => {
                    self.advance();
                    value.push(c as char);
                }
            }
        }

        Ok(Token::new(TokenKind::StringLiteral, span, value))
    }

    // -- chars --------------------------------------------------------------

    fn lex_char(&mut self) -> Result<Token, LexError> {
        let span = self.span();
        self.advance(); // consume opening '

        let ch = match self.peek() {
            None | Some(b'\n') => {
                return Err(LexError::new("unterminated char literal", span));
            }
            Some(b'\\') => {
                self.advance();
                match self.peek() {
                    Some(b'n') => { self.advance(); '\n' }
                    Some(b't') => { self.advance(); '\t' }
                    Some(b'r') => { self.advance(); '\r' }
                    Some(b'\\') => { self.advance(); '\\' }
                    Some(b'\'') => { self.advance(); '\'' }
                    Some(b'0') => { self.advance(); '\0' }
                    Some(c) => {
                        return Err(LexError::new(
                            format!("unknown escape sequence: \\{}", c as char),
                            self.span(),
                        ));
                    }
                    None => {
                        return Err(LexError::new("unterminated char literal", span));
                    }
                }
            }
            Some(c) => {
                self.advance();
                c as char
            }
        };

        if self.peek() != Some(b'\'') {
            return Err(LexError::new("unterminated char literal", span));
        }
        self.advance(); // consume closing '

        Ok(Token::new(TokenKind::CharLiteral, span, ch.to_string()))
    }

    // -- annotations --------------------------------------------------------

    fn lex_annotation(&mut self) -> Result<Token, LexError> {
        let span = self.span();
        self.advance(); // consume '@'

        if !matches!(self.peek(), Some(b'a'..=b'z' | b'A'..=b'Z' | b'_')) {
            return Err(LexError::new("expected annotation name after '@'", span));
        }

        let start = self.pos;
        while matches!(self.peek(), Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_')) {
            self.advance();
        }

        let name = std::str::from_utf8(&self.src[start..self.pos]).unwrap();
        Ok(Token::new(TokenKind::Annotation, span, format!("@{name}")))
    }

    // -- numbers ------------------------------------------------------------

    fn lex_number(&mut self) -> Result<Token, LexError> {
        let span = self.span();
        let start = self.pos;

        // Check for hex (0x) or binary (0b) prefix.
        if self.peek() == Some(b'0') {
            match self.peek_ahead(1) {
                Some(b'x' | b'X') => return self.lex_hex(span),
                Some(b'b' | b'B') => return self.lex_binary(span),
                _ => {}
            }
        }

        // Decimal integer or float.
        self.eat_digits();

        let mut is_float = false;

        // Fractional part – only if followed by a digit (to avoid consuming `..`).
        if self.peek() == Some(b'.') && matches!(self.peek_ahead(1), Some(b'0'..=b'9')) {
            is_float = true;
            self.advance(); // '.'
            self.eat_digits();
        }

        // Exponent part.
        if matches!(self.peek(), Some(b'e' | b'E')) {
            is_float = true;
            self.advance();
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.advance();
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(LexError::new("expected digit in exponent", self.span()));
            }
            self.eat_digits();
        }

        let text = std::str::from_utf8(&self.src[start..self.pos]).unwrap();
        let kind = if is_float {
            TokenKind::FloatLiteral
        } else {
            TokenKind::IntLiteral
        };

        Ok(Token::new(kind, span, text))
    }

    fn lex_hex(&mut self, span: Span) -> Result<Token, LexError> {
        let start = self.pos;
        self.advance(); // '0'
        self.advance(); // 'x'

        if !matches!(self.peek(), Some(b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F')) {
            return Err(LexError::new("expected hex digit after '0x'", self.span()));
        }

        while matches!(
            self.peek(),
            Some(b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F' | b'_')
        ) {
            self.advance();
        }

        let text = std::str::from_utf8(&self.src[start..self.pos]).unwrap();
        Ok(Token::new(TokenKind::IntLiteral, span, text))
    }

    fn lex_binary(&mut self, span: Span) -> Result<Token, LexError> {
        let start = self.pos;
        self.advance(); // '0'
        self.advance(); // 'b'

        if !matches!(self.peek(), Some(b'0' | b'1')) {
            return Err(LexError::new("expected binary digit after '0b'", self.span()));
        }

        while matches!(self.peek(), Some(b'0' | b'1' | b'_')) {
            self.advance();
        }

        let text = std::str::from_utf8(&self.src[start..self.pos]).unwrap();
        Ok(Token::new(TokenKind::IntLiteral, span, text))
    }

    fn eat_digits(&mut self) {
        while matches!(self.peek(), Some(b'0'..=b'9' | b'_')) {
            self.advance();
        }
    }

    // -- identifiers / keywords ---------------------------------------------

    fn lex_identifier(&mut self) -> Result<Token, LexError> {
        let span = self.span();
        let start = self.pos;

        while matches!(
            self.peek(),
            Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_')
        ) {
            self.advance();
        }

        let text = std::str::from_utf8(&self.src[start..self.pos]).unwrap();

        let kind = match text {
            "fn" => TokenKind::Fn,
            "return" => TokenKind::Return,
            "if" => TokenKind::If,
            "elif" => TokenKind::Elif,
            "else" => TokenKind::Else,
            "for" => TokenKind::For,
            "while" => TokenKind::While,
            "in" => TokenKind::In,
            "let" => TokenKind::Let,
            "mut" => TokenKind::Mut,
            "struct" => TokenKind::Struct,
            "enum" => TokenKind::Enum,
            "import" => TokenKind::Import,
            "true" => TokenKind::BoolTrue,
            "false" => TokenKind::BoolFalse,
            "new" => TokenKind::New,
            "break" => TokenKind::Break,
            "continue" => TokenKind::Continue,
            "match" => TokenKind::Match,
            "pub" => TokenKind::Pub,
            "self" => TokenKind::SelfKw,
            _ => TokenKind::Identifier,
        };

        Ok(Token::new(kind, span, text))
    }

    // -- operators & delimiters ---------------------------------------------

    fn lex_operator_or_delimiter(&mut self) -> Result<Token, LexError> {
        let span = self.span();
        let ch = self.peek().unwrap();
        let next = self.peek_ahead(1);

        // Two-character operators (checked first).
        let two_char = match (ch, next) {
            (b'=', Some(b'=')) => Some(TokenKind::EqEq),
            (b'!', Some(b'=')) => Some(TokenKind::NotEq),
            (b'<', Some(b'=')) => Some(TokenKind::LtEq),
            (b'>', Some(b'=')) => Some(TokenKind::GtEq),
            (b'<', Some(b'<')) => Some(TokenKind::Shl),
            (b'>', Some(b'>')) => Some(TokenKind::Shr),
            (b':', Some(b'=')) => Some(TokenKind::ColonEq),
            (b'+', Some(b'=')) => Some(TokenKind::PlusEq),
            (b'-', Some(b'=')) => Some(TokenKind::MinusEq),
            (b'*', Some(b'=')) => Some(TokenKind::StarEq),
            (b'/', Some(b'=')) => Some(TokenKind::SlashEq),
            (b'&', Some(b'&')) => Some(TokenKind::AmpAmp),
            (b'|', Some(b'|')) => Some(TokenKind::PipePipe),
            (b'.', Some(b'.')) => Some(TokenKind::DotDot),
            (b'-', Some(b'>')) => Some(TokenKind::Arrow),
            _ => None,
        };

        if let Some(kind) = two_char {
            let text = std::str::from_utf8(&self.src[self.pos..self.pos + 2]).unwrap();
            self.advance();
            self.advance();
            return Ok(Token::new(kind, span, text));
        }

        // Single-character operators & delimiters.
        self.advance();
        let (kind, text) = match ch {
            b'+' => (TokenKind::Plus, "+"),
            b'-' => (TokenKind::Minus, "-"),
            b'*' => (TokenKind::Star, "*"),
            b'/' => (TokenKind::Slash, "/"),
            b'%' => (TokenKind::Percent, "%"),
            b'<' => (TokenKind::Lt, "<"),
            b'>' => (TokenKind::Gt, ">"),
            b'=' => (TokenKind::Eq, "="),
            b'!' => (TokenKind::Bang, "!"),
            b'&' => (TokenKind::Ampersand, "&"),
            b'|' => (TokenKind::Pipe, "|"),
            b'^' => (TokenKind::Caret, "^"),
            b'~' => (TokenKind::Tilde, "~"),
            b'(' => (TokenKind::LParen, "("),
            b')' => (TokenKind::RParen, ")"),
            b'[' => (TokenKind::LBracket, "["),
            b']' => (TokenKind::RBracket, "]"),
            b'{' => (TokenKind::LBrace, "{"),
            b'}' => (TokenKind::RBrace, "}"),
            b':' => (TokenKind::Colon, ":"),
            b',' => (TokenKind::Comma, ","),
            b'.' => (TokenKind::Dot, "."),
            _ => {
                return Err(LexError::new(
                    format!("unexpected character: '{}'", ch as char),
                    span,
                ));
            }
        };

        Ok(Token::new(kind, span, text))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(input: &str) -> Vec<TokenKind> {
        Lexer::tokenize(input)
            .unwrap()
            .into_iter()
            .map(|t| t.kind)
            .collect()
    }

    fn texts(input: &str) -> Vec<String> {
        Lexer::tokenize(input)
            .unwrap()
            .into_iter()
            .map(|t| t.text)
            .collect()
    }

    // -- basic tokens -------------------------------------------------------

    #[test]
    fn empty_input() {
        assert_eq!(kinds(""), vec![TokenKind::Eof]);
    }

    #[test]
    fn identifier() {
        assert_eq!(
            kinds("foo"),
            vec![TokenKind::Identifier, TokenKind::Eof]
        );
        assert_eq!(texts("foo")[0], "foo");
    }

    #[test]
    fn keywords() {
        let input = "fn return if elif else for while in let mut struct enum import true false new break continue match pub self";
        let expected = vec![
            TokenKind::Fn, TokenKind::Return, TokenKind::If, TokenKind::Elif,
            TokenKind::Else, TokenKind::For, TokenKind::While, TokenKind::In,
            TokenKind::Let, TokenKind::Mut, TokenKind::Struct, TokenKind::Enum,
            TokenKind::Import, TokenKind::BoolTrue, TokenKind::BoolFalse,
            TokenKind::New, TokenKind::Break, TokenKind::Continue,
            TokenKind::Match, TokenKind::Pub, TokenKind::SelfKw,
            TokenKind::Eof,
        ];
        assert_eq!(kinds(input), expected);
    }

    // -- numbers ------------------------------------------------------------

    #[test]
    fn integer_literals() {
        assert_eq!(kinds("42"), vec![TokenKind::IntLiteral, TokenKind::Eof]);
        assert_eq!(texts("42")[0], "42");
    }

    #[test]
    fn float_literals() {
        assert_eq!(kinds("3.14"), vec![TokenKind::FloatLiteral, TokenKind::Eof]);
        assert_eq!(texts("3.14")[0], "3.14");
    }

    #[test]
    fn float_with_exponent() {
        assert_eq!(kinds("1e10"), vec![TokenKind::FloatLiteral, TokenKind::Eof]);
        assert_eq!(kinds("2.5E-3"), vec![TokenKind::FloatLiteral, TokenKind::Eof]);
    }

    #[test]
    fn hex_literal() {
        assert_eq!(kinds("0xFF"), vec![TokenKind::IntLiteral, TokenKind::Eof]);
        assert_eq!(texts("0xFF")[0], "0xFF");
    }

    #[test]
    fn binary_literal() {
        assert_eq!(kinds("0b1010"), vec![TokenKind::IntLiteral, TokenKind::Eof]);
        assert_eq!(texts("0b1010")[0], "0b1010");
    }

    #[test]
    fn number_with_underscores() {
        assert_eq!(texts("1_000_000")[0], "1_000_000");
        assert_eq!(texts("0xFF_FF")[0], "0xFF_FF");
        assert_eq!(texts("0b1111_0000")[0], "0b1111_0000");
    }

    // -- strings & chars ----------------------------------------------------

    #[test]
    fn string_literal() {
        assert_eq!(
            kinds(r#""hello""#),
            vec![TokenKind::StringLiteral, TokenKind::Eof]
        );
        assert_eq!(texts(r#""hello""#)[0], "hello");
    }

    #[test]
    fn string_escape_sequences() {
        let toks = Lexer::tokenize(r#""\n\t\\\"""#).unwrap();
        assert_eq!(toks[0].text, "\n\t\\\"");
    }

    #[test]
    fn unterminated_string() {
        assert!(Lexer::tokenize(r#""hello"#).is_err());
    }

    #[test]
    fn char_literal() {
        assert_eq!(
            kinds("'a'"),
            vec![TokenKind::CharLiteral, TokenKind::Eof]
        );
        assert_eq!(texts("'a'")[0], "a");
    }

    #[test]
    fn char_escape() {
        let toks = Lexer::tokenize(r"'\n'").unwrap();
        assert_eq!(toks[0].text, "\n");
    }

    #[test]
    fn unterminated_char() {
        assert!(Lexer::tokenize("'ab'").is_err());
    }

    // -- operators ----------------------------------------------------------

    #[test]
    fn single_char_operators() {
        let input = "+ - * / % < > = ! & | ^ ~";
        let expected = vec![
            TokenKind::Plus, TokenKind::Minus, TokenKind::Star,
            TokenKind::Slash, TokenKind::Percent, TokenKind::Lt,
            TokenKind::Gt, TokenKind::Eq, TokenKind::Bang,
            TokenKind::Ampersand, TokenKind::Pipe, TokenKind::Caret,
            TokenKind::Tilde, TokenKind::Eof,
        ];
        assert_eq!(kinds(input), expected);
    }

    #[test]
    fn two_char_operators() {
        let input = "== != <= >= := += -= *= /= && || << >> .. ->";
        let expected = vec![
            TokenKind::EqEq, TokenKind::NotEq, TokenKind::LtEq,
            TokenKind::GtEq, TokenKind::ColonEq, TokenKind::PlusEq,
            TokenKind::MinusEq, TokenKind::StarEq, TokenKind::SlashEq,
            TokenKind::AmpAmp, TokenKind::PipePipe, TokenKind::Shl,
            TokenKind::Shr, TokenKind::DotDot, TokenKind::Arrow,
            TokenKind::Eof,
        ];
        assert_eq!(kinds(input), expected);
    }

    // -- delimiters ---------------------------------------------------------

    #[test]
    fn delimiters() {
        let input = "( ) [ ] { } : , .";
        let expected = vec![
            TokenKind::LParen, TokenKind::RParen, TokenKind::LBracket,
            TokenKind::RBracket, TokenKind::LBrace, TokenKind::RBrace,
            TokenKind::Colon, TokenKind::Comma, TokenKind::Dot,
            TokenKind::Eof,
        ];
        assert_eq!(kinds(input), expected);
    }

    // -- annotations --------------------------------------------------------

    #[test]
    fn annotations() {
        let input = "@inline @pure @simd";
        let expected = vec![
            TokenKind::Annotation, TokenKind::Annotation,
            TokenKind::Annotation, TokenKind::Eof,
        ];
        assert_eq!(kinds(input), expected);
        let t = texts(input);
        assert_eq!(t[0], "@inline");
        assert_eq!(t[1], "@pure");
        assert_eq!(t[2], "@simd");
    }

    // -- comments -----------------------------------------------------------

    #[test]
    fn comments_are_skipped() {
        let input = "x # comment\ny";
        let k = kinds(input);
        assert_eq!(
            k,
            vec![
                TokenKind::Identifier,
                TokenKind::Newline,
                TokenKind::Identifier,
                TokenKind::Eof,
            ]
        );
    }

    // -- indentation --------------------------------------------------------

    #[test]
    fn indent_dedent() {
        let input = "if x:\n    y\n    z\nw\n";
        let k = kinds(input);
        assert_eq!(
            k,
            vec![
                TokenKind::If,
                TokenKind::Identifier,  // x
                TokenKind::Colon,
                TokenKind::Newline,
                TokenKind::Indent,
                TokenKind::Identifier,  // y
                TokenKind::Newline,
                TokenKind::Identifier,  // z
                TokenKind::Newline,
                TokenKind::Dedent,
                TokenKind::Identifier,  // w
                TokenKind::Newline,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn nested_indent() {
        let input = "a\n    b\n        c\n    d\ne\n";
        let k = kinds(input);
        assert_eq!(
            k,
            vec![
                TokenKind::Identifier,  // a
                TokenKind::Newline,
                TokenKind::Indent,
                TokenKind::Identifier,  // b
                TokenKind::Newline,
                TokenKind::Indent,
                TokenKind::Identifier,  // c
                TokenKind::Newline,
                TokenKind::Dedent,
                TokenKind::Identifier,  // d
                TokenKind::Newline,
                TokenKind::Dedent,
                TokenKind::Identifier,  // e
                TokenKind::Newline,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn dedent_at_eof() {
        let input = "a\n    b";
        let k = kinds(input);
        assert_eq!(
            k,
            vec![
                TokenKind::Identifier,  // a
                TokenKind::Newline,
                TokenKind::Indent,
                TokenKind::Identifier,  // b
                TokenKind::Dedent,
                TokenKind::Eof,
            ]
        );
    }

    // -- spans --------------------------------------------------------------

    #[test]
    fn span_tracking() {
        let input = "x\n  y";
        let toks = Lexer::tokenize(input).unwrap();
        assert_eq!(toks[0].span, Span::new(1, 1)); // x
        // toks: x(0), \n(1), INDENT(2), y(3), DEDENT(4), EOF(5)
        assert_eq!(toks[2].kind, TokenKind::Indent);
        assert_eq!(toks[3].span, Span::new(2, 3)); // y (after indent whitespace)
    }

    // -- realistic snippet --------------------------------------------------

    #[test]
    fn function_definition() {
        let input = "\
fn add(a: i32, b: i32) -> i32:
    return a + b
";
        let toks = Lexer::tokenize(input).unwrap();
        let k: Vec<_> = toks.iter().map(|t| &t.kind).collect();
        assert_eq!(
            k,
            vec![
                &TokenKind::Fn,
                &TokenKind::Identifier,  // add
                &TokenKind::LParen,
                &TokenKind::Identifier,  // a
                &TokenKind::Colon,
                &TokenKind::Identifier,  // i32
                &TokenKind::Comma,
                &TokenKind::Identifier,  // b
                &TokenKind::Colon,
                &TokenKind::Identifier,  // i32
                &TokenKind::RParen,
                &TokenKind::Arrow,
                &TokenKind::Identifier,  // i32
                &TokenKind::Colon,
                &TokenKind::Newline,
                &TokenKind::Indent,
                &TokenKind::Return,
                &TokenKind::Identifier,  // a
                &TokenKind::Plus,
                &TokenKind::Identifier,  // b
                &TokenKind::Newline,
                &TokenKind::Dedent,
                &TokenKind::Eof,
            ]
        );
    }

    // -- error cases --------------------------------------------------------

    #[test]
    fn unexpected_character() {
        assert!(Lexer::tokenize("$").is_err());
    }

    #[test]
    fn bad_annotation() {
        assert!(Lexer::tokenize("@").is_err());
    }

    #[test]
    fn bad_hex() {
        assert!(Lexer::tokenize("0xZZ").is_err());
    }

    #[test]
    fn bad_binary() {
        assert!(Lexer::tokenize("0b2").is_err());
    }

    #[test]
    fn bad_exponent() {
        assert!(Lexer::tokenize("1e").is_err());
    }

    #[test]
    fn mismatched_indent() {
        // Line indented to a level that doesn't match any outer level.
        let input = "a\n    b\n  c\n";
        assert!(Lexer::tokenize(input).is_err());
    }
}
