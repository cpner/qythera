"""Model definition DSL: lexer, parser, semantic analyzer, and code generator."""

import re
import sys
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np


class TokenType(Enum):
    KEYWORD = auto()
    IDENT = auto()
    NUMBER = auto()
    STRING = auto()
    LBRACE = auto()
    RBRACE = auto()
    SEMICOLON = auto()
    EQUALS = auto()
    COMMA = auto()
    COLON = auto()
    EOF = auto()


KEYWORDS = {
    "model", "vocab", "embed", "layers", "heads", "kv_heads",
    "ffn", "rope", "norm", "act", "ctx", "moe", "dtype",
    "expert", "top_k", "shared_experts", "bias",
    "activation", "hidden", "intermediate",
}


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    col: int

    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.col})"


class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens: List[Token] = []

    def tokenize(self) -> List[Token]:
        while self.pos < len(self.source):
            ch = self.source[self.pos]

            if ch in ' \t\r':
                self._advance()
            elif ch == '\n':
                self._advance()
                self.line += 1
                self.col = 1
            elif ch == '{':
                self.tokens.append(Token(TokenType.LBRACE, '{', self.line, self.col))
                self._advance()
            elif ch == '}':
                self.tokens.append(Token(TokenType.RBRACE, '}', self.line, self.col))
                self._advance()
            elif ch == ';':
                self.tokens.append(Token(TokenType.SEMICOLON, ';', self.line, self.col))
                self._advance()
            elif ch == '=':
                self.tokens.append(Token(TokenType.EQUALS, '=', self.line, self.col))
                self._advance()
            elif ch == ',':
                self.tokens.append(Token(TokenType.COMMA, ',', self.line, self.col))
                self._advance()
            elif ch == ':':
                self.tokens.append(Token(TokenType.COLON, ':', self.line, self.col))
                self._advance()
            elif ch == '"' or ch == "'":
                self._read_string(ch)
            elif ch.isdigit() or (ch == '-' and self.pos + 1 < len(self.source) and self.source[self.pos + 1].isdigit()):
                self._read_number()
            elif ch.isalpha() or ch == '_':
                self._read_ident()
            else:
                self._advance()

        self.tokens.append(Token(TokenType.EOF, '', self.line, self.col))
        return self.tokens

    def _advance(self):
        self.pos += 1
        self.col += 1

    def _read_string(self, quote: str):
        start_col = self.col
        self._advance()
        value = []
        while self.pos < len(self.source) and self.source[self.pos] != quote:
            if self.source[self.pos] == '\\':
                self._advance()
            value.append(self.source[self.pos])
            self._advance()
        if self.pos < len(self.source):
            self._advance()
        self.tokens.append(Token(TokenType.STRING, ''.join(value), self.line, start_col))

    def _read_number(self):
        start_col = self.col
        start = self.pos
        if self.source[self.pos] == '-':
            self._advance()
        while self.pos < len(self.source) and (self.source[self.pos].isdigit() or self.source[self.pos] == '.'):
            self._advance()
        value = self.source[start:self.pos]
        self.tokens.append(Token(TokenType.NUMBER, value, self.line, start_col))

    def _read_ident(self):
        start_col = self.col
        start = self.pos
        while self.pos < len(self.source) and (self.source[self.pos].isalnum() or self.source[self.pos] == '_'):
            self._advance()
        value = self.source[start:self.pos]
        if value in KEYWORDS:
            self.tokens.append(Token(TokenType.KEYWORD, value, self.line, start_col))
        else:
            self.tokens.append(Token(TokenType.IDENT, value, self.line, start_col))


class ASTNode:
    pass


@dataclass
class ModelDef(ASTNode):
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    blocks: List['Block'] = field(default_factory=list)


@dataclass
class Assignment(ASTNode):
    key: str
    value: Any


@dataclass
class Block(ASTNode):
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Program(ASTNode):
    models: List[ModelDef] = field(default_factory=list)


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> Program:
        program = Program()
        while self.current().type != TokenType.EOF:
            if self.current().type == TokenType.KEYWORD and self.current().value == "model":
                program.models.append(self._parse_model())
            else:
                self._advance()
        return program

    def _parse_model(self) -> ModelDef:
        self._expect(TokenType.KEYWORD, "model")
        name_token = self._expect(TokenType.IDENT)
        self._expect(TokenType.LBRACE)

        model = ModelDef(name=name_token.value)

        while self.current().type != TokenType.RBRACE:
            if self.current().type == TokenType.KEYWORD:
                if self.current().value in ("vocab", "embed", "layers", "heads", "kv_heads",
                                            "ffn", "rope", "ctx", "hidden", "intermediate"):
                    self._parse_property(model.properties)
                elif self.current().value in ("norm", "act", "bias"):
                    self._parse_property(model.properties)
                elif self.current().value == "moe":
                    model.blocks.append(self._parse_moe_block())
                else:
                    self._advance()
            elif self.current().type == TokenType.IDENT:
                self._parse_property(model.properties)
            else:
                self._advance()

        self._expect(TokenType.RBRACE)
        return model

    def _parse_property(self, props: Dict[str, Any]):
        key_token = self.current()
        self._advance()

        if self.current().type == TokenType.EQUALS:
            self._advance()
            value = self._parse_value()
            props[key_token.value] = value
        elif self.current().type == TokenType.NUMBER:
            value = self._parse_value()
            props[key_token.value] = value
        elif self.current().type == TokenType.STRING:
            value = self._parse_value()
            props[key_token.value] = value
        else:
            props[key_token.value] = True

        if self.current().type == TokenType.SEMICOLON:
            self._advance()

    def _parse_moe_block(self) -> Block:
        self._expect(TokenType.KEYWORD, "moe")
        self._expect(TokenType.LBRACE)

        block = Block(name="moe")

        while self.current().type != TokenType.RBRACE:
            if self.current().type == TokenType.KEYWORD:
                self._parse_property(block.properties)
            else:
                self._advance()

        self._expect(TokenType.RBRACE)
        if self.current().type == TokenType.SEMICOLON:
            self._advance()
        return block

    def _parse_value(self):
        if self.current().type == TokenType.NUMBER:
            val = self.current().value
            self._advance()
            return int(val) if '.' not in val else float(val)
        elif self.current().type == TokenType.STRING:
            val = self.current().value
            self._advance()
            return val
        elif self.current().type == TokenType.IDENT:
            val = self.current().value
            self._advance()
            return val
        return None

    def current(self) -> Token:
        return self.tokens[self.pos]

    def _advance(self):
        self.pos += 1

    def _expect(self, type: TokenType, value: str = None) -> Token:
        tok = self.current()
        if tok.type != type:
            raise SyntaxError(f"Expected {type.name}, got {tok.type.name} at {tok.line}:{tok.col}")
        if value and tok.value != value:
            raise SyntaxError(f"Expected '{value}', got '{tok.value}' at {tok.line}:{tok.col}")
        self._advance()
        return tok


class SemanticAnalyzer:
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def analyze(self, program: Program) -> bool:
        for model in program.models:
            self._validate_model(model)
        return len(self.errors) == 0

    def _validate_model(self, model: ModelDef):
        props = model.properties

        if "vocab" not in props:
            self.errors.append(f"Model '{model.name}' missing 'vocab'")
        if "embed" not in props:
            self.errors.append(f"Model '{model.name}' missing 'embed'")

        if "layers" in props:
            layers = props["layers"]
            if not isinstance(layers, int) or layers <= 0:
                self.errors.append(f"Model '{model.name}': 'layers' must be positive integer")

        if "heads" in props:
            heads = props["heads"]
            if not isinstance(heads, int) or heads <= 0:
                self.errors.append(f"Model '{model.name}': 'heads' must be positive integer")

        if "embed" in props and "heads" in props:
            embed = props["embed"]
            heads = props["heads"]
            if isinstance(embed, int) and isinstance(heads, int):
                if embed % heads != 0:
                    self.warnings.append(
                        f"Model '{model.name}': embed ({embed}) not divisible by heads ({heads})"
                    )

        if "ffn" in props and "embed" in props:
            ffn = props["ffn"]
            embed = props["embed"]
            if isinstance(ffn, int) and isinstance(embed, int):
                if ffn < embed:
                    self.warnings.append(
                        f"Model '{model.name}': ffn ({ffn}) < embed ({embed})"
                    )

        for block in model.blocks:
            if block.name == "moe":
                self._validate_moe(model.name, block, props)

    def _validate_moe(self, model_name: str, block: Block, model_props: Dict):
        moe_props = block.properties
        if "expert" in moe_props:
            expert = moe_props["expert"]
            if not isinstance(expert, int) or expert <= 0:
                self.errors.append(f"Model '{model_name}': moe 'expert' must be positive integer")

        if "top_k" in moe_props and "expert" in moe_props:
            top_k = moe_props["top_k"]
            expert = moe_props["expert"]
            if isinstance(top_k, int) and isinstance(expert, int):
                if top_k > expert:
                    self.errors.append(
                        f"Model '{model_name}': moe top_k ({top_k}) > expert ({expert})"
                    )


class CodeGenerator:
    def __init__(self):
        self.indent = 0

    def generate(self, program: Program) -> str:
        lines = []
        lines.append("import numpy as np")
        lines.append("")

        for model in program.models:
            lines.extend(self._generate_model(model))
            lines.append("")

        return "\n".join(lines)

    def _generate_model(self, model: ModelDef) -> List[str]:
        lines = []
        props = model.properties

        lines.append(f"class {model.name}:")
        self.indent += 1
        lines.append(f"{self._ind()}def __init__(self):")

        self.indent += 1
        vocab = props.get("vocab", 1000)
        embed = props.get("embed", 256)
        layers = props.get("layers", 4)
        heads = props.get("heads", 8)
        kv_heads = props.get("kv_heads", heads)
        ffn_dim = props.get("ffn", embed * 4)
        rope = props.get("rope", True)
        norm = props.get("norm", "layer")
        act = props.get("act", "swiglu")
        ctx = props.get("ctx", 2048)

        lines.append(f"{self._ind()}self.vocab_size = {vocab}")
        lines.append(f"{self._ind()}self.embed_dim = {embed}")
        lines.append(f"{self._ind()}self.num_layers = {layers}")
        lines.append(f"{self._ind()}self.num_heads = {heads}")
        lines.append(f"{self._ind()}self.kv_heads = {kv_heads}")
        lines.append(f"{self._ind()}self.ffn_dim = {ffn_dim}")
        lines.append(f"{self._ind()}self.ctx_len = {ctx}")
        lines.append(f"{self._ind()}self.rope = {rope}")
        lines.append(f"{self._ind()}self.norm_type = '{norm}'")
        lines.append(f"{self._ind()}self.activation = '{act}'")

        lines.append(f"{self._ind()}self.head_dim = self.embed_dim // self.num_heads")
        lines.append(f"")
        lines.append(f"{self._ind()}self.token_embedding = np.random.randn(self.vocab_size, self.embed_dim) * 0.02")
        lines.append(f"{self._ind()}self.position_encoding = np.zeros((self.ctx_len, self.embed_dim))")

        lines.append(f"")
        lines.append(f"{self._ind()}self.layers = []")
        lines.append(f"{self._ind()}for i in range(self.num_layers):")
        self.indent += 1
        lines.append(f"{self._ind()}layer = {{}}")
        lines.append(f"{self._ind()}layer['Wq'] = np.random.randn(self.embed_dim, self.embed_dim) * 0.02")
        lines.append(f"{self._ind()}layer['Wk'] = np.random.randn(self.embed_dim, self.kv_heads * self.head_dim) * 0.02")
        lines.append(f"{self._ind()}layer['Wv'] = np.random.randn(self.embed_dim, self.kv_heads * self.head_dim) * 0.02")
        lines.append(f"{self._ind()}layer['Wo'] = np.random.randn(self.embed_dim, self.embed_dim) * 0.02")
        lines.append(f"{self._ind()}layer['W_gate'] = np.random.randn(self.embed_dim, self.ffn_dim) * 0.02")
        lines.append(f"{self._ind()}layer['W_up'] = np.random.randn(self.embed_dim, self.ffn_dim) * 0.02")
        lines.append(f"{self._ind()}layer['W_down'] = np.random.randn(self.ffn_dim, self.embed_dim) * 0.02")
        lines.append(f"{self._ind()}layer['norm1'] = np.ones(self.embed_dim)")
        lines.append(f"{self._ind()}layer['norm2'] = np.ones(self.embed_dim)")
        lines.append(f"{self._ind()}self.layers.append(layer)")
        self.indent -= 1

        lines.append(f"{self._ind()}self.output_norm = np.ones(self.embed_dim)")

        for block in model.blocks:
            if block.name == "moe":
                lines.extend(self._generate_moe_block(block))

        self.indent -= 1

        lines.append(f"")
        lines.append(f"{self._ind()}def forward(self, tokens):")
        self.indent += 1
        lines.extend(self._generate_forward())
        self.indent -= 1

        return lines

    def _generate_moe_block(self, block: Block) -> List[str]:
        lines = []
        props = block.properties
        num_experts = props.get("expert", 8)
        top_k = props.get("top_k", 2)
        ffn_dim = props.get("hidden", 512)

        lines.append(f"{self._ind()}self.num_experts = {num_experts}")
        lines.append(f"{self._ind()}self.top_k = {top_k}")
        lines.append(f"{self._ind()}self.expert_dim = {ffn_dim}")
        lines.append(f"")
        lines.append(f"{self._ind()}self.expert_gates = []")
        lines.append(f"{self._ind()}self.expert_ups = []")
        lines.append(f"{self._ind()}self.expert_downs = []")
        lines.append(f"{self._ind()}for e in range(self.num_experts):")
        self.indent += 1
        lines.append(f"{self._ind()}self.expert_gates.append(np.random.randn(self.embed_dim, self.expert_dim) * 0.02)")
        lines.append(f"{self._ind()}self.expert_ups.append(np.random.randn(self.embed_dim, self.expert_dim) * 0.02)")
        lines.append(f"{self._ind()}self.expert_downs.append(np.random.randn(self.expert_dim, self.embed_dim) * 0.02)")
        self.indent -= 1
        return lines

    def _generate_forward(self) -> List[str]:
        lines = []
        ind = self._ind()

        lines.append(f"{ind}seq_len = len(tokens)")
        lines.append(f"{ind}x = self.token_embedding[tokens]")

        lines.append(f"")
        lines.append(f"{ind}for layer in self.layers:")
        self.indent += 1
        ind = self._ind()

        lines.append(f"{ind}residual = x")
        lines.append(f"{ind}x_norm = x / (np.sqrt(np.mean(x**2, axis=-1, keepdims=True)) + 1e-5)")

        lines.append(f"{ind}q = np.matmul(x_norm, layer['Wq'])")
        lines.append(f"{ind}k = np.matmul(x_norm, layer['Wk'])")
        lines.append(f"{ind}v = np.matmul(x_norm, layer['Wv'])")

        lines.append(f"{ind}attn = np.matmul(q, k.T) / np.sqrt(self.head_dim)")
        lines.append(f"{ind}attn = np.exp(attn - np.max(attn, axis=-1, keepdims=True))")
        lines.append(f"{ind}attn = attn / np.sum(attn, axis=-1, keepdims=True)")
        lines.append(f"{ind}attn_out = np.matmul(attn, v)")
        lines.append(f"{ind}x = residual + np.matmul(attn_out, layer['Wo'])")

        lines.append(f"")
        lines.append(f"{ind}residual = x")
        lines.append(f"{ind}x_norm = x / (np.sqrt(np.mean(x**2, axis=-1, keepdims=True)) + 1e-5)")
        lines.append(f"{ind}gate = np.maximum(0, np.matmul(x_norm, layer['W_gate']))")
        lines.append(f"{ind}up = np.matmul(x_norm, layer['W_up'])")
        lines.append(f"{ind}x = residual + np.matmul(gate * up, layer['W_down'])")

        self.indent -= 1

        lines.append(f"")
        lines.append(f"{ind}x = x / (np.sqrt(np.mean(x**2, axis=-1, keepdims=True)) + 1e-5)")
        lines.append(f"{ind}logits = np.matmul(x, self.token_embedding.T)")
        lines.append(f"{ind}return logits")

        return lines

    def _ind(self) -> str:
        return "    " * self.indent


class QytheraModule:
    def __init__(self, name: str, instance: Any):
        self.name = name
        self.instance = instance

    def forward(self, tokens):
        return self.instance.forward(tokens)

    def parameters(self) -> int:
        total = 0
        for attr in dir(self.instance):
            val = getattr(self.instance, attr)
            if isinstance(val, np.ndarray):
                total += val.size
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        for v in item.values():
                            if isinstance(v, np.ndarray):
                                total += v.size
                    elif isinstance(item, np.ndarray):
                        total += item.size
        return total

    def __repr__(self):
        return f"QytheraModule({self.name})"


def parse(source: str) -> Tuple[Program, List[str]]:
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    program = parser.parse()
    analyzer = SemanticAnalyzer()
    valid = analyzer.analyze(program)
    return program, analyzer.errors + analyzer.warnings


def compile_model(source: str) -> QytheraModule:
    program, issues = parse(source)

    if any("missing" in i.lower() or "must be" in i.lower() for i in issues):
        errors = [i for i in issues if "missing" in i.lower() or "must be" in i.lower()]
        raise ValueError(f"Semantic errors: {errors}")

    generator = CodeGenerator()
    code = generator.generate(program)

    namespace = {"np": np}
    exec(code, namespace)

    model_def = program.models[0] if program.models else None
    if not model_def:
        raise ValueError("No model definition found")

    model_class = namespace.get(model_def.name)
    if model_class is None:
        raise ValueError(f"Class {model_def.name} not found in generated code")

    instance = model_class()
    return QytheraModule(model_def.name, instance)


if __name__ == "__main__":
    source = """
    model Test {
        vocab 1000;
        embed 256;
        layers 4;
        heads 8;
        ffn 1024;
        rope true;
        ctx 2048;

        moe {
            expert 8;
            top_k 2;
            hidden 512;
        }
    }
    """

    print("=== Parsing ===")
    program, issues = parse(source)
    print(f"Models: {[m.name for m in program.models]}")
    print(f"Issues: {issues}")

    print("\n=== Compiling ===")
    module = compile_model(source)
    print(f"Module: {module}")
    print(f"Parameters: {module.parameters()}")

    print("\n=== Forward pass ===")
    tokens = np.array([1, 5, 3, 7, 2])
    logits = module.forward(tokens)
    print(f"Logits shape: {logits.shape}")
