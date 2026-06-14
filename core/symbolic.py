"""Symbolic reasoning modules: KnowledgeGraph, PropositionalLogic, FirstOrderLogic, SymbolicRegression."""

import re
import random
import operator
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np


class KnowledgeGraph:
    def __init__(self):
        self.triples: List[Tuple[str, str, str]] = []
        self.subject_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.predicate_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.object_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    def add_triple(self, subject: str, predicate: str, obj: str) -> None:
        triple = (subject, predicate, obj)
        if triple not in self.triples:
            self.triples.append(triple)
            self.subject_index[subject].append((predicate, obj))
            self.predicate_index[predicate].append((subject, obj))
            self.object_index[obj].append((subject, predicate))

    def query(self, subject: Optional[str] = None, predicate: Optional[str] = None,
              obj: Optional[str] = None) -> List[Tuple[str, str, str]]:
        results = self.triples
        if subject is not None:
            results = [t for t in results if t[0] == subject]
        if predicate is not None:
            results = [t for t in results if t[1] == predicate]
        if obj is not None:
            results = [t for t in results if t[2] == obj]
        return results

    def sparql_like(self, pattern: str) -> List[Dict[str, str]]:
        pattern = pattern.strip()
        if pattern.endswith("?"):
            pattern = pattern[:-1]
        match = re.match(r"SELECT\s+(\w+)\s+WHERE\s*\{\s*(\w+)\s+(\w+)\s+(\w+)\s*\}",
                         pattern, re.IGNORECASE)
        if not match:
            return []
        var_name = match.group(1)
        s, p, o = match.group(2), match.group(3), match.group(4)
        variables = {}
        if s.startswith("?"):
            variables["subject"] = s
        if p.startswith("?"):
            variables["predicate"] = p
        if o.startswith("?"):
            variables["object"] = o

        results = []
        for triple in self.triples:
            bind = {}
            if "subject" in variables:
                bind[variables["subject"][1:]] = triple[0]
            elif s != triple[0]:
                continue
            if "predicate" in variables:
                bind[variables["predicate"][1:]] = triple[1]
            elif p != triple[1]:
                continue
            if "object" in variables:
                bind[variables["object"][1:]] = triple[2]
            elif o != triple[2]:
                continue
            results.append(bind)
        return results


@dataclass
class PLFormula:
    pass


@dataclass
class PLAtom(PLFormula):
    name: str


@dataclass
class PLNot(PLFormula):
    child: PLFormula


@dataclass
class PLAnd(PLFormula):
    left: PLFormula
    right: PLFormula


@dataclass
class PLOr(PLFormula):
    left: PLFormula
    right: PLFormula


@dataclass
class PLImplies(PLFormula):
    left: PLFormula
    right: PLFormula


class PropositionalLogic:
    def __init__(self):
        self.variables: Set[str] = set()

    def parse(self, expr: str) -> PLFormula:
        expr = expr.strip()
        tokens = self._tokenize(expr)
        formula, _ = self._parse_expr(tokens, 0)
        return formula

    def _tokenize(self, expr: str) -> List[str]:
        tokens = []
        i = 0
        while i < len(expr):
            if expr[i].isspace():
                i += 1
                continue
            if expr[i] == '(':
                tokens.append('(')
                i += 1
            elif expr[i] == ')':
                tokens.append(')')
                i += 1
            elif expr[i:i+2] == '->':
                tokens.append('->')
                i += 2
            elif expr[i:i+2] == '&&':
                tokens.append('&&')
                i += 2
            elif expr[i:i+2] == '||':
                tokens.append('||')
                i += 2
            elif expr[i] == '!':
                tokens.append('!')
                i += 1
            elif expr[i].isalnum() or expr[i] == '_':
                j = i
                while j < len(expr) and (expr[j].isalnum() or expr[j] == '_'):
                    j += 1
                tokens.append(expr[i:j])
                i = j
            else:
                i += 1
        return tokens

    def _parse_expr(self, tokens: List[str], pos: int) -> Tuple[PLFormula, int]:
        return self._parse_impl(tokens, pos)

    def _parse_impl(self, tokens: List[str], pos: int) -> Tuple[PLFormula, int]:
        left, pos = self._parse_or(tokens, pos)
        if pos < len(tokens) and tokens[pos] == '->':
            pos += 1
            right, pos = self._parse_impl(tokens, pos)
            return PLImplies(left, right), pos
        return left, pos

    def _parse_or(self, tokens: List[str], pos: int) -> Tuple[PLFormula, int]:
        left, pos = self._parse_and(tokens, pos)
        while pos < len(tokens) and tokens[pos] == '||':
            pos += 1
            right, pos = self._parse_and(tokens, pos)
            left = PLOr(left, right)
        return left, pos

    def _parse_and(self, tokens: List[str], pos: int) -> Tuple[PLFormula, int]:
        left, pos = self._parse_not(tokens, pos)
        while pos < len(tokens) and tokens[pos] == '&&':
            pos += 1
            right, pos = self._parse_not(tokens, pos)
            left = PLAnd(left, right)
        return left, pos

    def _parse_not(self, tokens: List[str], pos: int) -> Tuple[PLFormula, int]:
        if pos < len(tokens) and tokens[pos] == '!':
            pos += 1
            child, pos = self._parse_not(tokens, pos)
            return PLNot(child), pos
        return self._parse_atom(tokens, pos)

    def _parse_atom(self, tokens: List[str], pos: int) -> Tuple[PLFormula, int]:
        if pos < len(tokens) and tokens[pos] == '(':
            pos += 1
            formula, pos = self._parse_expr(tokens, pos)
            if pos < len(tokens) and tokens[pos] == ')':
                pos += 1
            return formula, pos
        if pos < len(tokens) and tokens[pos].isalpha():
            name = tokens[pos]
            self.variables.add(name)
            return PLAtom(name), pos + 1
        raise ValueError(f"Unexpected token at position {pos}")

    def evaluate(self, formula: PLFormula, assignment: Dict[str, bool]) -> bool:
        if isinstance(formula, PLAtom):
            return assignment.get(formula.name, False)
        elif isinstance(formula, PLNot):
            return not self.evaluate(formula.child, assignment)
        elif isinstance(formula, PLAnd):
            return self.evaluate(formula.left, assignment) and self.evaluate(formula.right, assignment)
        elif isinstance(formula, PLOr):
            return self.evaluate(formula.left, assignment) or self.evaluate(formula.right, assignment)
        elif isinstance(formula, PLImplies):
            return not self.evaluate(formula.left, assignment) or self.evaluate(formula.right, assignment)
        raise ValueError(f"Unknown formula type: {type(formula)}")

    def _collect_vars(self, formula: PLFormula) -> Set[str]:
        if isinstance(formula, PLAtom):
            return {formula.name}
        elif isinstance(formula, PLNot):
            return self._collect_vars(formula.child)
        elif isinstance(formula, (PLAnd, PLOr, PLImplies)):
            return self._collect_vars(formula.left) | self._collect_vars(formula.right)
        return set()

    def _dpll(self, formula: PLFormula, assignment: Dict[str, bool]) -> Optional[Dict[str, bool]]:
        if isinstance(formula, PLAtom):
            return assignment
        if isinstance(formula, PLNot):
            result = self._dpll(formula.child, assignment)
            if result is None:
                return None
            return {k: not v if k in self._collect_vars(formula.child) else v
                    for k, v in result.items()}
        if isinstance(formula, PLAnd):
            left_result = self._dpll(formula.left, assignment)
            if left_result is None:
                return None
            right_result = self._dpll(formula.right, left_result)
            return right_result
        if isinstance(formula, PLOr):
            left_result = self._dpll(formula.left, assignment)
            if left_result is not None:
                return left_result
            return self._dpll(formula.right, assignment)
        if isinstance(formula, PLImplies):
            equiv = PLOr(PLNot(formula.left), formula.right)
            return self._dpll(equiv, assignment)
        return None

    def satisfiable(self, formula: PLFormula) -> Optional[Dict[str, bool]]:
        vars_in_formula = self._collect_vars(formula)
        n = len(vars_in_formula)
        var_list = sorted(vars_in_formula)
        for i in range(2 ** n):
            assignment = {}
            for j, var in enumerate(var_list):
                assignment[var] = bool((i >> j) & 1)
            if self.evaluate(formula, assignment):
                return assignment
        return None


@dataclass
class FOLTerm:
    pass


@dataclass
class FOLVariable(FOLTerm):
    name: str


@dataclass
class FOLConstant(FOLTerm):
    value: str


@dataclass
class FOLPredicate:
    name: str
    args: List[FOLTerm]


@dataclass
class FOLFormula:
    pass


@dataclass
class FOLAtom(FOLFormula):
    predicate: FOLPredicate


@dataclass
class FOLNot(FOLFormula):
    child: FOLFormula


@dataclass
class FOLAnd(FOLFormula):
    left: FOLFormula
    right: FOLFormula


@dataclass
class FOLOr(FOLFormula):
    left: FOLFormula
    right: FOLFormula


@dataclass
class FOLExists(FOLFormula):
    variable: FOLVariable
    body: FOLFormula


@dataclass
class FOLForall(FOLFormula):
    variable: FOLVariable
    body: FOLFormula


class FirstOrderLogic:
    def __init__(self):
        self.known_predicates: Dict[str, int] = {}

    def parse_term(self, term_str: str) -> FOLTerm:
        term_str = term_str.strip()
        if term_str[0].isupper() or term_str[0].isdigit():
            return FOLConstant(term_str)
        return FOLVariable(term_str)

    def parse_predicate(self, pred_str: str) -> FOLPredicate:
        match = re.match(r"(\w+)\((.*?)\)", pred_str.strip())
        if not match:
            raise ValueError(f"Cannot parse predicate: {pred_str}")
        name = match.group(1)
        args = [self.parse_term(a) for a in match.group(2).split(",")]
        return FOLPredicate(name, args)

    def unify(self, term1: FOLTerm, term2: FOLTerm,
              substitution: Optional[Dict[str, FOLTerm]] = None) -> Optional[Dict[str, FOLTerm]]:
        if substitution is None:
            substitution = {}
        if isinstance(term1, FOLVariable):
            if term1.name in substitution:
                return self.unify(substitution[term1.name], term2, substitution)
            substitution[term1.name] = term2
            return substitution
        if isinstance(term2, FOLVariable):
            if term2.name in substitution:
                return self.unify(term1, substitution[term2.name], substitution)
            substitution[term2.name] = term1
            return substitution
        if isinstance(term1, FOLConstant) and isinstance(term2, FOLConstant):
            if term1.value == term2.value:
                return substitution
            return None
        return None

    def unify_predicates(self, pred1: FOLPredicate, pred2: FOLPredicate) -> Optional[Dict[str, FOLTerm]]:
        if pred1.name != pred2.name or len(pred1.args) != len(pred2.args):
            return None
        substitution = {}
        for a1, a2 in zip(pred1.args, pred2.args):
            result = self.unify(a1, a2, substitution)
            if result is None:
                return None
            substitution = result
        return substitution

    def apply_substitution(self, term: FOLTerm, substitution: Dict[str, FOLTerm]) -> FOLTerm:
        if isinstance(term, FOLVariable):
            if term.name in substitution:
                return self.apply_substitution(substitution[term.name], substitution)
            return term
        return term

    def resolve(self, clause1: List[FOLPredicate], clause2: List[FOLPredicate]) -> List[List[FOLPredicate]]:
        resolvents = []
        for i, pred1 in enumerate(clause1):
            for j, pred2 in enumerate(clause2):
                negated = FOLPredicate(pred2.name, pred2.args)
                substitution = self.unify_predicates(pred1, negated)
                if substitution is not None:
                    new_clause = []
                    for k, p in enumerate(clause1):
                        if k != i:
                            new_args = [self.apply_substitution(a, substitution) for a in p.args]
                            new_clause.append(FOLPredicate(p.name, new_args))
                    for k, p in enumerate(clause2):
                        if k != j:
                            new_args = [self.apply_substitution(a, substitution) for a in p.args]
                            new_clause.append(FOLPredicate(p.name, new_args))
                    resolvents.append(new_clause)
        return resolvents


class SymbolicRegression:
    def __init__(self, population_size: int = 100, generations: int = 50,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.7):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.operators = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
        }
        self.unary_ops = {
            'sin': np.sin,
            'cos': np.cos,
            'exp': np.exp,
        }

    def _random_tree(self, max_depth: int = 4) -> Any:
        if max_depth <= 0 or random.random() < 0.3:
            return random.choice(['x', str(round(random.uniform(-5, 5), 2))])
        op = random.choice(list(self.operators.keys()) + list(self.unary_ops.keys()))
        if op in self.unary_ops:
            return (op, self._random_tree(max_depth - 1))
        return (op, self._random_tree(max_depth - 1), self._random_tree(max_depth - 1))

    def _evaluate_tree(self, tree: Any, x: float) -> float:
        if isinstance(tree, str):
            if tree == 'x':
                return x
            return float(tree)
        if isinstance(tree, tuple):
            op = tree[0]
            if op in self.unary_ops:
                return self.unary_ops[op](self._evaluate_tree(tree[1], x))
            left = self._evaluate_tree(tree[1], x)
            right = self._evaluate_tree(tree[2], x)
            return self.operators[op](left, right)
        return 0.0

    def _fitness(self, tree: Any, x_data: np.ndarray, y_data: np.ndarray) -> float:
        try:
            predictions = np.array([self._evaluate_tree(tree, float(x)) for x in x_data])
            mse = np.mean((predictions - y_data) ** 2)
            return 1.0 / (1.0 + mse)
        except (OverflowError, FloatingPointError):
            return 0.0

    def _mutate(self, tree: Any, depth: int = 3) -> Any:
        if random.random() < 0.2:
            return self._random_tree(depth)
        if isinstance(tree, str):
            return tree
        if isinstance(tree, tuple):
            if random.random() < self.mutation_rate:
                return self._random_tree(depth)
            op = tree[0]
            if op in self.unary_ops:
                return (op, self._mutate(tree[1], depth - 1))
            return (op, self._mutate(tree[1], depth - 1), self._mutate(tree[2], depth - 1))
        return tree

    def _crossover(self, tree1: Any, tree2: Any) -> Any:
        if random.random() < self.crossover_rate:
            return tree2
        return tree1

    def _tree_to_string(self, tree: Any) -> str:
        if isinstance(tree, str):
            return tree
        if isinstance(tree, tuple):
            op = tree[0]
            if op in self.unary_ops:
                return f"{op}({self._tree_to_string(tree[1])})"
            left = self._tree_to_string(tree[1])
            right = self._tree_to_string(tree[2])
            return f"({left} {op} {right})"
        return str(tree)

    def solve(self, x_data: np.ndarray, y_data: np.ndarray) -> Tuple[str, Any]:
        population = [self._random_tree() for _ in range(self.population_size)]
        best_tree = None
        best_fitness = -1.0

        for gen in range(self.generations):
            fitnesses = [self._fitness(tree, x_data, y_data) for tree in population]
            for i, f in enumerate(fitnesses):
                if f > best_fitness:
                    best_fitness = f
                    best_tree = population[i]

            sorted_pop = sorted(zip(fitnesses, population), key=lambda x: x[0], reverse=True)
            elite = [tree for _, tree in sorted_pop[:self.population_size // 4]]

            new_pop = list(elite)
            while len(new_pop) < self.population_size:
                tournament = random.sample(list(range(len(population))), 3)
                parent1 = population[max(tournament, key=lambda i: fitnesses[i])]
                tournament = random.sample(list(range(len(population))), 3)
                parent2 = population[max(tournament, key=lambda i: fitnesses[i])]
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_pop.append(child)

            population = new_pop

        return self._tree_to_string(best_tree), best_tree
