import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Callable
from itertools import product
import re


class PropositionalCalculus:
    def __init__(self):
        self.variables = set()
        self.operators = {'AND', 'OR', 'NOT', 'IMPLIES', 'IFF'}

    def parse(self, formula: str) -> str:
        formula = formula.replace('->', ' IMPLIES ').replace('<->', ' IFF ')
        formula = formula.replace('&', ' AND ').replace('|', ' OR ').replace('~', ' NOT ')
        formula = re.sub(r'\s+', ' ', formula).strip()
        self.variables.update(re.findall(r'[A-Z][A-Za-z0-9]*', formula))
        self.variables -= self.operators
        return formula

    def evaluate(self, formula: str, assignment: Dict[str, bool]) -> bool:
        expr = formula
        for var, val in assignment.items():
            expr = re.sub(r'\b' + var + r'\b', str(val), expr)
        expr = expr.replace('NOT True', 'False').replace('NOT False', 'True')
        for _ in range(10):
            expr = re.sub(r'True AND True', 'True', expr)
            expr = re.sub(r'True AND False', 'False', expr)
            expr = re.sub(r'False AND True', 'False', expr)
            expr = re.sub(r'False AND False', 'False', expr)
            expr = re.sub(r'True OR True', 'True', expr)
            expr = re.sub(r'True OR False', 'True', expr)
            expr = re.sub(r'False OR True', 'True', expr)
            expr = re.sub(r'False OR False', 'False', expr)
            expr = re.sub(r'True IMPLIES True', 'True', expr)
            expr = re.sub(r'True IMPLIES False', 'False', expr)
            expr = re.sub(r'False IMPLIES True', 'True', expr)
            expr = re.sub(r'False IMPLIES False', 'True', expr)
            expr = re.sub(r'True IFF True', 'True', expr)
            expr = re.sub(r'True IFF False', 'False', expr)
            expr = re.sub(r'False IFF True', 'False', expr)
            expr = re.sub(r'False IFF False', 'True', expr)
            expr = re.sub(r'\(True\)', 'True', expr)
            expr = re.sub(r'\(False\)', 'False', expr)
        if expr.strip() == 'True':
            return True
        if expr.strip() == 'False':
            return False
        raise ValueError(f"Cannot evaluate: {expr}")

    def truth_table(self, formula: str) -> List[Dict]:
        formula = self.parse(formula)
        vars_sorted = sorted(self.variables)
        n_vars = len(vars_sorted)
        table = []
        for values in product([False, True], repeat=n_vars):
            assignment = dict(zip(vars_sorted, values))
            try:
                result = self.evaluate(formula, assignment)
            except ValueError:
                result = None
            row = {**assignment, 'result': result}
            table.append(row)
        return table

    def resolution(self, clauses: List[Set[str]]) -> bool:
        clauses = [set(c) for c in clauses]
        while True:
            new_clauses = set()
            for i in range(len(clauses)):
                for j in range(i + 1, len(clauses)):
                    resolvents = self._resolve(clauses[i], clauses[j])
                    for r in resolvents:
                        if not r:
                            return True
                        new_clauses.add(frozenset(r))
            new_clauses_list = [set(c) for c in new_clauses]
            if not any(c not in [set(x) for x in [frozenset(cl) for cl in clauses]] for c in new_clauses_list):
                return False
            clauses.extend(new_clauses_list)

    def _resolve(self, c1: Set[str], c2: Set[str]) -> List[Set[str]]:
        resolvents = []
        for lit in c1:
            neg = 'NOT ' + lit if not lit.startswith('NOT ') else lit[4:]
            if neg in c2:
                new_clause = (c1 - {lit}) | (c2 - {neg})
                resolvents.append(new_clause)
        return resolvents

    def check(self, formula: str, knowledge_base: List[str] = None) -> bool:
        table = self.truth_table(formula)
        if knowledge_base is None:
            return all(row['result'] for row in table if row['result'] is not None)
        for row in table:
            if all(self.evaluate(kb, row) for kb in knowledge_base):
                if not row['result']:
                    return False
        return True


class FirstOrderLogic:
    def __init__(self):
        self.constants = set()
        self.predicates = set()
        self.functions = {}
        self.herbrand_universe = set()

    def set_herbrand_universe(self, constants: List[str]):
        self.herbrand_universe = set(constants)
        self.constants = set(constants)

    def unify(self, term1: str, term2: str, subst: Dict[str, str] = None) -> Optional[Dict[str, str]]:
        if subst is None:
            subst = {}
        t1 = self._apply_subst(term1, subst)
        t2 = self._apply_subst(term2, subst)
        if t1 == t2:
            return subst
        if self._is_variable(t1):
            return self._unify_var(t1, t2, subst)
        if self._is_variable(t2):
            return self._unify_var(t2, t1, subst)
        if self._is_function(t1) and self._is_function(t2):
            name1, args1 = self._parse_function(t1)
            name2, args2 = self._parse_function(t2)
            if name1 != name2 or len(args1) != len(args2):
                return None
            for a1, a2 in zip(args1, args2):
                subst = self.unify(a1, a2, subst)
                if subst is None:
                    return None
            return subst
        return None

    def _is_variable(self, term: str) -> bool:
        return term.isalpha() and term.islower() and len(term) == 1

    def _is_function(self, term: str) -> bool:
        return '(' in term

    def _parse_function(self, term: str) -> Tuple[str, List[str]]:
        name = term[:term.index('(')]
        args_str = term[term.index('(') + 1:term.rindex(')')]
        args = self._split_args(args_str)
        return name, args

    def _split_args(self, args_str: str) -> List[str]:
        args = []
        depth = 0
        current = ''
        for c in args_str:
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            elif c == ',' and depth == 0:
                args.append(current.strip())
                current = ''
                continue
            current += c
        if current.strip():
            args.append(current.strip())
        return args

    def _apply_subst(self, term: str, subst: Dict[str, str]) -> str:
        if self._is_variable(term):
            if term in subst:
                return self._apply_subst(subst[term], subst)
            return term
        if self._is_function(term):
            name, args = self._parse_function(term)
            new_args = [self._apply_subst(a, subst) for a in args]
            return name + '(' + ','.join(new_args) + ')'
        return term

    def _unify_var(self, var: str, term: str, subst: Dict[str, str]) -> Optional[Dict[str, str]]:
        if var in subst:
            return self.unify(subst[var], term, subst)
        if term in subst:
            return self.unify(var, subst[term], subst)
        if self._occurs_check(var, term, subst):
            return None
        new_subst = dict(subst)
        new_subst[var] = term
        return new_subst

    def _occurs_check(self, var: str, term: str, subst: Dict[str, str]) -> bool:
        if var == term:
            return True
        if self._is_function(term):
            _, args = self._parse_function(term)
            return any(self._occurs_check(var, a, subst) for a in args)
        return False

    def sld_resolve(self, goal: List[str], clauses: List[Tuple[List[str], Dict[str, str]]]) -> bool:
        if not goal:
            return True
        first = goal[0]
        rest = goal[1:]
        for clause_head, clause_body in clauses:
            subst = self.unify(first, clause_head)
            if subst is not None:
                new_body = [self._apply_subst(g, subst) for g in clause_body]
                new_goal = new_body + rest
                if self.sld_resolve(new_goal, clauses):
                    return True
        return False

    def evaluate(self, formula: str, interpretation: Dict[str, bool]) -> bool:
        if formula.startswith('NOT '):
            return not self.evaluate(formal[4:], interpretation)
        if ' AND ' in formula:
            parts = formula.split(' AND ')
            return all(self.evaluate(p.strip(), interpretation) for p in parts)
        if ' OR ' in formula:
            parts = formula.split(' OR ')
            return any(self.evaluate(p.strip(), interpretation) for p in parts)
        return interpretation.get(formula, False)

    def check(self, formula: str, knowledge_base: List[str] = None) -> bool:
        if knowledge_base is None:
            return self.evaluate(formula, {})
        for kb_formula in knowledge_base:
            if not self.evaluate(kb_formula, {}):
                return False
        return self.evaluate(formula, {})


class ModalLogic:
    def __init__(self):
        self.worlds: List[str] = []
        self.accessibility: Dict[str, Set[str]] = {}
        self-valuations: Dict[str, Dict[str, bool]] = {}

    def add_world(self, world: str):
        if world not in self.worlds:
            self.worlds.append(world)
            self.accessibility[world] = set()
            self.valuations[world] = {}

    def add_accessibility(self, from_world: str, to_world: str):
        if from_world in self.worlds and to_world in self.worlds:
            self.accessibility[from_world].add(to_world)

    def set_proposition(self, world: str, prop: str, value: bool):
        if world in self.valuations:
            self.valuations[world][prop] = bool(value)

    def evaluate_necessity(self, world: str, formula: str) -> bool:
        if world not in self.accessibility:
            return True
        for accessible in self.accessibility[world]:
            if not self.evaluate_formula(accessible, formula):
                return False
        return True

    def evaluate_possibility(self, world: str, formula: str) -> bool:
        if world not in self.accessibility:
            return False
        for accessible in self.accessibility[world]:
            if self.evaluate_formula(accessible, formula):
                return True
        return False

    def evaluate_formula(self, world: str, formula: str) -> bool:
        formula = formula.strip()
        if formula.startswith('BOX '):
            return self.evaluate_necessity(world, formula[4:])
        if formula.startswith('DIAMOND '):
            return self.evaluate_possibility(world, formula[8:])
        if formula.startswith('NOT '):
            return not self.evaluate_formula(world, formula[4:])
        if ' AND ' in formula:
            parts = formula.split(' AND ', 1)
            return self.evaluate_formula(world, parts[0]) and self.evaluate_formula(world, parts[1])
        if ' OR ' in formula:
            parts = formula.split(' OR ', 1)
            return self.evaluate_formula(world, parts[0]) or self.evaluate_formula(world, parts[1])
        if formula.startswith('(') and formula.endswith(')'):
            return self.evaluate_formula(world, formula[1:-1])
        return self.valuations.get(world, {}).get(formula, False)

    def check(self, formula: str, world: str = None) -> bool:
        if world is None:
            world = self.worlds[0] if self.worlds else None
        if world is None:
            return False
        return self.evaluate_formula(world, formula)


class TemporalLogic:
    def __init__(self):
        self.trace: List[Dict[str, bool]] = []
        self.num_states = 0

    def set_trace(self, trace: List[Dict[str, bool]]):
        self.trace = trace
        self.num_states = len(trace)

    def next_op(self, formula: str, time: int) -> bool:
        if time + 1 >= self.num_states:
            return False
        return self.evaluate_at(formula, time + 1)

    def until_op(self, phi: str, psi: str, time: int) -> bool:
        for t in range(time, self.num_states):
            if self.evaluate_at(psi, t):
                if all(self.evaluate_at(phi, k) for k in range(time, t)):
                    return True
                return False
        return False

    def globally_op(self, formula: str, time: int) -> bool:
        for t in range(time, self.num_states):
            if not self.evaluate_at(formula, t):
                return False
        return True

    def eventually_op(self, formula: str, time: int) -> bool:
        for t in range(time, self.num_states):
            if self.evaluate_at(formula, t):
                return True
        return False

    def evaluate_at(self, formula: str, time: int) -> bool:
        formula = formula.strip()
        if formula.startswith('X(') and formula.endswith(')'):
            return self.next_op(formula[2:-1], time)
        if formula.startswith('G(') and formula.endswith(')'):
            return self.globally_op(formula[2:-1], time)
        if formula.startswith('F(') and formula.endswith(')'):
            return self.eventually_op(formula[2:-1], time)
        if ' U ' in formula:
            parts = formula.split(' U ', 1)
            return self.until_op(parts[0], parts[1], time)
        if formula.startswith('NOT '):
            return not self.evaluate_at(formula[4:], time)
        if ' AND ' in formula:
            parts = formula.split(' AND ', 1)
            return self.evaluate_at(parts[0], time) and self.evaluate_at(parts[1], time)
        if ' OR ' in formula:
            parts = formula.split(' OR ', 1)
            return self.evaluate_at(parts[0], time) or self.evaluate_at(parts[1], time)
        if time < self.num_states:
            return self.trace[time].get(formula, False)
        return False

    def check(self, formula: str, start_time: int = 0) -> bool:
        return self.evaluate_at(formula, start_time)


class FuzzyLogic:
    def __init__(self):
        self.membership_functions: Dict[str, Callable[[float], float]] = {}
        self.rules: List[Tuple[str, float, str]] = []

    def add_membership_function(self, name: str, func: Callable[[float], float]):
        self.membership_functions[name] = func

    def triangular_mf(self, a: float, b: float, c: float) -> Callable[[float], float]:
        def mf(x):
            if x <= a or x >= c:
                return 0.0
            elif a < x <= b:
                return (x - a) / (b - a) if b != a else 1.0
            else:
                return (c - x) / (c - b) if c != b else 1.0
        return mf

    def trapezoidal_mf(self, a: float, b: float, c: float, d: float) -> Callable[[float], float]:
        def mf(x):
            if x <= a or x >= d:
                return 0.0
            elif a < x <= b:
                return (x - a) / (b - a) if b != a else 1.0
            elif b < x <= c:
                return 1.0
            else:
                return (d - x) / (d - c) if d != c else 1.0
        return mf

    def gaussian_mf(self, mean: float, sigma: float) -> Callable[[float], float]:
        def mf(x):
            return np.exp(-0.5 * ((x - mean) / sigma) ** 2)
        return mf

    def add_rule(self, antecedent: str, weight: float, consequent: str):
        self.rules.append((antecedent, weight, consequent))

    def fuzzify(self, value: float) -> Dict[str, float]:
        result = {}
        for name, mf in self.membership_functions.items():
            result[name] = float(mf(value))
        return result

    def infer(self, value: float) -> Dict[str, float]:
        fuzzified = self.fuzzify(value)
        outputs: Dict[str, float] = {}
        for antecedent, weight, consequent in self.rules:
            strength = fuzzified.get(antecedent, 0.0) * weight
            if consequent not in outputs:
                outputs[consequent] = 0.0
            outputs[consequent] = max(outputs[consequent], strength)
        return outputs

    def defuzzify(self, value: float) -> float:
        results = self.infer(value)
        if not results:
            return 0.0
        total_weight = sum(results.values())
        if total_weight == 0:
            return 0.0
        weighted_sum = 0.0
        for name, strength in results.items():
            mf = self.membership_functions.get(name)
            if mf is not None:
                centers = np.linspace(0, 100, 1000)
                memberships = np.array([mf(c) for c in centers])
                if memberships.sum() > 0:
                    center = np.average(centers, weights=memberships)
                else:
                    center = 50.0
            else:
                center = 50.0
            weighted_sum += strength * center
        return weighted_sum / total_weight

    def evaluate(self, formula: str, value: float) -> bool:
        fuzzified = self.fuzzify(value)
        if formula.startswith('NOT '):
            prop = formula[4:]
            return fuzzified.get(prop, 0.0) < 0.5
        if ' AND ' in formula:
            parts = formula.split(' AND ')
            return all(fuzzified.get(p.strip(), 0.0) > 0.5 for p in parts)
        if ' OR ' in formula:
            parts = formula.split(' OR ')
            return any(fuzzified.get(p.strip(), 0.0) > 0.5 for p in parts)
        return fuzzified.get(formula, 0.0) > 0.5

    def check(self, formula: str, value: float = None) -> bool:
        if value is None:
            value = 50.0
        return self.evaluate(formula, value)
