"""Planning modules: AStar, MCTS, IDAStar, STRIPSPlanner."""

import heapq
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


def _default_heuristic(a: Any, b: Any) -> float:
    try:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    except TypeError:
        return 0.0


class AStar:
    def __init__(self, heuristic: Optional[Callable] = None):
        self.heuristic = heuristic or _default_heuristic
        self.graph: Dict[Any, List[Tuple[Any, float]]] = defaultdict(list)

    def add_edge(self, from_node: Any, to_node: Any, cost: float) -> None:
        self.graph[from_node].append((to_node, cost))
        self.graph[to_node].append((from_node, cost))

    def _reconstruct(self, came_from: Dict[Any, Any], current: Any) -> List[Any]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def plan(self, start: Any, goal: Any) -> Optional[List[Any]]:
        open_set = [(0, start)]
        came_from: Dict[Any, Any] = {}
        g_score: Dict[Any, float] = {start: 0}
        f_score: Dict[Any, float] = {start: self.heuristic(start, goal)}
        closed_set: Set[Any] = set()

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                return self._reconstruct(came_from, current)
            if current in closed_set:
                continue
            closed_set.add(current)

            for neighbor, cost in self.graph[current]:
                if neighbor in closed_set:
                    continue
                tentative_g = g_score[current] + cost
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None


class MCTSNode:
    def __init__(self, state: Any, parent: Optional['MCTSNode'] = None, action: Any = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions: List[Any] = []

    def ucb1(self, exploration: float = 1.414) -> float:
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + exploration * math.sqrt(math.log(self.parent.visits) / self.visits)


class MCTS:
    def __init__(self, get_actions: Callable, simulate: Callable, iterations: int = 1000,
                 exploration: float = 1.414):
        self.get_actions = get_actions
        self.simulate = simulate
        self.iterations = iterations
        self.exploration = exploration

    def _expand(self, node: MCTSNode) -> MCTSNode:
        actions = self.get_actions(node.state)
        untried = [a for a in actions if a not in [c.action for c in node.children]]
        if not untried:
            return node
        action = random.choice(untried)
        new_state = self._apply_action(node.state, action)
        child = MCTSNode(new_state, parent=node, action=action)
        node.children.append(child)
        node.untried_actions.remove(action) if action in node.untried_actions else None
        return child

    def _apply_action(self, state: Any, action: Any) -> Any:
        if isinstance(state, dict) and isinstance(action, tuple):
            new_state = dict(state)
            new_state[action[0]] = action[1]
            return new_state
        return state

    def _simulate(self, state: Any) -> float:
        return self.simulate(state)

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        current = node
        while current is not None:
            current.visits += 1
            current.value += value
            current = current.parent

    def plan(self, start: Any, goal: Any) -> Optional[Any]:
        root = MCTSNode(start)
        root.untried_actions = list(self.get_actions(start))

        for _ in range(self.iterations):
            node = root
            while node.untried_actions and node.children:
                node = max(node.children, key=lambda c: c.ucb1(self.exploration))

            if node.untried_actions:
                node = self._expand(node)

            value = self._simulate(node.state)
            self._backpropagate(node, value)

        if not root.children:
            return None
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action


class IDAStar:
    def __init__(self, heuristic: Optional[Callable] = None, get_neighbors: Optional[Callable] = None):
        self.heuristic = heuristic or _default_heuristic
        self.get_neighbors = get_neighbors or (lambda state: [])

    def _search(self, node: Any, goal: Any, threshold: float, path: List[Any],
                visited: Set[Any]) -> Tuple[Optional[List[Any]], float]:
        f = len(path) - 1 + self.heuristic(node, goal)
        if f > threshold:
            return None, f
        if node == goal:
            return list(path), f

        min_threshold = float('inf')
        visited.add(node)

        for neighbor, cost in self.get_neighbors(node):
            if neighbor not in visited:
                path.append(neighbor)
                result, new_threshold = self._search(neighbor, goal, threshold, path, visited)
                if result is not None:
                    return result, new_threshold
                min_threshold = min(min_threshold, new_threshold)
                path.pop()

        visited.discard(node)
        return None, min_threshold

    def plan(self, start: Any, goal: Any) -> Optional[List[Any]]:
        threshold = self.heuristic(start, goal)
        path = [start]

        for _ in range(100):
            result, new_threshold = self._search(start, goal, threshold, path, set())
            if result is not None:
                return result
            if new_threshold == float('inf'):
                return None
            threshold = new_threshold

        return None


@dataclass
class STRIPSAction:
    name: str
    preconditions: Set[str]
    effects: Set[str]


class STRIPSPlanner:
    def __init__(self):
        self.actions: List[STRIPSAction] = []

    def add_action(self, name: str, preconditions: Set[str], effects: Set[str]) -> None:
        self.actions.append(STRIPSAction(name, preconditions, effects))

    def _get_applicable(self, state: Set[str]) -> List[STRIPSAction]:
        return [a for a in self.actions if a.preconditions.issubset(state)]

    def _apply_action(self, state: Set[str], action: STRIPSAction) -> Set[str]:
        new_state = set(state)
        for effect in action.effects:
            if effect.startswith("!"):
                new_state.discard(effect[1:])
            else:
                new_state.add(effect)
        return new_state

    def plan(self, start: Set[str], goal: Set[str]) -> Optional[List[str]]:
        queue: List[Tuple[int, Set[str], List[str]]] = [(0, set(start), [])]
        visited: Set[frozenset] = set()
        visited.add(frozenset(start))

        while queue:
            cost, state, plan_steps = queue.pop(0)

            if goal.issubset(state):
                return plan_steps

            for action in self._get_applicable(state):
                new_state = self._apply_action(state, action)
                state_key = frozenset(new_state)
                if state_key not in visited:
                    visited.add(state_key)
                    queue.append((cost + 1, new_state, plan_steps + [action.name]))

        return None
