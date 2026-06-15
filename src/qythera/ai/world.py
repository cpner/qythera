import numpy as np
from typing import List, Dict, Tuple, Optional, Set


class PhysicsObject:
    def __init__(self, position: np.ndarray, velocity: np.ndarray, mass: float, radius: float = 1.0):
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.mass = float(mass)
        self.radius = float(radius)
        self.force = np.zeros_like(self.position)
        self.id = id(self)

    def apply_force(self, force: np.ndarray):
        self.force += np.array(force, dtype=np.float64)

    def get_aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.position - self.radius, self.position + self.radius

    def integrate_euler(self, dt: float):
        acceleration = self.force / self.mass if self.mass > 0 else np.zeros_like(self.velocity)
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        self.force = np.zeros_like(self.force)

    def integrate_rk4(self, dt: float, force_fn=None):
        if force_fn is None:
            acceleration = self.force / self.mass if self.mass > 0 else np.zeros_like(self.velocity)
            self.velocity += acceleration * dt
            self.position += self.velocity * dt
            self.force = np.zeros_like(self.force)
            return

        mass = self.mass
        pos = self.position.copy()
        vel = self.velocity.copy()

        def accel(p, v, f):
            return f / mass if mass > 0 else np.zeros_like(v)

        f1 = force_fn(pos, vel)
        a1 = accel(pos, vel, f1)

        f2 = force_fn(pos + vel * dt / 2, vel + a1 * dt / 2)
        a2 = accel(pos + vel * dt / 2, vel + a1 * dt / 2, f2)

        f3 = force_fn(pos + (vel + a1 * dt / 2) * dt / 2, vel + a2 * dt / 2)
        a3 = accel(pos + (vel + a1 * dt / 2) * dt / 2, vel + a2 * dt / 2, f3)

        f4 = force_fn(pos + (vel + a2 * dt / 2) * dt, vel + a3 * dt)
        a4 = accel(pos + (vel + a2 * dt / 2) * dt, vel + a3 * dt, f4)

        self.velocity += (a1 + 2 * a2 + 2 * a3 + a4) * dt / 6
        self.position += (vel + (vel + a1 * dt) + (vel + a2 * dt) + (vel + a3 * dt)) * dt / 6
        self.force = np.zeros_like(self.force)


def aabb_collision(obj_a: PhysicsObject, obj_b: PhysicsObject) -> bool:
    a_min, a_max = obj_a.get_aabb()
    b_min, b_max = obj_b.get_aabb()
    return np.all(a_min <= b_max) and np.all(a_max >= b_min)


def elastic_collision_response(obj_a: PhysicsObject, obj_b: PhysicsObject):
    normal = obj_b.position - obj_a.position
    dist = np.linalg.norm(normal)
    if dist == 0:
        return
    normal = normal / dist

    rel_vel = obj_a.velocity - obj_b.velocity
    vel_along_normal = np.dot(rel_vel, normal)

    if vel_along_normal > 0:
        return

    e = 1.0
    j = -(1 + e) * vel_along_normal / (1 / obj_a.mass + 1 / obj_b.mass)
    impulse = j * normal

    obj_a.velocity += impulse / obj_a.mass
    obj_b.velocity -= impulse / obj_b.mass


class CausalDAG:
    def __init__(self):
        self.nodes: Set[str] = set()
        self.edges: Dict[str, Set[str]] = {}
        self.parents: Dict[str, Set[str]] = {}
        self.functions: Dict[str, callable] = {}
        self.priors: Dict[str, np.ndarray] = {}

    def add_node(self, name: str, func: callable = None, prior: np.ndarray = None):
        self.nodes.add(name)
        if name not in self.edges:
            self.edges[name] = set()
        if name not in self.parents:
            self.parents[name] = set()
        if func is not None:
            self.functions[name] = func
        if prior is not None:
            self.priors[name] = np.array(prior, dtype=np.float64)

    def add_edge(self, parent: str, child: str):
        self.edges[parent].add(child)
        self.parents[child].add(parent)

    def do_calculus(self, variable: str, value: float) -> Dict[str, float]:
        intervened = {}
        for node in self.nodes:
            if node == variable:
                intervened[node] = value
            elif variable not in self.parents[node]:
                if node in self.functions:
                    parent_vals = {p: intervened.get(p, self.priors.get(p, np.zeros(1))[0]) for p in self.parents[node]}
                    intervened[node] = self.functions[node](**parent_vals)
                elif node in self.priors:
                    intervened[node] = self.priors[node][0]
                else:
                    intervened[node] = 0.0
            else:
                if node in self.functions:
                    parent_vals = {p: value if p == variable else intervened.get(p, 0.0) for p in self.parents[node]}
                    intervened[node] = self.functions[node](**parent_vals)
                else:
                    intervened[node] = value
        return intervened

    def backdoor_check(self, x: str, y: str, z: Set[str]) -> bool:
        graph_copy = {n: set(self.edges[n]) for n in self.nodes}
        parents_copy = {n: set(self.parents[n]) for n in self.nodes}
        for node in list(z):
            graph_copy.pop(node, None)
            for parent in list(parents_copy.get(node, [])):
                graph_copy[parent].discard(node)
        visited = set()
        queue = [x]
        while queue:
            current = queue.pop(0)
            if current == y:
                return False
            if current in visited:
                continue
            visited.add(current)
            for neighbor in graph_copy.get(current, set()):
                if neighbor not in z:
                    queue.append(neighbor)
        visited = set()
        queue = [x]
        while queue:
            current = queue.pop(0)
            if current == y:
                return True
            if current in visited:
                continue
            visited.add(current)
            for parent in parents_copy.get(current, set()):
                if parent not in z:
                    queue.append(parent)
        return False

    def infer(self, evidence: Dict[str, float] = None) -> Dict[str, float]:
        if evidence is None:
            evidence = {}
        result = {}
        for node in self.nodes:
            if node in evidence:
                result[node] = evidence[node]
            elif node in self.functions:
                parent_vals = {p: result.get(p, self.priors.get(p, np.zeros(1))[0]) for p in self.parents[node]}
                result[node] = self.functions[node](**parent_vals)
            elif node in self.priors:
                result[node] = self.priors[node][0]
            else:
                result[node] = 0.0
        return result


class BDI_Agent:
    def __init__(self):
        self.beliefs: Dict[str, float] = {}
        self.desires: List[str] = []
        self.intentions: List[str] = []
        self.belief_revision_rate = 0.1

    def perceive(self, observations: Dict[str, float]):
        for key, value in observations.items():
            if key in self.beliefs:
                self.beliefs[key] = (1 - self.belief_revision_rate) * self.beliefs[key] + self.belief_revision_rate * value
            else:
                self.beliefs[key] = value

    def deliberate(self, utility_fn: callable = None):
        if utility_fn is None:
            self.desires = sorted(self.beliefs.keys(), key=lambda k: self.beliefs.get(k, 0), reverse=True)[:5]
        else:
            scored = [(k, utility_fn(k, self.beliefs.get(k, 0))) for k in self.beliefs]
            scored.sort(key=lambda x: x[1], reverse=True)
            self.desires = [k for k, s in scored[:5] if s > 0]

    def plan(self):
        self.intentions = [d for d in self.desires if self.beliefs.get(d, 0) > 0.3]

    def reason(self, observations: Dict[str, float], utility_fn: callable = None):
        self.perceive(observations)
        self.deliberate(utility_fn)
        self.plan()
        return self.intentions

    def execute(self) -> List[str]:
        actions = self.intentions.copy()
        self.intentions = []
        return actions

    def get_state(self) -> Dict:
        return {
            'beliefs': dict(self.beliefs),
            'desires': list(self.desires),
            'intentions': list(self.intentions)
        }


class RK4Integrator:
    def step(self, state: np.ndarray, dt: float, derivatives: callable) -> np.ndarray:
        k1 = derivatives(state)
        k2 = derivatives(state + dt / 2 * k1)
        k3 = derivatives(state + dt / 2 * k2)
        k4 = derivatives(state + dt * k3)
        return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class SPHSimulation:
    def __init__(self, particles: np.ndarray, h: float = 0.1):
        self.particles = np.array(particles, dtype=np.float64)
        self.h = float(h)
        self.density = np.zeros(len(particles))
        self.pressure = np.zeros(len(particles))
        self.velocities = np.zeros_like(particles)
        self.mass = np.ones(len(particles))
        self.rest_density = 1000.0
        self.gas_constant = 2000.0
        self.viscosity = 200.0
        self.gravity = np.array([0.0, -9.81])

    def _kernel(self, r: float) -> float:
        q = r / self.h
        if q > 2.0:
            return 0.0
        return (15.0 / (np.pi * self.h ** 3)) * max(0.0, 1.0 - q) ** 3

    def _kernel_gradient(self, r_ij: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(r_ij)
        if r < 1e-6 or r / self.h > 2.0:
            return np.zeros_like(r_ij)
        q = r / self.h
        return (-45.0 / (np.pi * self.h ** 4)) * (1.0 - q) ** 2 * (r_ij / r)

    def compute_density(self):
        n = len(self.particles)
        self.density = np.zeros(n)
        for i in range(n):
            for j in range(n):
                r_ij = self.particles[i] - self.particles[j]
                r = np.linalg.norm(r_ij)
                self.density[i] += self.mass[j] * self._kernel(r)

    def compute_pressure(self):
        self.pressure = self.gas_constant * (self.density - self.rest_density)

    def compute_forces(self) -> np.ndarray:
        n = len(self.particles)
        forces = np.zeros_like(self.particles)
        for i in range(n):
            f_pressure = np.zeros(2)
            f_viscosity = np.zeros(2)
            for j in range(n):
                if i == j:
                    continue
                r_ij = self.particles[i] - self.particles[j]
                grad = self._kernel_gradient(r_ij)
                f_pressure += -self.mass[j] * (self.pressure[i] + self.pressure[j]) / (2 * self.density[j] + 1e-6) * grad
                f_viscosity += self.mass[j] * (self.velocities[j] - self.velocities[i]) / (self.density[j] + 1e-6) * grad
            f_viscosity *= self.viscosity
            forces[i] = f_pressure + f_viscosity + self.mass[i] * self.gravity
        return forces

    def step(self, dt: float):
        self.compute_density()
        self.compute_pressure()
        forces = self.compute_forces()
        accelerations = forces / (self.density[:, np.newaxis] + 1e-6)
        self.velocities += accelerations * dt
        self.particles += self.velocities * dt
