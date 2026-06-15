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


class RigidBodyDynamics:
    def __init__(self, mass: float = 1.0, inertia: np.ndarray = None, position: np.ndarray = None,
                 orientation: np.ndarray = None):
        self.mass = mass
        self.inertia = inertia if inertia is not None else np.eye(3)
        self.inv_inertia = np.linalg.inv(self.inertia)
        self.position = np.array(position, dtype=np.float64) if position is not None else np.zeros(3)
        self.orientation = np.array(orientation, dtype=np.float64) if orientation is not None else np.array([1.0, 0.0, 0.0, 0.0])
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.force = np.zeros(3)
        self.torque = np.zeros(3)

    def apply_force(self, force: np.ndarray, contact_point: np.ndarray = None):
        self.force += np.array(force, dtype=np.float64)
        if contact_point is not None:
            r = contact_point - self.position
            self.torque += np.cross(r, force)

    def apply_torque(self, torque: np.ndarray):
        self.torque += np.array(torque, dtype=np.float64)

    @staticmethod
    def quaternion_multiply(q: np.ndarray, r: np.ndarray) -> np.ndarray:
        return np.array([
            q[0]*r[0] - q[1]*r[1] - q[2]*r[2] - q[3]*r[3],
            q[0]*r[1] + q[1]*r[0] + q[2]*r[3] - q[3]*r[2],
            q[0]*r[2] - q[1]*r[3] + q[2]*r[0] + q[3]*r[1],
            q[0]*r[3] + q[1]*r[2] - q[2]*r[1] + q[3]*r[0]
        ])

    @staticmethod
    def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def quaternion_normalize(q: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(q)
        return q / n if n > 1e-8 else q

    def rotate_vector(self, v: np.ndarray) -> np.ndarray:
        qv = np.array([0.0, v[0], v[1], v[2]])
        q_conj = self.quaternion_conjugate(self.orientation)
        return self.quaternion_multiply(
            self.quaternion_multiply(self.orientation, qv), q_conj
        )[1:]

    def angular_momentum(self) -> np.ndarray:
        return self.inertia @ self.angular_velocity

    def kinetic_energy(self) -> float:
        linear_ke = 0.5 * self.mass * np.dot(self.linear_velocity, self.linear_velocity)
        angular_ke = 0.5 * np.dot(self.angular_velocity, self.angular_momentum())
        return linear_ke + angular_ke

    def step(self, dt: float):
        linear_accel = self.force / self.mass if self.mass > 0 else np.zeros(3)
        self.linear_velocity += linear_accel * dt
        self.position += self.linear_velocity * dt

        angular_accel = self.inv_inertia @ self.torque - np.cross(
            self.angular_velocity, self.inv_inertia @ self.angular_velocity
        )
        self.angular_velocity += angular_accel * dt
        omega = self.angular_velocity
        omega_q = np.array([0.0, omega[0], omega[1], omega[2]])
        q_dot = 0.5 * self.quaternion_multiply(self.orientation, omega_q)
        self.orientation = self.quaternion_normalize(self.orientation + q_dot * dt)

        self.force = np.zeros(3)
        self.torque = np.zeros(3)


class FluidSimulation:
    def __init__(self, particles: np.ndarray, h: float = 0.1, rest_density: float = 1000.0,
                 gas_constant: float = 2000.0, viscosity: float = 200.0, particle_mass: float = 1.0):
        self.particles = np.array(particles, dtype=np.float64)
        self.h = float(h)
        self.rest_density = rest_density
        self.gas_constant = gas_constant
        self.viscosity = viscosity
        self.particle_mass = particle_mass
        self.n = len(particles)
        self.velocities = np.zeros_like(self.particles)
        self.density = np.zeros(self.n)
        self.pressure = np.zeros(self.n)
        self.gravity = np.array([0.0, -9.81])
        self.surface_tension = 0.0728

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

    def _kernel_laplacian(self, r: float) -> float:
        q = r / self.h
        if q > 2.0:
            return 0.0
        return (90.0 / (np.pi * self.h ** 5)) * (1.0 - q)

    def compute_density(self):
        for i in range(self.n):
            self.density[i] = 0.0
            for j in range(self.n):
                r = np.linalg.norm(self.particles[i] - self.particles[j])
                self.density[i] += self.particle_mass * self._kernel(r)

    def compute_pressure(self):
        self.pressure = self.gas_constant * (self.density - self.rest_density)

    def compute_viscosity_force(self) -> np.ndarray:
        forces = np.zeros_like(self.particles)
        for i in range(self.n):
            f_visc = np.zeros(2)
            for j in range(self.n):
                if i == j:
                    continue
                r_ij = self.particles[i] - self.particles[j]
                grad = self._kernel_gradient(r_ij)
                f_visc += (self.particle_mass / self.density[j]) * (
                    self.velocities[j] - self.velocities[i]
                ) * grad
            forces[i] = self.viscosity * f_visc
        return forces

    def compute_pressure_force(self) -> np.ndarray:
        forces = np.zeros_like(self.particles)
        for i in range(self.n):
            f_press = np.zeros(2)
            for j in range(self.n):
                if i == j:
                    continue
                r_ij = self.particles[i] - self.particles[j]
                grad = self._kernel_gradient(r_ij)
                f_press += -self.particle_mass * (
                    self.pressure[i] + self.pressure[j]
                ) / (2 * self.density[j] + 1e-6) * grad
            forces[i] = f_press
        return forces

    def compute_surface_force(self) -> np.ndarray:
        forces = np.zeros_like(self.particles)
        for i in range(self.n):
            normal = np.zeros(2)
            curvature = 0.0
            for j in range(self.n):
                if i == j:
                    continue
                r_ij = self.particles[i] - self.particles[j]
                r = np.linalg.norm(r_ij)
                grad = self._kernel_gradient(r_ij)
                normal += (self.particle_mass / self.density[j]) * grad
                curvature += self._kernel_laplacian(r)
            n_mag = np.linalg.norm(normal)
            if n_mag > 7.065:
                forces[i] = -self.surface_tension * curvature * (normal / n_mag)
        return forces

    def step(self, dt: float):
        self.compute_density()
        self.compute_pressure()
        f_press = self.compute_pressure_force()
        f_visc = self.compute_viscosity_force()
        f_surf = self.compute_surface_force()
        gravity_force = np.tile(self.gravity * self.particle_mass, (self.n, 1))
        total_force = f_press + f_visc + f_surf + gravity_force
        accel = total_force / (self.particle_mass * self.density[:, np.newaxis] + 1e-6)
        self.velocities += accel * dt
        self.particles += self.velocities * dt


class EconomicSimulation:
    def __init__(self, n_goods: int = 3, n_agents: int = 10):
        self.n_goods = n_goods
        self.n_agents = n_agents
        self.prices = np.ones(n_goods) * 10.0
        self.quantities = np.ones(n_goods) * 50.0
        self.supply = np.ones(n_goods) * 50.0
        self.demand = np.ones(n_goods) * 50.0
        self.agent_utility = np.random.rand(n_agents, n_goods) * 10.0
        self.agent_budget = np.ones(n_agents) * 100.0
        self.price_elasticity = np.ones(n_goods) * 1.5
        self.supply_elasticity = np.ones(n_goods) * 1.2

    def utility(self, agent_id: int, quantities: np.ndarray) -> float:
        alpha = self.agent_utility[agent_id]
        return float(np.sum(alpha * np.log(quantities + 1e-8)))

    def marginal_utility(self, agent_id: int, good: int, quantity: float) -> float:
        return float(self.agent_utility[agent_id, good] / (quantity + 1e-8))

    def demand_function(self, good: int) -> float:
        base_demand = self.agent_utilities_mean(good)
        return base_demand * (self.prices[good] / 10.0) ** (-self.price_elasticity[good])

    def supply_function(self, good: int) -> float:
        return self.supply[good] * (self.prices[good] / 10.0) ** self.supply_elasticity[good]

    def agent_utilities_mean(self, good: int) -> float:
        return float(np.mean(self.agent_utility[:, good]))

    def equilibrium_prices(self) -> np.ndarray:
        prices = self.prices.copy()
        for _ in range(100):
            for g in range(self.n_goods):
                excess_demand = self.demand_function(g) - self.supply_function(g)
                prices[g] *= 1.0 + 0.01 * excess_demand / self.supply[g]
                prices[g] = max(prices[g], 0.1)
        return prices

    def agent_optimal_bundle(self, agent_id: int) -> np.ndarray:
        budget = self.agent_budget[agent_id]
        alpha = self.agent_utility[agent_id]
        total_alpha = np.sum(alpha)
        if total_alpha < 1e-8:
            return np.zeros(self.n_goods)
        ideal = (alpha / total_alpha) * budget / self.prices
        actual_cost = np.sum(ideal * self.prices)
        if actual_cost > budget:
            ideal = ideal * budget / actual_cost
        return ideal

    def step(self, dt: float = 0.1):
        eq_prices = self.equilibrium_prices()
        self.prices += dt * (eq_prices - self.prices)
        for g in range(self.n_goods):
            self.demand[g] = self.demand_function(g)
            self.supply[g] = self.supply_function(g)
            self.quantities[g] += dt * 0.1 * (self.demand[g] - self.supply[g])
            self.quantities[g] = max(self.quantities[g], 1.0)


class SocialSimulation:
    def __init__(self, n_agents: int = 5, belief_dim: int = 3):
        self.n_agents = n_agents
        self.belief_dim = belief_dim
        self.opinions = np.random.uniform(-1, 1, (n_agents, belief_dim))
        self.adjacency = np.ones((n_agents, n_agents)) / n_agents
        self.trust = np.ones((n_agents, n_agents)) / n_agents
        self.confidence = np.ones(n_agents) * 0.5
        self.external_influence = np.zeros((n_agents, belief_dim))

    def set_network(self, adjacency: np.ndarray):
        self.adjacency = adjacency / (adjacency.sum(axis=1, keepdims=True) + 1e-8)

    def degroot_step(self):
        new_opinions = np.zeros_like(self.opinions)
        for i in range(self.n_agents):
            weighted = np.zeros(self.belief_dim)
            for j in range(self.n_agents):
                if self.adjacency[i, j] > 0:
                    diff = np.abs(self.opinions[j] - self.opinions[i])
                    belief_confidence = 1.0 - np.mean(diff) / 2.0
                    weight = self.adjacency[i, j] * max(belief_confidence, 0.0)
                    weighted += weight * self.opinions[j]
            total_weight = np.sum(self.adjacency[i]) + 1e-8
            new_opinions[i] = weighted / total_weight + self.external_influence[i]
        self.opinions = np.clip(new_opinions, -1, 1)

    def bounded_confidence_step(self, epsilon: float = 0.3):
        for i in range(self.n_agents):
            neighbors = []
            for j in range(self.n_agents):
                if i != j:
                    dist = np.linalg.norm(self.opinions[i] - self.opinions[j])
                    if dist < epsilon:
                        neighbors.append(j)
            if neighbors:
                new_op = self.opinions[i].copy()
                for j in neighbors:
                    new_op += self.confidence[i] * (self.opinions[j] - self.opinions[i])
                self.opinions[i] = np.clip(new_op, -1, 1)

    def polarization(self) -> float:
        mean_opinion = np.mean(self.opinions, axis=0)
        return float(np.mean(np.var(self.opinions - mean_opinion, axis=0)))

    def consensus(self) -> float:
        return float(1.0 - self.polarization())

    def step(self, method: str = "degroot"):
        if method == "degroot":
            self.degroot_step()
        elif method == "bounded":
            self.bounded_confidence_step()
        self.external_influence *= 0.9


class BDICycle:
    def __init__(self, n_states: int = 5, n_actions: int = 4):
        self.n_states = n_states
        self.n_actions = n_actions
        self.beliefs = np.ones(n_states) / n_states
        self.desires = np.zeros(n_actions)
        self.intentions = np.zeros(n_actions)
        self.transition = np.random.rand(n_states, n_states)
        self.transition /= self.transition.sum(axis=1, keepdims=True)
        self.reward = np.random.randn(n_states, n_actions)
        self.discount = 0.99
        self.desire_decay = 0.9
        self.intention_threshold = 0.3

    def bayesian_update(self, observation: int):
        likelihood = self.transition[:, observation]
        posterior = self.beliefs * likelihood
        self.beliefs = posterior / (posterior.sum() + 1e-8)

    def evaluate_desires(self, state: np.ndarray = None):
        if state is None:
            state = self.beliefs
        for a in range(self.n_actions):
            expected = np.dot(state, self.reward[:, a])
            self.desires[a] = self.desire_decay * self.desires[a] + (1 - self.desire_decay) * expected

    def filter_intentions(self):
        max_desire = np.max(self.desires)
        threshold = max(self.intention_threshold, max_desire * 0.5)
        self.intentions = np.where(self.desires >= threshold, self.desires, 0.0)

    def select_action(self) -> int:
        if np.sum(self.intentions) > 0:
            probs = self.intentions / (np.sum(self.intentions) + 1e-8)
        else:
            probs = np.ones(self.n_actions) / self.n_actions
        return int(np.random.choice(self.n_actions, p=probs))

    def reason(self, observation: int) -> int:
        self.bayesian_update(observation)
        self.evaluate_desires()
        self.filter_intentions()
        return self.select_action()

    def get_state(self) -> Dict:
        return {
            'beliefs': self.beliefs.tolist(),
            'desires': self.desires.tolist(),
            'intentions': self.intentions.tolist()
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
