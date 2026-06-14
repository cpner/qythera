import numpy as np
from typing import Optional


class GCNLayer:
    def __init__(self, in_features: int, out_features: int):
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features)

    def forward(self, node_features: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
        n = adjacency.shape[0]
        adj_hat = adjacency + np.eye(n)
        deg = adj_hat.sum(axis=1)
        deg_inv_sqrt = np.where(deg > 0, deg ** -0.5, 0)
        D_inv_sqrt = np.diag(deg_inv_sqrt)
        norm_adj = D_inv_sqrt @ adj_hat @ D_inv_sqrt
        return norm_adj @ node_features @ self.W + self.b


class GraphSAGELayer:
    def __init__(self, in_features: int, out_features: int, num_samples: int = 10):
        self.in_features = in_features
        self.out_features = out_features
        self.num_samples = num_samples
        scale = np.sqrt(2.0 / (2 * in_features + out_features))
        self.W_self = np.random.randn(in_features, out_features) * scale
        self.W_neigh = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features)

    def forward(self, node_features: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
        n = node_features.shape[0]
        out = np.zeros((n, self.out_features))
        for i in range(n):
            neighbors = np.where(adjacency[i] != 0)[0]
            if len(neighbors) == 0:
                neighbor_feat = np.zeros(self.in_features)
            else:
                if len(neighbors) > self.num_samples:
                    indices = np.random.choice(len(neighbors), self.num_samples, replace=False)
                    neighbors = neighbors[indices]
                neighbor_feat = np.mean(node_features[neighbors], axis=0)
            self_feat = node_features[i]
            out[i] = np.maximum(0, self_feat @ self.W_self + neighbor_feat @ self.W_neigh + self.b)
        return out


class GATLayer:
    def __init__(self, in_features: int, out_features: int, num_heads: int = 1):
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        scale = np.sqrt(2.0 / (in_features + self.head_dim))
        self.W = np.random.randn(in_features, out_features) * scale
        self.a = np.random.randn(2 * self.head_dim) * scale
        self.leaky_relu_slope = 0.2

    def forward(self, node_features: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
        n = node_features.shape[0]
        H = node_features @ self.W
        H = H.reshape(n, self.num_heads, self.head_dim)
        out = np.zeros_like(H)
        for i in range(n):
            for h in range(self.num_heads):
                scores = []
                neighbors = np.where(adjacency[i] != 0)[0]
                for j in neighbors:
                    e = np.dot(np.concatenate([H[i, h], H[j, h]]), self.a)
                    e = self.leaky_relu(e)
                    scores.append((j, e))
                if not scores:
                    continue
                attn_vals = np.array([s for _, s in scores])
                attn_weights = np.exp(attn_vals)
                attn_weights = attn_weights / attn_weights.sum()
                for idx, (j, _) in enumerate(scores):
                    out[i, h] += attn_weights[idx] * H[j, h]
        return out.reshape(n, self.out_features)

    @staticmethod
    def leaky_relu(x: float) -> float:
        return x if x > 0 else 0.2 * x


class TransE:
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int = 50):
        self.embedding_dim = embedding_dim
        scale = 0.1
        self.entity_embeddings = np.random.randn(num_entities, embedding_dim) * scale
        self.relation_embeddings = np.random.randn(num_relations, embedding_dim) * scale
        norms = np.linalg.norm(self.entity_embeddings, axis=1, keepdims=True)
        self.entity_embeddings = self.entity_embeddings / np.maximum(norms, 1e-8)

    def forward(self, heads: np.ndarray, relations: np.ndarray, tails: np.ndarray) -> np.ndarray:
        h = self.entity_embeddings[heads]
        r = self.relation_embeddings[relations]
        t = self.entity_embeddings[tails]
        return np.linalg.norm(h + r - t, axis=1)

    def score(self, head: int, relation: int, tail: int) -> float:
        h = self.entity_embeddings[head]
        r = self.relation_embeddings[relation]
        t = self.entity_embeddings[tail]
        return float(np.linalg.norm(h + r - t))

    def train_step(self, heads: np.ndarray, relations: np.ndarray, tails: np.ndarray,
                   neg_heads: np.ndarray = None, neg_tails: np.ndarray = None,
                   margin: float = 1.0, lr: float = 0.01):
        if neg_heads is None:
            neg_heads = heads.copy()
        if neg_tails is None:
            neg_tails = tails.copy()

        pos_scores = self.forward(heads, relations, tails)
        neg_scores = self.forward(neg_heads, relations, neg_tails)
        violations = pos_scores - neg_scores + margin > 0

        if violations.sum() == 0:
            return

        h_pos = self.entity_embeddings[heads[violations]]
        r_pos = self.relation_embeddings[relations[violations]]
        t_pos = self.entity_embeddings[tails[violations]]
        h_neg = self.entity_embeddings[neg_heads[violations]]
        t_neg = self.entity_embeddings[neg_tails[violations]]
        r_use = self.relation_embeddings[relations[violations]]

        pos_diff = h_pos + r_pos - t_pos
        neg_diff = h_neg + r_use - t_neg

        for i in range(len(h_pos)):
            pos_norm = np.linalg.norm(pos_diff[i])
            neg_norm = np.linalg.norm(neg_diff[i])
            if pos_norm > 0 and neg_norm > 0:
                grad_h_pos = pos_diff[i] / pos_norm
                grad_t_pos = -pos_diff[i] / pos_norm
                grad_h_neg = -(neg_diff[i] / neg_norm)
                grad_t_neg = neg_diff[i] / neg_norm

                self.entity_embeddings[heads[violations][i]] -= lr * grad_h_pos
                self.entity_embeddings[tails[violations][i]] -= lr * grad_t_pos
                self.entity_embeddings[neg_heads[violations][i]] -= lr * grad_h_neg
                self.entity_embeddings[neg_tails[violations][i]] -= lr * grad_t_neg
                self.relation_embeddings[relations[violations][i]] -= lr * (grad_h_pos - grad_h_neg)

        norms = np.linalg.norm(self.entity_embeddings, axis=1, keepdims=True)
        self.entity_embeddings = self.entity_embeddings / np.maximum(norms, 1e-8)


class MessagePassing:
    def __init__(self, in_features: int, hidden_features: int, out_features: int, num_layers: int = 2):
        self.num_layers = num_layers
        self.message_weights = []
        self.update_weights = []
        self.layer_norms = []
        scale = np.sqrt(2.0 / in_features)
        self.message_weights.append(np.random.randn(in_features, hidden_features) * scale)
        self.update_weights.append(np.random.randn(in_features + hidden_features, hidden_features) * scale)
        self.layer_norms.append((np.ones(hidden_features), np.zeros(hidden_features)))
        for _ in range(num_layers - 2):
            scale = np.sqrt(2.0 / hidden_features)
            self.message_weights.append(np.random.randn(hidden_features, hidden_features) * scale)
            self.update_weights.append(np.random.randn(hidden_features + hidden_features, hidden_features) * scale)
            self.layer_norms.append((np.ones(hidden_features), np.zeros(hidden_features)))
        scale = np.sqrt(2.0 / hidden_features)
        self.message_weights.append(np.random.randn(hidden_features, out_features) * scale)
        self.update_weights.append(np.random.randn(hidden_features + out_features, out_features) * scale)
        self.layer_norms.append((np.ones(out_features), np.zeros(out_features)))

    def aggregate(self, node_features: np.ndarray, adjacency: np.ndarray, layer_idx: int) -> np.ndarray:
        messages = node_features @ self.message_weights[layer_idx]
        n = node_features.shape[0]
        aggregated = np.zeros_like(messages)
        deg = adjacency.sum(axis=1, keepdims=True)
        deg = np.maximum(deg, 1)
        for i in range(n):
            neighbors = np.where(adjacency[i] != 0)[0]
            if len(neighbors) > 0:
                aggregated[i] = messages[neighbors].mean(axis=0)
        return aggregated

    def update(self, node_features: np.ndarray, aggregated: np.ndarray, layer_idx: int) -> np.ndarray:
        combined = np.concatenate([node_features, aggregated], axis=1)
        updated = combined @ self.update_weights[layer_idx]
        gamma, beta = self.layer_norms[layer_idx]
        mean = updated.mean(axis=1, keepdims=True)
        var = updated.var(axis=1, keepdims=True) + 1e-8
        normalized = (updated - mean) / np.sqrt(var)
        normalized = normalized * gamma + beta
        return np.maximum(0, normalized)

    def forward(self, node_features: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
        h = node_features
        for layer_idx in range(self.num_layers):
            aggregated = self.aggregate(h, adjacency, layer_idx)
            h = self.update(h, aggregated, layer_idx)
        return h
