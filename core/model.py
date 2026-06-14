import numpy as np, math

class RMSNorm:
    def __init__(self, dim, eps=1e-6):
        self.w = np.ones(dim, dtype=np.float32)
        self.eps = eps
    def forward(self, x):
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.w

class Attention:
    def __init__(self, dim, nheads, nkv=None, maxseq=2048):
        self.nh, self.nkv, self.hd = nheads, nkv or nheads, dim // nheads
        s = 1.0 / math.sqrt(self.hd)
        self.wq = np.random.randn(dim, self.nh * self.hd).astype(np.float32) * s
        self.wk = np.random.randn(dim, self.nkv * self.hd).astype(np.float32) * s
        self.wv = np.random.randn(dim, self.nkv * self.hd).astype(np.float32) * s
        self.wo = np.random.randn(self.nh * self.hd, dim).astype(np.float32) * s
        t = np.arange(maxseq * 2, dtype=np.float32)
        f = 1.0 / (10000 ** (np.arange(0, self.hd, 2).astype(np.float32) / self.hd))
        self.freqs = np.stack([np.cos(np.outer(t, f)), np.sin(np.outer(t, f))], axis=-1)
    def forward(self, x, kv=None, pos=0):
        B, L, _ = x.shape
        q = (x @ self.wq).reshape(B, L, self.nh, self.hd).transpose(0,2,1,3)
        k = (x @ self.wk).reshape(B, L, self.nkv, self.hd).transpose(0,2,1,3)
        v = (x @ self.wv).reshape(B, L, self.nkv, self.hd).transpose(0,2,1,3)
        fr = self.freqs[pos:pos+L]
        c, s = fr[:,:,0], fr[:,:,1]
        q1, q2 = q[...,:self.hd//2], q[...,self.hd//2:]
        k1, k2 = k[...,:self.hd//2], k[...,self.hd//2:]
        q = np.stack([q1*c - q2*s, q1*s + q2*c], axis=-1).reshape(B, self.nh, L, self.hd)
        k = np.stack([k1*c - k2*s, k1*s + k2*c], axis=-1).reshape(B, self.nkv, L, self.hd)
        if kv is not None:
            k = np.concatenate([kv[0], k], axis=2)
            v = np.concatenate([kv[1], v], axis=2)
        nc = (k, v)
        if self.nkv != self.nh:
            r = self.nh // self.nkv
            k = np.repeat(k, r, axis=1)
            v = np.repeat(v, r, axis=1)
        sc = math.sqrt(self.hd)
        a = (q @ k.transpose(0,1,3,2)) / sc
        tl = k.shape[2]
        mask = np.triu(np.full((L, tl), -1e9), k=tl - L + 1)
        a = np.exp(a + mask - (a + mask).max(axis=-1, keepdims=True))
        a = a / (a.sum(axis=-1, keepdims=True) + 1e-8)
        o = (a @ v).transpose(0,2,1,3).reshape(B, L, -1) @ self.wo
        return o, nc

class SwiGLU:
    def __init__(self, dim, ffd):
        s = 1.0 / math.sqrt(dim)
        self.w1 = np.random.randn(dim, ffd).astype(np.float32) * s
        self.w2 = np.random.randn(ffd, dim).astype(np.float32) * s
        self.w3 = np.random.randn(dim, ffd).astype(np.float32) * s
    def forward(self, x): return (np.maximum(0, x @ self.w1) * (x @ self.w3)) @ self.w2

class Layer:
    def __init__(self, dim, nh, ffd, nkv=None, maxseq=2048):
        self.n1 = np.ones(dim, dtype=np.float32)
        self.n2 = np.ones(dim, dtype=np.float32)
        self.attn = Attention(dim, nh, nkv, maxseq)
        self.ffn = SwiGLU(dim, ffd)
    def forward(self, x, kv=None, pos=0):
        h = x / (np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + 1e-6)) * self.n1
        o, nc = self.attn.forward(h, kv, pos)
        x = x + o
        h = x / (np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + 1e-6)) * self.n2
        return x + self.ffn.forward(h), nc

class Model:
    def __init__(self, vs=32000, dm=256, nh=8, nl=6, ffd=1024, nkv=4, ms=2048):
        s = 1.0 / math.sqrt(dm)
        self.emb = np.random.randn(vs, dm).astype(np.float32) * s
        self.pos = np.random.randn(ms, dm).astype(np.float32) * s * 0.1
        self.layers = [Layer(dm, nh, ffd, nkv, ms) for _ in range(nl)]
        self.fn = np.ones(dm, dtype=np.float32)
        self.head = np.random.randn(dm, vs).astype(np.float32) * s
        self.np = sum(p.size for l in self.layers for p in [l.attn.wq, l.attn.wk, l.attn.wv, l.attn.wo, l.ffn.w1, l.ffn.w2, l.ffn.w3])
        self.np += self.emb.size + self.head.size
    def forward(self, ids, kv=None):
        B, L = ids.shape
        x = self.emb[ids] + self.pos[:L]
        nkc = []
        for i, l in enumerate(self.layers):
            x, nc = l.forward(x, kv[i] if kv else None, 0)
            nkc.append(nc)
        x = x / (np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + 1e-6)) * self.fn
        return x @ self.head, nkc
    def generate(self, ids, mx=256, t=0.8, k=50, p=0.9):
        g = list(ids)
        kc = [None] * len(self.layers)
        for _ in range(mx):
            inp = np.array([g[-2048:]], dtype=np.int64)
            logits, kc = self.forward(inp, kc)
            nl = logits[0, -1] / max(t, 0.1)
            if k > 0:
                th = np.sort(nl)[-min(k, len(nl))]
                nl[nl < th] = -1e9
            if p < 1.0:
                si = np.argsort(nl)[::-1]
                sl = nl[si].copy()
                cp = np.cumsum(np.exp(sl) / (np.exp(sl).sum() + 1e-8))
                rm = cp > p; rm[1:] = rm[:-1]; rm[0] = False
                sl[rm] = -1e9
                nl[si] = sl
            prob = np.exp(nl - nl.max())
            prob = prob / (prob.sum() + 1e-8)
            g.append(int(np.random.choice(len(prob), p=prob)))
        return g
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        np.savez(os.path.join(path, "model.npz"), emb=self.emb, pos=self.pos, fn=self.fn, head=self.head,
                 **{f"L{i}_{k}": v for i, l in enumerate(self.layers) for k, v in [("n1",l.n1),("n2",l.n2),
                 ("wq",l.attn.wq),("wk",l.attn.wk),("wv",l.attn.wv),("wo",l.attn.wo),
                 ("w1",l.ffn.w1),("w2",l.ffn.w2),("w3",l.ffn.w3)]})
    @classmethod
    def load(cls, path):
        d = np.load(os.path.join(path, "model.npz"))
        nl = sum(1 for k in d if k.startswith("L0_"))
        m = cls(vs=d["emb"].shape[0], dm=d["emb"].shape[1], nl=nl)
        m.emb, m.pos, m.fn, m.head = d["emb"], d["pos"], d["fn"], d["head"]
        for i in range(nl):
            m.layers[i].n1, m.layers[i].n2 = d[f"L{i}_n1"], d[f"L{i}_n2"]
            m.layers[i].attn.wq, m.layers[i].attn.wk = d[f"L{i}_wq"], d[f"L{i}_wk"]
            m.layers[i].attn.wv, m.layers[i].attn.wo = d[f"L{i}_wv"], d[f"L{i}_wo"]
            m.layers[i].ffn.w1, m.layers[i].ffn.w2, m.layers[i].ffn.w3 = d[f"L{i}_w1"], d[f"L{i}_w2"], d[f"L{i}_w3"]
        return m
