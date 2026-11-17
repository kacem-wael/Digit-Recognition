import numpy as np


class MLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.W = [(rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32)
                  for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.mu = np.zeros(sizes[0], np.float32)
        self.sd = np.ones(sizes[0], np.float32)

    def standardize(self, X):
        return (X - self.mu) / self.sd

    def forward(self, X, keep=False):
        a = self.standardize(np.atleast_2d(X).astype(np.float32))
        acts = [a]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            a = np.maximum(z, 0) if i < len(self.W) - 1 else z
            acts.append(a)
        p = softmax(a)
        return (p, acts) if keep else p

    def predict(self, X, batch=8192):
        return np.concatenate([self.forward(X[i:i + batch]) for i in range(0, len(X), batch)])

    def fit(self, X, y, epochs=30, batch=128, lr=1e-3, wd=1e-4, dropout=(0.1, 0.3, 0.2),
            smooth=0.05, val=None, seed=0, log=print):
        rng = np.random.default_rng(seed)
        self.mu = X.mean(0)
        self.sd = np.sqrt(X.var(0) + 1e-2).astype(np.float32)
        Xs = self.standardize(X).astype(np.float32)
        Y = np.full((len(y), self.sizes[-1]), smooth / self.sizes[-1], np.float32)
        Y[np.arange(len(y)), y] += 1 - smooth
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1, b2, t = 0.9, 0.999, 0
        steps = epochs * int(np.ceil(len(X) / batch))
        history = []
        for ep in range(epochs):
            idx = rng.permutation(len(X))
            tot = 0.0
            for s in range(0, len(X), batch):
                j = idx[s:s + batch]
                xb, yb = Xs[j], Y[j]
                acts, masks = [], []
                a = xb
                for i, (W, b) in enumerate(zip(self.W, self.b)):
                    keep = 1 - dropout[i]
                    mk = (rng.random(a.shape, dtype=np.float32) < keep) / keep
                    a = a * mk
                    acts.append(a); masks.append(mk)
                    z = a @ W + b
                    a = np.maximum(z, 0) if i < len(self.W) - 1 else z
                p = softmax(a)
                tot += -np.sum(yb * np.log(p + 1e-9))
                d = (p - yb) / len(j)
                grads_W, grads_b = [None] * len(self.W), [None] * len(self.W)
                for i in reversed(range(len(self.W))):
                    grads_W[i] = acts[i].T @ d + wd * self.W[i]
                    grads_b[i] = d.sum(0)
                    if i:
                        d = (d @ self.W[i].T) * masks[i] * (acts[i] > 0)
                t += 1
                lr_t = lr * 0.5 * (1 + np.cos(np.pi * t / steps))
                for k, (p_, g) in enumerate(zip(params, grads_W + grads_b)):
                    m[k] = b1 * m[k] + (1 - b1) * g
                    v[k] = b2 * v[k] + (1 - b2) * g * g
                    p_ -= lr_t * (m[k] / (1 - b1 ** t)) / (np.sqrt(v[k] / (1 - b2 ** t)) + 1e-8)
            msg = {"epoch": ep + 1, "loss": float(tot / len(X))}
            if val is not None:
                msg["val_acc"] = float((self.predict(val[0]).argmax(1) == val[1]).mean())
            history.append(msg)
            log("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                 for k, v in msg.items()))
        return history

    def save(self, path):
        d = {"sizes": np.array(self.sizes), "mu": self.mu, "sd": self.sd}
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            d[f"W{i}"], d[f"b{i}"] = W, b
        np.savez_compressed(path, **d)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        net = cls(d["sizes"].tolist())
        net.mu, net.sd = d["mu"], d["sd"]
        net.W = [d[f"W{i}"] for i in range(len(net.W))]
        net.b = [d[f"b{i}"] for i in range(len(net.b))]
        return net


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)
