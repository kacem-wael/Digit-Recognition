import json, os, time
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from features import normalize_digit, extract, GROUPS, N_FEATURES
from nn import MLP

SEED, N_AUG, EPOCHS = 0, 2, 30
rng = np.random.default_rng(SEED)
os.makedirs("model", exist_ok=True)
os.makedirs("figures", exist_ok=True)


def load_mnist():
    if not os.path.exists("data/mnist.npz"):
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True, data_home="data")
        np.savez_compressed("data/mnist.npz", X=X.astype(np.uint8), y=y.astype(np.uint8))
    d = np.load("data/mnist.npz")
    return d["X"].reshape(-1, 28, 28), d["y"].astype(np.int64)


def augment(img):
    big = cv2.resize(img, (56, 56), interpolation=cv2.INTER_LINEAR)
    ang = rng.uniform(-15, 15)
    sc = rng.uniform(0.85, 1.1)
    sh = rng.uniform(-0.3, 0.3)
    M = cv2.getRotationMatrix2D((28, 28), ang, sc)
    S = np.array([[1, sh, -sh * 28], [0, 1, 0], [0, 0, 1]], np.float32)
    M = (np.vstack([M, [0, 0, 1]]) @ S)[:2]
    big = cv2.warpAffine(big, M, (56, 56), flags=cv2.INTER_LINEAR)
    r = rng.random()
    if r < 0.35:
        big = cv2.dilate(big, np.ones((int(rng.integers(2, 5)),) * 2, np.uint8))
    elif r < 0.55:
        big = cv2.erode(big, np.ones((2, 2), np.uint8))
    return big


def prepare(imgs):
    out = np.empty((len(imgs), 28, 28), np.float32)
    asp = np.empty(len(imgs), np.float32)
    for i, im in enumerate(imgs):
        out[i], asp[i] = normalize_digit(im)
    return out, asp


def main():
    t0 = time.time()
    X, y = load_mnist()
    Xtr, ytr, Xte, yte = X[:60000], y[:60000], X[60000:], y[60000:]

    print(f"[1/4] normalizing + augmenting ({N_AUG} extra copies)...")
    tr_imgs, tr_asp = prepare(Xtr.astype(np.float32))
    aug = [prepare([augment(im) for im in Xtr.astype(np.float32)]) for _ in range(N_AUG)]
    te_imgs, te_asp = prepare(Xte.astype(np.float32))

    print(f"[2/4] extracting descriptors ({N_FEATURES} per image)... ({time.time()-t0:.0f}s)")
    Ftr = np.concatenate([extract(tr_imgs, tr_asp)] + [extract(a, b) for a, b in aug])
    Ytr = np.concatenate([ytr] * (N_AUG + 1))
    Fte = extract(te_imgs, te_asp)

    print(f"[3/4] training on {len(Ftr)} samples... ({time.time()-t0:.0f}s)")
    net = MLP([N_FEATURES, 128, 64, 10], seed=SEED)
    hist = net.fit(Ftr, Ytr, epochs=EPOCHS, val=(Fte, yte), seed=SEED)
    net.save("model/digit_net.npz")

    print(f"[4/4] evaluating... ({time.time()-t0:.0f}s)")
    pred = net.predict(Fte).argmax(1)
    cm = np.zeros((10, 10), int)
    np.add.at(cm, (yte, pred), 1)
    metrics = {
        "test_accuracy": float((pred == yte).mean()),
        "per_class_accuracy": {str(c): float(cm[c, c] / cm[c].sum()) for c in range(10)},
        "confusion": cm.tolist(),
        "n_train": int(len(Ftr)), "n_test": int(len(yte)),
        "architecture": [N_FEATURES, 128, 64, 10],
        "groups": GROUPS, "history": hist,
        "baseline_hu_pca_knn": 0.4945,
    }
    json.dump(metrics, open("model/metrics.json", "w"), indent=2)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues")
    for i in range(10):
        for j in range(10):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if i == j else "black")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Neural network, test accuracy {metrics['test_accuracy']:.2%}")
    plt.tight_layout(); plt.savefig("figures/nn_confusion.png", dpi=200); plt.close()

    print(f"\nTest accuracy: {metrics['test_accuracy']:.4f}   (Hu+PCA+KNN baseline: 0.4945)")
    for c, a in metrics["per_class_accuracy"].items():
        print(f"  digit {c}: {a:.4f}")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
